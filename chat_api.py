# chat_api.py — CLIP search (lazy init) + Firestore (lazy init) + robust fallbacks
from __future__ import annotations
import base64, io, os, time, json
from typing import Any, Dict, List, Tuple

import numpy as np
from PIL import Image
from flask import Blueprint, request, jsonify
from runpod_client import run_query

# ---------------- ML optional (Torch/Transformers) ----------------
# Turn on by setting ENABLE_ML=1 in your environment (Render Settings → Environment)
ENABLE_ML = os.getenv("ENABLE_ML", "0") == "1"

# Flags & placeholders
_ML_AVAILABLE = False
_ML_ERROR: Exception | None = None
torch = None  # type: ignore
CLIPModel = None  # type: ignore
CLIPProcessor = None  # type: ignore
_DEVICE = None

if ENABLE_ML:
    try:
        import torch  # type: ignore
        from transformers import CLIPModel as _CLIPModel, CLIPProcessor as _CLIPProcessor  # type: ignore
        CLIPModel = _CLIPModel
        CLIPProcessor = _CLIPProcessor
        _DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _ML_AVAILABLE = True
    except Exception as e:
        _ML_AVAILABLE = False
        _ML_ERROR = e

# ---------------- Firebase (lazy) ----------------
_db = None
_db_err: Exception | None = None

def get_db():
    """
    Initialize Firestore from one of:
      1) FIREBASE_KEY = inline JSON (recommended on Railway)
      2) FIREBASE_KEY = file path to JSON
      3) GOOGLE_APPLICATION_CREDENTIALS = file path to JSON
    """
    global _db, _db_err
    if _db is not None or _db_err is not None:
        return _db

    try:
        import firebase_admin
        from firebase_admin import credentials, firestore

        raw = os.getenv("FIREBASE_KEY", "").strip()
        cred = None

        if raw:
            # If looks like JSON, parse; otherwise treat as a path
            if raw.startswith("{"):
                data = json.loads(raw)
                cred = credentials.Certificate(data)
            else:
                if not os.path.exists(raw):
                    raise RuntimeError(f"FIREBASE_KEY path not found: {raw}")
                cred = credentials.Certificate(raw)
        else:
            # Fallback to GOOGLE_APPLICATION_CREDENTIALS path
            gac = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "").strip()
            if not gac or not os.path.exists(gac):
                raise RuntimeError("FIREBASE_KEY not set, and GOOGLE_APPLICATION_CREDENTIALS missing or not found")
            cred = credentials.Certificate(gac)

        if not firebase_admin._apps:
            firebase_admin.initialize_app(cred)

        _db = firestore.client()
        return _db

    except Exception as e:
        _db_err = e
        return None



def _product_collection():
    db = get_db()
    if db is None:
        return None
    from firebase_admin import firestore as _fs  # type: ignore
    return db.collection("app").document("products").collection("items")

# ---------------- CLIP (lazy) ----------------
_MODEL_ID = "openai/clip-vit-base-patch32"  # 512-dim
_clip_model = None
_clip_proc = None

def _ensure_clip():
    if not _ML_AVAILABLE:
        raise RuntimeError("ML features are disabled on this deployment")
    global _clip_model, _clip_proc
    if _clip_model is None or _clip_proc is None:
        # Types ignored because we may run with ML disabled (placeholders above)
        _clip_model = CLIPModel.from_pretrained(_MODEL_ID).to(_DEVICE).eval()  # type: ignore
        _clip_proc  = CLIPProcessor.from_pretrained(_MODEL_ID)  # type: ignore

def _embed_text(text: str) -> np.ndarray | None:
    if not _ML_AVAILABLE or not text or not text.strip():
        return None
    _ensure_clip()
    ins = _clip_proc(text=[text], return_tensors="pt", padding=True, truncation=True)  # type: ignore
    ins = {k: v.to(_DEVICE) for k, v in ins.items()}  # type: ignore
    with torch.no_grad():  # type: ignore
        z = _clip_model.get_text_features(**ins)  # type: ignore
    z = z / z.norm(p=2, dim=-1, keepdim=True)  # type: ignore
    return z.squeeze(0).detach().cpu().numpy().astype("float32")  # type: ignore

def _embed_image_b64(img_b64: str) -> np.ndarray | None:
    if not _ML_AVAILABLE or not img_b64:
        return None
    try:
        pil = Image.open(io.BytesIO(base64.b64decode(img_b64))).convert("RGB")
    except Exception:
        return None
    _ensure_clip()
    ins = _clip_proc(images=[pil], return_tensors="pt")  # type: ignore
    ins = {k: v.to(_DEVICE) for k, v in ins.items()}  # type: ignore
    with torch.no_grad():  # type: ignore
        z = _clip_model.get_image_features(**ins)  # type: ignore
    z = z / z.norm(p=2, dim=-1, keepdim=True)  # type: ignore
    return z.squeeze(0).detach().cpu().numpy().astype("float32")  # type: ignore

# ---------------- Cache: products + embeddings ----------------
_CACHE = {"ts": 0.0, "arr": np.zeros((0,512), dtype="float32"), "items": []}

def _load_products() -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    now = time.time()
    if _CACHE["arr"] is not None and now - _CACHE["ts"] < 60:
        return _CACHE["arr"], _CACHE["items"]

    col = _product_collection()
    if col is None:
        _CACHE.update({"ts": now, "arr": np.zeros((0,512), dtype="float32"), "items": []})
        return _CACHE["arr"], _CACHE["items"]

    docs = list(col.limit(5000).stream())
    items, vecs = [], []
    for d in docs:
        x = d.to_dict() or {}
        item = {
            "id": x.get("id") or d.id,
            "title": x.get("title") or "",
            "brand": x.get("brand") or "",
            "price_text": x.get("price_text") or x.get("price") or "",
            "product_url": x.get("product_url") or x.get("url") or "",
            "image_url": x.get("image_url") or x.get("cropped_image_url") or "",
            "color_distribution": x.get("color_distribution") or {},
            "traits": x.get("traits") or {},
        }
        items.append(item)

        emb = x.get("clip_embedding")
        if isinstance(emb, list) and len(emb) == 512:
            vecs.append(emb)

    arr = np.asarray(vecs, dtype="float32")
    if arr.size:
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        arr = arr / np.clip(norms, 1e-8, None)

    _CACHE.update({"ts": now, "arr": arr, "items": items})
    return arr, items


def _cosine_topk(q: np.ndarray | None, arr: np.ndarray, k: int = 10):
    if arr is None or arr.size == 0 or q is None:
        return [], []
    sims = arr @ q
    k = min(k, arr.shape[0])
    idx = np.argpartition(-sims, k-1)[:k]
    idx = idx[np.argsort(-sims[idx])]
    return idx.tolist(), sims[idx].tolist()

def _apply_filters(cands, sims, filters):
    colors = {c.lower() for c in (filters.get("colors") or [])}
    brand  = (filters.get("brand") or "").lower().strip()
    max_price = filters.get("max_price")

    def price_to_float(p):
        if not p: return None
        s = "".join(ch for ch in str(p) if ch.isdigit() or ch==".")
        try: return float(s) if s else None
        except: return None

    out = []
    for item, sim in zip(cands, sims):
        bonus = 0.0
        if colors and isinstance(item.get("color_distribution"), dict):
            cd = {k.lower(): float(v) for k, v in item["color_distribution"].items() if isinstance(v,(int,float))}
            color_score = sum(cd.get(c, 0.0) for c in colors)
            bonus += 0.10 * color_score
        if brand and brand in (item.get("brand") or "").lower():
            bonus += 0.05
        score = float(sim) + bonus
        if max_price is not None:
            p = price_to_float(item.get("price_text",""))
            if p is None or p > max_price:
                continue
        out.append((score, item))
    out.sort(key=lambda x: x[0], reverse=True)
    return [dict(item, _score=round(score, 4)) for score, item in out]

def _get_ref_embedding(ref_id: str) -> np.ndarray | None:
    if not ref_id: return None
    col = _product_collection()
    if col is None:
        return None
    snap = col.document(ref_id).get()
    if not snap.exists: return None
    x = snap.to_dict() or {}
    emb = x.get("clip_embedding")
    if isinstance(emb, list) and len(emb) == 512:
        v = np.asarray(emb, dtype="float32")
        n = np.linalg.norm(v)
        return v / max(n, 1e-8)
    return None

bp = Blueprint("chat_api", __name__)

from runpod_client import run_query


def _ml_disabled_response():
    data = request.get_json(force=True) or {}
    try:
        result = run_query(data)
        return jsonify({
            "ok": True,
            "reply": result.get("reply") or "Here are some ideas.",
            "items": result.get("items") or []
        })
    except Exception as e:
        msg = "Semantic search is temporarily disabled and RunPod call failed."
        return jsonify({"ok": False, "reply": msg, "items": [], "error": str(e)}), 503

@bp.route("/ping")
def ping():
    return {"ok": True, "ml": _ML_AVAILABLE, "ml_error": str(_ML_ERROR) if _ML_ERROR else None}

@bp.route("/chat", methods=["POST"])
def chat_text_only():
    data = request.get_json(force=True) or {}
    msg = (data.get("message") or "").strip()

    if not _ML_AVAILABLE:
        return _ml_disabled_response()


    # Normal ML path
    top_k = int(data.get("top_k") or 8)
    filters = data.get("filters") or {}
    if not msg:
        return jsonify({"reply":"Tell me what you’re looking for.", "items":[]})
    arr, items = _load_products()
    if not getattr(arr, "size", 0):
        return jsonify({"reply":"Database not ready yet—try again in a moment.", "items":[]}), 503
    q = _embed_text(msg)
    idxs, sims = _cosine_topk(q, arr, k=max(top_k*3, top_k))
    cands = [items[i] for i in idxs]
    ranked = _apply_filters(cands, [float(s) for s in sims], filters)[:top_k]
    return jsonify({"reply": f'I looked for: “{msg}”.', "items": ranked})

    data = request.get_json(force=True) or {}
    msg = (data.get("message") or "").strip()
    top_k = int(data.get("top_k") or 8)
    filters = data.get("filters") or {}
    if not msg:
        return jsonify({"reply":"Tell me what you’re looking for.", "items":[]})
    arr, items = _load_products()
    if not getattr(arr, "size", 0):
        return jsonify({"reply":"Database not ready yet—try again in a moment.", "items":[]}), 503
    q = _embed_text(msg)
    idxs, sims = _cosine_topk(q, arr, k=max(top_k*3, top_k))
    cands = [items[i] for i in idxs]
    ranked = _apply_filters(cands, [float(s) for s in sims], filters)[:top_k]
    return jsonify({"reply": f'I looked for: “{msg}”.', "items": ranked})

@bp.route("/chat/messages", methods=["POST"])
def chat_messages():
    """
    Body:
    {
      "message": "blue denim jacket under $40",
      "image_base64": "<optional>",
      "ref_product_id": "<optional>",
      "top_k": 8,
      "filters": {"max_price": 40, "colors": ["blue"], "brand": "hm"}
    }
    """
    if not _ML_AVAILABLE:
        return _ml_disabled_response()

    data = request.get_json(force=True) or {}
    text = (data.get("message") or "").strip()
    img_b64 = data.get("image_base64")
    ref_id = data.get("ref_product_id")
    top_k = int(data.get("top_k") or 8)
    filters = data.get("filters") or {}

    e_text = _embed_text(text) if text else None
    e_img  = _embed_image_b64(img_b64) if img_b64 else None
    e_ref  = _get_ref_embedding(ref_id) if ref_id else None

    parts, weights = [], []
    if e_ref is not None and e_img is not None and e_text is not None:
        parts, weights = [e_ref, e_img, e_text], [0.55, 0.25, 0.20]
    elif e_ref is not None and e_text is not None:
        parts, weights = [e_ref, e_text], [0.70, 0.30]
    elif e_img is not None and e_text is not None:
        parts, weights = [e_img, e_text], [0.60, 0.40]
    elif e_ref is not None:
        parts, weights = [e_ref], [1.0]
    elif e_img is not None:
        parts, weights = [e_img], [1.0]
    elif e_text is not None:
        parts, weights = [e_text], [1.0]
    else:
        return jsonify({"reply":"Tell me what you’re looking for or upload a photo.", "items":[]})

    q = np.zeros(512, dtype="float32")
    for p, w in zip(parts, weights):
        q += w * p
    n = np.linalg.norm(q)
    if n > 0: q = q / n

    arr, items = _load_products()
    if not getattr(arr, "size", 0):
        return jsonify({"reply":"Database not ready yet—try again in a moment.", "items":[]}), 503

    idxs, sims = _cosine_topk(q, arr, k=max(top_k*3, top_k))
    cands = [items[i] for i in idxs]
    ranked = _apply_filters(cands, [float(s) for s in sims], filters)[:top_k]

    bits = []
    if text: bits.append(f'“{text}”')
    if ref_id: bits.append(f'like item {ref_id}')
    if img_b64: bits.append('your photo')
    looked = " + ".join(bits) if bits else "your request"
    if filters.get("max_price") is not None:
        try:
            looked += f' under ${float(filters["max_price"]):.0f}'
        except:
            pass
    if filters.get("colors"):
        looked += f' favoring {", ".join(filters["colors"])}'
    if filters.get("brand"):
        looked += f' and brand {filters["brand"]}'
    reply = f"I searched for {looked}. "
    reply += ("No good matches yet." if not ranked else f"Top match: {ranked[0].get('title') or 'Untitled'}.")
    return jsonify({"reply": reply, "items": ranked})
