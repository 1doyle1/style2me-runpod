# handler.py — RunPod GPU handler for StyleSnap
import os, json, numpy as np, torch
from PIL import Image
import firebase_admin
from firebase_admin import credentials, firestore
from transformers import CLIPModel, CLIPProcessor

# ---------------- Firestore init ----------------
cred = None
raw = os.getenv("FIREBASE_KEY", "").strip()
if raw.startswith("{"):
    cred = credentials.Certificate(json.loads(raw))
elif os.path.exists(raw):
    cred = credentials.Certificate(raw)
else:
    raise RuntimeError("FIREBASE_KEY missing or invalid")

if not firebase_admin._apps:
    firebase_admin.initialize_app(cred)
db = firestore.client()

# ---------------- CLIP init ----------------
MODEL_ID = "openai/clip-vit-base-patch32"
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_clip_model = CLIPModel.from_pretrained(MODEL_ID).to(_device).eval()
_clip_proc = CLIPProcessor.from_pretrained(MODEL_ID)

def _product_collection():
    return db.collection("app").document("products").collection("items")

def _load_products():
    docs = list(_product_collection().limit(2000).stream())
    items, vecs = [], []
    for d in docs:
        x = d.to_dict() or {}
        emb = x.get("clip_embedding")
        if isinstance(emb, list) and len(emb) == 512:
            items.append({
                "id": x.get("id") or d.id,
                "title": x.get("title") or "",
                "brand": x.get("brand") or "",
                "price_text": x.get("price_text") or x.get("price") or "",
                "product_url": x.get("product_url") or x.get("url") or "",
                "image_url": x.get("image_url") or x.get("cropped_image_url") or "",
            })
            vecs.append(emb)
    arr = np.asarray(vecs, dtype="float32")
    if arr.size:
        arr = arr / np.clip(np.linalg.norm(arr, axis=1, keepdims=True), 1e-8, None)
    return arr, items

def _embed_text(text: str):
    ins = _clip_proc(text=[text], return_tensors="pt", padding=True, truncation=True)
    ins = {k: v.to(_device) for k,v in ins.items()}
    with torch.no_grad():
        z = _clip_model.get_text_features(**ins)
    z = z / z.norm(p=2, dim=-1, keepdim=True)
    return z.squeeze(0).cpu().numpy().astype("float32")

def _cosine_topk(q, arr, k=8):
    sims = arr @ q
    idx = np.argpartition(-sims, k-1)[:k]
    idx = idx[np.argsort(-sims[idx])]
    return idx.tolist(), sims[idx].tolist()

def handler(event):
    data = event.get("input") or {}
    query = (data.get("message") or "").strip()
    top_k = int(data.get("top_k") or 8)
    if not query:
        return {"reply": "Tell me what you’re looking for.", "items": []}
    arr, items = _load_products()
    if not arr.size or not items:
        return {"reply": "Database is empty.", "items": []}
    q = _embed_text(query)
    idxs, sims = _cosine_topk(q, arr, k=min(top_k, len(items)))
    cands = [items[i] for i in idxs]
    return {"reply": f"Here are some results for '{query}'.", "items": cands}
