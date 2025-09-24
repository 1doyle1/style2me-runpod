from __future__ import annotations
import json, os, re, pprint
from typing import Any, Dict, List, Tuple
from openai import OpenAI

# pull helpers from chat_api (catalog + embeddings + filters)
from chat_api import _load_products, _embed_text, _cosine_topk, _apply_filters

def _load_openai_client() -> OpenAI:
    raw = os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENAI_APIKEY") or ""
    key = raw.strip()
    if not key.startswith("sk-"):
        head = (key[:6] + "…") if key else "<empty>"
        raise RuntimeError(f"OPENAI_API_KEY not set or malformed (got: {head}).")
    print(f"[openai] using key: {key[:8]}…{key[-6:]}")
    return OpenAI(api_key=key)

client = _load_openai_client()

SYSTEM_PROMPT = """You are StyleSnap — a friendly assistant who can also help with fashion.
Keep replies short (1–4 sentences). Only search products when the user clearly wants items."""

# -------- simple keyword fallback --------
def _simple_keyword_search(query: str, items: List[Dict[str, Any]], top_k: int, filters: Dict[str, Any]):
    q = (query or "").lower().strip()
    if not q or not items:
        return []
    toks = [t for t in re.split(r"[^a-z0-9]+", q) if t]
    def score(it):
        s = 0
        text = f"{(it.get('title') or '').lower()} {(it.get('brand') or '').lower()}"
        for t in toks:
            if t in text: s += 1
        return s
    ranked = sorted(items, key=score, reverse=True)
    ranked = _apply_filters(ranked[:max(top_k*5, top_k)], [1.0]*max(top_k*5, top_k), filters)[:top_k]
    return ranked

def tool_search_similar(query: str, top_k: int = 8, filters: Dict[str, Any] | None = None):
    filters = filters or {}
    arr, items = _load_products()

    q = _embed_text(query or "")
    if q is not None and getattr(arr, "size", 0):
        idxs, sims = _cosine_topk(q, arr, k=max(top_k*3, top_k))
        cands = [items[i] for i in idxs]
        return _apply_filters(cands, [float(s) for s in sims], filters)[:top_k]

    return _simple_keyword_search(query or "", items, top_k, filters)

def llm_complete(messages: List[Dict[str, str]], tools: List[Dict[str, Any]] | None = None):
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    kwargs = {
        "model": model,
        "temperature": 0.5,
        "messages": messages,
    }
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "auto"
    return client.chat.completions.create(**kwargs)

def run(messages: List[Dict[str, str]]) -> Tuple[str, List[Dict[str, Any]]]:
    tools = [{
        "type": "function",
        "function": {
            "name": "search_similar",
            "description": "Search the product catalog (semantic if enabled; keyword fallback otherwise).",
            "parameters": {
                "type": "object",
                "properties": {
                    "query":   {"type": "string"},
                    "top_k":   {"type": "integer", "default": 8},
                    "filters": {"type": "object"}
                },
                "required": ["query"]
            }
        }
    }]

    sys = {"role": "system", "content": SYSTEM_PROMPT}

    # First pass
    r1 = llm_complete([sys, *messages], tools=tools)
    choice = r1.choices[0]
    msg = getattr(choice, "message", None)

    if not msg or getattr(msg, "tool_calls", None) in (None, []):
        text = (getattr(msg, "content", None) or getattr(choice, "text", None) or "Got it.").strip()
        return text, []

    # Handle tool calls
    tool_calls = msg.tool_calls or []
    items: List[Dict[str, Any]] = []
    follow_messages: List[Dict[str, str]] = [sys, *messages]

    for call in tool_calls:
        try:
            if getattr(call, "type", "") != "function":
                continue
            fn = call.function
            name = getattr(fn, "name", "")
            argstr = getattr(fn, "arguments", "") or "{}"
            try:
                args = json.loads(argstr)
            except Exception:
                last_user = next((m["content"] for m in reversed(messages) if m.get("role")=="user" and m.get("content")), "")
                args = {"query": last_user, "top_k": 8, "filters": {}}

            if name == "search_similar":
                q = (args.get("query") or "").strip()
                top_k = int(args.get("top_k") or 8)
                filters = args.get("filters") or {}
                if not q:
                    last_user = next((m["content"] for m in reversed(messages) if m.get("role")=="user" and m.get("content")), "")
                    q = last_user.strip()

                try:
                    items = tool_search_similar(q, top_k=top_k, filters=filters) or []
                except Exception as e:
                    print("[agent] tool_search_similar failed:", repr(e))
                    items = []

                # ✅ Only append tool role if assistant actually gave tool_calls + id
                if getattr(call, "id", None) and getattr(msg, "tool_calls", None):
                    tool_payload = {"ok": True, "query": q, "count": len(items), "items": items}
                    follow_messages.append({
                        "role": "tool",
                        "tool_call_id": call.id,
                        "name": "search_similar",
                        "content": json.dumps(tool_payload, default=str)
                    })
                else:
                    print("[agent] Skipping tool message append — no valid tool_call.id")
        except Exception as e:
            print("[agent] tool_call handler error:", repr(e))

    # Debug log payload before second pass
    print("[DEBUG] follow_messages before second pass:")
    pprint.pprint(follow_messages)

    # Second pass
    try:
        r2 = llm_complete(follow_messages, tools=None)
        text = (r2.choices[0].message.content or "Here are some ideas.").strip()
    except Exception as e:
        print("[agent] second completion failed:", repr(e))
        text = "Here are some ideas."
    return text, items
