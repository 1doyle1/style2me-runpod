# handler.py — RunPod serverless entrypoint using agent_orchestrator
import runpod
from agent_orchestrator import run as agent_run

def handler(event):
    data = event.get("input") or {}
    # Accept either a whole conversation or just a single message
    messages = data.get("messages")
    if not messages:
        msg = (data.get("message") or "").strip()
        if msg:
            messages = [{"role": "user", "content": msg}]
        else:
            return {"reply": "No input provided.", "items": []}

    try:
        reply, items = agent_run(messages)
        return {"reply": reply, "items": items}
    except Exception as e:
        return {"reply": "Agent error.", "error": str(e), "items": []}

# Start RunPod serverless loop
runpod.serverless.start({"handler": handler})
