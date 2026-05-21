import os
import time
from typing import Any, Optional

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

BACKEND_URL = os.environ["BACKEND_URL"].rstrip("/")  # e.g. http://vllm-large:8000/v1
MODEL_NAME = os.environ["MODEL_NAME"]

app = FastAPI(title="Altos Router")


# --------------------- Simple RESTful endpoint ---------------------

class ChatRequest(BaseModel):
    question: str
    max_tokens: Optional[int] = 1024
    temperature: Optional[float] = 0.7


class ChatResponse(BaseModel):
    answer: str
    model_used: str
    elapsed_ms: int


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    start = time.perf_counter()
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": req.question}],
        "max_tokens": req.max_tokens,
        "temperature": req.temperature,
    }
    async with httpx.AsyncClient(timeout=600.0) as client:
        r = await client.post(f"{BACKEND_URL}/chat/completions", json=payload)
        if r.status_code >= 400:
            raise HTTPException(status_code=r.status_code, detail=r.text)
        data = r.json()

    return ChatResponse(
        answer=data["choices"][0]["message"]["content"],
        model_used=MODEL_NAME,
        elapsed_ms=int((time.perf_counter() - start) * 1000),
    )


# --------------------- OpenAI-compatible endpoints ---------------------

def _model_entry(model_id: str, created: int) -> dict:
    return {
        "id": model_id,
        "object": "model",
        "created": created,
        "owned_by": "vllm",
        "permission": [
            {
                "id": f"perm-{model_id}",
                "object": "model_permission",
                "created": created,
                "allow_create_engine": False,
                "allow_sampling": True,
                "allow_logprobs": False,
                "allow_search_indices": False,
                "allow_view": True,
                "allow_fine_tuning": False,
                "organization": "*",
                "group": None,
                "is_blocking": False,
            }
        ],
        "root": model_id,
        "parent": None,
    }


@app.get("/v1/models")
async def list_models():
    now = int(time.time())
    return {"object": "list", "data": [_model_entry(MODEL_NAME, now)]}


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    payload: dict[str, Any] = await request.json()
    forwarded = dict(payload)
    forwarded["model"] = MODEL_NAME  # 永遠轉送到唯一後端
    stream = bool(payload.get("stream", False))

    if not stream:
        async with httpx.AsyncClient(timeout=600.0) as client:
            r = await client.post(f"{BACKEND_URL}/chat/completions", json=forwarded)
            if r.status_code >= 400:
                raise HTTPException(status_code=r.status_code, detail=r.text)
            return r.json()

    async def event_stream():
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream("POST", f"{BACKEND_URL}/chat/completions", json=forwarded) as r:
                if r.status_code >= 400:
                    body = await r.aread()
                    yield f"data: {body.decode('utf-8', 'replace')}\n\n".encode()
                    return
                async for chunk in r.aiter_raw():
                    if chunk:
                        yield chunk

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/health")
async def health():
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            r = await client.get(f"{BACKEND_URL}/models")
            backend_ok = r.status_code < 500
    except Exception:
        backend_ok = False
    return {"status": "ok", "backend": backend_ok, "model": MODEL_NAME}


@app.get("/")
async def root():
    return {
        "service": "altos-router",
        "endpoints": ["/chat", "/v1/chat/completions", "/v1/models", "/health"],
        "model": MODEL_NAME,
    }
