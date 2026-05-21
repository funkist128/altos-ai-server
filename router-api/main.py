import json
import os
import time
import uuid
from typing import Any, Optional

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

LARGE_URL = os.environ["LARGE_MODEL_URL"].rstrip("/")
SMALL_URL = os.environ["SMALL_MODEL_URL"].rstrip("/")
LARGE_NAME = os.environ["LARGE_MODEL_NAME"]
SMALL_NAME = os.environ["SMALL_MODEL_NAME"]

ROUTE_LARGE = "LARGE"
ROUTE_SMALL = "SMALL"

CLASSIFIER_SYSTEM = (
    "You are a routing classifier. You must reply with exactly one word: "
    "either LARGE or SMALL. No punctuation, no explanation."
)

CLASSIFIER_USER_TEMPLATE = """Decide which model should answer the user's question.

Reply LARGE if the question needs strong reasoning, such as:
- code generation, debugging, refactoring
- multi-step math, proofs, scientific reasoning
- long-form writing, deep analysis, structured planning
- nuanced translation or complex summarization
- agentic / tool-use planning

Reply SMALL if the question is:
- greetings, small talk, casual chat
- simple factual lookup or definition
- short single-turn Q&A
- formatting / trivial text rewrite

Question:
\"\"\"{question}\"\"\"

Answer with one word only (LARGE or SMALL):"""


app = FastAPI(title="Altos Gemma Router")


async def classify(question: str) -> str:
    """Ask the small model to classify; fall back to SMALL on any failure."""
    payload = {
        "model": SMALL_NAME,
        "messages": [
            {"role": "system", "content": CLASSIFIER_SYSTEM},
            {"role": "user", "content": CLASSIFIER_USER_TEMPLATE.format(question=question[:4000])},
        ],
        "max_tokens": 4,
        "temperature": 0.0,
    }
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            r = await client.post(f"{SMALL_URL}/chat/completions", json=payload)
            r.raise_for_status()
            text = r.json()["choices"][0]["message"]["content"].strip().upper()
    except Exception:
        return ROUTE_SMALL
    return ROUTE_LARGE if "LARGE" in text else ROUTE_SMALL


def target_for(route: str) -> tuple[str, str]:
    if route == ROUTE_LARGE:
        return LARGE_URL, LARGE_NAME
    return SMALL_URL, SMALL_NAME


# --------------------- Simple RESTful endpoint ---------------------

class ChatRequest(BaseModel):
    question: str
    force_model: Optional[str] = None  # "large" | "small" | None
    max_tokens: Optional[int] = 1024
    temperature: Optional[float] = 0.7


class ChatResponse(BaseModel):
    answer: str
    model_used: str
    route: str
    elapsed_ms: int


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    start = time.perf_counter()

    if req.force_model in ("large", "small"):
        route = ROUTE_LARGE if req.force_model == "large" else ROUTE_SMALL
    else:
        route = await classify(req.question)

    url, name = target_for(route)
    payload = {
        "model": name,
        "messages": [{"role": "user", "content": req.question}],
        "max_tokens": req.max_tokens,
        "temperature": req.temperature,
    }
    async with httpx.AsyncClient(timeout=600.0) as client:
        r = await client.post(f"{url}/chat/completions", json=payload)
        if r.status_code >= 400:
            raise HTTPException(status_code=r.status_code, detail=r.text)
        data = r.json()

    return ChatResponse(
        answer=data["choices"][0]["message"]["content"],
        model_used=name,
        route=route,
        elapsed_ms=int((time.perf_counter() - start) * 1000),
    )


# --------------------- OpenAI-compatible endpoints (for Open WebUI) ---------------------

@app.get("/v1/models")
async def list_models():
    now = int(time.time())
    return {
        "object": "list",
        "data": [
            {"id": "auto", "object": "model", "created": now, "owned_by": "router"},
            {"id": LARGE_NAME, "object": "model", "created": now, "owned_by": "vllm"},
            {"id": SMALL_NAME, "object": "model", "created": now, "owned_by": "vllm"},
        ],
    }


def _last_user_message(messages: list[dict]) -> str:
    for m in reversed(messages):
        if m.get("role") == "user":
            content = m.get("content", "")
            if isinstance(content, list):
                # OpenAI vision-style array content -> join text parts
                return " ".join(
                    part.get("text", "") for part in content if isinstance(part, dict) and part.get("type") == "text"
                )
            return content or ""
    return ""


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    payload: dict[str, Any] = await request.json()
    requested = payload.get("model", "auto")

    if requested == LARGE_NAME:
        url, name, route = LARGE_URL, LARGE_NAME, ROUTE_LARGE
    elif requested == SMALL_NAME:
        url, name, route = SMALL_URL, SMALL_NAME, ROUTE_SMALL
    else:
        question = _last_user_message(payload.get("messages", []))
        route = await classify(question)
        url, name = target_for(route)

    forwarded = dict(payload)
    forwarded["model"] = name

    stream = bool(payload.get("stream", False))

    if not stream:
        async with httpx.AsyncClient(timeout=600.0) as client:
            r = await client.post(f"{url}/chat/completions", json=forwarded)
            if r.status_code >= 400:
                raise HTTPException(status_code=r.status_code, detail=r.text)
            data = r.json()
            data.setdefault("x_router", {})["route"] = route
            return data

    async def event_stream():
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream("POST", f"{url}/chat/completions", json=forwarded) as r:
                if r.status_code >= 400:
                    body = await r.aread()
                    yield f"data: {json.dumps({'error': body.decode('utf-8', 'replace')})}\n\n".encode()
                    return
                async for chunk in r.aiter_raw():
                    if chunk:
                        yield chunk

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/health")
async def health():
    async def ping(url: str) -> bool:
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                r = await client.get(f"{url}/models")
                return r.status_code < 500
        except Exception:
            return False

    return {
        "status": "ok",
        "large_backend": await ping(LARGE_URL),
        "small_backend": await ping(SMALL_URL),
    }


@app.get("/")
async def root():
    return {
        "service": "altos-gemma-router",
        "endpoints": ["/chat", "/v1/chat/completions", "/v1/models", "/health"],
        "models": {"large": LARGE_NAME, "small": SMALL_NAME},
    }
