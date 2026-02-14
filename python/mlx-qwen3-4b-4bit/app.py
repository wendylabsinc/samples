#!/usr/bin/env python3
"""
MLX Chat backend — Qwen3 4B (4-bit) on Apple Silicon.
FastAPI server with mlx-lm for local inference.
"""
import logging
import re
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_NAME = "Qwen/Qwen3-4B-MLX-4bit"

# Globals populated at startup
model = None
tokenizer = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, tokenizer
    from mlx_lm import load

    logger.info(f"Loading {MODEL_NAME} …")
    model, tokenizer = load(MODEL_NAME)
    logger.info("Model ready.")
    yield


app = FastAPI(lifespan=lifespan)

DIST_DIR = Path(__file__).parent / "frontend" / "dist"


class ChatRequest(BaseModel):
    messages: list[dict]
    system: str | None = None
    maxTokens: int = 256
    temperature: float = 0.6
    topP: float = 0.95
    topK: int = 20


@app.post("/api/chat")
async def chat(req: ChatRequest):
    from mlx_lm import generate
    from mlx_lm.generate import make_sampler

    conversation = []
    if req.system:
        conversation.append({"role": "system", "content": req.system})
    conversation.extend(req.messages)

    prompt = tokenizer.apply_chat_template(
        conversation, tokenize=False, add_generation_prompt=True
    )

    sampler = make_sampler(temp=req.temperature, top_p=req.topP, top_k=req.topK)

    t0 = time.perf_counter()
    reply = generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=req.maxTokens,
        sampler=sampler,
        verbose=False,
    )
    duration_ms = int((time.perf_counter() - t0) * 1000)

    # Strip <think>...</think> reasoning block from Qwen3 output
    reply = re.sub(r"<think>.*?</think>\s*", "", reply, flags=re.DOTALL).strip()

    return {"reply": reply, "model": MODEL_NAME, "durationMs": duration_ms}


@app.get("/status")
async def status():
    return {
        "model": MODEL_NAME,
        "loaded": model is not None,
    }


@app.get("/logo.svg")
async def logo():
    return FileResponse(Path(__file__).parent / "logo.svg", media_type="image/svg+xml")


@app.get("/{path:path}")
async def spa(request: Request, path: str):
    """Serve the React SPA — try the exact file, fall back to index.html."""
    file = DIST_DIR / path
    if path and file.is_file():
        return FileResponse(file)
    return FileResponse(DIST_DIR / "index.html", media_type="text/html")
