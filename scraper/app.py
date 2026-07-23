"""
FastAPI wrapper around query.py -- exposes the same retrieve() + generate_answer()
logic as a web API, and serves a small chat UI on top of it.

query.py itself is untouched and still works standalone from the CLI --
this file imports its functions rather than duplicating them, so there's
exactly one place that knows how retrieval/generation actually works.


Run:
    uvicorn app:app --reload

Then open http://127.0.0.1:8000 in a browser.
"""

import os
import time
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import query  # your existing Phase 4 script -- retrieve(), build_prompt(), generate_answer()

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "data" / "static"

app = FastAPI(title="IMSciences Assistant API")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# loaded once at startup, reused for every request -- loading the model
# and FAISS index per-request would make every answer take several extra
# seconds for no reason
_state = {"model": None, "index": None, "chunks": None}


@app.on_event("startup")
def load_everything():
    from sentence_transformers import SentenceTransformer
    import faiss

    print("Loading embedding model and FAISS index...")
    _state["model"] = SentenceTransformer(query.EMBED_MODEL_NAME)
    _state["index"] = faiss.read_index(str(query.INDEX_FILE))
    _state["chunks"] = query.load_chunks()
    print(f"Ready -- {len(_state['chunks'])} chunks loaded.")


class ChatRequest(BaseModel):
    question: str


class SourceOut(BaseModel):
    rank: int
    score: float
    preview: str
    source_url: str = ""
    title: str = ""
    type: str = ""


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceOut]
    warning: str = ""
    latency_seconds: float = 0.0


@app.get("/")
def serve_index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health():
    ready = all(_state[k] is not None for k in _state)
    return {"ready": ready, "chunks_loaded": len(_state["chunks"]) if ready else 0}


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    start = time.time()
    question = req.question.strip()

    if not question:
        return ChatResponse(answer="Please type a question.", sources=[])

    retrieved = query.retrieve(question, _state["model"], _state["index"], _state["chunks"])

    sources = []
    for r in retrieved:
        chunk = _state["chunks"][r["chunk_index"]]
        sources.append(SourceOut(
            rank=r["rank"],
            score=r["score"],
            preview=r["text"][:220].replace("\n", " "),
            source_url=chunk.get("source_url", ""),
            title=chunk.get("title", ""),
            type=chunk.get("type", ""),
        ))

    prompt = query.build_prompt(question, retrieved)

    if not os.environ.get("GROQ_API_KEY"):
        return ChatResponse(
            answer="",
            sources=sources,
            warning="GROQ_API_KEY is not set on the server, so I can show you "
                    "the retrieved sources but can't generate a written answer. "
                    "Set the key and restart the server.",
            latency_seconds=time.time() - start,
        )

    try:
        answer = query.generate_answer(prompt)
    except Exception as e:
        return ChatResponse(
            answer="",
            sources=sources,
            warning=f"The answer-generation step failed: {e}",
            latency_seconds=time.time() - start,
        )

    return ChatResponse(answer=answer, sources=sources, latency_seconds=time.time() - start)