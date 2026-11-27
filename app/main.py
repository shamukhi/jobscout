from __future__ import annotations

import json
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .agent import JobAgent, RecommendedJob
from .config import ROOT, Settings
from .embeddings import Embedder, get_embedder
from .index import JobIndex
from .llm import make_client
from .models import Job, Profile, SearchFilters, SearchHit
from .retrieval import Mode, Searcher
from .store import Store


@dataclass
class State:
    settings: Settings
    store: Store
    index: JobIndex
    searcher: Searcher
    agent: JobAgent | None


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=500)
    filters: SearchFilters = Field(default_factory=SearchFilters)
    user_id: str | None = None
    k: int = Field(default=10, ge=1, le=50)
    mode: Mode = "personalized"


class ChatRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=64, pattern=r"^[\w\-.@]+$")
    message: str = Field(min_length=1, max_length=2000)


class ChatResponse(BaseModel):
    reply: str
    recommendations: list[RecommendedJob]
    tool_calls: list[dict]


class IngestRequest(BaseModel):
    jobs: list[Job]
    prune: bool = False  # True = this payload is the full feed; anything missing is removed


def build_state(settings: Settings, embedder: Embedder | None = None, llm_client: Any = None) -> State:
    store = Store(settings.db_path)
    embedder = embedder or get_embedder(settings.embed_backend, settings.embed_model)
    index = JobIndex(settings.chroma_dir, embedder, store)
    searcher = Searcher(index, store)
    model = settings.llm_model or "injected-client"
    if llm_client is None:
        llm_client, model = make_client(settings)
    agent = JobAgent(store, searcher, llm_client, model) if llm_client else None
    # First boot: load the seed feed so the demo has data.
    if index.count() == 0 and settings.seed_file.exists():
        jobs = [Job.model_validate(j) for j in json.loads(settings.seed_file.read_text())]
        index.sync(jobs, prune=False)
    return State(settings, store, index, searcher, agent)


def create_app(settings: Settings | None = None, embedder: Embedder | None = None, llm_client: Any = None) -> FastAPI:
    settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.s = build_state(settings, embedder, llm_client)
        yield

    app = FastAPI(title="Jobscout", version="0.1.0", lifespan=lifespan)

    def st() -> State:
        return app.state.s

    def require_admin(x_admin_token: str | None = Header(default=None)) -> None:
        if settings.admin_token and x_admin_token != settings.admin_token:
            raise HTTPException(401, "bad admin token")

    @app.get("/health")
    def health(s: State = Depends(st)):
        return {"status": "ok", "jobs": s.store.job_count(), "indexed": s.index.count(),
                "embedder": s.index.embedder.name, "agent": s.agent is not None,
                "llm": s.agent.model if s.agent else None}

    @app.post("/search", response_model=list[SearchHit])
    def search(req: SearchRequest, s: State = Depends(st)):
        profile = s.store.get_profile(req.user_id) if req.user_id else None
        return s.searcher.search(req.query, req.filters, profile, req.user_id, req.k, req.mode)

    @app.post("/chat", response_model=ChatResponse)
    def chat(req: ChatRequest, s: State = Depends(st)):
        if s.agent is None:
            raise HTTPException(503, "LLM not configured: set a provider API key (see README)")
        try:
            res = s.agent.chat(req.user_id, req.message)
        except Exception as exc:  # provider/network errors must not leak as bare 500s
            if getattr(exc, "status_code", None) == 429:
                raise HTTPException(429, "The model provider is rate-limiting this key. Try again in a minute.")
            resp = getattr(exc, "response", None)
            detail = (getattr(resp, "text", "") or "")[:400] or str(exc)[:300]
            raise HTTPException(502, f"LLM provider error: {type(exc).__name__}: {detail}")
        return ChatResponse(reply=res.reply, recommendations=res.recommendations, tool_calls=res.tool_calls)

    @app.get("/jobs/{job_id}", response_model=Job)
    def get_job(job_id: str, s: State = Depends(st)):
        job = s.store.get_jobs([job_id]).get(job_id)
        if not job:
            raise HTTPException(404, "job not found")
        return job

    @app.post("/ingest", dependencies=[Depends(require_admin)])
    def ingest(req: IngestRequest, s: State = Depends(st)):
        return s.index.sync(req.jobs, prune=req.prune).dict()

    @app.delete("/jobs/{job_id}", dependencies=[Depends(require_admin)])
    def delete_job(job_id: str, s: State = Depends(st)):
        return {"removed": s.index.remove([job_id])}

    @app.get("/users/{user_id}/profile", response_model=Profile)
    def get_profile(user_id: str, s: State = Depends(st)):
        return s.store.get_profile(user_id)

    @app.put("/users/{user_id}/profile", response_model=Profile)
    def put_profile(user_id: str, profile: Profile, s: State = Depends(st)):
        s.store.save_profile(user_id, profile)
        return profile

    @app.get("/")
    def home():
        return FileResponse(ROOT / "static" / "index.html")

    return app


app = create_app()
