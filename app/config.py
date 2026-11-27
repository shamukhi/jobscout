from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _env(name: str, default: str) -> str:
    return os.getenv(name, default)


@dataclass(frozen=True)
class Settings:
    # Where SQLite + Chroma files live (mount a volume here in production).
    state_dir: Path = field(default_factory=lambda: Path(_env("STATE_DIR", str(ROOT / "var"))))
    seed_file: Path = field(default_factory=lambda: Path(_env("SEED_FILE", str(ROOT / "data" / "jobs.json"))))
    # "fastembed" (default, light), "st" (sentence-transformers), or "hash" (offline, tests only)
    embed_backend: str = field(default_factory=lambda: _env("EMBED_BACKEND", "fastembed"))
    embed_model: str = field(default_factory=lambda: _env("EMBED_MODEL", "BAAI/bge-small-en-v1.5"))
    # LLM provider: auto | anthropic | gemini | groq | cerebras | openrouter | mistral | ollama | custom
    llm_provider: str = field(default_factory=lambda: _env("LLM_PROVIDER", "auto").lower())
    llm_model: str | None = field(default_factory=lambda: os.getenv("LLM_MODEL"))  # None = provider default
    llm_base_url: str | None = field(default_factory=lambda: os.getenv("LLM_BASE_URL"))
    llm_api_key: str | None = field(default_factory=lambda: os.getenv("LLM_API_KEY"))
    admin_token: str | None = field(default_factory=lambda: os.getenv("ADMIN_TOKEN"))

    @property
    def db_path(self) -> Path:
        return self.state_dir / "jobscout.sqlite3"

    @property
    def chroma_dir(self) -> Path:
        return self.state_dir / "chroma"
