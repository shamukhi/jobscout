"""SQLite = source of truth for jobs, user profiles, conversation memory, saved/dismissed jobs.
Chroma only holds vectors; it can always be rebuilt from here."""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path

from .models import Job, Profile

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    json TEXT NOT NULL,
    updated_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS users (
    user_id TEXT PRIMARY KEY,
    profile_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('user','assistant')),
    content TEXT NOT NULL,
    ts REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_user ON messages(user_id, id);
CREATE TABLE IF NOT EXISTS interactions (
    user_id TEXT NOT NULL,
    job_id TEXT NOT NULL,
    action TEXT NOT NULL CHECK (action IN ('saved','dismissed')),
    ts REAL NOT NULL,
    PRIMARY KEY (user_id, job_id)
);
"""


class Store:
    def __init__(self, path: Path | str):
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._db = sqlite3.connect(str(path), check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        with self._lock:
            self._db.executescript(SCHEMA)

    # ---- jobs -------------------------------------------------------
    def upsert_jobs(self, jobs: list[Job]) -> None:
        now = time.time()
        with self._lock, self._db:
            self._db.executemany(
                "INSERT INTO jobs(id, json, updated_at) VALUES (?,?,?) "
                "ON CONFLICT(id) DO UPDATE SET json=excluded.json, updated_at=excluded.updated_at",
                [(j.id, j.model_dump_json(), now) for j in jobs],
            )

    def delete_jobs(self, ids: list[str]) -> None:
        with self._lock, self._db:
            self._db.executemany("DELETE FROM jobs WHERE id=?", [(i,) for i in ids])

    def all_jobs(self) -> list[Job]:
        with self._lock:
            rows = self._db.execute("SELECT json FROM jobs ORDER BY id").fetchall()
        return [Job.model_validate_json(r["json"]) for r in rows]

    def get_jobs(self, ids: list[str]) -> dict[str, Job]:
        if not ids:
            return {}
        marks = ",".join("?" * len(ids))
        with self._lock:
            rows = self._db.execute(f"SELECT json FROM jobs WHERE id IN ({marks})", ids).fetchall()
        jobs = [Job.model_validate_json(r["json"]) for r in rows]
        return {j.id: j for j in jobs}

    def job_count(self) -> int:
        with self._lock:
            return self._db.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]

    def fingerprint(self) -> str:
        """Changes whenever the jobs table changes; lets BM25 know to rebuild."""
        with self._lock:
            n, last = self._db.execute("SELECT COUNT(*), COALESCE(MAX(updated_at),0) FROM jobs").fetchone()
        return f"{n}:{last}"

    # ---- profile / memory ------------------------------------------
    def get_profile(self, user_id: str) -> Profile:
        with self._lock:
            row = self._db.execute("SELECT profile_json FROM users WHERE user_id=?", (user_id,)).fetchone()
        return Profile.model_validate_json(row["profile_json"]) if row else Profile()

    def save_profile(self, user_id: str, profile: Profile) -> None:
        with self._lock, self._db:
            self._db.execute(
                "INSERT INTO users(user_id, profile_json) VALUES (?,?) "
                "ON CONFLICT(user_id) DO UPDATE SET profile_json=excluded.profile_json",
                (user_id, profile.model_dump_json()),
            )

    def add_message(self, user_id: str, role: str, content: str) -> None:
        with self._lock, self._db:
            self._db.execute(
                "INSERT INTO messages(user_id, role, content, ts) VALUES (?,?,?,?)",
                (user_id, role, content, time.time()),
            )

    def recent_messages(self, user_id: str, n: int = 12) -> list[dict]:
        with self._lock:
            rows = self._db.execute(
                "SELECT role, content FROM messages WHERE user_id=? ORDER BY id DESC LIMIT ?", (user_id, n)
            ).fetchall()
        msgs = [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]
        while msgs and msgs[0]["role"] != "user":  # API requires the first turn to be the user's
            msgs.pop(0)
        return msgs

    # ---- saved / dismissed -----------------------------------------
    def set_interaction(self, user_id: str, job_id: str, action: str) -> None:
        with self._lock, self._db:
            self._db.execute(
                "INSERT INTO interactions(user_id, job_id, action, ts) VALUES (?,?,?,?) "
                "ON CONFLICT(user_id, job_id) DO UPDATE SET action=excluded.action, ts=excluded.ts",
                (user_id, job_id, action, time.time()),
            )

    def interaction_ids(self, user_id: str, action: str) -> set[str]:
        with self._lock:
            rows = self._db.execute(
                "SELECT job_id FROM interactions WHERE user_id=? AND action=?", (user_id, action)
            ).fetchall()
        return {r["job_id"] for r in rows}

    def dump_profile_json(self, user_id: str) -> str:
        return json.dumps(self.get_profile(user_id).model_dump(), indent=None)
