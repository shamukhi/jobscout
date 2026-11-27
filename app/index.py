"""Vector index over jobs with incremental sync.

The jobs feed changes constantly, so `sync()` diffs against what is already indexed using two hashes:
  - embed hash: hash of the text that gets embedded -> only then pay for a new embedding
  - full hash:  hash of every field -> metadata-only changes are a cheap update
Jobs missing from the feed (filled / expired) are removed when prune=True.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import chromadb

from .embeddings import Embedder
from .models import Job
from .store import Store

BATCH = 64


@dataclass
class SyncReport:
    added: int = 0
    re_embedded: int = 0
    metadata_updated: int = 0
    unchanged: int = 0
    removed: int = 0

    def dict(self) -> dict:
        return asdict(self)


def _meta(job: Job) -> dict:
    return {
        "job_type": job.job_type,
        "seniority": job.seniority,
        "remote": job.remote,
        "ehash": job.embed_hash(),
        "fhash": job.full_hash(),
    }


class JobIndex:
    def __init__(self, chroma_dir, embedder: Embedder, store: Store):
        self.embedder = embedder
        self.store = store
        self._client = chromadb.PersistentClient(path=str(chroma_dir))
        # one collection per embedding model: switching models never mixes vector spaces
        safe = "".join(c if c.isalnum() else "_" for c in embedder.name)[:50]
        self._col = self._client.get_or_create_collection(f"jobs_{safe}", metadata={"hnsw:space": "cosine"})

    def count(self) -> int:
        return self._col.count()

    def _existing(self) -> dict[str, dict]:
        if self._col.count() == 0:
            return {}
        got = self._col.get(include=["metadatas"])
        return dict(zip(got["ids"], got["metadatas"]))

    def sync(self, jobs: list[Job], prune: bool = False) -> SyncReport:
        report = SyncReport()
        existing = self._existing()
        to_embed: list[Job] = []
        meta_only: list[Job] = []
        for job in jobs:
            old = existing.get(job.id)
            if old is None:
                to_embed.append(job)
                report.added += 1
            elif old.get("ehash") != job.embed_hash():
                to_embed.append(job)
                report.re_embedded += 1
            elif old.get("fhash") != job.full_hash():
                meta_only.append(job)
                report.metadata_updated += 1
            else:
                report.unchanged += 1

        for i in range(0, len(to_embed), BATCH):
            chunk = to_embed[i : i + BATCH]
            self._col.upsert(
                ids=[j.id for j in chunk],
                embeddings=self.embedder.embed([j.embed_text() for j in chunk]),
                metadatas=[_meta(j) for j in chunk],
            )
        if meta_only:
            self._col.update(ids=[j.id for j in meta_only], metadatas=[_meta(j) for j in meta_only])

        self.store.upsert_jobs(jobs)

        if prune:
            gone = sorted(set(existing) - {j.id for j in jobs})
            if gone:
                self._col.delete(ids=gone)
                self.store.delete_jobs(gone)
            report.removed = len(gone)
        return report

    def remove(self, ids: list[str]) -> int:
        present = [i for i in ids if i in self._existing()]
        if present:
            self._col.delete(ids=present)
            self.store.delete_jobs(present)
        return len(present)

    def query(self, text: str, n: int, filters=None) -> list[str]:
        """Return job ids ranked by cosine similarity. Exact-match filters are pushed into Chroma."""
        total = self._col.count()
        if total == 0:
            return []
        clauses = []
        if filters is not None:
            if filters.remote is not None:
                clauses.append({"remote": filters.remote})
            if filters.job_type:
                clauses.append({"job_type": filters.job_type})
            if filters.seniority:
                clauses.append({"seniority": filters.seniority})
        where = None if not clauses else clauses[0] if len(clauses) == 1 else {"$and": clauses}
        res = self._col.query(
            query_embeddings=self.embedder.embed([text]),
            n_results=min(n, total),
            where=where,
        )
        return res["ids"][0]
