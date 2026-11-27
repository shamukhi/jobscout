"""Hybrid retrieval: dense (Chroma) + BM25 -> Reciprocal Rank Fusion -> personalised re-rank.

`mode` exists so the eval harness can ablate each stage:
  dense | bm25 | hybrid | personalized
"""
from __future__ import annotations

import math
from typing import Literal

from rank_bm25 import BM25Okapi

from .index import JobIndex
from .models import Job, Profile, SearchFilters, SearchHit
from .store import Store
from .textutil import norm_skill, tokenize

Mode = Literal["dense", "bm25", "hybrid", "personalized"]

RRF_K = 60
CANDIDATES = 50
# weights for the personalised score (sum to 1)
W_REL, W_SKILL, W_PREF, W_RECENT = 0.60, 0.20, 0.12, 0.08


def _rrf(rankings: list[list[str]]) -> dict[str, float]:
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, jid in enumerate(ranking):
            scores[jid] = scores.get(jid, 0.0) + 1.0 / (RRF_K + rank + 1)
    return scores


def _doc_tokens(job: Job) -> list[str]:
    # title and skills repeated: cheap field boosting for BM25
    return tokenize(f"{job.title} {job.title} {' '.join(job.skills)} {' '.join(job.skills)} {job.description}")


class Searcher:
    def __init__(self, index: JobIndex, store: Store):
        self.index = index
        self.store = store
        self._bm25: BM25Okapi | None = None
        self._bm25_ids: list[str] = []
        self._bm25_fp = ""

    # ---- BM25 (rebuilt only when the jobs table changed) -------------
    def _ensure_bm25(self) -> None:
        fp = self.store.fingerprint()
        if fp == self._bm25_fp and self._bm25 is not None:
            return
        jobs = self.store.all_jobs()
        self._bm25_ids = [j.id for j in jobs]
        self._bm25 = BM25Okapi([_doc_tokens(j) for j in jobs]) if jobs else None
        self._bm25_fp = fp

    def _bm25_rank(self, query: str, n: int) -> list[str]:
        self._ensure_bm25()
        if self._bm25 is None:
            return []
        toks = tokenize(query)
        if not toks:
            return []
        scores = self._bm25.get_scores(toks)
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n]
        return [self._bm25_ids[i] for i in order if scores[i] > 0]

    # ---- personalisation --------------------------------------------
    @staticmethod
    def _personal_features(job: Job, p: Profile) -> tuple[dict[str, float], list[str]]:
        reasons: list[str] = []

        user_skills = {norm_skill(s) for s in p.skills}
        job_skills = set(job.skills)
        overlap = sorted(user_skills & job_skills)
        skill = len(overlap) / len(job_skills) if job_skills and user_skills else 0.0
        if overlap:
            reasons.append(f"matches your skills: {', '.join(overlap[:4])}")

        checks: list[float] = []
        if p.preferred_locations:
            ok = job.remote or any(loc.lower() in job.location.lower() for loc in p.preferred_locations)
            checks.append(1.0 if ok else 0.0)
            if ok:
                reasons.append("in a preferred location" if not job.remote else "remote-friendly")
        if p.remote_preference != "any":
            ok = job.remote if p.remote_preference == "remote" else not job.remote
            checks.append(1.0 if ok else 0.0)
            if ok and p.remote_preference == "remote":
                reasons.append("remote, as you prefer")
        if p.job_types:
            ok = job.job_type in p.job_types
            checks.append(1.0 if ok else 0.0)
            if ok:
                reasons.append(f"{job.job_type} role")
        if p.seniority:
            ok = job.seniority == p.seniority
            checks.append(1.0 if ok else 0.0)
        pref = sum(checks) / len(checks) if checks else 0.5

        recent = math.exp(-job.age_days() / 45.0)
        if job.age_days() <= 7:
            reasons.append("posted this week")
        return {"skills": skill, "preferences": pref, "recency": recent}, reasons

    # ---- main entry --------------------------------------------------
    def search(
        self,
        query: str,
        filters: SearchFilters | None = None,
        profile: Profile | None = None,
        user_id: str | None = None,
        k: int = 10,
        mode: Mode = "personalized",
    ) -> list[SearchHit]:
        filters = filters or SearchFilters()
        profile = profile or Profile()

        dense = self.index.query(query, CANDIDATES, filters) if mode in ("dense", "hybrid", "personalized") else []
        sparse = self._bm25_rank(query, CANDIDATES * 2) if mode in ("bm25", "hybrid", "personalized") else []

        if mode == "dense":
            fused = {jid: 1.0 / (r + 1) for r, jid in enumerate(dense)}
        elif mode == "bm25":
            fused = {jid: 1.0 / (r + 1) for r, jid in enumerate(sparse)}
        else:
            fused = _rrf([dense, sparse])

        jobs = self.store.get_jobs(list(fused))
        dismissed = self.store.interaction_ids(user_id, "dismissed") if user_id else set()
        saved = self.store.interaction_ids(user_id, "saved") if user_id else set()

        pool = [jid for jid in fused if jid in jobs and filters.matches(jobs[jid]) and jid not in dismissed]
        if not pool:
            return []
        top = max(fused[j] for j in pool)

        hits: list[SearchHit] = []
        for jid in pool:
            job = jobs[jid]
            rel = fused[jid] / top
            breakdown = {"relevance": round(rel, 4)}
            reasons: list[str] = []
            score = rel
            if mode == "personalized":
                feats, reasons = self._personal_features(job, profile)
                score = (
                    W_REL * rel
                    + W_SKILL * feats["skills"]
                    + W_PREF * feats["preferences"]
                    + W_RECENT * feats["recency"]
                )
                breakdown.update({k_: round(v, 4) for k_, v in feats.items()})
            hits.append(SearchHit(job=job, score=round(score, 4), breakdown=breakdown, reasons=reasons, saved=jid in saved))

        hits.sort(key=lambda h: (-h.score, h.job.id))
        return hits[:k]
