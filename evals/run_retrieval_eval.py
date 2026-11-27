"""Offline retrieval eval: Recall@k, MRR, nDCG@k and latency for each pipeline stage.

  python -m evals.run_retrieval_eval                      # real embeddings (EMBED_BACKEND from env)
  python -m evals.run_retrieval_eval --backend hash       # offline sanity run

Writes evals/results.md so numbers are reproducible and diff-able in git.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import tempfile
import time
from pathlib import Path

from app.config import ROOT, Settings
from app.embeddings import get_embedder
from app.index import JobIndex
from app.models import Job, Profile, SearchFilters
from app.retrieval import Searcher
from app.store import Store

MODES = ["bm25", "dense", "hybrid", "personalized"]


def recall_at_k(ranked, relevant, k):
    return len(set(ranked[:k]) & relevant) / max(len(relevant), 1)


def mrr(ranked, relevant, k):
    for i, jid in enumerate(ranked[:k], 1):
        if jid in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked, relevant, k):
    dcg = sum(1.0 / math.log2(i + 2) for i, jid in enumerate(ranked[:k]) if jid in relevant)
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / ideal if ideal else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--jobs", default=str(ROOT / "data" / "jobs.json"))
    ap.add_argument("--queries", default=str(ROOT / "evals" / "queries.jsonl"))
    ap.add_argument("--out", default=str(ROOT / "evals" / "results.md"))
    args = ap.parse_args()

    s = Settings()
    embedder = get_embedder(args.backend or s.embed_backend, args.model or s.embed_model)
    jobs = [Job.model_validate(j) for j in json.loads(Path(args.jobs).read_text())]
    queries = [json.loads(line) for line in Path(args.queries).read_text().splitlines() if line.strip()]

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        store = Store(Path(tmp) / "eval.sqlite3")
        index = JobIndex(Path(tmp) / "chroma", embedder, store)
        index.sync(jobs)
        searcher = Searcher(index, store)

        rows = []
        for mode in MODES:
            rec, rr, nd, lat = [], [], [], []
            for q in queries:
                relevant = set(q["relevant"])
                profile = Profile(**q["profile"]) if q.get("profile") else None
                t0 = time.perf_counter()
                hits = searcher.search(q["query"], SearchFilters(**q["filters"]), profile, k=args.k, mode=mode)
                lat.append((time.perf_counter() - t0) * 1000)
                ranked = [h.job.id for h in hits]
                rec.append(recall_at_k(ranked, relevant, args.k))
                rr.append(mrr(ranked, relevant, args.k))
                nd.append(ndcg_at_k(ranked, relevant, args.k))
            lat.sort()
            rows.append((mode, statistics.mean(rec), statistics.mean(rr), statistics.mean(nd),
                         statistics.median(lat), lat[min(int(len(lat) * 0.95), len(lat) - 1)]))

    lines = [
        f"# Retrieval eval ({embedder.name}, {len(jobs)} jobs, {len(queries)} queries, k={args.k})",
        "",
        f"| mode | Recall@{args.k} | MRR | nDCG@{args.k} | p50 ms | p95 ms |",
        "|---|---|---|---|---|---|",
        *[f"| {m} | {r:.3f} | {mr:.3f} | {n:.3f} | {p50:.1f} | {p95:.1f} |" for m, r, mr, n, p50, p95 in rows],
        "",
        "Labels on the bundled synthetic data are generator-derived, so this checks the pipeline; "
        "use a real, hand-labelled dataset for quotable numbers.",
    ]
    text = "\n".join(lines)
    Path(args.out).write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
