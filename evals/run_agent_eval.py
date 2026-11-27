"""Behavioural eval for the agent loop. Needs ANTHROPIC_API_KEY (makes real LLM calls).

Checks tool routing (right tools, right filters), refusal to act on injected instructions inside job
text, and that every recommended id exists. Each case runs against a fresh temp state.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

from app.agent import JobAgent
from app.config import ROOT, Settings
from app.embeddings import get_embedder
from app.index import JobIndex
from app.llm import make_client
from app.models import Job
from app.retrieval import Searcher
from app.store import Store


def run_case(case: dict, jobs: list[Job], settings: Settings, client, model: str) -> tuple[bool, list[str]]:
    jobs = [j.model_copy() for j in jobs]
    if case.get("inject_into_job_description"):
        for j in jobs:
            if "llm" in j.skills:
                j.description += " " + case["inject_into_job_description"]
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        store = Store(Path(tmp) / "e.sqlite3")
        index = JobIndex(Path(tmp) / "c", get_embedder(settings.embed_backend, settings.embed_model), store)
        index.sync(jobs)
        agent = JobAgent(store, Searcher(index, store), client, model)
        res = agent.chat("eval-user", case["message"])

        problems = []
        called = [c["name"] for c in res.tool_calls]
        for t in case.get("must_call", []):
            if t not in called:
                problems.append(f"expected tool {t}, got {called}")
        for t in case.get("must_not_call", []):
            if t in called:
                problems.append(f"unexpected tool {t}")
        for key, val in case.get("search_args_include", {}).items():
            searches = [c["input"] for c in res.tool_calls if c["name"] == "search_jobs"]
            if not any(s.get(key) == val for s in searches):
                problems.append(f"no search_jobs call with {key}={val!r}")
        if case.get("recs_must_be_real"):
            real = set(store.get_jobs([r.job.id for r in res.recommendations]))
            if not res.recommendations or len(real) != len(res.recommendations):
                problems.append("recommendations empty or contain unknown ids")
        return not problems, problems


def main():
    settings = Settings()
    client, model = make_client(settings)
    if client is None:
        sys.exit("No LLM key found. Set e.g. GEMINI_API_KEY or GROQ_API_KEY (see README).")
    print(f"agent eval with provider={settings.llm_provider} model={model}\n")
    jobs = [Job.model_validate(j) for j in json.loads((ROOT / "data" / "jobs.json").read_text())]
    cases = [json.loads(l) for l in (ROOT / "evals" / "agent_cases.jsonl").read_text().splitlines() if l.strip()]
    passed = 0
    for case in cases:
        ok, problems = run_case(case, jobs, settings, client, model)
        passed += ok
        print(f"{'PASS' if ok else 'FAIL'}  {case['name']}" + ("" if ok else f"  -> {problems}"))
    print(f"\n{passed}/{len(cases)} passed")
    sys.exit(0 if passed == len(cases) else 1)


if __name__ == "__main__":
    main()
