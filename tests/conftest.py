import json
from datetime import datetime, timedelta, timezone

import pytest

from app.embeddings import HashEmbedder
from app.index import JobIndex
from app.models import Job
from app.retrieval import Searcher
from app.store import Store


def make_job(i: int, **kw) -> Job:
    base = dict(
        id=f"j{i}", title="Backend Engineer", company="Acme", location="Pune", remote=False,
        job_type="full-time", seniority="junior",
        description="Build python services and rest apis with postgres.",
        skills=["python", "postgresql"], posted_at=datetime.now(timezone.utc) - timedelta(days=3),
    )
    base.update(kw)
    return Job(**base)


@pytest.fixture
def jobs():
    return [
        make_job(1, title="LLM Engineer", description="Build rag pipelines and agents on large language models.",
                 skills=["python", "llm", "rag", "embeddings"], location="Bengaluru", remote=True,
                 job_type="internship", seniority="intern"),
        make_job(2),
        make_job(3, title="Frontend Engineer", description="Build react user interfaces in typescript.",
                 skills=["react", "typescript"], location="Gurugram"),
        make_job(4, title="Data Engineer", description="Airflow spark pipelines and warehouse modelling.",
                 skills=["python", "sql", "airflow", "spark"]),
    ]


@pytest.fixture
def env(tmp_path, jobs):
    store = Store(tmp_path / "t.sqlite3")
    index = JobIndex(tmp_path / "chroma", HashEmbedder(), store)
    index.sync(jobs)
    return store, index, Searcher(index, store)
