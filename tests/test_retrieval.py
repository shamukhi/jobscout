from app.models import Profile, SearchFilters
from tests.conftest import make_job


def ids(hits):
    return [h.job.id for h in hits]


def test_hybrid_finds_relevant_job_first(env):
    _, _, s = env
    assert ids(s.search("rag pipelines with language models", mode="hybrid"))[0] == "j1"
    assert ids(s.search("react typescript", mode="bm25"))[0] == "j3"


def test_filters_apply_in_every_mode(env):
    _, _, s = env
    for mode in ("dense", "bm25", "hybrid", "personalized"):
        hits = s.search("engineer", SearchFilters(remote=True), mode=mode)
        assert hits and all(h.job.remote for h in hits)
    hits = s.search("engineer", SearchFilters(location="gurugram"), mode="hybrid")
    assert ids(hits) == ["j3"] or all("Gurugram" in h.job.location for h in hits)


def test_personalization_uses_skills_and_explains(env):
    _, _, s = env
    profile = Profile(skills=["python", "airflow", "spark", "sql"])
    hits = s.search("engineer", profile=profile, mode="personalized")
    assert hits[0].job.id == "j4"
    assert any("airflow" in r for r in hits[0].reasons)
    assert set(hits[0].breakdown) == {"relevance", "skills", "preferences", "recency"}


def test_dismissed_excluded_saved_flagged(env):
    store, _, s = env
    store.set_interaction("u1", "j1", "dismissed")
    store.set_interaction("u1", "j2", "saved")
    hits = s.search("engineer", user_id="u1")
    assert "j1" not in ids(hits)
    assert next(h for h in hits if h.job.id == "j2").saved


def test_bm25_rebuilds_when_index_changes(env):
    _, index, s = env
    assert ids(s.search("kubernetes terraform", mode="bm25")) == []
    index.sync([make_job(9, title="DevOps", description="kubernetes terraform", skills=["kubernetes"])])
    assert ids(s.search("kubernetes terraform", mode="bm25")) == ["j9"]


def test_profile_merge_unions_lists():
    p = Profile(skills=["Python"]).merge({"skills": ["fastapi", "python"], "remote_preference": "remote"})
    assert p.skills == ["python", "fastapi"] or set(p.skills) == {"python", "fastapi"}
    assert p.remote_preference == "remote"
