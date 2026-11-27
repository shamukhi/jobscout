from types import SimpleNamespace as NS

from fastapi.testclient import TestClient

from app.config import Settings
from app.embeddings import HashEmbedder
from app.main import create_app
from tests.test_agent import FakeClient, text


def client_for(tmp_path, llm=None):
    settings = Settings(state_dir=tmp_path, seed_file=tmp_path / "missing.json", admin_token="secret")
    return TestClient(create_app(settings, HashEmbedder(), llm))


def job_payload(i, **kw):
    d = dict(id=f"x{i}", title="Search Engineer", company="Acme", location="Pune",
             description="Improve ranking and relevance.", skills=["python"], posted_at="2026-11-20T00:00:00Z")
    d.update(kw)
    return d


def test_ingest_search_and_auth(tmp_path):
    with client_for(tmp_path) as c:
        assert c.post("/ingest", json={"jobs": [job_payload(1)]}).status_code == 401
        h = {"X-Admin-Token": "secret"}
        r = c.post("/ingest", json={"jobs": [job_payload(1), job_payload(2, title="Frontend Engineer",
                 description="react ui", skills=["react"])]}, headers=h)
        assert r.json()["added"] == 2
        hits = c.post("/search", json={"query": "ranking relevance", "k": 5}).json()
        assert hits[0]["job"]["id"] == "x1" and "breakdown" in hits[0]
        assert c.get("/jobs/x2").json()["title"] == "Frontend Engineer"
        assert c.get("/jobs/none").status_code == 404
        assert c.get("/health").json()["indexed"] == 2


def test_chat_requires_llm_then_works(tmp_path):
    with client_for(tmp_path) as c:
        assert c.post("/chat", json={"user_id": "u", "message": "hi"}).status_code == 503
    fake = FakeClient([([text("Hello!")], "end_turn")])
    with client_for(tmp_path / "b", fake) as c:
        r = c.post("/chat", json={"user_id": "u", "message": "hi"})
        assert r.status_code == 200 and r.json()["reply"] == "Hello!"
        assert c.post("/chat", json={"user_id": "bad id!", "message": "x"}).status_code == 422


def test_provider_errors_map_to_clean_http_codes(tmp_path):
    class Boom:
        def __init__(self, status):
            self.status, self.messages = status, self

        def create(self, **kw):
            e = RuntimeError("provider said no")
            e.status_code = self.status
            raise e

    with client_for(tmp_path / "r", Boom(429)) as c:
        assert c.post("/chat", json={"user_id": "u", "message": "hi"}).status_code == 429
    with client_for(tmp_path / "e", Boom(500)) as c:
        r = c.post("/chat", json={"user_id": "u", "message": "hi"})
        assert r.status_code == 502 and "RuntimeError" in r.json()["detail"]
