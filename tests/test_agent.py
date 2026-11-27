from types import SimpleNamespace as NS

from app.agent import JobAgent


def tool(name, **inp):
    return NS(type="tool_use", id=f"tu_{name}", name=name, input=inp)


def text(t):
    return NS(type="text", text=t)


class FakeClient:
    """Scripted LLM: returns one canned response per call and records what it was sent."""

    def __init__(self, script):
        self.script, self.calls = list(script), []
        self.messages = self

    def create(self, **kw):
        self.calls.append({**kw, "messages": list(kw["messages"])})  # snapshot, like serialising the request
        content, stop = self.script.pop(0)
        return NS(content=content, stop_reason=stop)


def test_full_tool_loop_with_memory_and_structured_output(env):
    store, _, searcher = env
    client = FakeClient([
        ([tool("update_profile", skills=["Python", "RAG"], job_types=["internship"])], "tool_use"),
        ([tool("search_jobs", query="rag pipelines", job_type="internship")], "tool_use"),
        ([tool("present_recommendations", summary="Found one.",
               picks=[{"job_id": "j1", "reason": "RAG role matches your skills"},
                      {"job_id": "ghost", "reason": "hallucinated"},
                      {"job_id": "j2", "reason": "never surfaced by a tool"}])], "tool_use"),
    ])
    agent = JobAgent(store, searcher, client, "fake-model")
    res = agent.chat("u1", "I know Python and RAG, find internships")

    assert [c["name"] for c in res.tool_calls] == ["update_profile", "search_jobs", "present_recommendations"]
    assert res.reply == "Found one."
    assert [r.job.id for r in res.recommendations] == ["j1"]  # ghost + unseen ids dropped
    assert set(store.get_profile("u1").skills) == {"python", "rag"}  # long-term memory written

    # second turn: profile + prior conversation are carried into the prompt
    client2 = FakeClient([([text("Anything else?")], "end_turn")])
    JobAgent(store, searcher, client2, "fake-model").chat("u1", "thanks")
    sent = client2.calls[0]
    assert "rag" in sent["system"]
    assert [m["role"] for m in sent["messages"]] == ["user", "assistant", "user"]


def test_tool_errors_are_returned_to_the_model_not_raised(env):
    store, _, searcher = env
    client = FakeClient([
        ([tool("get_job_details", job_id="nope")], "tool_use"),
        ([text("Sorry, that job doesn't exist.")], "end_turn"),
    ])
    res = JobAgent(store, searcher, client, "m").chat("u2", "details on nope")
    err = client.calls[1]["messages"][-1]["content"][0]
    assert err["is_error"] and "unknown job_id" in err["content"]
    assert "doesn't exist" in res.reply


def test_step_limit(env):
    store, _, searcher = env
    client = FakeClient([([tool("list_saved_jobs")], "tool_use")] * 10)
    res = JobAgent(store, searcher, client, "m").chat("u3", "loop")
    assert res.steps == 6 and "trouble" in res.reply
