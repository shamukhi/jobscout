import json
from types import SimpleNamespace as NS

import pytest

from app.agent import TOOLS, JobAgent
from app.config import Settings
from app.llm import OpenAICompatClient, make_client, to_openai_messages, to_openai_tools


def oa_response(content=None, tool_calls=None):
    calls = [NS(id=i, function=NS(name=n, arguments=a)) for i, n, a in (tool_calls or [])]
    return NS(choices=[NS(message=NS(content=content, tool_calls=calls or None))])


class FakeOpenAI:
    """Stands in for openai.OpenAI: records payloads, returns scripted chat-completion responses."""

    def __init__(self, script):
        self.script, self.payloads = list(script), []
        self.chat = NS(completions=NS(create=self._create))

    def _create(self, **kw):
        self.payloads.append({**kw, "messages": list(kw["messages"])})
        return self.script.pop(0)


def make_adapter(script):
    c = OpenAICompatClient.__new__(OpenAICompatClient)  # skip real SDK construction
    c._client = FakeOpenAI(script)
    from app.llm import _Messages
    c.messages = _Messages(c)
    return c


def test_tool_schema_conversion_covers_every_agent_tool():
    oa = to_openai_tools(TOOLS)
    assert [t["function"]["name"] for t in oa] == [t["name"] for t in TOOLS]
    assert all(t["type"] == "function" and t["function"]["parameters"]["type"] == "object" for t in oa)


def test_message_conversion_handles_all_turn_shapes():
    blocks = [NS(type="text", text="let me look"), NS(type="tool_use", id="c1", name="search_jobs", input={"query": "x"})]
    msgs = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": blocks},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "c1", "content": "{}"}]},
    ]
    out = to_openai_messages("SYS", msgs)
    assert out[0] == {"role": "system", "content": "SYS"}
    assert out[2]["tool_calls"][0]["function"]["arguments"] == json.dumps({"query": "x"})
    assert out[2]["content"] == "let me look"
    assert out[3] == {"role": "tool", "tool_call_id": "c1", "content": "{}"}


def test_agent_runs_unchanged_on_openai_compatible_client(env):
    store, _, searcher = env
    client = make_adapter([
        oa_response(tool_calls=[("a1", "update_profile", json.dumps({"skills": ["Python"]}))]),
        oa_response(tool_calls=[(None, "search_jobs", json.dumps({"query": "rag pipelines"}))]),  # missing id (Gemini quirk)
        oa_response(tool_calls=[("a3", "present_recommendations", json.dumps(
            {"summary": "Here.", "picks": [{"job_id": "j1", "reason": "fits"}]}))]),
    ])
    res = JobAgent(store, searcher, client, "any-model").chat("u9", "I know Python, find RAG roles")
    assert [c["name"] for c in res.tool_calls] == ["update_profile", "search_jobs", "present_recommendations"]
    assert [r.job.id for r in res.recommendations] == ["j1"]
    sent = client._client.payloads
    assert sent[0]["tools"][0]["type"] == "function" and sent[0]["messages"][0]["role"] == "system"
    assert any(m["role"] == "tool" for m in sent[1]["messages"])  # tool results fed back correctly


def test_malformed_tool_arguments_become_empty_input_not_a_crash(env):
    store, _, searcher = env
    client = make_adapter([
        oa_response(tool_calls=[("a1", "list_saved_jobs", "{not json")]),
        oa_response(content="ok"),
    ])
    assert JobAgent(store, searcher, client, "m").chat("u8", "saved?").reply == "ok"


def test_make_client_selection(monkeypatch):
    for k in ("ANTHROPIC_API_KEY", "GEMINI_API_KEY", "GROQ_API_KEY", "CEREBRAS_API_KEY", "OPENROUTER_API_KEY", "MISTRAL_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    assert make_client(Settings(llm_provider="auto")) == (None, None)

    monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
    client, model = make_client(Settings(llm_provider="auto"))
    assert isinstance(client, OpenAICompatClient) and model == "llama-3.3-70b-versatile"

    client, model = make_client(Settings(llm_provider="groq", llm_model="other-model"))
    assert model == "other-model"

    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    with pytest.raises(ValueError, match="LLM_MODEL is required"):
        make_client(Settings(llm_provider="openrouter"))
    with pytest.raises(ValueError, match="unknown LLM_PROVIDER"):
        make_client(Settings(llm_provider="nope"))
    # local Ollama needs no key
    client, model = make_client(Settings(llm_provider="ollama"))
    assert isinstance(client, OpenAICompatClient) and model == "llama3.1"
