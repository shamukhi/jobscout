"""LLM provider layer.

The agent is written against Anthropic's `client.messages.create(...)` shape (content blocks with
type text / tool_use). `OpenAICompatClient` exposes the same method on top of any OpenAI-compatible
chat-completions endpoint, so switching providers is a config change, not a code change.
"""
from __future__ import annotations

import json
import os
from types import SimpleNamespace as NS
from typing import Any

# provider -> (base_url, api-key env var, default model).
# Free-tier model ids change often: override with LLM_MODEL if a default is retired.
PRESETS: dict[str, tuple[str | None, str | None, str | None]] = {
    "anthropic": (None, "ANTHROPIC_API_KEY", "claude-sonnet-5-5"),
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai/", "GEMINI_API_KEY", "gemini-2.5-flash"),
    "groq": ("https://api.groq.com/openai/v1", "GROQ_API_KEY", "llama-3.3-70b-versatile"),
    "cerebras": ("https://api.cerebras.ai/v1", "CEREBRAS_API_KEY", None),
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY", None),
    "mistral": ("https://api.mistral.ai/v1", "MISTRAL_API_KEY", "mistral-small-latest"),
    "ollama": ("http://localhost:11434/v1", None, "llama3.1"),
    "custom": (None, "LLM_API_KEY", None),  # set LLM_BASE_URL + LLM_MODEL yourself
}
AUTO_ORDER = ["anthropic", "gemini", "groq", "cerebras", "openrouter", "mistral"]


# ---------- Anthropic-shaped <-> OpenAI-shaped conversion ---------------------------------
def _get(obj: Any, key: str, default=None):
    return obj.get(key, default) if isinstance(obj, dict) else getattr(obj, key, default)


def to_openai_tools(tools: list[dict]) -> list[dict]:
    out = []
    for t in tools:
        fn = {"name": t["name"], "description": t.get("description", "")}
        schema = t["input_schema"]
        if schema.get("properties"):  # Gemini rejects an OBJECT schema with empty properties
            fn["parameters"] = schema
        out.append({"type": "function", "function": fn})
    return out


def to_openai_messages(system: str, messages: list[dict]) -> list[dict]:
    out: list[dict] = [{"role": "system", "content": system}]
    for m in messages:
        content = m["content"]
        if isinstance(content, str):
            out.append({"role": m["role"], "content": content})
        elif m["role"] == "assistant":
            text = "".join(_get(b, "text", "") for b in content if _get(b, "type") == "text")
            calls = []
            for b in content:
                if _get(b, "type") != "tool_use":
                    continue
                call = {"id": _get(b, "id"), "type": "function",
                        "function": {"name": _get(b, "name"), "arguments": json.dumps(_get(b, "input", {}))}}
                if _get(b, "extra_content"):  # e.g. Gemini thought_signature must be echoed back
                    call["extra_content"] = _get(b, "extra_content")
                calls.append(call)
            msg: dict = {"role": "assistant", "content": text or None}
            if calls:
                msg["tool_calls"] = calls
            out.append(msg)
        else:  # user turn made of tool_result blocks -> one "tool" message each
            for b in content:
                if _get(b, "type") == "tool_result":
                    out.append({"role": "tool", "tool_call_id": _get(b, "tool_use_id"), "content": _get(b, "content", "")})
    return out


def from_openai_response(resp: Any) -> NS:
    msg = resp.choices[0].message
    blocks = []
    if getattr(msg, "content", None):
        blocks.append(NS(type="text", text=msg.content))
    for i, tc in enumerate(getattr(msg, "tool_calls", None) or []):
        try:
            args = json.loads(tc.function.arguments or "{}")
        except json.JSONDecodeError:
            args = {}
        extra = getattr(tc, "extra_content", None)
        if extra is None:
            extra = (getattr(tc, "model_extra", None) or {}).get("extra_content")
        blocks.append(NS(type="tool_use", id=tc.id or f"call_{i}", name=tc.function.name, input=args,
                         extra_content=extra))
    has_tools = any(b.type == "tool_use" for b in blocks)
    return NS(content=blocks, stop_reason="tool_use" if has_tools else "end_turn")


class _Messages:
    def __init__(self, outer: "OpenAICompatClient"):
        self._o = outer

    def create(self, *, model: str, max_tokens: int, system: str, tools: list[dict], messages: list[dict]):
        resp = self._o._client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            messages=to_openai_messages(system, messages),
            tools=to_openai_tools(tools),
        )
        return from_openai_response(resp)


class OpenAICompatClient:
    def __init__(self, base_url: str, api_key: str):
        from openai import OpenAI  # lazy import

        self._client = OpenAI(base_url=base_url, api_key=api_key or "none", max_retries=2, timeout=90)
        self.messages = _Messages(self)


# ---------- factory ------------------------------------------------------------------------
def make_client(settings) -> tuple[Any, str] | tuple[None, None]:
    """Return (client, model) for the configured provider, or (None, None) if no key is available."""
    provider = settings.llm_provider
    if provider == "auto":
        provider = next((p for p in AUTO_ORDER if os.getenv(PRESETS[p][1] or "")), None)
        if provider is None:
            return None, None
    if provider not in PRESETS:
        raise ValueError(f"unknown LLM_PROVIDER {provider!r}; choose from {sorted(PRESETS)} or 'auto'")

    base_url, key_env, default_model = PRESETS[provider]
    base_url = settings.llm_base_url or base_url
    api_key = settings.llm_api_key or (os.getenv(key_env) if key_env else None)
    model = settings.llm_model or default_model
    if key_env and not api_key:
        return None, None
    if not model:
        raise ValueError(f"LLM_MODEL is required for provider {provider!r}")

    if provider == "anthropic":
        import anthropic

        return anthropic.Anthropic(api_key=api_key), model
    if not base_url:
        raise ValueError("LLM_BASE_URL is required for provider 'custom'")
    return OpenAICompatClient(base_url, api_key or "none"), model
