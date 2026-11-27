from __future__ import annotations

import re

_TOKEN = re.compile(r"[a-z0-9][a-z0-9+#.]*")

_SKILL_ALIASES = {
    "js": "javascript",
    "ts": "typescript",
    "postgres": "postgresql",
    "k8s": "kubernetes",
    "ml": "machine learning",
    "nodejs": "node.js",
    "node": "node.js",
    "large language models": "llm",
    "llms": "llm",
    "genai": "llm",
    "restful apis": "rest apis",
    "rest": "rest apis",
    "spring": "spring boot",
    "powerbi": "power bi",
}


def tokenize(text: str) -> list[str]:
    return [t.rstrip(".") for t in _TOKEN.findall(text.lower())]


def norm_skill(skill: str) -> str:
    s = skill.strip().lower()
    return _SKILL_ALIASES.get(s, s)
