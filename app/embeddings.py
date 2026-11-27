"""Embedding backends behind one tiny interface.

- fastembed: ONNX runtime, ~100MB, fits free-tier hosts (default)
- st:        sentence-transformers (needs torch)
- hash:      deterministic hashing-trick vectors; lexical only. For tests/offline CI.
"""
from __future__ import annotations

import hashlib
from typing import Protocol

import numpy as np

from .textutil import tokenize


class Embedder(Protocol):
    name: str

    def embed(self, texts: list[str]) -> list[list[float]]: ...


def _l2(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    return m / np.clip(norms, 1e-12, None)


class HashEmbedder:
    name = "hash-512"

    def __init__(self, dim: int = 512):
        self.dim = dim

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        toks = tokenize(text)
        feats = toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:])]
        for f in feats:
            h = int.from_bytes(hashlib.md5(f.encode()).digest()[:8], "little")
            v[h % self.dim] += 1.0 if (h >> 63) & 1 else -1.0
        return v

    def embed(self, texts: list[str]) -> list[list[float]]:
        return _l2(np.stack([self._vec(t) for t in texts])).tolist()


class FastEmbedder:
    def __init__(self, model: str):
        from fastembed import TextEmbedding  # lazy: heavy import

        self.name = f"fastembed:{model}"
        self._m = TextEmbedding(model_name=model)

    def embed(self, texts: list[str]) -> list[list[float]]:
        return _l2(np.array(list(self._m.embed(texts)), dtype=np.float32)).tolist()


class STEmbedder:
    def __init__(self, model: str):
        from sentence_transformers import SentenceTransformer

        self.name = f"st:{model}"
        self._m = SentenceTransformer(model)

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self._m.encode(texts, normalize_embeddings=True).tolist()


def get_embedder(backend: str, model: str) -> Embedder:
    if backend == "hash":
        return HashEmbedder()
    if backend == "st":
        return STEmbedder(model)
    if backend == "fastembed":
        return FastEmbedder(model)
    raise ValueError(f"unknown EMBED_BACKEND: {backend}")
