"""Stable local embeddings and BM25. No network, no model download."""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field

TOKEN_RE = re.compile(r"[a-z0-9_./+-]+")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall((text or "").lower())


class HashingEmbedder:
    """Character n-gram hashing vector. Deterministic across processes."""

    def __init__(self, dim: int = 256, ngram_range: tuple[int, int] = (3, 5)) -> None:
        self.dim = dim
        self.ngram_range = ngram_range

    def embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        blob = f" {text.lower()} "
        low, high = self.ngram_range
        for n in range(low, high + 1):
            if len(blob) < n:
                continue
            for index in range(len(blob) - n + 1):
                gram = blob[index : index + n]
                digest = hashlib.blake2b(gram.encode("utf-8"), digest_size=8).digest()
                slot = int.from_bytes(digest, "little") % self.dim
                sign = 1.0 if digest[-1] % 2 == 0 else -1.0
                vector[slot] += sign
        return _l2_normalize(vector)


def cosine(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    return sum(a * b for a, b in zip(left, right))


def _l2_normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]


@dataclass
class BM25Index:
    k1: float = 1.5
    b: float = 0.75
    documents: list[list[str]] = field(default_factory=list)
    doc_freq: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    avg_len: float = 0.0

    def fit(self, texts: list[str]) -> None:
        self.documents = [tokenize(text) for text in texts]
        self.doc_freq = defaultdict(int)
        total = 0
        for tokens in self.documents:
            total += len(tokens)
            for token in set(tokens):
                self.doc_freq[token] += 1
        self.avg_len = (total / len(self.documents)) if self.documents else 0.0

    def score(self, query: str) -> list[float]:
        q_tokens = tokenize(query)
        if not self.documents:
            return []
        n_docs = len(self.documents)
        scores = [0.0] * n_docs
        query_counts = Counter(q_tokens)
        for token, qtf in query_counts.items():
            df = self.doc_freq.get(token, 0)
            if df == 0:
                continue
            idf = math.log(1 + (n_docs - df + 0.5) / (df + 0.5))
            for index, tokens in enumerate(self.documents):
                tf = tokens.count(token)
                if tf == 0:
                    continue
                denom = tf + self.k1 * (1 - self.b + self.b * (len(tokens) / (self.avg_len or 1.0)))
                scores[index] += idf * qtf * ((tf * (self.k1 + 1)) / denom)
        return scores
