"""
Embeddings — turn text into vectors so we can measure semantic similarity.

LEARNING:
  An embedding maps text -> a vector; similar meaning -> nearby vectors (cosine sim).
  Production uses a neural model (OpenAI text-embedding-3, bge, e5). To run with ZERO
  dependencies we implement a classic TF-IDF vectorizer here — same retrieval mechanics,
  no downloads. Swap in a neural embedder (e.g. sentence-transformers) later without
  changing the retriever interface.
"""
from __future__ import annotations
import math, re
from collections import Counter

_TOKEN = re.compile(r"[a-z0-9]+")

# Tiny English stopword list — removing these sharpens retrieval and stops filler words
# like "the/is/of" from creating spurious matches (a real engine uses a fuller list).
STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "is", "are", "was", "were", "be", "been", "being",
    "to", "of", "in", "on", "for", "with", "as", "by", "at", "from", "this", "that", "these",
    "those", "it", "its", "i", "you", "we", "they", "he", "she", "do", "does", "did", "how",
    "what", "when", "where", "why", "which", "who", "can", "could", "would", "should", "my",
    "me", "your", "our", "their", "if", "then", "so", "not", "no", "yes", "have", "has", "had",
    "will", "shall", "may", "might", "about", "into", "out", "up", "down", "over", "under",
}


def stem(token: str) -> str:
    """Plural normalization so 'refunds' matches 'refund'.
    We deliberately do NOT strip '-ing'/'-ed': over-stemming collapses unrelated words
    (e.g. 'meaning' -> 'mean' == 'means') and creates false matches. Conservative is safer."""
    for suf in ("es", "s"):
        if len(token) > len(suf) + 2 and token.endswith(suf):
            return token[: -len(suf)]
    return token


def tokenize(text: str, keep_stop: bool = False) -> list[str]:
    toks = _TOKEN.findall(text.lower())
    return [stem(t) for t in toks if keep_stop or t not in STOPWORDS]


class TfidfEmbedder:
    """Fit a vocabulary + IDF on a corpus, then embed any text as a sparse TF-IDF vector."""

    def __init__(self):
        self.idf: dict[str, float] = {}
        self.vocab: set[str] = set()

    def fit(self, docs: list[str]) -> "TfidfEmbedder":
        n = len(docs)
        df: Counter = Counter()
        for d in docs:
            for term in set(tokenize(d)):
                df[term] += 1
        # smoothed idf (avoids divide-by-zero, dampens common words)
        self.idf = {t: math.log((1 + n) / (1 + c)) + 1.0 for t, c in df.items()}
        self.vocab = set(self.idf)
        return self

    def embed(self, text: str) -> dict[str, float]:
        toks = tokenize(text)
        if not toks:
            return {}
        tf = Counter(toks)
        length = len(toks)
        vec = {t: (cnt / length) * self.idf.get(t, 0.0) for t, cnt in tf.items() if t in self.idf}
        # L2 normalize -> cosine similarity becomes a simple dot product
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        return {t: v / norm for t, v in vec.items()}


def cosine(a: dict[str, float], b: dict[str, float]) -> float:
    """Cosine similarity of two sparse, L2-normalized vectors = dot product."""
    if len(a) > len(b):
        a, b = b, a
    return sum(v * b.get(t, 0.0) for t, v in a.items())
