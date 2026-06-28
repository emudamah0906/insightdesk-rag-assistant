"""
Retriever — find the most relevant chunks for a query.

LEARNING:
  - Chunking: split docs into overlapping windows so retrieval is precise.
  - Dense retrieval: cosine similarity over TF-IDF/embedding vectors (semantic-ish).
  - Sparse retrieval: BM25 (exact keyword match, great for IDs/jargon like "401", "2FA").
  - Hybrid: fuse both ranked lists with Reciprocal Rank Fusion (RRF) for best recall.
"""
from __future__ import annotations
import math, re, pathlib
from collections import Counter
from dataclasses import dataclass, field
from embeddings import TfidfEmbedder, cosine, tokenize


@dataclass
class Chunk:
    id: str
    text: str
    source: str
    vec: dict = field(default_factory=dict)


def chunk_text(text: str, size: int = 60, overlap: int = 15) -> list[str]:
    """Word-based chunking with overlap. Keeps adjacent context together."""
    words = text.split()
    if len(words) <= size:
        return [text]
    out, i = [], 0
    while i < len(words):
        out.append(" ".join(words[i:i + size]))
        i += size - overlap
    return out


def load_chunks(kb_dir: str) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(pathlib.Path(kb_dir).glob("*.md")):
        text = path.read_text()
        # split on markdown ## sections first, then window each section
        sections = re.split(r"\n##\s+", text)
        for s_idx, sec in enumerate(sections):
            for c_idx, ch in enumerate(chunk_text(sec.strip())):
                if ch.strip():
                    chunks.append(Chunk(id=f"{path.stem}#{s_idx}.{c_idx}", text=ch, source=path.name))
    return chunks


class BM25:
    """Classic BM25 ranking. Rewards rare matching terms, normalizes by length."""

    def __init__(self, docs_tokens: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs = docs_tokens
        self.N = len(docs_tokens)
        self.avgdl = sum(len(d) for d in docs_tokens) / max(self.N, 1)
        df: Counter = Counter()
        for d in docs_tokens:
            for t in set(d):
                df[t] += 1
        self.idf = {t: math.log(1 + (self.N - c + 0.5) / (c + 0.5)) for t, c in df.items()}
        self.tfs = [Counter(d) for d in docs_tokens]

    def score(self, query: str, idx: int) -> float:
        tf, dl, s = self.tfs[idx], len(self.docs[idx]), 0.0
        for t in tokenize(query):
            if t not in tf:
                continue
            num = tf[t] * (self.k1 + 1)
            den = tf[t] + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
            s += self.idf.get(t, 0.0) * num / den
        return s


def rrf(rankings: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion — combine multiple ranked id lists into one."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, cid in enumerate(ranking):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: -x[1])


class HybridRetriever:
    def __init__(self, kb_dir: str):
        self.chunks = load_chunks(kb_dir)
        self.embedder = TfidfEmbedder().fit([c.text for c in self.chunks])
        for c in self.chunks:
            c.vec = self.embedder.embed(c.text)
        self.bm25 = BM25([tokenize(c.text) for c in self.chunks])
        self.by_id = {c.id: c for c in self.chunks}

    def search(self, query: str, k: int = 5, mode: str = "hybrid") -> list[Chunk]:
        qv = self.embedder.embed(query)
        dense = sorted(self.chunks, key=lambda c: -cosine(qv, c.vec))
        dense_ids = [c.id for c in dense]
        sparse_ids = [c.id for _, c in sorted(
            ((self.bm25.score(query, i), c) for i, c in enumerate(self.chunks)),
            key=lambda x: -x[0])]
        if mode == "dense":
            order = dense_ids
        elif mode == "sparse":
            order = sparse_ids
        else:  # hybrid
            order = [cid for cid, _ in rrf([dense_ids[:20], sparse_ids[:20]])]
        return [self.by_id[cid] for cid in order[:k]]
