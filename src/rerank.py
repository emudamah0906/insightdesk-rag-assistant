"""
Reranking — reorder the shortlist for precision.

In production a CROSS-ENCODER (e.g. bge-reranker, Cohere Rerank) jointly encodes
(query, chunk) and scores relevance — far more accurate than first-stage retrieval,
but too slow to run over the whole corpus, so we only rerank the top-k shortlist.

Offline default here = a lightweight lexical-overlap + phrase-bonus reranker that
mimics the SHAPE of a cross-encoder (query-document interaction). Swap in a real
cross-encoder (e.g. bge-reranker, Cohere Rerank) behind the same interface.
"""
from __future__ import annotations
from embeddings import tokenize


def rerank(query: str, chunks: list, top_n: int = 3) -> list:
    q_terms = set(tokenize(query))
    ql = query.lower()

    def score(ch) -> float:
        terms = tokenize(ch.text)
        if not terms:
            return 0.0
        overlap = sum(1 for t in terms if t in q_terms)
        coverage = len(q_terms & set(terms)) / max(len(q_terms), 1)  # how many query terms hit
        density = overlap / len(terms)                                # how focused the chunk is
        phrase_bonus = 0.3 if any(qt in ch.text.lower() for qt in ql.split() if len(qt) > 4) else 0.0
        return coverage * 2.0 + density + phrase_bonus

    return sorted(chunks, key=score, reverse=True)[:top_n]
