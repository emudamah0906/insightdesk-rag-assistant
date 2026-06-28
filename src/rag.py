"""
RAG orchestrator — the full pipeline.

Flow:  query -> retrieve (hybrid) -> rerank -> generate (grounded, cited) -> answer

Run:
    python src/rag.py "How long do refunds take?"
    python src/rag.py            # interactive REPL
"""
from __future__ import annotations
import sys, pathlib, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))  # allow direct execution

import config  # noqa: F401  -- loads .env into os.environ on import
from retriever import HybridRetriever
from rerank import rerank
from llm import generate, IDK, active_provider
from embeddings import cosine

KB_DIR = str(pathlib.Path(__file__).resolve().parent.parent / "data" / "kb")

# Minimum retrieval similarity to attempt an answer. Below this the query is treated
# as out-of-scope and escalated rather than answered (groundedness guard).
CONF_THRESHOLD = 0.15


class RAGPipeline:
    def __init__(self, kb_dir: str = KB_DIR, mode: str = "hybrid"):
        self.retriever = HybridRetriever(kb_dir)
        self.mode = mode

    def answer(self, query: str, k: int = 5, top_n: int = 3, verbose: bool = True) -> dict:
        t0 = time.perf_counter()
        retrieved = self.retriever.search(query, k=k, mode=self.mode)
        reranked = rerank(query, retrieved, top_n=top_n)
        qv = self.retriever.embedder.embed(query)
        confidence = max((cosine(qv, c.vec) for c in reranked), default=0.0)
        if confidence < CONF_THRESHOLD:
            answer = IDK
        else:
            answer = generate(query, reranked)
        latency_ms = (time.perf_counter() - t0) * 1000
        result = {
            "query": query,
            "answer": answer,
            "sources": list(dict.fromkeys(c.source for c in reranked)),
            "retrieved_ids": [c.id for c in reranked],
            "confidence": round(confidence, 3),
            "provider": active_provider(),
            "latency_ms": round(latency_ms, 1),
        }
        if verbose:
            print(f"\nQ: {query}")
            print(f"A: {answer}")
            print(f"   provider={result['provider']}  sources={result['sources']}  "
                  f"conf={result['confidence']}  chunks={result['retrieved_ids']}  "
                  f"{result['latency_ms']}ms")
        return result


def main():
    rag = RAGPipeline()
    if len(sys.argv) > 1:
        rag.answer(" ".join(sys.argv[1:]))
        return
    print(f"InsightDesk RAG — provider={active_provider()} — ask a question (blank line to "
          f"quit).\n(Set ANTHROPIC_API_KEY or LLM_PROVIDER=bedrock for real generation.)")
    while True:
        try:
            q = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q:
            break
        rag.answer(q)


if __name__ == "__main__":
    main()
