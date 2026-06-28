"""
Evaluation harness — measure RAG quality the way you'd defend it to a client.

LEARNING: evaluate RETRIEVAL and GENERATION separately.
  - Retrieval hit-rate: did the expected source document appear in the retrieved set?
  - Context precision: fraction of retrieved chunks from the right source.
  - Faithfulness (proxy): does the answer contain the required ground-truth facts?
  - Latency: p50 / p95 (an SLA the client cares about).

This is a "golden set" regression test (eval/golden.jsonl). Run it on every change to
prompts, chunking, embeddings, or model — exactly what eval-driven LLMOps looks like.
With ANTHROPIC_API_KEY set, you could add an LLM-as-judge faithfulness score too.

Run:  python src/eval.py
"""
from __future__ import annotations
import sys, json, pathlib, statistics
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from rag import RAGPipeline

GOLDEN = pathlib.Path(__file__).resolve().parent.parent / "eval" / "golden.jsonl"


def load_golden():
    return [json.loads(l) for l in GOLDEN.read_text().splitlines() if l.strip()]


def evaluate():
    rag = RAGPipeline()
    rows = load_golden()
    hits, faithful, precisions, latencies = 0, 0, [], []
    print(f"\n{'Q':45} {'srcHit':7} {'facts':6} {'ms':6}")
    print("-" * 70)
    for r in rows:
        res = rag.answer(r["question"], verbose=False)
        retrieved_sources = res["sources"]
        src_hit = r["expected_source"] in retrieved_sources
        # context precision over reranked chunk sources
        chunk_sources = [cid.split("#")[0] + ".md" for cid in res["retrieved_ids"]]
        precision = sum(s == r["expected_source"] for s in chunk_sources) / max(len(chunk_sources), 1)
        ans = res["answer"].lower()
        facts_ok = all(tok.lower() in ans for tok in r["must_contain"])
        hits += src_hit
        faithful += facts_ok
        precisions.append(precision)
        latencies.append(res["latency_ms"])
        flag = "✓" if src_hit else "✗"
        fflag = "✓" if facts_ok else "✗"
        print(f"{r['question'][:44]:45} {flag:^7} {fflag:^6} {res['latency_ms']:>5}")

    n = len(rows)
    p95 = sorted(latencies)[max(0, int(0.95 * n) - 1)]
    print("-" * 70)
    print(f"Retrieval hit-rate     : {hits}/{n}  ({hits/n:.0%})")
    print(f"Faithfulness (facts)   : {faithful}/{n}  ({faithful/n:.0%})")
    print(f"Mean context precision : {statistics.mean(precisions):.0%}")
    print(f"Latency p50 / p95 (ms) : {statistics.median(latencies):.1f} / {p95:.1f}")
    print("\nInterpretation: high hit-rate but lower faithfulness => retrieval is fine, the "
          "GENERATION step (or chunking) is the bottleneck. That diagnosis is the whole point.")
    return {"hit_rate": hits / n, "faithfulness": faithful / n}


if __name__ == "__main__":
    evaluate()
