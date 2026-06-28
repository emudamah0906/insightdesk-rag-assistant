"""
FastAPI service — serve the RAG assistant over HTTP.

Endpoints:
  GET  /health        -> liveness probe (for load balancers / k8s)
  POST /ask  {query}  -> grounded answer + sources + confidence + latency
  GET  /metrics       -> simple in-process counters (a stand-in for Prometheus)

This is the thin, stateless serving layer you'd containerize (see Dockerfile) and put
behind an API Gateway (see aws/architecture.md). Model/index load once at startup.

Setup:  pip install fastapi uvicorn pydantic
Run:    uvicorn src.api:app --reload --port 8080
Test:   curl -s localhost:8080/ask -H 'content-type: application/json' \
             -d '{"query":"how long do refunds take?"}' | python -m json.tool
"""
from __future__ import annotations
import sys, pathlib, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

try:
    from fastapi import FastAPI
    from pydantic import BaseModel
except ImportError:  # pragma: no cover
    print("Install serving deps:  pip install fastapi uvicorn pydantic")
    raise

from rag import RAGPipeline

app = FastAPI(title="InsightDesk RAG API", version="1.0")
_rag = RAGPipeline()              # load KB + index once at startup
_metrics = {"requests": 0, "escalations": 0, "total_latency_ms": 0.0}


class Ask(BaseModel):
    query: str
    k: int = 5


@app.get("/health")
def health():
    return {"status": "ok", "chunks_indexed": len(_rag.retriever.chunks)}


@app.post("/ask")
def ask(req: Ask):
    res = _rag.answer(req.query, k=req.k, verbose=False)
    _metrics["requests"] += 1
    _metrics["total_latency_ms"] += res["latency_ms"]
    if res["answer"].startswith("I don't have that"):
        _metrics["escalations"] += 1
    return res


@app.get("/metrics")
def metrics():
    n = max(_metrics["requests"], 1)
    return {
        "requests": _metrics["requests"],
        "escalation_rate": round(_metrics["escalations"] / n, 3),
        "avg_latency_ms": round(_metrics["total_latency_ms"] / n, 2),
    }
