"""
LLM layer — generate a grounded answer from retrieved context.

Provider is chosen by env var LLM_PROVIDER (or auto-detected):
  - "mock"      (default, offline) — extractive: returns the most query-relevant
                sentences from context. Deterministic, free, good enough to run eval.
  - "anthropic" — real Claude. `pip install anthropic`, set ANTHROPIC_API_KEY.
  - "bedrock"   — Claude on AWS Bedrock via boto3. `pip install boto3`, configure AWS creds.

AUTO mode (LLM_PROVIDER unset): use anthropic if ANTHROPIC_API_KEY is present, else mock.

Why a provider interface? It's exactly the build-vs-buy / data-residency lever you offer a
client: start on a hosted API, move to Bedrock-in-VPC for compliance, all
without touching the RAG pipeline. Any real provider FALLS BACK to mock on error so demos
never break.
"""
from __future__ import annotations
import os, re
from embeddings import tokenize

SYSTEM = (
    "You are InsightDesk, an enterprise support assistant. Answer ONLY using the provided "
    "context. If the answer is not in the context, say you don't know and offer to escalate "
    "to a human. Be concise (2-4 sentences). Cite the source file(s) in square brackets, "
    "e.g. [billing.md]. Do not invent policies, numbers, or dates."
)

IDK = ("I don't have that in the knowledge base. I can escalate this to a human "
       "agent if you'd like.")


def active_provider() -> str:
    p = os.getenv("LLM_PROVIDER", "").lower()
    if p in {"mock", "anthropic", "bedrock"}:
        return p
    return "anthropic" if os.getenv("ANTHROPIC_API_KEY") else "mock"


# --------------------------------------------------------------------------- mock
def _split_sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n", text) if len(s.strip()) > 15]


def _mock_generate(query: str, contexts: list) -> str:
    q_terms = set(tokenize(query))
    if not q_terms:
        return IDK
    scored = []
    for ch in contexts:
        for sent in _split_sentences(ch.text):
            overlap = len(q_terms & set(tokenize(sent)))
            if overlap:
                scored.append((overlap, sent, ch.source))
    scored.sort(key=lambda x: -x[0])
    if not scored:
        return IDK
    seen, picked, sources = set(), [], []
    for _, sent, src in scored:
        if sent not in seen:
            seen.add(sent); picked.append(sent); sources.append(src)
        if len(picked) == 2:
            break
    cites = " ".join(f"[{s}]" for s in dict.fromkeys(sources))
    return " ".join(picked) + f" {cites}"


# ------------------------------------------------------------------- real providers
def _build_prompt(query: str, contexts: list) -> str:
    ctx = "\n\n".join(f"[{c.source}] {c.text}" for c in contexts)
    return f"Context:\n{ctx}\n\nQuestion: {query}"


def _anthropic_generate(query: str, contexts: list) -> str:
    import anthropic  # pip install anthropic
    client = anthropic.Anthropic()
    msg = client.messages.create(
        model=os.getenv("ANTHROPIC_MODEL", "claude-opus-4-8"),
        max_tokens=400, temperature=0, system=SYSTEM,
        messages=[{"role": "user", "content": _build_prompt(query, contexts)}],
    )
    return msg.content[0].text.strip()


def _bedrock_generate(query: str, contexts: list) -> str:
    import boto3, json  # pip install boto3
    brt = boto3.client("bedrock-runtime", region_name=os.getenv("AWS_REGION", "us-east-1"))
    model_id = os.getenv("BEDROCK_MODEL_ID", "anthropic.claude-3-5-sonnet-20241022-v2:0")
    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 400, "temperature": 0, "system": SYSTEM,
        "messages": [{"role": "user", "content": _build_prompt(query, contexts)}],
    }
    resp = brt.invoke_model(modelId=model_id, body=json.dumps(body))
    return json.loads(resp["body"].read())["content"][0]["text"].strip()


_REAL = {"anthropic": _anthropic_generate, "bedrock": _bedrock_generate}


def generate(query: str, contexts: list) -> str:
    provider = active_provider()
    if provider in _REAL:
        try:
            return _REAL[provider](query, contexts)
        except Exception as e:  # never break the demo; degrade to mock
            return f"[{provider} unavailable: {e}; using offline mock]\n" + _mock_generate(query, contexts)
    return _mock_generate(query, contexts)
