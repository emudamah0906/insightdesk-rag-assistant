# InsightDesk — Enterprise RAG Support Assistant

An end-to-end retrieval-augmented generation (RAG) system that answers customer support
questions from a company knowledge base, with intent routing, an evaluation harness, a
serving layer, and a clickable chat UI.

> **Runs fully offline with zero external services.** The retrieval core uses pure-Python
> TF-IDF embeddings + BM25, so it works anywhere with Python + pandas. Real Claude (API or
> AWS Bedrock) can be switched on with one environment variable.

## What's inside

| Area | File(s) | Description |
|---|---|---|
| Knowledge base | `data/kb/*.md` | Source policy docs (billing, technical, account, shipping) |
| Tickets dataset | `data/tickets.csv` | Labeled support tickets for EDA + the classifier |
| Embeddings | `src/embeddings.py` | TF-IDF vectorizer + cosine similarity |
| Retrieval | `src/retriever.py` | Chunking, dense + BM25 sparse search, RRF hybrid fusion |
| Reranking | `src/rerank.py` | Lexical-overlap reranker (cross-encoder-shaped interface) |
| Generation | `src/llm.py` | Grounded prompt; mock / Claude API / Bedrock providers |
| Orchestrator | `src/rag.py` | Full pipeline + confidence gate + CLI/REPL |
| Evaluation | `src/eval.py`, `eval/golden.jsonl` | Golden-set: hit-rate, faithfulness, precision, latency |
| Intent classifier | `src/classifier_torch.py` | PyTorch MLP routing tickets to an intent |
| EDA | `scripts/01_eda.py` | Class balance, crosstabs, priority KPIs |
| API | `src/api.py` | FastAPI service (`/ask`, `/health`, `/metrics`) |
| Chat UI | `src/app.py` | Streamlit chat with sources, confidence, latency |
| Deployment design | `aws/architecture.md` | AWS production architecture + security/cost notes |
| Solution doc | `consulting/proposal.md` | Business problem, ROI model, phased delivery |

## Quickstart (works immediately)

```bash
cd InsightDesk-AI-Capstone

make eda          # explore the tickets dataset
make demo         # ask the RAG assistant a sample question (offline)
make chat         # interactive REPL
make eval         # run the golden-set evaluation

pip install torch && make classify                  # PyTorch intent classifier
pip install fastapi uvicorn pydantic && make api    # serve the API
pip install streamlit && make ui                    # chat UI
```

No `make`? Run the commands directly, e.g. `python3 src/rag.py "how long do refunds take?"`.

## LLM providers

The generation backend is selected by `src/llm.py::active_provider()`:

- **mock** (default, offline) — extractive answers from retrieved context. Deterministic, free.
- **anthropic** — real Claude. `pip install anthropic`, set `ANTHROPIC_API_KEY`.
- **bedrock** — Claude on AWS Bedrock. `pip install boto3`, set `LLM_PROVIDER=bedrock` + AWS creds.

Auto mode (no `LLM_PROVIDER` set) uses Anthropic if `ANTHROPIC_API_KEY` is present, else mock.
Any real provider **falls back to the mock on error** so the system never hard-fails.

```bash
pip install anthropic
export ANTHROPIC_API_KEY=sk-ant-...     # or copy .env.example -> .env
make demo                                # output shows provider=anthropic
```

## Architecture

```
query ─▶ retrieve (dense + BM25, RRF) ─▶ rerank ─▶ confidence gate ─▶ generate ─▶ answer + citations
                                                          │
                                              below threshold ─▶ escalate to human
```

## How it works (key design choices)

- **Hybrid retrieval:** dense (TF-IDF cosine) catches paraphrase; BM25 catches exact terms
  (IDs, codes like `401`/`429`); Reciprocal Rank Fusion combines them.
- **Confidence gate:** if the best retrieved chunk is below a cosine threshold, the query is
  treated as out-of-scope and escalated rather than answered — a guard against hallucination.
- **Separated evaluation:** `eval.py` measures retrieval and generation independently, so a
  quality regression points at the right stage.
- **Provider interface:** swapping hosted API ↔ in-VPC Bedrock ↔ offline mock requires no
  changes to the RAG pipeline.

## Project layout

```
InsightDesk-AI-Capstone/
├── data/            kb/*.md (knowledge base) · tickets.csv
├── src/             embeddings · retriever · rerank · llm · rag · eval · classifier_torch · api · app
├── scripts/         01_eda.py
├── eval/            golden.jsonl
├── aws/             architecture.md
├── consulting/      proposal.md
├── Dockerfile · Makefile · requirements.txt · .env.example
```

## Current limitations

- Offline embeddings are TF-IDF, not neural — chosen so the system runs with zero setup.
  Production would use neural embeddings + a real cross-encoder reranker.
- The mock LLM is extractive: it surfaces the relevant sentence but does not reason over it
  (e.g. "refund after 40 days" returns the 30-day policy without concluding "no"). Switching
  to a real provider resolves this.
