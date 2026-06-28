# InsightDesk — Client Solution Proposal (1-pager)

## 1. Business problem
The client's support team handles ~10,000 tickets/month. ~60% are repetitive FAQ-style
questions (refunds, login, shipping) that consume senior-agent time and slow response SLAs.

## 2. Proposed solution
An AI support assistant ("InsightDesk") that **deflects repetitive tickets** by answering
from the client's own knowledge base with citations, **routes** high-priority intents
(billing, complaints) to humans, and **escalates** anything it isn't confident about.

- Grounded in client docs via **RAG** → answers are accurate and cite sources (no hallucination free-for-all).
- **Intent classifier** routes/prioritizes before answering.
- **Guardrails + human-in-the-loop** for trust and compliance.

## 3. Why this approach (trade-offs stated)
- **RAG over fine-tuning:** the KB changes often and answers must be citable → RAG. Tone can
  be fine-tuned later if needed.
- **Start on Bedrock (managed):** fastest time-to-value, data stays in-account; revisit
  self-hosting an open model once volume/cost justifies it.

## 4. ROI model (illustrative — replace with client numbers)
| Driver | Assumption | Annual value |
|---|---|---|
| Tickets/month | 10,000 | — |
| Automatable (FAQ) share | 60% → 6,000 | — |
| Realistic deflection | 50% of automatable → 3,000/mo | — |
| Fully-loaded cost per human-handled ticket | $5 | — |
| **Gross savings** | 3,000 × $5 × 12 | **$180,000** |
| Run cost (LLM + infra + maintenance) | ~$30,000 | −$30,000 |
| **Net annual value** | | **≈ $150,000** |
| Faster response SLA | hours → seconds on FAQs | CSAT / retention upside |

**ROI ≈ (180k − 30k) / 30k ≈ 5×.  Payback ≈ 2–3 months** after a 4–6 week build.
> Present conservatively, give ranges, and tie savings to a metric the client already tracks.

## 5. Phased delivery (de-risked)
1. **Discovery (1 wk):** KB audit, success metrics, data access, compliance bar.
2. **POC (2–3 wks):** this prototype on real docs; measure deflection + faithfulness on a golden set.
3. **Pilot (3–4 wks):** one support queue, shadow then canary, human-in-the-loop, monitor CSAT.
4. **Production (4 wks):** AWS deploy, guardrails, observability, runbooks, knowledge transfer.
5. **Scale/optimize:** more queues, caching, cost tuning, retraining loop.

Each phase has a **go/no-go gate** tied to measured value.

## 6. Risks & mitigations
| Risk | Mitigation |
|---|---|
| Hallucination / wrong answer | RAG + citations + confidence gate + human escalation + eval gate |
| Stale KB | Scheduled re-indexing (EventBridge) |
| Data privacy | In-VPC Bedrock, KMS, ACL filtering, PII redaction, audit logs |
| Low adoption | Change management, agent-in-the-loop UX, measure & iterate |
| Cost creep | Semantic caching, model cascade, token budgets, dashboards |

## 7. Success metrics
Deflection rate · faithfulness (eval set) · p95 latency · CSAT · cost per ticket · escalation rate.

## Executive summary
"We give your support team an assistant that answers customers from *your own* documented
policies — with citations — deflecting about half your repetitive tickets, paying for itself
in under a quarter, while routing the sensitive ones to humans. We start with a 3-week proof
on your real data and only scale what the metrics justify."
