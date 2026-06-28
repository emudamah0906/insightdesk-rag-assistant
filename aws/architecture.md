# InsightDesk — AWS Production Architecture (Solution Design)

> Maps the local prototype to a secure, scalable AWS deployment.

## Reference architecture

```
                         ┌──────────────────────── Security ────────────────────────┐
                         │ IAM least-privilege · VPC + PrivateLink · KMS · CloudTrail │
                         └────────────────────────────────────────────────────────────┘
 User ─▶ CloudFront/WAF ─▶ API Gateway ─▶ Lambda / ECS Fargate (RAG orchestrator)
                                                │
              ┌──────────────────┬──────────────┼───────────────────┬────────────────┐
              ▼                  ▼              ▼                   ▼                ▼
        Bedrock LLM        OpenSearch       ElastiCache         DynamoDB         Bedrock
        + Guardrails      (vector+BM25)     (semantic cache)   (chat history)   Knowledge Base
              ▲                  ▲
              │                  │  ingestion (offline)
        S3 (raw KB docs) ─▶ Lambda/Glue (parse→chunk→embed) ─▶ OpenSearch index
                                                │
                              EventBridge schedule → re-index on new docs

 Observability: CloudWatch (logs/metrics/alarms) · X-Ray traces · cost + drift dashboards
```

## Component mapping (prototype → AWS)

| Local prototype | AWS production service | Why |
|---|---|---|
| `embeddings.py` (TF-IDF) | Bedrock Titan / Cohere embeddings (or self-host on SageMaker) | Neural semantic quality |
| `retriever.py` (cosine + BM25 + RRF) | **OpenSearch** (k-NN vector + BM25 in one engine) | Managed hybrid search, ACL filters |
| `rerank.py` | Cohere Rerank / `bge-reranker` on SageMaker endpoint | Cross-encoder precision |
| `llm.py` (mock) | **Bedrock** (Claude) with **Guardrails** | Managed FM, data stays in-account |
| `rag.py` orchestrator | **Lambda** (spiky) or **ECS Fargate** (steady) | Stateless, autoscaled |
| `api.py` (FastAPI) | API Gateway + container | Auth, throttling, TLS |
| `eval.py` golden set | SageMaker Pipelines / CodeBuild eval gate | Block deploys on regression |
| chat history | DynamoDB | Serverless, low-latency |
| semantic cache | ElastiCache (Redis) | Cut cost/latency on repeats |

## Inference sizing & cost levers
- Right-size the model (cascade: small model first, escalate hard queries).
- **Semantic + prefix caching** for FAQ traffic — biggest cost win for support.
- Bedrock on-demand to start; **Provisioned Throughput** only once volume justifies it.
- Quote the client **p95 latency** and **cost per 1k tickets**, not just accuracy.

## Security & compliance (lead with this)
- Per-document **ACL metadata filtering** at retrieval — never surface a doc a user can't see.
- **PII redaction** + Bedrock Guardrails (denied topics, prompt-injection defense).
- KMS encryption at rest, TLS in transit, VPC endpoints (no public traffic), CloudTrail audit.
- Human-in-the-loop escalation for low-confidence / high-priority (billing, complaints).

## High availability & scaling
- Multi-AZ; autoscaling Fargate/Lambda; OpenSearch multi-node; DynamoDB on-demand.
- Async re-indexing via EventBridge; circuit-breaker + fallback to keyword search if LLM down.

## Bedrock toggle already wired
`src/llm.py` already includes a `_bedrock_generate` provider using `boto3`
**bedrock-runtime** `invoke_model`. Set `LLM_PROVIDER=bedrock` (with AWS creds + region)
to run generation on AWS Bedrock — no changes to the RAG pipeline.
