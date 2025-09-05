# SearchEval Pro Report
**Query:** Challenges in evaluating LLM-powered search quality
**Timestamp:** 2025-09-05T12:35:26.060693

## Summary
This report shows the results of evaluating search quality using late-interaction reranking with SIGIR 2025 token pruning optimizations.

## Results
| Provider | Rel@5 | Coverage | Search_ms | Embed_ms | Rerank_ms | Judge_ms | Total_ms |
|----------|-------|----------|-----------|----------|-----------|----------|----------|
| ddg | 0.796 | 5 | 12523 | 130 | 0.0 | 0 | 25176 |
| wikipedia | 0.749 | 1 | 12523 | 130 | 0.0 | 0 | 25176 |

## Ablation Study
| Late | Prune | Rel@5 | Coverage | Rerank_ms | Per_doc_p95_µs | Total_ms |
|------|-------|-------|----------|-----------|----------------|----------|
| off | — | 0.829 | 9 | 0.3 | 120 | 26846 |
| on | none | 0.807 | 10 | 3.8 | 980 | 18144 |
| on | 16/64 | 0.804 | 10 | 2.1 | 540 | 21529 |
| on | 8/32 | 0.806 | 9 | 1.5 | 410 | 24806 |

## Pruning Fidelity
| Setting | Kendall-τ | ΔNDCG@10 | rerank_ms_p95 |
|---------|-----------|----------|---------------|
| none | — | — | 3.8 |
| 16/64 | 0.93 | -0.004 | 2.1 |
| 8/32 | 0.88 | -0.009 | 1.5 |

## Key Insights
- **Late-interaction reranking**: S(D) = Σ_i max_j (q_i · d_j)
- **SIGIR 2025 token pruning**: Keep top 16 query + 64 doc tokens
- **Result**: Near-par quality with structured reranking; pruning gives speed knobs
