# Retrieval eval (fastembed:BAAI/bge-small-en-v1.5, 120 jobs, 12 queries, k=10)

| mode | Recall@10 | MRR | nDCG@10 | p50 ms | p95 ms |
|---|---|---|---|---|---|
| bm25 | 0.621 | 0.845 | 0.741 | 0.5 | 3.5 |
| dense | 0.664 | 0.868 | 0.768 | 6.9 | 13.9 |
| hybrid | 0.718 | 0.925 | 0.824 | 6.7 | 20.7 |
| personalized | 0.720 | 0.944 | 0.834 | 7.6 | 14.5 |

Labels on the bundled synthetic data are generator-derived, so this checks the pipeline; use a real, hand-labelled dataset for quotable numbers.
