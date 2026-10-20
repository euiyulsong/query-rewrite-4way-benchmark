# GPT-6 Luna query-rewriting and retrieval results

- **Model:** `gpt-6-luna`
- **Evaluation:** 100 frozen cases each from QReCC (NQ-derived subset) and CANARD Wiki-augmented; seed 42
- **Conditions:** raw question, zero-shot, same-data 3-shot, cross-data 3-shot, fixed generic 3-shot
- **Rewrite API calls:** 800 successful unique rewrites; cached responses were reused for retrieval/QA evaluation
- **Retrieval:** BM25 + `all-MiniLM-L6-v2`, reciprocal-rank fusion, `ms-marco-MiniLM-L-6-v2` reranker, top-10 ranking metrics; top five documents sent to GPT-6 Luna answer reader
- **Full protocol:** `python3 run.py --model gpt-6-luna --workers 5 --qrecc-mode proxy --abort-on-error`
- **Generated:** 2026-10-20 (environment date)

## Retrieval ranking: CANARD closed pool

This is the principal retrieval result. The corpus contains each test example's gold-containing article context chunk plus up to three provided distractors per case (281 deduplicated documents total). It is a small closed-pool contrast, not open-Wikipedia retrieval. Recall@5 and Hit@5 are identical here because each case has one positive document.

| Condition | MRR@10 | Δ vs zero-shot (95% paired bootstrap CI) | Hit/Recall@5 | nDCG@10 |
| --- | ---: | ---: | ---: | ---: |
| Raw question | 0.30158 | -0.65759 [-0.74444, -0.55657] | 0.52000 | 0.37847 |
| Zero-shot | 0.95917 | baseline | 1.00000 | 0.96954 |
| Same-data 3-shot | 0.96417 | +0.00500 [0.00000, +0.01500] | 1.00000 | 0.97323 |
| Cross-data 3-shot | 0.95278 | -0.00639 [-0.02639, +0.01000] | 0.99000 | 0.96456 |
| Generic 3-shot | 0.95000 | -0.00917 [-0.02500, 0.00000] | 0.99000 | 0.96024 |

### Does rewriting improve retrieval over the raw question?

On this CANARD pool, all four rewritten-query conditions substantially outperform the raw question: zero-shot improves MRR@10 by **+0.65759** (95% paired CI **[+0.55657, +0.74444]**), same-data 3-shot by +0.66259, cross-data 3-shot by +0.65120, and generic 3-shot by +0.64842. The paired intervals for these raw-question comparisons are all above zero. (For the score deltas, these are the published paired-bootstrap results; the full per-condition values and intervals are in `summary.json`.)

Among rewritten queries, few-shot prompting does **not** show a reliable retrieval gain over zero-shot on this sample: each paired MRR interval includes zero. The point estimate is +0.00500 for same-data, -0.00639 for cross-data, and -0.00917 for generic 3-shot.

## Reader answer quality: CANARD

| Condition | Answer EM | Answer Subspan EM | Answer token F1 | Δ Subspan EM vs zero-shot (95% paired bootstrap CI) |
| --- | ---: | ---: | ---: | --- |
| Raw question | 0.01 | 0.01 | 0.09239 | -0.10 [-0.17, -0.04] |
| Zero-shot | 0.03 | 0.11 | 0.24429 | baseline |
| Same-data 3-shot | 0.02 | 0.10 | 0.24897 | -0.01 [-0.05, +0.02] |
| Cross-data 3-shot | 0.04 | 0.10 | 0.23403 | -0.01 [-0.05, +0.02] |
| Generic 3-shot | 0.02 | 0.10 | 0.24300 | -0.01 [-0.05, +0.02] |

The closed-pool retriever ranks gold contexts highly, but the answer reader's strict EM, subspan EM, and token F1 remain low. Few-shot subspan-EM differences from zero-shot have paired intervals including zero.

## QReCC reader output (proxy evidence only)

Reader metrics below use the same synthetic answer-as-document proxy corpus; they are not official QReCC QA scores.

| Condition | Proxy MRR@10 | Recall@5 | Answer EM | Answer Subspan EM | Answer token F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Raw question | 0.82160 | 0.94 | 0.05 | 0.06 | 0.29898 |
| Zero-shot | 0.98833 | 1.00 | 0.02 | 0.03 | 0.48005 |
| Same-data 3-shot | 0.97833 | 0.99 | 0.01 | 0.02 | 0.45129 |
| Cross-data 3-shot | 0.98667 | 1.00 | 0.03 | 0.04 | 0.49679 |
| Generic 3-shot | 0.98500 | 0.99 | 0.03 | 0.04 | 0.49952 |

## QReCC: synthetic answer-as-document proxy (not benchmark retrieval)

| Condition | Proxy MRR@10 | Δ vs zero-shot (95% paired bootstrap CI) | Reader Answer EM | Reader Subspan EM |
| --- | ---: | ---: | ---: | ---: |
| Raw question | 0.82160 | proxy only | 0.05 | 0.06 |
| Zero-shot | 0.98833 | baseline | 0.02 | 0.03 |
| Same-data 3-shot | 0.97833 | -0.01000 [-0.03000, 0.00000] | 0.01 | 0.02 |
| Cross-data 3-shot | 0.98667 | -0.00167 [-0.00500, 0.00000] | 0.03 | 0.04 |
| Generic 3-shot | 0.98500 | -0.00333 [-0.03000, +0.01842] | 0.03 | 0.04 |

QReCC's corpus here consists only of synthetic documents made from the 100 gold answer strings. These near-ceiling ranking metrics are **not real QReCC retrieval scores** and must not be presented as such. They only describe ranking behavior within this toy proxy corpus.

## Protocol and limitations

- MRR@10, Recall/Hit@5 and nDCG@10 evaluate retrieval ranking; Answer EM/Subspan EM/F1 evaluate a separate GPT answer reader on the top five retrieved passages. Subspan EM is 1 when the normalized gold answer occurs as a contiguous token span in the prediction, allowing surrounding explanation. Rewrite F1 is lexical overlap against the gold rewrite and is not the retrieval result.
- Pairwise deltas are candidate minus baseline on the same case IDs; 95% intervals use paired bootstrap with seed 42 (1,500 resamples).
- CANARD uses one fixed 100-case sample and a closed pool with incomplete labels; this is not a full-corpus Wikipedia retrieval benchmark.
- QReCC uses answer-as-document proxy evidence, not its official 54M-passage retrieval collection. Do not use QReCC proxy scores as benchmark numbers.
- Few-shot retrieval changes are small and uncertain here. Repeated samples and alternative demo sets are needed to generalize.
- Frozen source dataset JSON and OpenAI response cache are excluded from version control. Per-case output records and aggregate result files are committed for audit; API keys are not stored.
