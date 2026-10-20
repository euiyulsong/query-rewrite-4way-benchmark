# Experiment run record

- **Model:** `gpt-6-luna`
- **Run type:** rewrite-only; retrieval and answer generation skipped
- **Evaluation data:** 100 frozen examples each from QReCC (NQ-derived subset) and CANARD Wiki-augmented; seed 42
- **Conditions:** raw-question baseline, zero-shot, same-data 3-shot, cross-data 3-shot, and fixed generic 3-shot
- **Rewrite API calls:** 800 successful calls; 0 rewrite errors
- **Date:** 2026-10-20 (environment date)
- **Protocol:** `python3 run.py --model gpt-6-luna --workers 5 --qrecc-mode rewrite_only --skip-retrieval --skip-answer --abort-on-error`
- **Artifacts:** `rewrites.jsonl`, `records.jsonl`, `summary.json`, `summary.csv`, and `report.html`

## Rewrite F1

| Dataset | Raw question | Zero-shot | Same-data 3-shot | Cross-data 3-shot | Generic 3-shot |
| --- | ---: | ---: | ---: | ---: | ---: |
| QReCC | 0.63913 | 0.79004 | 0.79394 | 0.78816 | **0.82848** |
| CANARD | 0.72911 | 0.69143 | 0.72570 | 0.72294 | **0.77484** |

## Paired bootstrap difference vs zero-shot

| Dataset | Condition | Rewrite F1 delta | 95% paired bootstrap CI |
| --- | --- | ---: | --- |
| QReCC | Same-data 3-shot | +0.00390 | [-0.02304, +0.03194] |
| QReCC | Cross-data 3-shot | -0.00188 | [-0.02316, +0.01896] |
| QReCC | Generic 3-shot | +0.03844 | [+0.01206, +0.06632] |
| CANARD | Same-data 3-shot | +0.03428 | [+0.00765, +0.06327] |
| CANARD | Cross-data 3-shot | +0.03151 | [+0.00357, +0.05978] |
| CANARD | Generic 3-shot | +0.08341 | [+0.05477, +0.11518] |

## Interpretation

Generic 3-shot had the highest mean rewrite F1 on both datasets in this run. Its paired interval was above zero on both datasets. Same-data and cross-data 3-shot intervals also were above zero on CANARD, while the QReCC intervals for these conditions included zero. This is one frozen 100-example sample per dataset and one fixed demonstration set; it does not establish generalization across samples, model runs, or demonstration sets.

This is a **rewrite-only** experiment. Retrieval ranking and answer-generation metrics are intentionally absent and must not be interpreted as zero. QReCC's synthetic answer-as-document proxy was not used. Dataset preparation and the evaluation code are described in the repository README.

## Reproducibility and privacy

The frozen evaluation JSONs and API response cache are excluded from version control. The manifest records the frozen IDs and sampling seed; regenerate the data with `python prepare.py --n 100 --shots 3 --seed 42`. Individual output records may contain dataset text and model rewrites and are published here for result audit. No API key is included.
