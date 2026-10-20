# Multi-turn Query Rewriting: Zero-shot / Same-data / Cross-data / Generic 3-shot

Reproducible experiment using **GPT-6 Luna** (`gpt-6-luna`) on **100 distinct conversations from each of two public datasets**:

1. [QReCC](https://huggingface.co/datasets/slupart/qrecc): NQ-derived subset only, to avoid QuAC overlap with CANARD.
2. [CANARD Wiki-augmented](https://huggingface.co/datasets/gaussalgo/Canard_Wiki-augmented): extractive, answerable questions only, with original article contexts and hard negatives.

The evaluation set is frozen with seed 42 and includes 100 samples *per dataset*, not 100 total. All conditions receive exactly the same 100 questions and history. Same-data and cross-data Few-shot demonstrations come from original **training splits**. Generic Few-shot uses separately authored fixed examples.

**Important:** A first GPT-6 Luna run is recorded under [`results/`](results/EXPERIMENT.md). It includes retrieval/QA evaluation using the explicitly described CANARD closed pool and QReCC answer-as-document proxy; QReCC proxy scores are not official retrieval results. Reproduction requires an OpenAI API key, model access, internet, and downloaded retriever weights.

## 1. Experimental conditions

| Condition | Prompt examples | Retrieval query |
| --- | --- | --- |
| `raw_question` (extra baseline) | N/A | Original current turn, no rewrite |
| `zero_shot` | None | GPT-6 Luna standalone rewrite |
| `same_3shot` | 3 TRAIN examples from evaluation dataset | GPT-6 Luna standalone rewrite |
| `cross_3shot` | 3 TRAIN examples from the **other** dataset | GPT-6 Luna standalone rewrite |
| `generic_3shot` | 3 synthetic general-purpose examples, same for both datasets | GPT-6 Luna standalone rewrite |

The rewrite system instruction and model settings remain identical. All few-shot examples contain **history + user question + gold rewrite**, and are fixed across queries rather than dynamically retrieved. The cross-domain comparison swaps only the source of demonstrations. The generic condition uses the **same 3 hand-authored, dataset-independent demonstrations** on both datasets (coreference, ellipsis, topic switch). This is an additional held-constant example style comparison, not trained on either corpus. For rigorous follow-up, vary 3 independently sampled demonstration sets and report variance (the initial protocol uses one fixed set).

### Generic 3-shot: independent of the evaluation datasets

The file [`generic_demos.json`](generic_demos.json) contains exactly three **synthetic, hand-written** examples shared by QReCC and CANARD:

1. **Coreference:** “Where does it live?” → “Where does the red panda live?”
2. **Ellipsis:** “What about the James Webb Space Telescope?” → “When was the James Webb Space Telescope launched?”
3. **Topic switch:** “How long is it?” → “How long is the Amazon River?”

The generic examples have conversation history and the same prompt serialization as dataset-derived examples. The instructions, model settings, history, query, and answer-generation pipeline remain unchanged across conditions. We do **not** choose generic examples using validation/test-set outcomes. You can provide a different three-example file with `--generic-demos /path/to/generic_demos.json`; this records a new fingerprint and defines a **different experiment**.

**Fair comparison:** all few-shot methods contain **exactly 3 demonstrations**. Their token lengths and linguistic content are not identical, so analyze input tokens/latency and repeat with different demo sets before making production decisions.

## 2. Install

Python 3.10+ is recommended. A CPU works; a GPU speeds up dense embedding and cross-encoder inference. Internet is needed to download datasets/models.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt

export OPENAI_API_KEY='YOUR_API_KEY'   # Windows PowerShell: $env:OPENAI_API_KEY='...'
```

Never paste the key into source files. The code uses OpenAI Responses API with `reasoning.effort=none`, which `gpt-6-luna` supports.

## 3. Prepare and freeze 200 test examples

```bash
python prepare.py --n 100 --shots 3 --seed 42
```

On macOS/Linux, once dependencies and the API key are configured, you can instead run everything with `./run_all.sh` (downloads + all API calls + results report).

Creates `data/qrecc.json`, `data/canard.json`, and `data/manifest.json`. Eval examples are sampled without selecting multiple turns from one conversation. QReCC filters `Conversation_source=nq` (QReCC also includes QuAC-derived data, overlapping with CANARD). CANARD is filtered to answerable extractive examples. We verify that few-shot examples do not share evaluation conversations and that exact questions/rewrites do not overlap across demonstrations and tests. At run time, `generic_demos.json` is validated against both evaluation sets, and its content fingerprint is stored in `results/summary.json`.

The prompt uses the complete provided dialog history for a sample (the same history in each test condition). Input length is recorded via OpenAI token usage so relative prompt sizes remain visible.

## 4. Run the entire experiment

```bash
python run.py --model gpt-6-luna --workers 5 --qrecc-mode proxy
```

- **800 GPT-6 Luna rewrite calls** (2 datasets x 100 cases x 4 prompting conditions).
- **1,000 retrieval evaluations** and up to **1,000 GPT-6 Luna answer calls** (including the optional no-rewrite baseline).
- Fixed BM25 + `sentence-transformers/all-MiniLM-L6-v2` dense retrieval, fused by RRF with `k=60`, top 30 candidates, `cross-encoder/ms-marco-MiniLM-L-6-v2` reranking.
- Top-10 retrieved document results for ranking metrics, top-5 context documents for final generation.
- Same local retrieval model, answer model, answer instructions, and gold labels for all prompting conditions.
- OpenAI calls are cached by hash of the model/prompt/settings under `results/api_cache`; subsequent runs reuse successful responses.
- Output includes `results/rewrites.jsonl`, `results/records.jsonl`, `results/summary.json`, `results/summary.csv`, and a browsable `results/report.html`.

Fast rewrite-only check (no retrieval/model weights downloaded, still uses GPT Luna API):

```bash
python run.py --skip-retrieval --no-raw
```

Disable answer generation, while still measuring retrieval:

```bash
python run.py --skip-answer
```

Change model via `--model MODEL_ID`. For initial 2x2 smoke run, prepare a separate data directory with `python prepare.py --n 2 --out-dir data_smoke` then pass `--expected-n 2 --data-dir data_smoke --out-dir results_smoke` to run.py.

## 5. Truthfulness about retrieval protocols (read before interpreting MRR)

### CANARD: small closed-pool retrieval, **not full Wikipedia search**

Corpus: gold article context chunk for each of the 100 test questions plus up to three benchmark-provided distractors per question, de-duplicated. The chunk is selected at corpus build time to contain the gold answer if the original context exceeds 3,600 characters; **that choice is the same across all conditions**. The answer itself is *never included in the prompt except as naturally occurring within the retrieved document*. There may be additional relevant-but-unlabeled passages. This is a **closed-pool contrast**, not official open-domain Wikipedia retrieval.

### QReCC: default **answer-as-document proxy**, NOT real retrieval or real QA

QReCC human gold answers and rewrite labels are downloadable as small metadata, but its **official retrieval corpus has 54 million passages**. By default, QReCC assigns each of the 100 answer strings to a *synthetic document* to make the entire 200-sample pipeline runnable on a laptop. It will artificially inflate top-k or answer metrics. Results are clearly labeled:

`ANSWER_AS_DOCUMENT_PROXY_NOT_OFFICIAL_RETRIEVAL`

**Do not publish proxy QReCC MRR/Answer EM as QReCC benchmark scores.** They are suitable only for comparing the four rewrite strategies within one fixed toy corpus. Even within proxy conditions, be cautious with generalization to production.

For **rewrite-only**, avoiding any proxy ranking:

```bash
python run.py --qrecc-mode rewrite_only
```

For **real document retrieval** over an explicitly supplied fixed QReCC corpus:

```bash
python run.py --qrecc-mode external --qrecc-docs /path/to/qrecc_passages_subset.jsonl
```

That JSONL must contain `{ "id": "<original QReCC Truth_passages ID>", "text": "source passage text" }` on each line. Include **every evaluation example's gold passage (at least one)** plus plenty of negatives and ideally use the full original retrieval collection. The loader **aborts** if any case has no gold passage in the supplied corpus. The public passage collection is available at [slupart/qrecc-passages](https://huggingface.co/datasets/slupart/qrecc-passages) and is very large; loading it in memory with this *small-scale* runner is not supported. If your goal is the official 54M-passage benchmark, replace the simple in-memory retriever with an indexed search system (Lucene/FAISS/OpenSearch). Supplied-passage evaluation is still a closed-corpus setup unless the original full corpus and qrels are used.

### Other interpretation caveats

- n=100 **per dataset** is a small benchmark; use **paired bootstrap 95% CI** and report confidence intervals.
- Results are paired by case ID with missing/failed calls excluded (coverage and error counts are exported). Failed cases are not silently scored as zero.
- Rewrite **EM/F1** is reference-text overlap, not semantic equivalence; rank and QA quality matter more.
- Reader EM is strict normalized answer match. Answer Subspan EM is 1 when the normalized gold answer occurs as a contiguous token span in the prediction; it allows surrounding explanation but does not allow reordered or partial answer tokens. Inspect token-F1 and qualitative examples as well.
- Model API latency is measured at runtime and cached, so re-runs preserve the **original** measured latency, not a zero-latency cache hit.
- For CANARD, extracted answer strings may have multiple valid paraphrases; strict answer EM underestimates semantic correctness.
- Fixed demonstration quality, prompt length, and domain differences are potential confounders. Generic demonstrations are manually authored (not sampled), which controls train-set overlap but introduces an example-style confound. Repeat with alternative demo sets for stronger conclusions.

## 6. How to analyze decreases

Open `results/report.html` locally. It contains rewrite quality, retrieval ranking and answer metrics by dataset, paired difference confidence intervals, and searchable per-query rewrite/answer comparisons with a **Few-shot loss** filter and **Few-shot type** selector (Same-data, Cross-data, Generic). For the recorded retrieval outcome and protocol caveats, see [`results/EXPERIMENT.md`](results/EXPERIMENT.md).

The summary JSON includes `case_level_comparison` with wins, losses and ties for same-shot, cross-shot, and generic-shot relative to zero-shot on `rewrite_f1`, `mrr_at_10` and `answer_em`. Paired bootstrap intervals are computed separately for each dataset.

```bash
python compare.py --out-dir results
python report.py --out-dir results
```

This does not require rerunning the API or neural retriever.

## 7. Offline code tests

```bash
pytest -q
```

The tests validate parsing, split isolation, 3-shot formatting including generic examples, proxy labels, retrieval ranking arithmetic and paired bootstrap logic. The actual HF/OpenAI pipeline cannot be executed without network + an API key; unit tests are **not** experimental results.

## Sources

- [QReCC: Open-Domain Question Answering Goes Conversational via Question Rewriting](https://aclanthology.org/2021.naacl-main.44/)
- [CANARD: Can You Unpack That? Learning to Rewrite Questions-in-Context](https://aclanthology.org/D19-1605/)
- [CANARD Wiki-augmented dataset](https://huggingface.co/datasets/gaussalgo/Canard_Wiki-augmented)
- [QReCC HF dataset with passage qrels](https://huggingface.co/datasets/slupart/qrecc)
- [QReCC full passage collection](https://huggingface.co/datasets/slupart/qrecc-passages)
- [OpenAI model docs](https://developers.openai.com/api/docs/models)
