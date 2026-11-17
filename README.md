# Originality Detector (MVP)

Self-hosted similarity (plagiarism) + AI-writing evidence checker for universities. English only in this MVP.
No humanizer / detector-evasion features, by design.

## Run
```bash
cd /workspace/detector && . .venv/bin/activate
uvicorn app.main:app --host 0.0.0.0 --port 8100      # open http://localhost:8100
```
Fresh install: `./download_models.sh` (creates venv, installs requirements.txt, pre-downloads v2 models + v1 models for rollback).

Env vars: `DETECTOR_MANIFEST_URL` (remote model manifest, http(s):// or file://), `WEB_SEARCH_API_KEY`
(legacy, unused); `WEB_SOURCES=wikipedia,openalex[,brave]` opt-in web-source check (`BRAVE_API_KEY` for brave), `DETECTOR_WORKERS` (default 2), `DETECTOR_DATA`, `DETECTOR_LIGHT=1` (no models; tests).

## API
| Method | Path | Purpose |
|---|---|---|
| POST | /api/submit (file, title, add_to_library) | queue PDF/DOCX/TXT, returns job_id |
| GET | /api/jobs/{id} | status + progress |
| GET | /api/reports/{id} , /api/reports/{id}/pdf | JSON / PDF report |
| POST/GET | /api/library | add reference doc / list library |
| POST | /api/appeals/{id} ; GET /api/appeals | appeal workflow |
| GET | /api/registry ; /api/calibration | manifest; evaluation history |
| POST | /api/registry/check , /update , /rollback | update mechanism |

## Tests & evaluation
```bash
python -m pytest -q                                   # fast tests (no models)
RUN_MODEL_TESTS=1 DETECTOR_LIGHT=0 python -m pytest -q tests/test_models.py
python eval/evaluate.py --n 150 [--apply]             # v1 MAGE benchmark
python eval/build_dataset.py; python eval/run_features.py fakespot test   # v2 eval set + features
python eval/eval_ai_v2.py; python eval/calibrate_v2.py fakespot fakespot-ai/roberta-base-ai-text-detection-v1 AI
python eval/build_sim_dataset.py; python eval/eval_similarity.py; python eval/winnow_grid.py
```
See RESEARCH_AND_DESIGN.md for research, architecture and measured results.

## v2 results (measured 16 Oct 2026; real numbers only)

### AI-writing detection
**Evaluation data:**
- **Held-out test:** 1,500 texts (MAGE test + MAGE OOD GPT-4 + GPT-4-paraphrased + RAID stratified by model/attack). Long texts were cut to the first 400 words.
- **Calibration split:** 800 texts, used for thresholds.
- **ESL / native human essays:** 91 TOEFL essays (Liang et al.), 248 ELLIPSE/Feedback-Prize ELL essays, 80 ICNALE essays and 88 native Hewlett essays.
- **Threshold method:** thresholds are chosen on the calibration split and then applied unchanged to the test and ESL sets. The test FPR can therefore differ from the 1%/2% target.

**Document-level results:**

| Detector (doc level, single pass ≤512 tokens) | Test AUROC | TPR @1% FPR thr | TPR @2% FPR thr | Test FPR @1% / @2% thr | TOEFL FPR @1% / @2% thr |
|---|---|---|---|---|---|
| **v1** ensemble (distilgpt2 + HC3-RoBERTa) | 0.615 | 0.049 | 0.105 | 0.002 / 0.012 | 0.077 / 0.099 |
| distilgpt2 log-prob alone | 0.668 | 0.291 | 0.377 | 0.032 / 0.078 | 0.055 / 0.121 |
| SmolLM2-135M Binoculars (zero-shot) | 0.635 | 0.306 | 0.379 | – | – / 0.088 |
| SmolLM2-135M Fast-DetectGPT (1-model) | 0.667 | 0.381 | 0.418 | – | – / 0.066 |
| **fakespot-ai/roberta-base-ai-text-detection-v1 (promoted)** | **0.963** | **0.785** | **0.802** | 0.010 / 0.017 | 0.044 / 0.055 |
| Logistic stack (fakespot + SmolLM features) | 0.963 | 0.785 | 0.802 | same as fakespot | 0.055 @2% |
| GBM stack | 0.962 | 0.753 | 0.797 | – | 0.066 @2% |

**Notes on these results:**
- **ESL false-positive rate for the promoted model:**
  - At the 1% threshold: ELLIPSE 0.0%, ICNALE 0.0%, TOEFL 4.4%, native Hewlett 0.0%.
  - v1 at the same thresholds: ELLIPSE 2.0%, ICNALE 2.5%, TOEFL 7.7%.
- **Per-source AUROC (promoted model vs v1):**

  | Source | Promoted | v1 |
  |---|---|---|
  | MAGE test | 0.926 | 0.718 |
  | MAGE OOD GPT-4 | 0.989 | 0.909 |
  | MAGE GPT-4-paraphrased | 0.927 | 0.830 |
  | RAID | 0.995 | 0.645 |

  The model's training data is undisclosed, so the RAID figure may be inflated by overlap.
- **Avoid a plain 0.5 probability cut-off.** With isotonic calibration fitted on the AI-heavy calibration split, a 0.5 cut-off flagged 69% of TOEFL essays (promoted model) and 100% (v1). v2 therefore uses FPR-targeted thresholds and class-balanced calibration.

**As deployed in the app:** documents are scored in ~300-word chunks with a word-weighted logit, plus ~3-sentence windows for highlighting (`eval/calibrate_v2.py`). Results on the first 600 test texts, with the ESL set:
- **Document level:**
  - AUROC 0.983.
  - At the 1% threshold: TPR 0.891, test FPR 0.036, TOEFL FPR 12.1%.
  - At the 2% threshold: TPR 0.901, test FPR 0.042, TOEFL FPR 14.3%.
- **Window level (2,114 test windows):**
  - AUROC 0.926.
  - At the 1% threshold: TPR 0.446, FPR 0.004.
  - At the 2% threshold: TPR 0.575, FPR 0.013.
  - Window FPR at the 2% threshold on ESL: TOEFL 3.8%, ICNALE 2.4%, ELLIPSE 0.5%.
- **Speed:**
  - Classifier only: ~0.8–1.3 s per 1,000 words.
  - With highlighting windows: 3.95 s per 1,000 words, measured while other jobs shared the CPU.
- **TOEFL false positives stay well above the 1–2% target.** Treat AI flags as prompts for a conversation, not as evidence.

**Promoted or rejected:**
- **Promoted:** fakespot RoBERTa, as the single classifier with FPR thresholds. The stacks were no better, so the simpler model won.
- **Rejected (measured):**
  - The v1 ensemble.
  - SmolLM2-135M Binoculars and Fast-DetectGPT: too weak at this model size.
  - The GBM stack.
- **Not finished (CPU time):**
  - Qwen2.5-0.5B pair: 8.8 s per 1,000 words, dropped.
  - desklib DeBERTa-v3-large (MIT, trained on RAID): ~7.7 s per 1,000 words, partial features only, not evaluated.
  - DeBERTa-v3-small fine-tune: the first run hit an fp16 NaN; the fixed fp32 run was stopped before finishing.

### Similarity
**Evaluation data:** a synthetic PAN-style set built from Wikipedia (CC BY-SA).
- **Library:** 2,479 documents (all sources plus 2,000 distractors).
- **Documents:** 100 dev and 200 test documents, with 1,004 plagiarised test sentences.
- **Obfuscation types:** verbatim, light edits, back-translation (opus-mt en→de→en) and light edits plus sentence shuffling.
- **Also included:** quoted-and-cited passages and reference lists.
- **Thresholds** were tuned on dev.

| Config (test) | Precision | Recall | F1 | Recall: verbatim / light / backtransl. / shuffle | Quoted flagged | Bibliography flagged | Check s/1000 words |
|---|---|---|---|---|---|---|---|
| **v1** MiniLM, thr 0.82, no exclusions | 0.947 | 0.920 | 0.933 | 0.974 / 0.896 / 0.899 / 0.907 | 14/60 | 4/220 | 0.55 |
| MiniLM + exclusions + common-phrase filter | 0.961 | 0.914 | 0.937 | 0.963 / 0.896 / 0.891 / 0.903 | 3/60 | 0/220 | 0.56 |
| bge-small-en-v1.5 | 0.946 | 0.948 | 0.947 | 0.963 / 0.939 / 0.946 / 0.944 | 3/60 | 0/220 | 0.95 |
| **e5-small-v2 (promoted), thr 0.92** | **0.960** | **0.947** | **0.953** | 0.963 / 0.939 / 0.938 / 0.948 | 2/60 | 0/220 | 2.96* |

\*Measured while other experiments ran in parallel, so timings are inflated and noisy. The gte-small, e5+k4w3 and bge+k4w3 runs were stopped before finishing.

**Winnowing grid (exact channel only, test F1):**
- v1 setting (k=5, w=4, coverage 0.3): **0.745**.
- Best on dev (k=4, w=3, coverage 0.2): **0.832** (precision 0.881, recall 0.788).
- Not promoted: it was never measured together with the embedding channel, and it lowers exact-channel precision.

**Other v2 changes:**
- FAISS vector index (exact inner product), persisted, with re-embedding when the registry switches embedding model.
- Document-frequency whitelist for common phrases.
- Quote and bibliography exclusion.
- The report shows the matched source passage.
- Opt-in web-source check (`WEB_SOURCES=wikipedia,openalex`, both free with no key, or `brave` with `BRAVE_API_KEY`). It was verified live against Wikipedia.
- **Not done:** the 12k-document and 100+ page throughput test. The script is `eval/throughput.py`, but it was not run before the stop.

### Registry
- **Manifest v2 actives:** `fakespot-roberta@2.0.0`, calibration `v2-fakespot-20261016`, similarity `e5-small-w5@2.0.0`.
- **Rollback:** `POST /api/registry/rollback` restores the v1 detectors, calibration and similarity settings. This is tested, and the library re-embeds on rollback.


## v3 (2026-10-30) — active
- AI writing: fakespot RoBERTa + desklib DeBERTa-v3-large. Both must agree (AND rule) at thresholds that include ESL essays in the human pool.
- Measured on held-out sets: test FPR 0.035→0.009, TOEFL ESL FPR 0.121→0.055, real CS224N student-report FPR 0.572→0.103. Strict recall 0.895→0.832.
- Similarity is unchanged from v2. The v3 paraphrase path is in the code but not activated (it has not been evaluated).
- Full numbers and caveats: **V3_RESULTS.md**.
- Rollback to v2: `POST /api/registry/rollback` (or copy `models/manifest_v2_backup.json`). A full v2 copy is in `/workspace/detector_v2_backup`.
- v3 needs ~1.7 GB more RAM and the `desklib/ai-text-detector-v1.01` download. Real-model tests: `RUN_MODEL_TESTS=1 pytest`.

## v3.1 (2026-10-30)
Active similarity matcher `e5-small-v3match@3.1.0` (seed/extend paraphrase matching + v31 quote/bibliography exclusion); see V3_RESULTS.md. Roll back with `POST /api/registry/rollback`.
Optional library growth: `.venv/bin/python scripts/ingest_sources.py --help` (OpenAlex / arXiv / Wikipedia; use `--dry-run` first).

**v3.2:** similarity matcher `e5-small-v3match@3.2.0` adds a gated condense stage and fixes quote exclusion. See V3_RESULTS.md. The free-library ingest raised false positives, so it is kept out of the live DB (`data/corpus_with_library_v32.db`).
