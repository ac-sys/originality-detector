# AI-Writing & Similarity Detector — Research, Design and MVP Report
*Prepared 16 October 2026. Scope: English-only MVP; multilingual (Shona/Ndebele, French, Portuguese, Swahili) is future work.*

## 1. Research findings

### 1.1 Turnitin
- **What it does:** similarity checking against its web/publication/student-paper databases plus an AI-writing indicator. It scores text in segments, each sentence gets a score, and these are averaged into a document percentage ([Turnitin Originality](https://www.turnitin.com/products/feedback-studio/originality/)).
- **Published accuracy claims:** under 1% document false-positive rate (FPR) **only for documents with ≥20% AI writing**. Sentence-level FPR is about 4%, and it is higher at the boundary between human and AI text ([Turnitin CPO update](https://www.turnitin.com/blog/ai-writing-detection-update-from-turnitins-chief-product-officer), [FPR explainer](https://www.turnitin.com/blog/understanding-the-false-positive-rate-for-sentences-of-our-ai-writing-detection-capability)). The AIW-2 white paper reports a 0.5% document FPR and a 0.33% sentence FPR on more than 700k pre-2019 papers, a 300-word minimum, and a "<20%" band where false positives are more common ([white paper, hosted by Univ. at Buffalo](https://www.buffalo.edu/content/www/lms/guides-instructors/integrations/turnitin/_jcr_content/root/maincontent/par/download/file.res/Turnitin%E2%80%99s%20AI%20Writing%20Detection%20Model%20Architecture%20and%20Testing%20Protocol.pdf)). These figures are vendor-reported, not independently audited.
- **Weaknesses we can target:**
  - The model is closed and gives no per-feature explanation.
  - Papers sit in Turnitin's repository, so the institution's corpus is locked in.
  - Pricing is quote-based per institution and in USD, which is hard for African universities. No public price list was found, so we quote no figure.
  - It is hosted offshore, which raises data-sovereignty questions.
  - It needs good bandwidth.

### 1.2 Bias and reliability of AI detectors
- Liang et al. tested 7 GPT detectors. They misclassified human TOEFL essays by non-native writers as AI-generated with an average FPR of 61.3%, while near-perfectly classifying US 8th-grade essays. The bias is tied to low perplexity ([Patterns 2023](https://www.cell.com/patterns/fulltext/S2666-3899(23)00130-7), [arXiv 2304.02819](https://arxiv.org/abs/2304.02819)). This matters directly for African universities, where most students write in a second language.
- RAID (ACL 2024) tested detectors on more than 6M generations, 11 models, 11 adversarial attacks and 4 decoding strategies. It found detectors are "easily fooled" by attacks, sampling changes, repetition penalties and unseen models ([ACL Anthology](https://aclanthology.org/2024.acl-long.674/), [dataset](https://huggingface.co/datasets/liamdugan/raid)).
- **Design consequence:** report a likelihood with uncertainty and evidence, never a verdict. Build in an appeal workflow and human review.

### 1.3 Competitors (pricing as published; checked via public pages/search 16 Oct 2026)
| Product | Model | Published pricing |
|---|---|---|
| Copyleaks | AI + plagiarism, LMS for education | Personal $16.99/mo ($13.99 annual), Pro $99.99/mo ($74.99 annual), 1 credit = 250 words; Education is a custom quote ([pricing](https://copyleaks.com/pricing)) |
| Originality.ai | AI + plagiarism, publisher-focused | Pro $14.95/mo ($12.95 annual), 2,000 credits (1 credit = 100 words); Enterprise $179/mo ([pricing](https://originality.ai/pricing)) |
| GPTZero | AI detection, education | Free 10k words/mo; Premium $23.99/mo; Professional $45.99/mo (third-party capture: [solocontenttools](https://solocontenttools.com/guides/originality-ai-vs-gptzero-freelancers); prices render client-side, so verify live) |
| Winston AI, Grammarly (AI detector + plagiarism) | consumer / prosumer | not verified today; check before quoting |
| PlagScan, Unicheck | plagiarism checkers | reported to have been absorbed into Turnitin (Ouriginal and Unicheck acquisitions). Not verified today; confirm before citing |
| Open source | Binoculars, Fast-DetectGPT, HF RoBERTa detectors | free; need engineering and evaluation |

Every competitor is a per-word/credit or per-seat USD subscription, hosted abroad, with a closed model.

### 1.4 State-of-the-art AI-text detection
- **Perplexity / burstiness:** AI text is more predictable under a language model. This is cheap, but it is exactly the signal that penalises non-native writers (Liang et al.).
- **DetectGPT** compares the probability curvature of the text against perturbed versions ([arXiv 2301.11305](https://arxiv.org/abs/2301.11305), link not re-checked today). **Fast-DetectGPT** replaces perturbation with sampling. It reports ~75% relative AUROC improvement and 340× speedup, with black-box AUROC of 0.93 on ChatGPT/GPT-4 text using GPT-J/Neo-2.7B surrogates ([arXiv 2310.05130](https://arxiv.org/abs/2310.05130), [code](https://github.com/baoguangsheng/fast-detect-gpt)).
- **Binoculars** takes the ratio of perplexity to cross-perplexity between two related LLMs. It reports detecting >90% of ChatGPT samples at 0.01% FPR, zero-shot ([arXiv 2401.12070](https://arxiv.org/abs/2401.12070)). It needs two ~7B models, so it requires a GPU or a quantized CPU build.
- **Fine-tuned classifiers (RoBERTa/DeBERTa):** strong in-domain, but they degrade on unseen generators and domains (RAID).
- **Watermarking** only works if the generator cooperates, so it can't be relied on for student submissions ([Kirchenbauer et al., arXiv 2301.10226](https://arxiv.org/abs/2301.10226), link not re-checked today).

### 1.5 Similarity methods
- **Shingling + winnowing** (Schleimer, Wilkerson & Aiken, SIGMOD 2003, the MOSS algorithm) gives a guaranteed detection of any shared run of ≥ k+w−1 words. It is robust to case, punctuation and whitespace. Used in the MVP.
- **MinHash/LSH** is the next step for document-level near-duplicate search at millions of documents.
- **Semantic embedding search** (sentence-transformers, multilingual models such as LaBSE/multilingual-E5) catches paraphrase and, later, **cross-language** plagiarism (e.g. Shona source translated into English). This is future work.
- **Web sources:** a search API (Bing/Brave/Google CSE) queried with distinctive sentences, with results fetched and fingerprinted. The MVP has a stub behind `WEB_SEARCH_API_KEY`.

### 1.6 Open datasets
- RAID ([HF](https://huggingface.co/datasets/liamdugan/raid))
- MAGE ([HF yaful/MAGE](https://huggingface.co/datasets/yaful/MAGE), used in our evaluation)
- HC3 (Hello-SimpleAI/HC3, training data of the classifier we use, so **not** used for evaluation)
- M4 (multi-generator, multi-domain, multilingual)

## 2. Differentiators for the African university market
1. **Cost:**
   - Self-hosted on one CPU server (this MVP runs on 8 cores/15 GB, no GPU).
   - Flat institutional licence in local terms instead of per-word USD credits.
   - Runs on the existing campus server or a local cloud.
2. **Data sovereignty:**
   - Student work stays on campus, which supports compliance with national data-protection law (e.g. Zimbabwe's Cyber and Data Protection Act; legal review needed).
   - **No corpus lock-in:** the library belongs to the university and exports as SQLite/JSON.
3. **Low bandwidth:** processing is local, there are no uploads abroad, the UI is lightweight, and the PDF report can be downloaded and printed.
4. **Fairness-first AI detection:**
   - Calibrated probability with an uncertainty band and a low-confidence warning for short texts.
   - A bias disclaimer and published, reproducible evaluation numbers.
   - The perplexity feature is down-weighted automatically when it doesn't help (see §4).
5. **Explainability:**
   - Sentence-level evidence: exact match, paraphrase match with the matched source sentence, and AI-likelihood.
   - Model versions are printed on every report.
6. **Appeal workflow:** students or staff lodge an appeal against a report, and it goes into a review queue (an "evidence, not verdict" policy).
7. **Always current:** a model registry with versioned detectors, remote update checks, recalibration on fresh benchmarks, an accuracy history, and one-click rollback.
8. **LMS integration (next):** a Moodle plagiarism plugin (Moodle has a plagiarism-plugin API) calling our REST API.
9. **Multilingual (future):** Shona/Ndebele similarity via multilingual embeddings, which is where global vendors are weakest.

## 3. Architecture
```
 Browser UI / Moodle plugin ──REST──> FastAPI (app/main.py)
                                       ├─ Extract (pypdf, python-docx, txt) → normalise → sentence spans
                                       ├─ JobManager (thread pool; doc split into 120-sentence blocks; progress)
                                       │    ├─ AI Ensemble (registry-selected plugins)
                                       │    │    ├─ PerplexityDetector (distilgpt2, 512-token sliding window)
                                       │    │    └─ ClassifierDetector (RoBERTa HC3, ~150-word groups)
                                       │    │    → logistic calibration (weights in manifest) → p ± band
                                       │    └─ Similarity: winnowing index (SQLite) + MiniLM embeddings (cosine ≥0.82)
                                       ├─ Library (SQLite): docs, fingerprints, sentence embeddings; grows with submissions
                                       ├─ Reports (JSON + ReportLab PDF), Appeals (JSON)
                                       └─ Registry (models/manifest.json): versions, active set, calibration, history
                 remote manifest (DETECTOR_MANIFEST_URL) ──daily check──> stage (snapshot_download) → activate → hot-swap; rollback
                 eval/evaluate.py (benchmark + recalibration) → data/eval_history.json → /api/calibration
```
**Update mechanism:**
1. The vendor (you) publishes a signed `manifest.json` listing detector specs (`id`, `slot`, `type`, `hf_id`, `version`) and calibration.
2. Each campus server checks it daily.
3. `/api/registry/update` pre-downloads the weights and activates the new set.
4. The ensemble builds new models first, then swaps under a lock, so in-flight jobs finish on the old models.
5. Every activation pushes the previous state onto `history`, so `/api/registry/rollback` restores it.
6. `eval/evaluate.py --apply` re-evaluates on a fresh benchmark sample, refits calibration and logs metrics for the dashboard.

**Production gate (planned):** auto-activate only if the new version beats the current one on AUROC and FPR at a fixed threshold on the held-out benchmark plus a local human-written (pre-2022) campus corpus.

**Scaling path:** Redis/RQ or Celery workers, Postgres + pgvector or FAISS for embeddings, MinHash-LSH for candidate retrieval, and ONNX/int8 quantized models.

## 4. MVP: measured evaluation (real numbers, 16 Oct 2026)
- **Setup:** MAGE test split, streamed and shuffled (seed 7), texts of 120–500 words. 150 human + 150 AI texts; 100 for calibration, 200 for testing (101 human, 99 AI). Models: distilgpt2 perplexity + Hello-SimpleAI/chatgpt-detector-roberta, on CPU.
- **Speed:** 0.91 s per document (~300 words) on 8 CPU cores.

| Metric (test, n=200) | Value |
|---|---|
| AUROC, perplexity feature alone | 0.688 |
| AUROC, RoBERTa classifier alone | 0.838 |
| Ensemble, default calibration @0.5: accuracy / TPR / FPR / AUROC | 0.770 / 0.727 / 0.188 / 0.842 |
| Ensemble, fitted calibration @0.5: accuracy / TPR / FPR / AUROC | 0.770 / 0.758 / 0.218 / 0.838 |
| Ensemble at threshold set for ≤1% FPR on calibration split (0.822): TPR / FPR | 0.222 / 0.020 |

**Interpretation:**
- Both detectors are small, 2023-era open models, and MAGE spans many generators and domains. An FPR of ~19–22% at the 0.5 threshold is **far too high** for misconduct decisions.
- At a strict threshold, FPR drops to 2% but only 22% of AI text is caught.
- The fitted calibration gives the perplexity feature almost no weight (0.011), which matches the finding that it is weak and biased.
- This is the honest baseline that the update mechanism exists to improve. The sample is small (n=200), so confidence intervals are wide.

**Tests:** 8 fast tests (extraction PDF/DOCX/TXT, sentence split, winnowing robustness, exact match, registry activate/rollback/remote check, full API flow including a 1,200-sentence document, PDF and appeal) plus 1 real-model test (ensemble + paraphrase detection). All pass.

## 5. Limitations
- The AI detector's accuracy is modest (above). Don't present it as competitive with Turnitin's published claims until stronger models are in place.
- AI highlighting is at ~150-word group granularity for the classifier, not true per-sentence.
- The corpus starts empty. Similarity is only as good as the library, and web search is a stub.
- English only. There is no authentication, user roles or LMS plugin yet. The SQLite and threads setup suits a single server.
- Embedding search is brute force (fine up to roughly 10^5–10^6 sentences).
- The manifest is not yet cryptographically signed.

## 6. Next steps
1. **Stronger detectors in the registry:**
   - Fast-DetectGPT or Binoculars with ~1–2B models (e.g. Qwen2.5-1.5B pair, int8) on CPU.
   - A DeBERTa-v3 classifier fine-tuned on RAID + MAGE + M4 plus local pre-2022 student essays.
   - Re-run `eval/evaluate.py` and only promote a model if it improves.
2. **Local fairness benchmark:** collect consented, pre-ChatGPT essays from Zimbabwean students and report FPR on them specifically.
3. **Corpus seeding:** past theses/dissertations from the institutional repository, MinHash-LSH, FAISS.
4. **Web-source check:** a real search provider.
5. **Integration and access:** Moodle plagiarism plugin, SSO/LDAP, roles (student pre-check vs. staff), and a review UI for the appeal queue.
6. **Multilingual:** multilingual embeddings for Shona/Ndebele and cross-language similarity.
7. **Signed manifests** and an admin dashboard for calibration history and rollback.

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
