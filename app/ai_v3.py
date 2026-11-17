"""AI-writing detection v3 (CPU, self-hosted).
Changes vs v2 (see V3_RESULTS.md for measured effect):
 1. Two-model fusion: fakespot RoBERTa-base (v2) + desklib DeBERTa-v3-large (MIT, RAID-trained), combined by a
    logistic model fitted on the calibration split (eval/v3/fit_v3.py).
 2. Chunking without tiny tails (a trailing chunk < min_tail words is merged into the previous one).
 3. ESL-aware thresholds: the human pool used to set the 1%/2% FPR thresholds includes 300 extra ELLIPSE learner
    essays (disjoint from every reported ESL test set).
 4. Sentence highlighting from overlapping sliding windows (stride 1 sentence) averaged per sentence, then
    segment smoothing (isolated single-sentence flags are dropped) - better mixed human+AI localisation."""
import math, os, threading
import numpy as np
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class _Base:
    batch_size = 16
    def logits(self, texts):
        if not texts: return []
        out = []
        with torch.inference_mode():
            for b in range(0, len(texts), self.batch_size):
                out += self._batch(texts[b:b + self.batch_size])
        return out

class HFClassifier(_Base):
    batch_size = 16
    def __init__(self, spec):
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        hf = spec["hf_id"]; self.spec = spec; self.device = DEVICE
        try:
            self.tok = AutoTokenizer.from_pretrained(hf, local_files_only=True)
            self.model = AutoModelForSequenceClassification.from_pretrained(hf, local_files_only=True).to(self.device).eval()
        except Exception:
            self.tok = AutoTokenizer.from_pretrained(hf)
            self.model = AutoModelForSequenceClassification.from_pretrained(hf).to(self.device).eval()
        lab = {v: int(k) for k, v in self.model.config.id2label.items()}; self.ai_idx = lab.get(spec.get("ai_label", "LABEL_1"), 1)
    def _batch(self, texts):
        x = self.tok(texts, truncation=True, max_length=512, padding=True, return_tensors="pt").to(self.device)
        lg = self.model(**x).logits.float(); return (lg[:, self.ai_idx] - lg[:, 1 - self.ai_idx]).cpu().tolist()

class Desklib(_Base):
    """desklib/ai-text-detector-v1.01 (MIT): DeBERTa-v3-large encoder + mean pooling + linear head."""
    batch_size = 8
    def __init__(self, spec):
        from transformers import AutoTokenizer, AutoConfig, AutoModel
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file
        hf = spec["hf_id"]; self.spec = spec; self.device = DEVICE
        try:
            self.tok = AutoTokenizer.from_pretrained(hf, local_files_only=True)
            cfg = AutoConfig.from_pretrained(hf, local_files_only=True)
            st_path = hf_hub_download(hf, "model.safetensors", local_files_only=True)
        except Exception:
            self.tok = AutoTokenizer.from_pretrained(hf)
            cfg = AutoConfig.from_pretrained(hf)
            st_path = hf_hub_download(hf, "model.safetensors")
        self.enc = AutoModel.from_config(cfg); self.head = torch.nn.Linear(cfg.hidden_size, 1)
        sd = load_file(st_path)
        self.enc.load_state_dict({k[6:]: v for k, v in sd.items() if k.startswith("model.")}, strict=False)
        self.head.load_state_dict({k[11:]: v for k, v in sd.items() if k.startswith("classifier.")})
        self.enc = self.enc.to(self.device).eval()
        self.head = self.head.to(self.device).eval()
    def _batch(self, texts):
        x = self.tok(texts, truncation=True, max_length=512, padding=True, return_tensors="pt").to(self.device)
        h = self.enc(**x).last_hidden_state; m = x["attention_mask"].unsqueeze(-1).float()
        return self.head((h * m).sum(1) / m.sum(1).clamp_min(1e-9))[:, 0].cpu().tolist()

PLUGINS_V3 = {"hf_sequence_classifier": HFClassifier, "desklib_deberta": Desklib}

# ---------- grouping ----------
def chunks(sentences, target=300, min_tail=120):
    """Consecutive sentence groups of ~target words; a short tail is merged into the previous chunk."""
    groups, cur, n = [], [], 0
    for i, s in enumerate(sentences):
        cur.append(i); n += len(s.split())
        if n >= target: groups.append(cur); cur, n = [], 0
    if cur:
        if groups and n < min_tail: groups[-1] += cur
        else: groups.append(cur)
    return groups
def windows(sentences, n=3, max_words=90):
    """Sliding windows, stride 1 sentence: window i = sentences i..i+n-1 (stops early at max_words)."""
    out = []
    L = len(sentences)
    starts = range(max(1, L - n + 1)) if L >= n else [0]
    for i in starts:
        g, w = [], 0
        for j in range(i, min(L, i + n)):
            g.append(j); w += len(sentences[j].split())
            if w >= max_words: break
        out.append(g)
    return out

def fuse(cal, vals):
    """Logistic fusion z = b + sum w_k * clip(v_k)."""
    z = cal["bias"]
    for k, w in cal["weights"].items(): z += w * max(-15.0, min(15.0, vals[k]))
    return z

def block_features(dets, cal, sentences):
    """dets: {model_key: detector with .logits}; sentences: list[str] (one block).
    Returns (A, B): A[i] = fused sliding-window score averaged over windows covering sentence i;
    B[i] = {model_key: logit of the ~300-word chunk containing sentence i} (doc score = word-weighted mean)."""
    n = len(sentences); B = [dict() for _ in range(n)]
    ch = chunks(sentences, cal.get("chunk_words", 300), cal.get("min_tail", 120))
    ch_texts = [" ".join(sentences[i] for i in g) for g in ch]
    early_exit = os.environ.get("DETECTOR_FAST_CPU", "1") == "1" and cal["doc"].get("rule") == "and"
    for k in cal["doc"]["weights"]:
        if k == "dk" and early_exit and "fk" in B[0] and isinstance(dets.get("dk"), Desklib):
            # AND-rule early exit: only run heavy DeBERTa-v3-large on chunks where fakespot is not strongly negative
            cand_idx = [ci for ci, g in enumerate(ch) if B[g[0]].get("fk", 0.0) >= -2.0]
            dk_vals = [-4.0] * len(ch)
            if cand_idx:
                for ci, v in zip(cand_idx, dets["dk"].logits([ch_texts[ci] for ci in cand_idx])):
                    dk_vals[ci] = float(v)
            for v, g in zip(dk_vals, ch):
                for i in g: B[i][k] = float(v)
        else:
            for v, g in zip(dets[k].logits(ch_texts), ch):
                for i in g: B[i][k] = float(v)
    wi = windows(sentences, cal.get("window_sents", 3), cal.get("window_max_words", 90))
    wl = {k: dets[k].logits([" ".join(sentences[i] for i in g) for g in wi]) for k in cal["win"]["weights"]}
    acc = np.zeros(n); cnt = np.zeros(n)
    for j, g in enumerate(wi):
        z = fuse(cal["win"], {k: wl[k][j] for k in wl})
        for i in g: acc[i] += z; cnt[i] += 1
    A = [float(acc[i] / cnt[i]) if cnt[i] else -10.0 for i in range(n)]
    return A, B

def summarise_v3(A, B, cal, sentences):
    wts = np.array([len(s.split()) for s in sentences], float); words = int(wts.sum())
    names = list(cal["doc"]["weights"])
    doc_vals = {k: float(np.average([b[k] for b in B], weights=np.maximum(wts, 1))) for k in names}
    doc = fuse(cal["doc"], doc_vals)
    t1, t2 = cal["doc"]["thresholds"]["fpr_1pct"], cal["doc"]["thresholds"]["fpr_2pct"]
    if cal["doc"].get("rule") == "and":   # all detectors must agree, each at its own FPR-targeted threshold
        MT = cal["doc"]["model_thresholds"]
        hit = lambda lvl: all(doc_vals[k] > MT[k][lvl] for k in names)
        verdict = "likely_ai_strict" if hit("fpr_1pct") else "possible_ai" if hit("fpr_2pct") else "no_strong_signal"
    else:
        verdict = "likely_ai_strict" if doc > t1 else "possible_ai" if doc > t2 else "no_strong_signal"
    flags = smooth_flags(A, cal["win"]["thresholds"][cal["win"].get("use", "fpr_2pct")], cal["win"].get("min_run", 2))
    p = float(np.interp(doc, *cal["doc"]["isotonic"]))
    sp = [float(np.interp(s, *cal["win"]["isotonic"])) for s in A]
    band = round(min(0.35, 0.05 + 25 / max(words, 1)), 2)
    return {"probability": round(min(0.99, max(0.01, p)), 3), "uncertainty": band, "words": words, "score": round(doc, 3),
            "verdict": verdict, "verdict_text": VERDICT_TEXT[verdict],
            "share_flagged": round(float(sum(wts[i] for i, f in enumerate(flags) if f)) / max(words, 1), 3),
            "features": {k: round(v, 3) for k, v in doc_vals.items()}, "thresholds": cal["doc"]["thresholds"],
            "sentence_probs": [round(x, 3) for x in sp], "sentence_flagged": flags, "low_confidence": words < 150}

def smooth_flags(sent, thr, min_run=2):
    f = [s > thr for s in sent]
    if min_run <= 1: return f
    out = list(f); i = 0
    while i < len(f):
        if f[i]:
            j = i
            while j < len(f) and f[j]: j += 1
            if j - i < min_run:
                for k in range(i, j): out[k] = False
            i = j
        else: i += 1
    return out

VERDICT_TEXT = {"likely_ai_strict": "Strong AI-writing signal (threshold set for ~1% false positives on held-out human text incl. learner essays)",
                "possible_ai": "Possible AI-writing signal (threshold set for ~2% false positives)",
                "no_strong_signal": "No strong AI-writing signal"}
