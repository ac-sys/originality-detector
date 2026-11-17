"""AI-writing detection ensemble. Each detector is a plugin selected from the model registry."""
import math, threading
import numpy as np
from .config import LIGHT_MODE

def _sig(x): return 1/(1+math.exp(-max(-50, min(50, x))))

class PerplexityDetector:
    """Per-sentence mean token log-probability under a small causal LM (AI text tends to be
    more predictable => higher log-prob). Burstiness = std of sentence log-probs."""
    def __init__(self, spec):
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
        self.torch = torch; self.spec = spec
        try:
            self.tok = AutoTokenizer.from_pretrained(spec["hf_id"], local_files_only=True)
            self.model = AutoModelForCausalLM.from_pretrained(spec["hf_id"], local_files_only=True).eval()
        except Exception:
            self.tok = AutoTokenizer.from_pretrained(spec["hf_id"])
            self.model = AutoModelForCausalLM.from_pretrained(spec["hf_id"]).eval()
        self.ctx = min(512, getattr(self.model.config, "n_positions", 512) or 512)
    def score(self, text, sentences):
        torch = self.torch
        enc = self.tok(text, return_offsets_mapping=True, add_special_tokens=False)
        ids, offs = enc["input_ids"], enc["offset_mapping"]
        lp = np.full(len(ids), np.nan)
        stride = self.ctx // 2
        with torch.no_grad():
            start = 0
            while start < len(ids):
                end = min(len(ids), start + self.ctx)
                x = torch.tensor([ids[start:end]])
                logits = self.model(x).logits[0, :-1]
                l = torch.log_softmax(logits.float(), -1).gather(1, x[0, 1:, None])[:, 0].numpy()
                first = 1 if start == 0 else stride  # only keep tokens with context
                for t in range(first, end - start):
                    lp[start + t] = l[t - 1]
                if end == len(ids): break
                start += stride
        starts = np.array([o[0] for o in offs]) if offs else np.array([])
        out = []
        for s, e, _ in sentences:
            m = (starts >= s) & (starts < e) & ~np.isnan(lp)
            out.append(float(lp[m].mean()) if m.any() else float("nan"))
        return out

class ClassifierDetector:
    """Pretrained HF sequence classifier, applied to groups of consecutive sentences (~200 tokens)."""
    def __init__(self, spec):
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import torch
        self.torch = torch; self.spec = spec
        try:
            self.tok = AutoTokenizer.from_pretrained(spec["hf_id"], local_files_only=True)
            self.model = AutoModelForSequenceClassification.from_pretrained(spec["hf_id"], local_files_only=True).eval()
        except Exception:
            self.tok = AutoTokenizer.from_pretrained(spec["hf_id"])
            self.model = AutoModelForSequenceClassification.from_pretrained(spec["hf_id"]).eval()
        lab = {v: int(k) for k, v in self.model.config.id2label.items()}
        self.ai_idx = lab.get(spec.get("ai_label", "LABEL_1"), 1)
    def score(self, text, sentences, max_words=150, max_sents=None):
        groups, cur, n = [], [], 0
        for i, (_, _, s) in enumerate(sentences):
            cur.append(i); n += len(s.split())
            if n >= max_words or (max_sents and len(cur) >= max_sents): groups.append(cur); cur, n = [], 0
        if cur:
            if groups and max_sents and n < 25: groups[-1] += cur   # avoid tiny trailing window
            else: groups.append(cur)
        out = [0.0] * len(sentences)
        with self.torch.no_grad():
            for b in range(0, len(groups), 8):
                batch = groups[b:b+8]
                texts = [" ".join(sentences[i][2] for i in g) for g in batch]
                x = self.tok(texts, truncation=True, max_length=512, padding=True, return_tensors="pt")
                lg = self.model(**x).logits.float()
                logit = (lg[:, self.ai_idx] - lg[:, 1 - self.ai_idx]).numpy()
                for g, v in zip(batch, logit):
                    for i in g: out[i] = float(v)
        return out

PLUGINS = {"causal_lm_perplexity": PerplexityDetector, "hf_sequence_classifier": ClassifierDetector}

class Ensemble:
    def __init__(self, registry):
        self.registry = registry; self.lock = threading.RLock(); self.detectors = {}; self.reload()
    def reload(self):
        """Hot-swap: build new detectors first, then switch atomically."""
        new = {}
        if not LIGHT_MODE:
            v3 = self.registry.manifest["calibration"].get("type") == "threshold_v3"
            if v3: from .ai_v3 import PLUGINS_V3
            for slot, did in self.registry.manifest["active"].items():
                spec = self.registry.spec(did)
                new[slot] = (PLUGINS_V3 if v3 else PLUGINS)[spec["type"]](spec)
        with self.lock:
            self.detectors = new; self.cal = dict(self.registry.manifest["calibration"])
    def versions(self):
        return dict(self.registry.manifest["active"]) | {"calibration": self.cal.get("version")}
    def features(self, text, sentences):
        with self.lock: dets, cal = dict(self.detectors), self.cal
        if cal.get("type") == "threshold_v3":
            from .ai_v3 import block_features
            keys = set(cal["doc"]["weights"]) | set(cal["win"]["weights"])
            if any(cal["slots"][k] not in dets for k in keys):   # light mode / missing model: neutral output
                return [-10.0] * len(sentences), [{k: -10.0 for k in cal["doc"]["weights"]} for _ in sentences], cal
            A, B = block_features({k: dets[cal["slots"][k]] for k in keys}, cal, [s[2] for s in sentences]); return A, B, cal
        if cal.get("type") == "threshold_v2":
            # A = window (~3 sentences) logits for highlighting; B = document-chunk (~300 words) logits
            c = dets.get("classifier")
            if c is None: return [0.0] * len(sentences), [0.0] * len(sentences), cal
            A = c.score(text, sentences, max_words=cal.get("window_max_words", 90), max_sents=cal.get("window_sents", 3))
            B = c.score(text, sentences, max_words=cal.get("chunk_words", 300))
            return A, B, cal
        lp = dets["perplexity"].score(text, sentences) if "perplexity" in dets else [float("nan")] * len(sentences)
        cl = dets["classifier"].score(text, sentences) if "classifier" in dets else [0.0] * len(sentences)
        return lp, cl, cal
    @staticmethod
    def combine(lp, cl, cal):
        w = cal["weights"]; z = cal["bias"]
        if not math.isnan(lp): z += w["logprob"] * lp
        else: z += w["logprob"] * cal.get("lp_mean", -4.0)
        z += w["clf_logit"] * cl
        return _sig(z)
    def analyse(self, text, sentences):
        if not sentences:
            return {"probability": None, "sentences": [], "disclaimer": DISCLAIMER}
        lp, cl, cal = self.features(text, sentences)
        return self.summarise(lp, cl, cal, sentences)
    def summarise(self, lp, cl, cal, sentences):
        if not sentences:
            return {"probability": None, "sentence_probs": [], "disclaimer": DISCLAIMER}
        if cal.get("type") == "threshold_v3":
            from .ai_v3 import summarise_v3
            return summarise_v3(lp, cl, cal, [s[2] for s in sentences]) | {"models": self.versions(), "disclaimer": DISCLAIMER}
        if cal.get("type") == "threshold_v2":
            return self._summarise_v2(lp, cl, cal, sentences)
        sent_p = [self.combine(a, b, cal) for a, b in zip(lp, cl)]
        wts = np.array([len(s[2].split()) for s in sentences], float)
        lpv = np.array(lp); ok = ~np.isnan(lpv)
        doc_lp = float(np.average(lpv[ok], weights=wts[ok])) if ok.any() else float("nan")
        doc_cl = float(np.average(np.array(cl), weights=wts))
        burst = float(np.std(lpv[ok])) if ok.sum() > 2 else None
        p = self.combine(doc_lp, doc_cl, cal)
        words = int(wts.sum())
        # uncertainty: wider band for short texts; probabilities are clipped away from 0/1
        band = round(min(0.35, 0.08 + 25 / max(words, 1)), 2)
        flagged = [i for i, x in enumerate(sent_p) if x >= 0.5]
        return {"probability": round(min(0.98, max(0.02, p)), 3), "uncertainty": band, "words": words,
                "share_flagged": round(sum(wts[i] for i in flagged) / max(words, 1), 3),
                "features": {"mean_logprob": None if math.isnan(doc_lp) else round(doc_lp, 3),
                             "classifier_logit": round(doc_cl, 3), "burstiness": None if burst is None else round(burst, 3)},
                "sentence_probs": [round(x, 3) for x in sent_p],
                "low_confidence": words < 150, "models": self.versions(), "disclaimer": DISCLAIMER}

def _iso(x, pts):
    xs, ys = pts
    return float(np.interp(x, xs, ys))
def _v2(self, A, B, cal, sentences):
    wts = np.array([len(s[2].split()) for s in sentences], float); words = int(wts.sum())
    doc = float(np.average(np.array(B), weights=wts))
    p = _iso(doc, cal["doc_isotonic"])
    t1, t2 = cal["doc_thresholds"]["fpr_1pct"], cal["doc_thresholds"]["fpr_2pct"]
    verdict = ("likely_ai_strict" if doc > t1 else "possible_ai" if doc > t2 else "no_strong_signal")
    sp = [_iso(a, cal["window_isotonic"]) for a in A]
    wt2 = cal["window_thresholds"]["fpr_2pct"]
    flagged = [i for i, a in enumerate(A) if a > wt2]
    band = round(min(0.35, 0.05 + 25 / max(words, 1)), 2)
    return {"probability": round(min(0.99, max(0.01, p)), 3), "uncertainty": band, "words": words, "score": round(doc, 3),
            "verdict": verdict, "verdict_text": {"likely_ai_strict": "Strong AI-writing signal (threshold set for ~1% false positives on held-out human text)",
                                                  "possible_ai": "Possible AI-writing signal (threshold set for ~2% false positives)",
                                                  "no_strong_signal": "No strong AI-writing signal"}[verdict],
            "share_flagged": round(sum(wts[i] for i in flagged) / max(words, 1), 3),
            "features": {"classifier_logit": round(doc, 3)}, "thresholds": cal["doc_thresholds"],
            "sentence_probs": [round(x, 3) for x in sp], "sentence_flagged": [a > wt2 for a in A],
            "low_confidence": words < 150, "models": self.versions(), "disclaimer": DISCLAIMER}
Ensemble._summarise_v2 = _v2

DISCLAIMER = ("AI-writing scores are statistical estimates, not proof. Detectors make mistakes, are known to "
              "flag non-native English writers more often, and can be wrong on short, formulaic or heavily edited text. "
              "Never use this score alone to make an academic-misconduct decision; review the evidence with the student.")
