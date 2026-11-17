"""Measure int8 dynamic quantisation (torch.ao.quantization.quantize_dynamic on nn.Linear) for fakespot (fk) and desklib (dk).
Accuracy: on the first N test docs (held-out), int8 chunk logits vs cached fp32 logits (data/eval/v3raw/test.*.jsonl);
doc-level AUROC and AND-rule verdict agreement using the deployed v3 thresholds. Speed: fp32 vs int8 on the same chunks.
Usage: quant_eval.py [N=200] -> eval/v3/quant_results.json"""
import sys, os, json, time, pathlib, resource
B = pathlib.Path(__file__).resolve().parents[2]; sys.path.insert(0, str(B))
import numpy as np, torch; torch.set_num_threads(int(os.environ.get("NTHREADS", "6")))
from sklearn.metrics import roc_auc_score
from app.ai_v3 import HFClassifier, Desklib
from app.extract import split_sentences
N = int(sys.argv[1]) if len(sys.argv) > 1 else 200
D = B / "data/eval"; rows = [json.loads(l) for l in open(D / "test.jsonl")][:N]
cal = json.load(open(B / "models/manifest.json"))["calibration"]; MT = cal["doc"]["model_thresholds"]
SPECS = {"fk": (HFClassifier, {"hf_id": "fakespot-ai/roberta-base-ai-text-detection-v1", "ai_label": "AI"}), "dk": (Desklib, {"hf_id": "desklib/ai-text-detector-v1.01"})}
def rss(): return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
res = {"n_docs": N}; docval = {}
for key, (cls, spec) in SPECS.items():
    cache = [json.loads(l) for l in open(D / f"v3raw/test.{key}.jsonl")][:N]
    texts, owner, fp = [], [], []
    for i, (r, c) in enumerate(zip(rows, cache)):
        sents = r.get("sentences") or [s[2] for s in split_sentences(r["text"])]
        for v, g in c["chunks"]: texts.append(" ".join(sents[j] for j in g)); owner.append(i); fp.append(v)
    words = sum(len(t.split()) for t in texts); m0 = rss()
    M = cls(spec); m1 = rss()
    sub = texts[:40]; sw = sum(len(t.split()) for t in sub)
    t = time.time(); f32 = M.logits(sub); t32 = time.time() - t
    target = M.model if hasattr(M, "model") else M.enc
    q = torch.ao.quantization.quantize_dynamic(target, {torch.nn.Linear}, dtype=torch.qint8)
    if hasattr(M, "model"): M.model = q
    else: M.enc = q
    t = time.time(); _ = M.logits(sub); t8 = time.time() - t
    t = time.time(); i8 = []
    for k in range(0, len(texts), 8): i8 += M.logits(texts[k:k + 8])
    fp, i8 = np.array(fp), np.array(i8)
    def doc(vals):
        out = np.zeros(N); wsum = np.zeros(N)
        for o, v, tx in zip(owner, vals, texts): w = max(len(tx.split()), 1); out[o] += v * w; wsum[o] += w
        return out / np.maximum(wsum, 1)
    docval[key] = (doc(fp), doc(i8))
    y = np.array([r["label"] for r in rows])
    res[key] = {"chunks": len(texts), "corr": round(float(np.corrcoef(fp, i8)[0, 1]), 4), "mean_abs_diff": round(float(np.abs(fp - i8).mean()), 3),
                "auroc_fp32": round(roc_auc_score(y, docval[key][0]), 4), "auroc_int8": round(roc_auc_score(y, docval[key][1]), 4),
                "s_per_1000w_fp32": round(t32 / sw * 1000, 2), "s_per_1000w_int8": round(t8 / sw * 1000, 2),
                "rss_model_load_mb": round(m1 - m0)}
    print(key, res[key], flush=True); del M, q; json.dump(res, open(B / "eval/v3/quant_results.json", "w"), indent=1)
y = np.array([r["label"] for r in rows])
for lvl in ("fpr_1pct", "fpr_2pct"):
    a = np.all([docval[k][0] > MT[k][lvl] for k in docval], 0); b = np.all([docval[k][1] > MT[k][lvl] for k in docval], 0)
    res[f"and_rule_{lvl}"] = {"agreement": round(float((a == b).mean()), 4), "tpr_fp32": round(float(a[y == 1].mean()), 3), "tpr_int8": round(float(b[y == 1].mean()), 3),
                             "fpr_fp32": round(float(a[y == 0].mean()), 3), "fpr_int8": round(float(b[y == 0].mean()), 3)}
json.dump(res, open(B / "eval/v3/quant_results.json", "w"), indent=1); print(json.dumps(res, indent=1))
