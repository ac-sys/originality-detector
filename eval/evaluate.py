"""Evaluate the active detector ensemble on a sample of the MAGE test set (Li et al., 2024,
https://huggingface.co/datasets/yaful/MAGE), fit calibration on a held-out calibration split,
and append results to data/eval_history.json (feeds the calibration dashboard /api/calibration).
Usage: python eval/evaluate.py [--n 150] [--apply]   (--apply writes fitted calibration to the manifest)"""
import sys, json, time, random, argparse, math, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import numpy as np
from datasets import load_dataset
from app.config import MANIFEST, DATA
from app.registry import Registry
from app.ai_detect import Ensemble
from app.extract import split_sentences, normalise

def auroc(y, s):
    y, s = np.array(y), np.array(s); pos, neg = s[y == 1], s[y == 0]
    return float(((pos[:, None] > neg[None]).sum() + 0.5 * (pos[:, None] == neg[None]).sum()) / (len(pos) * len(neg)))
def fit_logreg(X, y, l2=1e-2, it=3000, lr=0.1):
    mu, sd = X.mean(0), X.std(0) + 1e-9; Z = (X - mu) / sd; w = np.zeros(Z.shape[1]); b = 0.0
    for _ in range(it):
        p = 1 / (1 + np.exp(-(Z @ w + b))); g = p - y
        w -= lr * (Z.T @ g / len(y) + l2 * w); b -= lr * g.mean()
    return w / sd, b - (w / sd) @ mu   # weights in raw feature space
def metrics(y, p, thr=0.5):
    y, p = np.array(y), np.array(p); pred = p >= thr
    tp = int((pred & (y == 1)).sum()); fp = int((pred & (y == 0)).sum())
    return {"accuracy": round(float((pred == y).mean()), 3), "tpr": round(tp / max(1, (y == 1).sum()), 3),
            "fpr": round(fp / max(1, (y == 0).sum()), 3), "auroc": round(auroc(y, p), 3)}

ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=150); ap.add_argument("--apply", action="store_true")
ap.add_argument("--seed", type=int, default=7); a = ap.parse_args()
random.seed(a.seed)
hum, ai = [], []
for x in load_dataset("yaful/MAGE", split="test", streaming=True).shuffle(seed=a.seed, buffer_size=5000):
    t = normalise(x["text"]); w = len(t.split())
    if not 120 <= w <= 500: continue
    (hum if str(x["label"]) == "1" else ai).append((t, x["src"]))   # MAGE: label 1 = human, 0 = machine
    if len(hum) >= a.n and len(ai) >= a.n: break
hum, ai = hum[:a.n], ai[:a.n]
data = [(t, 0, s) for t, s in hum] + [(t, 1, s) for t, s in ai]; random.shuffle(data)
reg = Registry(MANIFEST); ens = Ensemble(reg)
feats, t0 = [], time.time()
for t, y, s in data:
    sents = split_sentences(t); lp, cl, cal = ens.features(t, sents)
    w = np.array([len(x[2].split()) for x in sents], float); lpv = np.array(lp); ok = ~np.isnan(lpv)
    feats.append([float(np.average(lpv[ok], weights=w[ok])), float(np.average(cl, weights=w))])
secs = (time.time() - t0) / len(data)
X, Y = np.array(feats), np.array([d[1] for d in data])
nc = len(data) // 3   # first third = calibration split, rest = test
w, b = fit_logreg(X[:nc], Y[:nc])
Xt, Yt = X[nc:], Y[nc:]
p_ens = 1 / (1 + np.exp(-(Xt @ w + b)))
p_def = np.array([Ensemble.combine(f[0], f[1], reg.manifest["calibration"]) for f in Xt])
# threshold for <=1% FPR on calibration split
pc = 1 / (1 + np.exp(-(X[:nc] @ w + b))); thr1 = float(np.quantile(pc[Y[:nc] == 0], 0.99))
res = {"time": time.time(), "dataset": "yaful/MAGE test (streamed, shuffled seed %d), 120-500 words" % a.seed,
       "n_total": len(data), "n_calibration": nc, "n_test": len(Yt), "test_human": int((Yt == 0).sum()), "test_ai": int((Yt == 1).sum()),
       "models": ens.versions(), "seconds_per_doc": round(secs, 2),
       "auroc_perplexity_only": round(auroc(Yt, Xt[:, 0]), 3), "auroc_classifier_only": round(auroc(Yt, Xt[:, 1]), 3),
       "ensemble_default_calibration": metrics(Yt, p_def), "ensemble_fitted_calibration": metrics(Yt, p_ens),
       "ensemble_at_1pct_fpr_threshold": dict(metrics(Yt, p_ens, thr1), threshold=round(thr1, 3)),
       "fitted": {"weights": {"logprob": round(float(w[0]), 4), "clf_logit": round(float(w[1]), 4)}, "bias": round(float(b), 4)}}
print(json.dumps(res, indent=2))
hist = DATA / "eval_history.json"; h = json.loads(hist.read_text()) if hist.exists() else []; h.append(res); hist.write_text(json.dumps(h, indent=2))
if a.apply:
    cal = {"version": time.strftime("mage-%Y%m%d-%H%M"), "weights": res["fitted"]["weights"], "bias": res["fitted"]["bias"],
           "lp_mean": round(float(X[:, 0].mean()), 3), "note": "Fitted by eval/evaluate.py on MAGE calibration split"}
    reg.activate(calibration=cal, reason="recalibration from evaluation"); print("calibration applied:", cal["version"])
