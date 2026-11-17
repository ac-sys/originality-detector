"""Calibrate a classifier detector exactly as the app runs it (doc = ~300-word chunks, word-weighted mean
logit; highlighting = ~3-sentence windows). Fits class-balanced isotonic maps and FPR-targeted thresholds on
CALIB, then measures doc- and window-level metrics on TEST and ESL. -> data/eval/calibration_v2_<name>.json
Usage: calibrate_v2.py NAME HF_ID_OR_PATH [AI_LABEL]"""
import os, sys, json, pathlib, time
os.environ.setdefault("OMP_NUM_THREADS", "3")
B = pathlib.Path(__file__).resolve().parent.parent; sys.path.insert(0, str(B))
import numpy as np, torch
torch.set_num_threads(int(os.environ.get("NTHREADS", "3")))
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score
from app.ai_detect import ClassifierDetector
from app.extract import split_sentences
name, hf = sys.argv[1], sys.argv[2]; lab = sys.argv[3] if len(sys.argv) > 3 else "AI"
det = ClassifierDetector({"hf_id": hf, "ai_label": lab})
D = B / "data/eval"; LIM = {"calib": 800, "test": 600, "esl": 600}
OUT = D / f"calib_scores_{name}.json"; cache = json.load(open(OUT)) if OUT.exists() else {}
t0 = time.time(); words = 0
for sp in ("calib", "test", "esl"):
    if sp in cache: continue
    rows = [json.loads(l) for l in open(D / f"{sp}.jsonl")][:LIM[sp]]; out = []
    for r in rows:
        s = split_sentences(r["text"]); w = np.array([len(x[2].split()) for x in s], float); words += w.sum()
        doc = float(np.average(det.score(r["text"], s, max_words=300), weights=w))
        win = det.score(r["text"], s, max_words=90, max_sents=3)
        groups = sorted(set(win), key=win.index)
        out.append({"label": r["label"], "source": r["source"], "doc": doc, "win": list(dict.fromkeys(win))})
    cache[sp] = out; json.dump(cache, open(OUT, "w"))
    print(sp, "done", round(time.time() - t0), "s", flush=True)
spw = (time.time() - t0) / max(words, 1) * 1000 if words else None
def bal_iso(x, y):
    w = np.where(y == 1, 0.5 / max(1, (y == 1).sum()), 0.5 / max(1, (y == 0).sum()))
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(x, y, sample_weight=w)
    xs = np.linspace(np.percentile(x, 0.5), np.percentile(x, 99.5), 60); return iso, [xs.tolist(), iso.predict(xs).tolist()]
res = {"name": name, "hf_id": hf, "s_per_1000_words_doc_plus_windows": round(spw, 2) if spw else None}
for level in ("doc", "win"):
    def arr(sp):
        if level == "doc": return np.array([r["doc"] for r in cache[sp]]), np.array([r["label"] for r in cache[sp]]), [r["source"] for r in cache[sp]]
        x, y, s = [], [], []
        for r in cache[sp]:
            for v in r["win"]: x.append(v); y.append(r["label"]); s.append(r["source"])
        return np.array(x), np.array(y), s
    cx, cy, _ = arr("calib"); tx, ty, ts = arr("test"); ex, ey, es = arr("esl")
    iso, pts = bal_iso(cx, cy)
    r = {"n_calib": len(cx), "n_test": len(tx), "auroc_test": round(roc_auc_score(ty, tx), 4), "isotonic": pts, "thresholds": {}}
    for f in (0.01, 0.02):
        thr = float(np.quantile(cx[cy == 0], 1 - f)); k = f"fpr_{int(f*100)}pct"; r["thresholds"][k] = round(thr, 4)
        r[f"test_tpr_{k}"] = round(float((tx[ty == 1] > thr).mean()), 3); r[f"test_fpr_{k}"] = round(float((tx[ty == 0] > thr).mean()), 3)
        r[f"esl_fpr_{k}"] = {s: round(float((ex[np.array(es) == s] > thr).mean()), 3) for s in sorted(set(es))}
    r["esl_fpr_at_balanced_p0.5"] = {s: round(float((iso.predict(ex[np.array(es) == s]) >= 0.5).mean()), 3) for s in sorted(set(es))}
    res[level] = r
json.dump(res, open(D / f"calibration_v2_{name}.json", "w"), indent=2)
print(json.dumps({l: {k: v for k, v in res[l].items() if k != "isotonic"} for l in ("doc", "win")}, indent=1))
