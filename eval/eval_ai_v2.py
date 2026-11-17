"""Compare AI detectors (v1 vs candidates) on held-out data.
Thresholds for 1%/2% FPR are chosen on the CALIB split and then applied to TEST and ESL (honest held-out).
Stacked models are trained on TRAIN only, calibrated (isotonic) on CALIB."""
import os; os.environ.setdefault("OMP_NUM_THREADS","2")
import json, pathlib, sys, numpy as np, math
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score
D = pathlib.Path(__file__).resolve().parent.parent / "data" / "eval"
LIM = {"train": 1500, "calib": 800, "test": 1500, "esl": 10000}
def load(split):
    rows = [json.loads(l) for l in open(D / f"{split}.jsonl")][:LIM[split]]
    for f in D.glob(f"feats_*_{split}.jsonl"):
        for r, l in zip(rows, open(f)): r.update(json.loads(l))
    return rows
S = {s: load(s) for s in ["train", "calib", "test", "esl"]}
man = json.load(open(D.parent.parent / "models/manifest.json"))
cal = next((h["calibration"] for h in man["history"] if h["calibration"]["version"].startswith("mage")), man["calibration"]) if False else man["calibration"]
def v1score(r):  # exactly the v1 production combination with the calibration that was active in v1
    w = cal["weights"]; z = cal["bias"] + w["logprob"] * r["v1_lp"] + w["clf_logit"] * r["v1_clf"]; return 1 / (1 + math.exp(-z))
SINGLE = {"v1_ensemble": v1score, "v1_roberta_hc3": lambda r: r["v1_clf"], "v1_distilgpt2_logprob": lambda r: r["v1_lp"],
          "fakespot_roberta": lambda r: r.get("fakespot"), "desklib_deberta_large": lambda r: r.get("desklib"),
          "deberta_small_finetuned": lambda r: r.get("deberta"),
          "smol_binoculars": lambda r: -r["s_binoculars"] if "s_binoculars" in r else None,
          "smol_fastdetect_1model": lambda r: r.get("s_fastdetect1"), "smol_fastdetect_ref_inst": lambda r: r.get("s_fastdetect2"),
          "qwen05_binoculars": lambda r: -r["q_binoculars"] if "q_binoculars" in r else None,
          "qwen05_fastdetect_1model": lambda r: r.get("q_fastdetect1"), "qwen05_fastdetect_ref_inst": lambda r: r.get("q_fastdetect2")}
FEATS = ["v1_lp", "v1_clf", "fakespot", "s_lp", "s_ent", "s_lp_std", "s_fastdetect1", "s_binoculars", "s_fastdetect2"]
def stack(kind, feats):
    tr = [r for r in S["train"] if all(f in r for f in feats)]
    X = np.array([[r[f] for f in feats] for r in tr]); y = np.array([r["label"] for r in tr])
    m = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced") if kind == "lr" else HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, class_weight="balanced", random_state=0)
    mu, sd = X.mean(0), X.std(0) + 1e-9
    m.fit((X - mu) / sd, y)
    cache = {}
    for sp in ("calib", "test", "esl"):
        rows = [r for r in S[sp] if all(f in r for f in feats)]
        if not rows: continue
        Z = (np.array([[r[f] for f in feats] for r in rows]) - mu) / sd
        sc = m.decision_function(Z) if kind == "lr" else m.predict_proba(Z)[:, 1]
        for r, v in zip(rows, sc): cache[id(r)] = float(v)
    stack.last = (m, mu, sd)
    return lambda r: cache.get(id(r))
def evaluate(name, fn):
    out = {}
    def arr(split):
        rows = [r for r in S[split]]; sc = [fn(r) for r in rows]
        keep = [i for i, v in enumerate(sc) if v is not None and not (isinstance(v, float) and math.isnan(v))]
        return [rows[i] for i in keep], np.array([sc[i] for i in keep])
    cr, cs = arr("calib"); tr, ts = arr("test"); er, es = arr("esl")
    if len(ts) < 50 or len(cs) < 50: return None
    cy = np.array([r["label"] for r in cr]); ty = np.array([r["label"] for r in tr])
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(cs, cy)
    out["n_test"] = int(len(ty)); out["auroc_test"] = round(roc_auc_score(ty, ts), 4)
    for fpr in (0.01, 0.02):
        thr = float(np.quantile(cs[cy == 0], 1 - fpr))
        k = f"{int(fpr*100)}pct"
        out[f"thr_{k}"] = round(thr, 4)
        out[f"test_tpr_at_{k}"] = round(float((ts[ty == 1] > thr).mean()), 3)
        out[f"test_fpr_at_{k}"] = round(float((ts[ty == 0] > thr).mean()), 3)
        if len(es):
            out[f"esl_fpr_at_{k}"] = {src: round(float((es[[r["source"] == src for r in er]] > thr).mean()), 3) for src in sorted({r["source"] for r in er})}
    # at 0.5 calibrated probability
    tp = iso.predict(ts); out["test_acc_at_p0.5"] = round(float(((tp >= 0.5) == ty).mean()), 3)
    if len(es): out["esl_fpr_at_p0.5"] = {src: round(float((iso.predict(es[[r["source"] == src for r in er]]) >= 0.5).mean()), 3) for src in sorted({r["source"] for r in er})}
    out["auroc_by_source"] = {}
    for src in sorted({r["source"] for r in tr}):
        mask = np.array([r["source"] == src for r in tr])
        if len(set(ty[mask])) == 2: out["auroc_by_source"][src] = round(roc_auc_score(ty[mask], ts[mask]), 3)
    return out
res = {"splits": {k: len(v) for k, v in S.items()}, "v1_calibration": cal.get("version")}
for n, f in SINGLE.items():
    r = evaluate(n, f)
    if r: res[n] = r; print(n, r["auroc_test"], r["test_tpr_at_1pct"], r["test_tpr_at_2pct"], r.get("esl_fpr_at_2pct"), flush=True)
have = [f for f in FEATS if f in S["train"][0]]
for kind in ("lr", "gbm"):
    for nm, feats in [("all", have), ("no_v1", [f for f in have if not f.startswith("v1")]), ("fakespot+smol", ["fakespot"] + [f for f in have if f.startswith("s_")])]:
        if not feats: continue
        name = f"stack_{kind}_{nm}"; r = evaluate(name, stack(kind, feats))
        if r: r["features"] = feats; res[name] = r; print(name, r["auroc_test"], r["test_tpr_at_1pct"], r["test_tpr_at_2pct"], r.get("esl_fpr_at_2pct"), flush=True)
json.dump(res, open(D / "ai_v2_results.json", "w"), indent=2)
