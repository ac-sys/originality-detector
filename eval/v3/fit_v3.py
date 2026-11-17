"""Fit v3 AI-detector calibration from cached raw logits (score_raw.py) and compare v2 vs v3 on held-out sets.
Fit data: calib split (+ esl_calib human essays for thresholds). Tuning of min_run/window threshold: mixed_dev.
Held-out: test, esl, academic, mixed. Writes data/eval/calibration_v3.json and data/eval/v3_results.json."""
import json, pathlib, sys, collections
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score
B = pathlib.Path(__file__).resolve().parents[2]; sys.path.insert(0, str(B))
from app.ai_v3 import fuse, smooth_flags
D = B / "data/eval"; R = D / "v3raw"
V2 = json.load(open(D / "calibration_v2_fakespot.json"))
V2D, V2W = V2["doc"]["thresholds"], V2["win"]["thresholds"]
MODELS = sys.argv[1].split(",") if len(sys.argv) > 1 else ["fk", "dk"]
WMODELS = sys.argv[2].split(",") if len(sys.argv) > 2 else MODELS
TAG = sys.argv[3] if len(sys.argv) > 3 else "v3"
clip = lambda v: max(-15.0, min(15.0, v))

def rows(split): return [json.loads(l) for l in open(D / f"{split}.jsonl")]
def raw(split, m, win=False):
    p = R / f"{split}.{m}{'.win' if win else ''}.jsonl"
    return [json.loads(l) for l in open(p)] if p.exists() else []
def wmean(ch): return float(np.average([c[0] for c in ch], weights=[max(1, len(c[1])) and c[2] if len(c) > 2 else 1 for c in ch])) if ch else None
def docval(rec, key, w):  # word-weighted mean of chunk logits
    ch = rec[key]; ws = [sum(w[i] for i in g) for _, g in ch]
    return float(np.average([v for v, _ in ch], weights=ws))

def load_docs(split, models, limit=None):
    """-> list of (row, {m: docval}, v2val) for rows where all models are scored."""
    R0 = rows(split); raws = {m: raw(split, m) for m in models}; fk = raw(split, "fk")
    n = min([len(R0)] + [len(v) for v in raws.values()] + [len(fk)])
    if limit: n = min(n, limit)
    out = []
    for i in range(n):
        if any(raws[m][i] is None for m in models) or fk[i] is None: continue
        out.append((R0[i], {m: docval(raws[m][i], "chunks", raws[m][i]["w"]) for m in models}, docval(fk[i], "v2chunks", fk[i]["w"])))
    return out

def fit_lr(X, y):
    lr = LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced").fit(np.clip(X, -15, 15), y)
    return float(lr.intercept_[0]), [float(c) for c in lr.coef_[0]]
def thr_at(h, fpr): return float(np.quantile(h, 1 - fpr))
def iso(x, y):
    ir = IsotonicRegression(out_of_bounds="clip", y_min=0.01, y_max=0.99).fit(x, y)
    g = np.linspace(np.percentile(x, 1), np.percentile(x, 99), 60); return [g.tolist(), ir.predict(g).tolist()]

# ---------- document level ----------
cal = load_docs("calib", MODELS, 800); escal = load_docs("esl_calib", MODELS)
X = np.array([[d[1][m] for m in MODELS] for d in cal]); y = np.array([d[0]["label"] for d in cal])
b, w = fit_lr(X, y); C = {"bias": b, "weights": dict(zip(MODELS, w))}
z = lambda d: fuse(C, d[1])
hum = [z(d) for d in cal if d[0]["label"] == 0] + [z(d) for d in escal]
C["thresholds"] = {"fpr_1pct": thr_at(hum, .01), "fpr_2pct": thr_at(hum, .02)}
pool_docs = [d for d in cal if d[0]["label"] == 0] + escal
C["model_thresholds"] = {m: {"fpr_1pct": thr_at([d[1][m] for d in pool_docs], .01), "fpr_2pct": thr_at([d[1][m] for d in pool_docs], .02)} for m in MODELS}
C["rule"] = "and"  # verdict needs every model above its own FPR-targeted threshold (see eval/v3/rules_v3.py)
C["isotonic"] = iso(np.array([z(d) for d in cal]), y)
C["n_fit"] = len(cal); C["n_human_threshold_pool"] = len(hum)
res = {"models": MODELS, "win_models": WMODELS, "doc": {}, "win": {}}

def doc_eval(name, docs, pos_labels=(1,)):
    out = {"n": len(docs)}
    for ver in ("v2", "v3"):
        s = np.array([d[2] if ver == "v2" else z(d) for d in docs]); T = V2D if ver == "v2" else C["thresholds"]
        lab = np.array([d[0]["label"] for d in docs]); o = {}
        if len(set(lab[lab >= 0])) == 2: o["auroc"] = round(roc_auc_score(lab[lab >= 0], s[lab >= 0]), 4)
        for k in ("fpr_1pct", "fpr_2pct"):
            fl = s > T[k] if ver == "v2" else np.array([all(d[1][m] > C["model_thresholds"][m][k] for m in MODELS) for d in docs])
            if (lab == 1).any(): o[f"tpr@{k}"] = round(float(fl[lab == 1].mean()), 3)
            if (lab == 0).any(): o[f"fpr@{k}"] = round(float(fl[lab == 0].mean()), 3)
            if (lab == -1).any(): o[f"flag_rate_gpt4polished@{k}"] = round(float(fl[lab == -1].mean()), 3)
            bys = collections.defaultdict(list)
            for d, f in zip(docs, fl): bys[d[0].get("source", "?")].append(f)
            o[f"by_source@{k}"] = {sname: round(float(np.mean(v)), 3) for sname, v in sorted(bys.items())}
        out[ver] = o
    res["doc"][name] = out
for sp in ("test", "esl", "academic"):
    docs = load_docs(sp, MODELS, 600 if sp == "test" else None)
    if docs: doc_eval(sp, docs)

# ---------- window / sentence level ----------
def load_win(split, models, limit=None):
    R0 = rows(split); raws = {m: raw(split, m, True) for m in models}; fk = raw(split, "fk", True)
    n = min([len(R0)] + [len(v) for v in raws.values()] + [len(fk)])
    if limit: n = min(n, limit)
    return [(R0[i], {m: raws[m][i] for m in models}, fk[i]) for i in range(n)
            if all(raws[m][i] is not None for m in models) and fk[i] is not None]
cw = load_win("calib", WMODELS, 300)
CW = None
if cw:
    ecw = load_win("esl_calib", WMODELS, 150)
    Xw, yw = [], []
    for r, rr, _ in cw:
        for j in range(len(rr[WMODELS[0]]["win3"])):
            Xw.append([rr[m]["win3"][j][0] for m in WMODELS]); yw.append(r["label"])
    bw, ww = fit_lr(np.array(Xw), np.array(yw)); CW = {"bias": bw, "weights": dict(zip(WMODELS, ww))}
    def sent_scores(rr, n):
        acc = np.zeros(n); cnt = np.zeros(n); W = rr[WMODELS[0]]["win3"]
        for j in range(len(W)):
            zz = fuse(CW, {m: rr[m]["win3"][j][0] for m in WMODELS})
            for i in W[j][1]: acc[i] += zz; cnt[i] += 1
        return [float(acc[i] / cnt[i]) if cnt[i] else -10.0 for i in range(n)]
    hs = []
    for r, rr, fk in cw + ecw:
        if r["label"] == 0: hs += sent_scores(rr, fk["n"])
    CW["thresholds"] = {"fpr_1pct": thr_at(hs, .01), "fpr_2pct": thr_at(hs, .02)}
    allS, allY = [], []
    for r, rr, fk in cw:
        s = sent_scores(rr, fk["n"]); allS += s; allY += [r["label"]] * len(s)
    CW["isotonic"] = iso(np.array(allS), np.array(allY))
    def v2_flags(fk, k="fpr_2pct"):
        f = [False] * fk["n"]
        for v, g in fk["v2win"]:
            if v > V2W[k]:
                for i in g: f[i] = True
        return f
    def v3_flags(rr, n, use, mr): return smooth_flags(sent_scores(rr, n), CW["thresholds"][use], mr)
    def prf(pred, gold):
        tp = sum(p and g for p, g in zip(pred, gold)); fp = sum(p and not g for p, g in zip(pred, gold)); fn = sum(g and not p for p, g in zip(pred, gold))
        P = tp / max(tp + fp, 1); Rr = tp / max(tp + fn, 1)
        return {"precision": round(P, 3), "recall": round(Rr, 3), "f1": round(2 * P * Rr / max(P + Rr, 1e-9), 3),
                "human_sentence_fpr": round(fp / max(sum(not g for g in gold), 1), 3)}
    def mixed_eval(split, use, mr):
        docs = load_win(split, WMODELS); P2, P3, G = [], [], []
        for r, rr, fk in docs:
            P2 += v2_flags(fk); P3 += v3_flags(rr, fk["n"], use, mr); G += [bool(x) for x in r["sent_labels"]]
        return len(docs), prf(P2, G), prf(P3, G)
    grid = {}
    for use in ("fpr_1pct", "fpr_2pct"):
        for mr in (1, 2, 3):
            n, _, p3 = mixed_eval("mixed_dev", use, mr); grid[f"{use}/min_run={mr}"] = p3
    ok = [k for k in grid if grid[k]["human_sentence_fpr"] <= 0.015] or list(grid)  # ESL-safety: cap human-sentence FPR on dev
    best = max(ok, key=lambda k: grid[k]["f1"]); use, mr = best.split("/")[0], int(best.split("=")[1])
    CW.update({"use": use, "min_run": mr})
    res["win"]["mixed_dev_grid_v3"] = grid; res["win"]["chosen"] = best
    n, p2, p3 = mixed_eval("mixed", use, mr); res["win"]["mixed_test"] = {"n_docs": n, "v2": p2, "v3": p3}
    def share_eval(split, limit=None, pos=None):
        docs = load_win(split, WMODELS, limit); out = {"n_docs": len(docs)}
        for ver in ("v2", "v3"):
            by = collections.defaultdict(lambda: [0, 0])
            for r, rr, fk in docs:
                f = v2_flags(fk) if ver == "v2" else v3_flags(rr, fk["n"], use, mr)
                key = f"{r.get('source','?')}|label={r['label']}"
                by[key][0] += sum(fk["w"][i] for i in range(fk["n"]) if f[i]); by[key][1] += sum(fk["w"])
            out[ver] = {k: round(a / max(b_, 1), 3) for k, (a, b_) in sorted(by.items())}
        return out
    res["win"]["share_flagged_words_test"] = share_eval("test", 300)
    res["win"]["share_flagged_words_esl"] = share_eval("esl")
CAL = {"version": 3, "models": MODELS, "doc": C, "win": CW, "weights": C["weights"]}
json.dump(CAL, open(D / f"calibration_{TAG}.json", "w"), indent=1)
json.dump(res, open(D / f"{TAG}_results.json", "w"), indent=1)
json.dump(res, open(B / f"eval/v3/{TAG}_results.json", "w"), indent=1)
print(json.dumps(res, indent=1))
