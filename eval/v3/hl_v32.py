"""v3.2 sentence highlighting: hysteresis segments on the existing fakespot sliding-window sentence scores (no new model runs).
Seed sentences: score > t_hi; grow each seed's segment over neighbours with score > t_lo; drop segments shorter than min_run.
Optional doc gate: growth only when the doc's mean sentence score > g. Thresholds = quantiles of HUMAN calib+esl_calib
sentence scores (same data as v3). Chosen on mixed_dev: max F1 s.t. human-sentence FPR <= 1.5% (same cap as v3).
Reported on mixed (test, n=150) and on TOEFL/ESL human essays (share of words flagged). -> eval/v3/hl_v32_results.json"""
import json, pathlib, itertools, numpy as np, sys
B = pathlib.Path(__file__).resolve().parents[2]; D = B / "data/eval"; R = D / "v3raw"
CW = json.load(open(B / "models/manifest.json"))["calibration"]["win"]
fz = lambda v: CW["bias"] + CW["weights"]["fk"] * v
rows = lambda s: [json.loads(l) for l in open(D / f"{s}.jsonl")]
win = lambda s: [json.loads(l) for l in open(R / f"{s}.fk.win.jsonl")]
def sscore(w):
    acc = np.zeros(w["n"]); cnt = np.zeros(w["n"])
    for v, g in w["win3"]:
        for i in g: acc[i] += fz(v); cnt[i] += 1
    return np.where(cnt > 0, acc / np.maximum(cnt, 1), -10.0)
def runs(f):
    i = 0; out = []
    while i < len(f):
        if f[i]:
            j = i
            while j < len(f) and f[j]: j += 1
            out.append((i, j)); i = j
        else: i += 1
    return out
def flags(s, t_hi, t_lo, mr, g=None):
    seed = s > t_hi
    if t_lo is None or (g is not None and s.mean() <= g): grow = seed
    else: grow = s > t_lo
    out = np.zeros(len(s), bool)
    for a, b in runs(grow):
        if seed[a:b].any() and b - a >= mr: out[a:b] = True
    return out
def load(split, n=None):
    W = win(split); X = rows(split)[:len(W)]
    return [(x, w) for x, w in zip(X, W) if w is not None][:n]
hum = []
for sp, n in (("calib", 300), ("esl_calib", 150)):
    for x, w in load(sp, n):
        if x["label"] == 0: hum += list(sscore(w))
hum = np.array(hum); q = lambda f: float(np.quantile(hum, 1 - f))
def prf(split, p):
    tp = fp = fn = neg = 0
    for x, w in load(split):
        f = flags(sscore(w), *p); gold = np.array([bool(v) for v in x["sent_labels"]])[:len(f)]
        tp += int((f & gold).sum()); fp += int((f & ~gold).sum()); fn += int((~f & gold).sum()); neg += int((~gold).sum())
    P = tp / max(1, tp + fp); Rr = tp / max(1, tp + fn)
    return {"precision": round(P, 3), "recall": round(Rr, 3), "f1": round(2 * P * Rr / max(1e-9, P + Rr), 3), "human_sentence_fpr": round(fp / max(1, neg), 4)}
def share(split, p):
    fl = tot = 0
    for x, w in load(split):
        if x.get("label", 0) != 0: continue
        f = flags(sscore(w), *p); ww = np.array(w["w"])[:len(f)]; fl += int(ww[f].sum()); tot += int(ww.sum())
    return round(fl / max(1, tot), 4)
v3 = (CW["thresholds"][CW.get("use", "fpr_1pct")], None, CW.get("min_run", 2), None)
grid = []
for hi, lo, mr, g in itertools.product((0.005, 0.01, 0.02), (None, 0.03, 0.05, 0.08, 0.12, 0.16, 0.20), (1, 2, 3), (None, 0.0, 1.0, 2.0)):
    if lo is not None and lo <= hi: continue
    p = (q(hi), None if lo is None else q(lo), mr, g); grid.append(({"hi": hi, "lo": lo, "min_run": mr, "gate": g}, p, prf("mixed_dev", p)))
ok = [x for x in grid if x[2]["human_sentence_fpr"] <= 0.015]
best = max(ok, key=lambda x: x[2]["f1"])
out = {"v3_dev": prf("mixed_dev", v3), "chosen": best[0], "chosen_thresholds": best[1], "chosen_dev": best[2],
       "mixed_test": {"v3": prf("mixed", v3), "v32": prf("mixed", best[1])},
       "esl_word_share_flagged": {"v3": share("esl", v3), "v32": share("esl", best[1])},
       "grid_top": sorted([(x[0], x[2]) for x in ok], key=lambda x: -x[1]["f1"])[:10]}
json.dump(out, open(B / "eval/v3/hl_v32_results.json", "w"), indent=1); print(json.dumps({k: v for k, v in out.items() if k != "grid_top"}, indent=1))
