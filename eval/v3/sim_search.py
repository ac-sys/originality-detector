"""Stage 2 (fast): exact re-implementation of Corpus._paraphrase_v3 on cached features; grid search on DEV
(orig_dev + hard_dev) maximising micro-F1 subject to dev clean-sentence FPR <= 1.2%; report on TEST.
-> eval/v3/sim_v31_results.json (+ partial grid in data/simeval/cache/grid.json)"""
import pickle, json, itertools, pathlib, time
B = pathlib.Path(__file__).resolve().parents[2]; CD = B / "data/simeval/cache"
F = {k: pickle.load(open(CD / f"feats_{k}.pkl", "rb")) for k in ("orig_dev", "hard_dev", "orig_test", "hard_test") if (CD / f"feats_{k}.pkl").exists()}
def simulate(doc, p):
    S = doc["sents"]; n = len(S); res = [None] * n; seeds = {}
    if p["version"] == 2:
        for i, s in enumerate(S):
            if s["exact"] is not None: res[i] = s["exact"]
            elif not s["excluded"] and s["nw"] >= 6 and s["top"][0][0] >= p["para"]: res[i] = s["top"][0][1]
        return res
    ok = [not s["excluded"] and s["nw"] >= 6 for s in S]
    for i, s in enumerate(S):
        if s["exact"] is not None: res[i] = s["exact"]; seeds[s["exact"]] = seeds.get(s["exact"], 0) + 2; continue
        if not ok[i]: continue
        for sc, d in s["top"]:
            if sc >= p["para"] and res[i] is None: res[i] = d; seeds[d] = seeds.get(d, 0) + 2
            elif sc >= p["seed"]: seeds[d] = seeds.get(d, 0) + 1
    cands = [d for d, c in sorted(seeds.items(), key=lambda x: -x[1]) if c >= p["seed_min"]][:8]
    if cands:
        for i, s in enumerate(S):
            if res[i] is not None or not ok[i]: continue
            best, bd = -1, None
            for d in cands:
                v = s["cand"].get(d)
                if v is None: continue
                x = max(v[0], v[1]) if p["pairs"] else v[0]
                if x > best: best, bd = x, d
            if best >= p["align"]: res[i] = bd
    return res
def score(docs, p):
    tp = fp = fn = 0; per = {}; cl = [0, 0]; q = [0, 0]
    for doc in docs:
        r = simulate(doc, p)
        for l, s, m in zip(doc["labs"], doc["sents"], r):
            if s["excluded"]: m = None
            if l["type"] == "quoted": q[0] += 1; q[1] += m is not None
            if l["src"] is not None:
                good = m == l["src"]; per.setdefault(l["type"], [0, 0]); per[l["type"]][0] += 1; per[l["type"]][1] += good
                tp += good; fn += not good; fp += (m is not None and not good)
            elif m is not None: fp += 1
            if l["type"] == "clean": cl[0] += 1; cl[1] += m is not None
    P = tp / max(1, tp + fp); R = tp / max(1, tp + fn)
    return {"precision": round(P, 3), "recall": round(R, 3), "f1": round(2 * P * R / max(1e-9, P + R), 3), "clean_sentence_fpr": round(cl[1] / max(1, cl[0]), 4),
            "recall_by_type": {k: round(v[1] / v[0], 3) for k, v in sorted(per.items())}, "quoted_flagged": f"{q[1]}/{q[0]}", "_c": (tp, fp, fn, cl)}
def dev(p):
    a, b = score(F["orig_dev"], p), score(F["hard_dev"], p)
    tp, fp, fn = (a["_c"][k] + b["_c"][k] for k in range(3)); cl = a["_c"][3][1] + b["_c"][3][1], a["_c"][3][0] + b["_c"][3][0]
    P = tp / max(1, tp + fp); R = tp / max(1, tp + fn)
    return round(2 * P * R / max(1e-9, P + R), 4), round(cl[0] / max(1, cl[1]), 4), a["f1"], b["f1"]
V2 = {"version": 2, "para": 0.92}
out = {"v2_dev": dev(V2)}; print("v2 dev f1/cleanFPR/orig/hard", out["v2_dev"], flush=True)
grid = []; t = time.time()
for para, seed, sm_, pairs in itertools.product((0.90, 0.92, 0.94, 0.96), (0.83, 0.86, 0.89, 0.91, 0.93), (1, 2, 3, 4, 5), (True, False)):
    for align in [round(0.80 + 0.01 * k, 2) for k in range(13)]:
        p = {"version": 3, "para": para, "seed": seed, "seed_min": sm_, "pairs": pairs, "align": align}
        grid.append((dev(p), p))
    json.dump(grid, open(CD / "grid.json", "w"))
print("grid", len(grid), round(time.time() - t), "s", flush=True)
okg = [g for g in grid if g[0][1] <= out["v2_dev"][1] + 0.003]  # no more than +0.3pt dev clean FPR vs v2
best = max(okg, key=lambda g: g[0][0]); out["chosen"] = best; print("best", best, flush=True)
out["best_unconstrained"] = max(grid, key=lambda g: g[0][0])
for k in ("orig_test", "hard_test"):
    out.setdefault("v2", {})[k] = {a: b for a, b in score(F[k], V2).items() if a != "_c"}
    out.setdefault("v3", {})[k] = {a: b for a, b in score(F[k], best[1]).items() if a != "_c"}
json.dump(out, open(B / "eval/v3/sim_v31_results.json", "w"), indent=1); print(json.dumps({k: out[k] for k in ("v2", "v3")}, indent=1))
