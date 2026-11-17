"""v3.2 condense stage, simulated on cached features on top of the active v3.1 matcher.
Stage 4 (only for sentences still unmatched): top-1 library doc d with cos >= c_top; best(single, adjacent-pair) vs d >= c_sim;
and entity/number overlap gate: >= c_nent distinct entity tokens (capitalised non-initial words, numbers) of the sentence,
fraction found in d's text >= c_ent; OR (passage gate) an adjacent sentence has top-1 = d with cos >= r_top. Chosen on DEV (orig_dev+hard_dev): max micro-F1 s.t. dev clean FPR <= v3.1 dev FPR (no increase).
-> eval/v3/sim_v32_results.json"""
import pickle, json, itertools, pathlib, re, sys
B = pathlib.Path(__file__).resolve().parents[2]; sys.path.insert(0, str(B))
exec(open(B / "eval/v3/sim_search.py").read().split("V2 = {")[0].replace("__file__", repr(str(B / "eval/v3/sim_search.py"))))
from app.similarity import entity_tokens
D = B / "data/simeval"
corpus = [json.loads(l) for l in open(D / "corpus.jsonl")]
DOCTOK = {}
def dtok(d):
    if d not in DOCTOK: DOCTOK[d] = set(entity_tokens(corpus[d]["text"], all_tokens=True))
    return DOCTOK[d]
srcf = {"orig_dev": "susp_dev.jsonl", "orig_test": "susp_test.jsonl", "hard_dev": "hard_dev.jsonl", "hard_test": "hard_test.jsonl"}
for k, f in srcf.items():
    raw = [json.loads(l) for l in open(D / f)]
    for doc, r in zip(F[k], raw):
        sents = r["sentences"] + [""] * (len(doc["sents"]) - len(r["sentences"]))
        for s, t in zip(doc["sents"], sents):
            e = set(entity_tokens(t)); d = s["top"][0][1]
            s["ent"] = (len(e), len(e & dtok(d)) / max(1, len(e)))
V31 = {"version": 3, "para": 0.94, "seed": 0.91, "seed_min": 3, "pairs": True, "align": 0.85}
base_sim = simulate
def sim32(doc, p):
    res = base_sim(doc, V31)
    if p.get("c_top") is None: return res
    for i, s in enumerate(doc["sents"]):
        if res[i] is not None or s["excluded"] or s["nw"] < p["c_nw"]: continue
        sc, d = s["top"][0]; v = s["cand"].get(d)
        if sc < p["c_top"] or v is None or max(v) < p["c_sim"]: continue
        run = any(0 <= j < len(doc["sents"]) and doc["sents"][j]["top"][0][1] == d and doc["sents"][j]["top"][0][0] >= p["r_top"]
                  for j in (i - 1, i + 1))   # passage-level: a neighbouring sentence points to the same source doc
        if (s["ent"][0] >= p["c_nent"] and s["ent"][1] >= p["c_ent"]) or (p["r_top"] < 1 and run): res[i] = d
    return res
simulate = sim32
def dv(p):
    a, b = score(F["orig_dev"], p), score(F["hard_dev"], p)
    tp, fp, fn = (a["_c"][k] + b["_c"][k] for k in range(3)); cl = (a["_c"][3][1] + b["_c"][3][1], a["_c"][3][0] + b["_c"][3][0])
    P = tp / max(1, tp + fp); R = tp / max(1, tp + fn)
    return round(2 * P * R / max(1e-9, P + R), 4), round(cl[0] / max(1, cl[1]), 4), b["recall_by_type"].get("llm_condense")
base = dv({}); print("v3.1 dev", base, flush=True)
grid = []
for ct, cs, cn, ce, nw, rt in itertools.product((0.84, 0.86, 0.88, 0.90), (0.86, 0.88, 0.90, 0.92), (1, 2, 3, 4, 99), (0.75, 0.9, 1.0), (6, 10), (0.86, 0.88, 0.90, 1.0)):
    p = {"c_top": ct, "c_sim": cs, "c_nent": cn, "c_ent": ce, "c_nw": nw, "r_top": rt}; grid.append((dv(p), p))
json.dump(grid, open(CD / "grid32.json", "w"))
ok = [g for g in grid if g[0][1] <= base[1]]  # no dev clean-FPR increase vs v3.1
best = max(ok, key=lambda g: (g[0][0], g[0][2] or 0, g[1]["c_nent"], g[1]["c_ent"], g[1]["r_top"], g[1]["c_sim"], g[1]["c_top"]))  # ties -> strictest gate; print("best", best, flush=True)
strip = lambda r: {k: v for k, v in r.items() if k != "_c"}
out = {"v31_dev": base, "chosen": best, "v31": {k: strip(score(F[k], {})) for k in ("orig_test", "hard_test")},
       "v32": {k: strip(score(F[k], best[1])) for k in ("orig_test", "hard_test")}}
json.dump(out, open(B / "eval/v3/sim_v32_results.json", "w"), indent=1); print(json.dumps({k: out[k] for k in ("v31", "v32")}, indent=1))
