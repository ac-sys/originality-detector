"""v2 vs v3 similarity on the original synthetic set (verbatim/light/backtranslation/shuffle) AND the harder v3 set
(LLM paraphrase / LLM condense / mosaic). One shared library: all true sources + 2000 distractors.
v2 = deployed v2 params (fixed). v3 = parameters chosen on DEV (old dev + hard dev) by F1, reported on TEST.
-> data/simeval/results_v3.json"""
import json, sys, time, pathlib, tempfile, os, random, itertools
B = pathlib.Path(__file__).resolve().parents[2]; sys.path.insert(0, str(B)); os.environ["DETECTOR_LIGHT"] = "0"
import numpy as np, torch; torch.set_num_threads(int(os.environ.get("NTHREADS", "4")))
import app.similarity as sm
from app.extract import split_sentences
D = B / "data/simeval"
corpus = [json.loads(l) for l in open(D / "corpus.jsonl")]
L = lambda f: [json.loads(l) for l in open(D / f)]
sets = {"orig_dev": L("susp_dev.jsonl"), "orig_test": L("susp_test.jsonl"), "hard_dev": L("hard_dev.jsonl"), "hard_test": L("hard_test.jsonl")}
need = sorted({l["src"] for v in sets.values() for d in v for l in d["labels"] if l["src"] is not None})
# filler ("clean") sentences of the hard set come from corpus docs that must NOT be in the library
clean = {s for v in sets.values() for d in v for s, l in zip(d["sentences"], d["labels"]) if l["src"] is None}
filler_docs = {i for i, c in enumerate(corpus) if any(x[2] in clean for x in split_sentences(c["text"])[:60])}
print("excluded filler docs", len(filler_docs), flush=True)
R = random.Random(1); others = [i for i in range(len(corpus)) if i not in set(need) and i not in filler_docs]
use = sorted(set(need) | set(R.sample(others, 2000)))
def spans(doc):
    out, pos, t = [], 0, doc["text"]
    for s in doc["sentences"]:
        i = t.find(s, pos); out.append((i, i + len(s), s)); pos = i + len(s)
    for m in __import__("re").finditer(r"[^\n]+", t[pos:]):
        s = m.group().strip()
        if s: out.append((pos + m.start(), pos + m.end(), s))
    labs = doc["labels"] + [{"src": None, "type": "bib" if doc["text"][pos:].strip() else "clean"}] * (len(out) - len(doc["labels"]))
    return out, labs
V2 = dict(sm.P); V2.update(embed_model="intfloat/e5-small-v2", dim=384, query_prefix="query: ", k=5, w=4, para_threshold=0.92, exact_coverage=0.3,
           df_max_frac=0.01, df_min=20, min_words=6, version=2)
t0 = time.time()
lib_sents = {i: split_sentences(corpus[i]["text"]) for i in use}
flat = [s[2] for i in use for s in lib_sents[i]]
EMB = sm.Embedder.encode(flat, V2["embed_model"], V2["query_prefix"]); print("library embedded", len(flat), round(time.time() - t0), "s", flush=True)
qemb = {}
def build(p):
    c = sm.Corpus(pathlib.Path(tempfile.mkdtemp()) / "c.db", params=p); idmap = {}; k = 0
    for i in use:
        n = len(lib_sents[i]); d = c.add(corpus[i]["title"], corpus[i]["text"], lib_sents[i], source="wiki", embeddings=EMB[k:k + n], save=False); k += n; idmap[d] = i
    return c, idmap
def run(c, docs, key):
    out = []; t = time.time()
    for j, d in enumerate(docs):
        sp, labs = spans(d)
        if (key, j) not in qemb: qemb[(key, j)] = sm.Embedder.encode([s for _, _, s in sp], c.p["embed_model"], c.p["query_prefix"])
        out.append((labs, c.check(d["text"], sp, embeddings=qemb[(key, j)])))
    return out, time.time() - t
def score(results, idmap):
    tp = fp = fn = 0; per = {}; quoted = [0, 0]; bib = [0, 0]; clean = [0, 0]
    for labs, r in results:
        for l, x in zip(labs, r):
            m = None if x["excluded"] else (x["exact"] or x["paraphrase"])
            if l["type"] == "quoted": quoted[0] += 1; quoted[1] += bool(m)
            if l["type"] == "bib": bib[0] += 1; bib[1] += bool(m)
            if l["src"] is not None:
                ok = m is not None and idmap[m["doc"]] == l["src"]
                per.setdefault(l["type"], [0, 0]); per[l["type"]][0] += 1; per[l["type"]][1] += ok
                tp += ok; fn += not ok; fp += (m is not None and not ok)
            elif m is not None: fp += 1
            if l["type"] == "clean": clean[0] += 1; clean[1] += m is not None
    P_ = tp / max(1, tp + fp); R_ = tp / max(1, tp + fn)
    return {"precision": round(P_, 3), "recall": round(R_, 3), "f1": round(2 * P_ * R_ / max(1e-9, P_ + R_), 3),
            "recall_by_type": {k: round(v[1] / v[0], 3) for k, v in sorted(per.items())}, "n_plag": tp + fn,
            "quoted_flagged": f"{quoted[1]}/{quoted[0]}", "bibliography_flagged": f"{bib[1]}/{bib[0]}", "clean_sentence_fpr": round(clean[1] / max(1, clean[0]), 4)}
res = {"library_docs": len(use), "library_sentences": len(flat), "sets": {k: len(v) for k, v in sets.items()}}
words = {k: sum(len(d["text"].split()) for d in v) for k, v in sets.items()}
# v2
c, idmap = build(V2)
for k in ("orig_test", "hard_test"):
    r, t = run(c, sets[k], k); res.setdefault("v2", {})[k] = score(r, idmap); res["v2"][k]["check_s_per_1000w"] = round(t / words[k] * 1000, 3)
print("v2", json.dumps(res["v2"]), flush=True)
# v3 grid on dev
grid = []
for (kk, ww, cov) in [(5, 4, 0.3), (4, 3, 0.2)]:
    p = dict(V2); p.update(k=kk, w=ww, exact_coverage=cov, version=3); c, idmap = build(p)
    for pt, st, at, sm_ in itertools.product((0.92,), (0.85, 0.87, 0.89), (0.84, 0.86, 0.88), (1, 2)):
        c.p.update(para_threshold=pt, seed_threshold=st, align_threshold=at, seed_min=sm_)
        rd = run(c, sets["orig_dev"], "orig_dev")[0] + run(c, sets["hard_dev"], "hard_dev")[0]
        f1 = score(rd, idmap)["f1"]; grid.append((f1, kk, ww, cov, pt, st, at, sm_))
        json.dump({"v2": res["v2"], "grid": grid}, open(D / "results_v3_partial.json", "w"))
    print("grid done for", kk, ww, max(grid), flush=True)
best = max(grid); f1, kk, ww, cov, pt, st, at, sm_ = best
p = dict(V2); p.update(k=kk, w=ww, exact_coverage=cov, version=3, para_threshold=pt, seed_threshold=st, align_threshold=at, seed_min=sm_)
c, idmap = build(p); res["v3_params"] = {k: p[k] for k in ("k", "w", "exact_coverage", "para_threshold", "seed_threshold", "align_threshold", "seed_min")}
res["v3_dev_f1"] = f1
for k in ("orig_test", "hard_test"):
    r, t = run(c, sets[k], k); res.setdefault("v3", {})[k] = score(r, idmap); res["v3"][k]["check_s_per_1000w"] = round(t / words[k] * 1000, 3)
# ablation: v3 exact params without doc-alignment stage
c.p.update(align_threshold=9.0)
for k in ("orig_test", "hard_test"):
    res.setdefault("v3_no_align", {})[k] = score(run(c, sets[k], k)[0], idmap)
print(json.dumps(res, indent=1)); json.dump(res, open(D / "results_v3.json", "w"), indent=2)
