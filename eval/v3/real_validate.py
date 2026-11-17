"""Validate matchers through the REAL Corpus.check code (not the simulator) on full held-out orig_test + hard_test.
Same library as sim_feats.py (true sources + 2000 distractors, cached embeddings). Configs: v2, v3.1, v3.2 (all with
their deployed exclusion setting). Partial results saved every 20 docs -> eval/v3/real_validate.json"""
import json, sys, time, pathlib, tempfile, os, random, copy
B = pathlib.Path(__file__).resolve().parents[2]; sys.path.insert(0, str(B)); os.environ["DETECTOR_LIGHT"] = "0"
import numpy as np, torch; torch.set_num_threads(int(os.environ.get("NTHREADS", "6")))
src = open(B / "eval/v3/sim_feats.py").read()
exec(src.split("rows_of = {}")[0].replace("__file__", repr(str(B / "eval/v3/sim_feats.py"))))
c = sm.Corpus(pathlib.Path(tempfile.mkdtemp()) / "c.db", params=V2); dbid = {}; k = 0
for i in use:
    n = len(lib_sents[i]); dbid[c.add(corpus[i]["title"], corpus[i]["text"], lib_sents[i], source="wiki", embeddings=EMB[k:k + n], save=False)] = i; k += n
EXTRA = os.environ.get("EXTRA")   # optional: JSONL of ingested docs [title, text, src] added to the library (ingestion effect)
if EXTRA:
    from app.extract import normalise
    ex = [json.loads(l) for l in open(EXTRA)]; t = time.time(); ns = 0
    for title, text, sid in ex:
        text = normalise(text); ss = split_sentences(text)
        if ss: c.add(title, text, ss, source=sid.split(":")[0], save=False); ns += len(ss)
    print("extra docs", len(ex), "sentences", ns, "ingest_s", round(time.time() - t), "library sentences", c.index().ntotal, flush=True)
man = json.load(open(B / "models/manifest.json"))["similarity"]
V31 = dict(V2); V31.update({k: v for k, v in man.items() if k not in ("condense",)})
CFG = {"v2": V2, "v3.1": V31}
if man.get("condense"): V32 = dict(V31); V32["condense"] = man["condense"]; CFG["v3.2"] = V32
if os.environ.get("CONDENSE"): V32 = dict(V31); V32["condense"] = json.loads(os.environ["CONDENSE"]); CFG["v3.2"] = V32
if os.environ.get("CFGS"): CFG = {k: v for k, v in CFG.items() if k in os.environ["CFGS"].split(",")}
outf = B / os.environ.get("OUT", "eval/v3/real_validate.json"); acc = {}
def add(cfg, name, labs, res):
    a = acc.setdefault(cfg, {}).setdefault(name, {"tp": 0, "fp": 0, "fn": 0, "clean": [0, 0], "per": {}, "sec": 0.0})
    for l, r in zip(labs, res):
        m = None if r["excluded"] else (r["exact"] or r["paraphrase"] or {}).get("doc")
        m = dbid.get(m, -1) if m is not None else None   # -1 = matched an ingested (extra) doc
        if l["src"] is not None:
            g = m == l["src"]; pt = a["per"].setdefault(l["type"], [0, 0]); pt[0] += 1; pt[1] += g
            a["tp"] += g; a["fn"] += not g; a["fp"] += (m is not None and not g)
        elif m is not None: a["fp"] += 1
        if l["type"] == "clean": a["clean"][0] += 1; a["clean"][1] += m is not None
def summary():
    out = {}
    for cfg, v in acc.items():
        for name, a in v.items():
            P = a["tp"] / max(1, a["tp"] + a["fp"]); R = a["tp"] / max(1, a["tp"] + a["fn"])
            out.setdefault(cfg, {})[name] = {"n_docs": a.get("n", 0), "precision": round(P, 3), "recall": round(R, 3), "f1": round(2 * P * R / max(1e-9, P + R), 3),
                "clean_sentence_fpr": round(a["clean"][1] / max(1, a["clean"][0]), 4), "clean_n": a["clean"][0],
                "recall_by_type": {k: round(x[1] / x[0], 3) for k, x in sorted(a["per"].items())}, "s_per_doc_match": round(a["sec"] / max(1, a.get("n", 1)), 3)}
    return out
for name in ("hard_test", "orig_test"):
    t0 = time.time()
    for j, doc in enumerate(sets[name]):
        sp, labs = spans(doc); Q = enc([s for _, _, s in sp])
        for cfg, p in CFG.items():
            c.p = p; t = time.time(); r = c.check(doc["text"], sp, embeddings=Q, paraphrase=True)
            add(cfg, name, labs, r); a = acc[cfg][name]; a["sec"] += time.time() - t; a["n"] = a.get("n", 0) + 1
        if j % 20 == 19 or j == len(sets[name]) - 1:
            json.dump({"configs": {k: {x: y for x, y in v.items() if x != "embed_model"} for k, v in CFG.items()}, "results": summary()}, open(outf, "w"), indent=1)
            print(name, j + 1, round(time.time() - t0), "s", flush=True)
print(json.dumps(summary(), indent=1)); print("ALLDONE", flush=True)
