"""Evaluate similarity configs on the synthetic plagiarism set. Sentence-level precision/recall/F1,
recall per obfuscation type, quotation/bibliography handling. Thresholds tuned on dev, reported on test.
Usage: eval_similarity.py [--full] (full = all 12k corpus docs; default = sources + 2000 distractors)"""
import json, sys, time, pathlib, tempfile, os, random, argparse
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
os.environ["DETECTOR_LIGHT"] = "0"
import numpy as np
import torch; torch.set_num_threads(int(os.environ.get("NTHREADS","3")))
from app.similarity import Corpus, Embedder, P as DEFAULT
from app.extract import split_sentences
D = pathlib.Path(__file__).resolve().parent.parent / "data" / "simeval"
ap = argparse.ArgumentParser(); ap.add_argument("--full", action="store_true"); ap.add_argument("--configs", default="all"); a = ap.parse_args()
corpus = [json.loads(l) for l in open(D / "corpus.jsonl")]
dev = [json.loads(l) for l in open(D / "susp_dev.jsonl")]; test = [json.loads(l) for l in open(D / "susp_test.jsonl")]
def spans(doc):
    out, pos, t = [], 0, doc["text"]
    for s, lab in zip(doc["sentences"], doc["labels"]):
        i = t.find(s, pos); out.append((i, i + len(s), s)); pos = i + len(s)
    for m in __import__("re").finditer(r"[^\n]+", t[pos:]):
        s = m.group().strip()
        if s: out.append((pos + m.start(), pos + m.end(), s))
    labs = doc["labels"] + [{"src": None, "type": "bib" if doc["text"][pos:].strip() else "clean"}] * (len(out) - len(doc["labels"]))
    return out, labs
need = sorted({l["src"] for d in dev + test for l in d["labels"] if l["src"] is not None})
R = random.Random(1)
if a.full: use = list(range(len(corpus)))
else:
    others = [i for i in range(len(corpus)) if i not in set(need)]; use = sorted(set(need) | set(R.sample(others, 2000)))
CONFIGS = {
 "v1_minilm":  dict(embed_model="sentence-transformers/all-MiniLM-L6-v2", dim=384, query_prefix="", k=5, w=4, para_threshold=0.82, df_max_frac=1.0, df_min=10**9, excl=False),
 "minilm_v2":  dict(embed_model="sentence-transformers/all-MiniLM-L6-v2", dim=384, query_prefix="", excl=True),
 "bge_small":  dict(embed_model="BAAI/bge-small-en-v1.5", dim=384, query_prefix="", excl=True),
 "e5_small":   dict(embed_model="intfloat/e5-small-v2", dim=384, query_prefix="query: ", excl=True),
 "e5_k4w3":    dict(embed_model="intfloat/e5-small-v2", dim=384, query_prefix="query: ", k=4, w=3, exact_coverage=0.2, excl=True),
 "bge_k4w3":   dict(embed_model="BAAI/bge-small-en-v1.5", dim=384, query_prefix="", k=4, w=3, exact_coverage=0.2, excl=True),
 "gte_small":  dict(embed_model="thenlper/gte-small", dim=384, query_prefix="", excl=True),
}
if a.configs != "all": CONFIGS = {k: v for k, v in CONFIGS.items() if k in a.configs.split(",")}
def run_checks(c, docs, para=True):
    out = []
    for d in docs:
        sp, labs = spans(d)
        E = Embedder.encode([s for _, _, s in sp], c.p["embed_model"], c.p["query_prefix"]) if para else None
        r = c.check(d["text"], sp, embeddings=E, exclude_quotes=c.p["excl"], exclude_bibliography=c.p["excl"], paraphrase=para)
        out.append((labs, r))
    return out
def score(results, thr=None, idmap=None):
    tp = fp = fn = 0; per = {}; quoted = [0, 0]; bib = [0, 0]
    for labs, r in results:
        for l, x in zip(labs, r):
            m = None
            if not x["excluded"]:
                m = x["exact"] or (x["paraphrase"] if x["paraphrase"] and (thr is None or x["paraphrase"]["score"] >= thr) else None)
            if l["type"] == "quoted": quoted[0] += 1; quoted[1] += bool(m)
            if l["type"] == "bib": bib[0] += 1; bib[1] += bool(m)
            truth = l["src"]
            if truth is not None:
                ok = m is not None and idmap[m["doc"]] == truth
                per.setdefault(l["type"], [0, 0]); per[l["type"]][0] += 1; per[l["type"]][1] += ok
                tp += ok; fn += not ok
                if m is not None and not ok: fp += 1
            elif m is not None: fp += 1
    P_ = tp / max(1, tp + fp); R_ = tp / max(1, tp + fn)
    return {"precision": round(P_, 3), "recall": round(R_, 3), "f1": round(2 * P_ * R_ / max(1e-9, P_ + R_), 3),
            "recall_by_type": {k: round(v[1] / v[0], 3) for k, v in sorted(per.items())},
            "quoted_flagged": f"{quoted[1]}/{quoted[0]}", "bibliography_flagged": f"{bib[1]}/{bib[0]}"}
OUTF = D / ("results_full.json" if a.full else "results.json")
prev = json.load(open(OUTF)) if OUTF.exists() else {}
results = {"corpus_docs": len(use), "dev_docs": len(dev), "test_docs": len(test),
           "test_plag_sentences": sum(1 for d in test for l in d["labels"] if l["src"] is not None)}
results.update({k: v for k, v in prev.items() if k in CONFIGS})
for name, cfg in CONFIGS.items():
    if name in results: print("cached", name); continue
    p = dict(DEFAULT); p.update(cfg); p.setdefault("excl", True)
    tmp = tempfile.mkdtemp(); c = Corpus(pathlib.Path(tmp) / "c.db", params=p); idmap = {}
    t0 = time.time()
    for i in use:
        s = split_sentences(corpus[i]["text"]); d = c.add(corpus[i]["title"], corpus[i]["text"], s, source="wiki", save=False); idmap[d] = i
    c.save_index(); t_index = time.time() - t0
    t0 = time.time(); rd = run_checks(c, dev); rt = run_checks(c, test); t_check = time.time() - t0
    nw = sum(len(d["text"].split()) for d in dev + test)
    best = max([x / 100 for x in range(60, 96)], key=lambda t: score(rd, t, idmap)["f1"]) if name != "v1_minilm" else p["para_threshold"]
    res = score(rt, best, idmap); res.update(threshold=best, index_seconds=round(t_index, 1), check_s_per_1000w=round(t_check / nw * 1000, 3),
                                             exact_only=score([(l, [dict(x, paraphrase=None) for x in r]) for l, r in rt], None, idmap))
    results[name] = res; print(name, json.dumps(res), flush=True)
    json.dump(results, open(OUTF, "w"), indent=2)

