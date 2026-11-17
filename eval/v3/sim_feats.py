"""Stage 1 (slow, run once, resumable): cache everything the v3 paraphrase matcher needs so that threshold
search is pure numpy. Library = all true sources + 2000 distractors (filler docs excluded), as eval_sim_v3.py.
Per query sentence stores: exact-match doc (v2 fingerprint stage), excluded flag, ok flag, top-5 (score, doc),
and for each candidate doc (any top-5 doc with cos>=0.80, or exact doc) the best single-sentence and
best adjacent-pair cosine. -> data/simeval/cache/feats_<set>.pkl"""
import json, sys, time, pathlib, tempfile, os, random, pickle
B = pathlib.Path(__file__).resolve().parents[2]; sys.path.insert(0, str(B)); os.environ["DETECTOR_LIGHT"] = "0"
import numpy as np, torch; torch.set_num_threads(int(os.environ.get("NTHREADS", "4")))
import app.similarity as sm
from app.extract import split_sentences
D = B / "data/simeval"; CD = D / "cache"
corpus = [json.loads(l) for l in open(D / "corpus.jsonl")]
L = lambda f: [json.loads(l) for l in open(D / f)]
sets = {"orig_dev": L("susp_dev.jsonl"), "orig_test": L("susp_test.jsonl"), "hard_dev": L("hard_dev.jsonl"), "hard_test": L("hard_test.jsonl")}
need = sorted({l["src"] for v in sets.values() for d in v for l in d["labels"] if l["src"] is not None})
clean = {s for v in sets.values() for d in v for s, l in zip(d["sentences"], d["labels"]) if l["src"] is None}
filler_docs = {i for i, c in enumerate(corpus) if any(x[2] in clean for x in split_sentences(c["text"])[:60])}
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
enc = lambda xs: sm.Embedder.encode(xs, V2["embed_model"], V2["query_prefix"]).astype(np.float32)
lib_sents = {i: split_sentences(corpus[i]["text"]) for i in use}
owner = np.array([i for i in use for _ in lib_sents[i]]); flat = [s[2] for i in use for s in lib_sents[i]]
if (CD / "lib_emb.npy").exists() and len(np.load(CD / "lib_emb.npy", mmap_mode="r")) == len(flat): EMB = np.load(CD / "lib_emb.npy")
else: t = time.time(); EMB = enc(flat); np.save(CD / "lib_emb.npy", EMB); print("lib embedded", round(time.time() - t), flush=True)
rows_of = {}
for k, d in enumerate(owner): rows_of.setdefault(int(d), []).append(k)
pf = CD / "pair_emb.pkl"; PAIR = pickle.load(open(pf, "rb")) if pf.exists() else {}
def pairs(d):
    if d not in PAIR:
        s = [x[2] for x in lib_sents[d]]; PAIR[d] = enc([s[k] + " " + s[k + 1] for k in range(len(s) - 1)]) if len(s) > 1 else np.zeros((0, 384), np.float32)
    return PAIR[d]
c = sm.Corpus(pathlib.Path(tempfile.mkdtemp()) / "c.db", params=V2); dbid = {}; k = 0
for i in use:
    n = len(lib_sents[i]); dbid[c.add(corpus[i]["title"], corpus[i]["text"], lib_sents[i], source="wiki", embeddings=EMB[k:k + n], save=False)] = i; k += n
for name, docs in sets.items():
    out_f = CD / f"feats_{name}.pkl"
    if out_f.exists(): print("have", name); continue
    out = []; t = time.time()
    for j, doc in enumerate(docs):
        sp, labs = spans(doc); Q = enc([s for _, _, s in sp])
        r = c.check(doc["text"], sp, embeddings=Q, paraphrase=False)
        S = Q @ EMB.T; top = np.argsort(-S, 1)[:, :5]
        sents = []
        for i in range(len(sp)):
            ex = dbid[r[i]["exact"]["doc"]] if r[i]["exact"] else None
            sents.append({"exact": ex, "excluded": bool(r[i]["excluded"]), "nw": len(sp[i][2].split()),
                          "top": [(float(S[i, q]), int(owner[q])) for q in top[i]]})
        cands = {d for s in sents for sc, d in s["top"] if sc >= 0.80} | {s["exact"] for s in sents if s["exact"] is not None}
        for s in sents: s["cand"] = {}
        for d in cands:
            single = S[:, rows_of[d]].max(1); P = pairs(d); pair = (Q @ P.T).max(1) if len(P) else np.full(len(sp), -1.0)
            for i, s in enumerate(sents): s["cand"][d] = (float(single[i]), float(pair[i]))
        out.append({"labs": labs, "sents": sents, "words": len(doc["text"].split())})
        if j % 20 == 19:
            print(name, j + 1, len(docs), round(time.time() - t), "s", flush=True); pickle.dump(PAIR, open(pf, "wb"))
    pickle.dump(out, open(out_f, "wb")); pickle.dump(PAIR, open(pf, "wb")); print("done", name, round(time.time() - t), flush=True)
print("ALLDONE", flush=True)
