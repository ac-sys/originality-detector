"""Grid search for winnowing k/w/coverage (exact channel only) on dev; report best on test. -> data/simeval/winnow_grid.json"""
import os, sys, json, pathlib, tempfile, importlib
os.environ["DETECTOR_LIGHT"] = "1"
sys.argv = [sys.argv[0], "--configs", "none"]
B = pathlib.Path(__file__).resolve().parent.parent; sys.path.insert(0, str(B)); sys.path.insert(0, str(B / "eval"))
import app.similarity as sm
from app.extract import split_sentences
ev = importlib.import_module("eval_similarity")   # reuses data + scoring (runs no configs)
res = {}
for k in (3, 4, 5, 6):
    for w in (3, 4, 6):
        p = dict(sm.P); p.update(k=k, w=w, excl=True)
        c = sm.Corpus(pathlib.Path(tempfile.mkdtemp()) / "c.db", params=p); idmap = {}
        for i in ev.use:
            d = c.add("", ev.corpus[i]["text"], split_sentences(ev.corpus[i]["text"])); idmap[d] = i
        for cov in (0.2, 0.3, 0.5):
            c.p["exact_coverage"] = cov
            rd = ev.run_checks(c, ev.dev, para=False); rt = ev.run_checks(c, ev.test, para=False)
            res[f"k{k}_w{w}_cov{cov}"] = {"dev": ev.score(rd, None, idmap), "test": ev.score(rt, None, idmap)}
            print(k, w, cov, res[f"k{k}_w{w}_cov{cov}"]["dev"]["f1"], res[f"k{k}_w{w}_cov{cov}"]["test"]["f1"], flush=True)
best = max(res, key=lambda x: res[x]["dev"]["f1"]); res["best_on_dev"] = best
json.dump(res, open(B / "data/simeval/winnow_grid.json", "w"), indent=2); print("best", best, res[best]["test"])
