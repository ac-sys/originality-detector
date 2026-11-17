"""Compute and cache features for each (extractor, split). Resumable. Usage: run_features.py EXTRACTOR SPLIT [LIMIT]"""
import sys, json, time, pathlib, os
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent)); sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import numpy as np
import torch, os as _os; torch.set_num_threads(int(_os.environ.get("NTHREADS", "8")))
D = pathlib.Path(__file__).resolve().parent.parent / "data" / "eval"
LIMITS = {"train": 1500, "calib": 800, "test": 1500, "esl": 10000}
def load(split):
    rows = [json.loads(l) for l in open(D / f"{split}.jsonl")]
    return rows[:LIMITS[split]]
def make(name):
    from features import LMPair, Classifier, Desklib
    if name == "smol": m = LMPair("HuggingFaceTB/SmolLM2-135M", "HuggingFaceTB/SmolLM2-135M-Instruct", "s_"); return lambda ts: [m.features(t) for t in ts]
    if name == "qwen": m = LMPair("Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-0.5B-Instruct", "q_"); return lambda ts: [m.features(t) for t in ts]
    if name == "fakespot": m = Classifier("fakespot-ai/roberta-base-ai-text-detection-v1", "fs"); return lambda ts: [{"fakespot": v} for v in m.batch(ts)]
    if name == "deberta":
        m = Classifier(str(pathlib.Path(__file__).resolve().parent.parent / "models/local/deberta-small-detector-v2"), "deb", ai_label="ai", maxlen=256)
        return lambda ts: [{"deberta": v} for v in m.batch(ts)]
    if name == "desklib": m = Desklib(); return lambda ts: [{"desklib": v} for v in m.batch(ts)]
    if name == "v1":
        os.environ["DETECTOR_LIGHT"] = "0"
        from app.registry import Registry; from app.ai_detect import Ensemble; from app.extract import split_sentences
        import app.config as cfg
        reg = Registry(cfg.MANIFEST); e = Ensemble(reg); out = []
        def f(ts):
            res = []
            for t in ts:
                s = split_sentences(t); lp, cl, _ = e.features(t, s); w = np.array([len(x[2].split()) for x in s], float)
                lpv = np.array(lp); ok = ~np.isnan(lpv)
                res.append({"v1_lp": float(np.average(lpv[ok], weights=w[ok])) if ok.any() else -4.0, "v1_clf": float(np.average(cl, weights=w))})
            return res
        return f
name, split = sys.argv[1], sys.argv[2]; lim = int(sys.argv[3]) if len(sys.argv) > 3 else None
rows = load(split)[:lim] if lim else load(split)
out_p = D / f"feats_{name}_{split}.jsonl"
done = sum(1 for _ in open(out_p)) if out_p.exists() else 0
if done >= len(rows): print("cached", name, split); sys.exit()
fn = make(name); t0 = time.time(); words = 0
with open(out_p, "a") as f:
    for i in range(done, len(rows), 8):
        b = [r["text"] for r in rows[i:i+8]]; words += sum(len(x.split()) for x in b)
        for r in fn(b): f.write(json.dumps(r) + "\n")
        f.flush()
        if (i // 8) % 10 == 0: print(name, split, i + len(b), "/", len(rows), "s/1000w=%.2f" % ((time.time() - t0) / words * 1000), flush=True)
print("done", name, split, "s/1000w=%.2f" % ((time.time() - t0) / max(words, 1) * 1000), flush=True)
