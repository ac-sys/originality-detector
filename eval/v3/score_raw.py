"""Cache raw logits for v2 (exact app grouping) and v3 (chunks w/o tiny tails + sliding windows), per text.
Usage: score_raw.py SPLIT MODEL(fk|dk) [--windows] [--limit N]  -> data/eval/v3raw/SPLIT.MODEL[.win].jsonl (resumable)"""
import sys, os, json, pathlib, time, argparse
B = pathlib.Path(__file__).resolve().parents[2]; sys.path.insert(0, str(B))
import torch; torch.set_num_threads(int(os.environ.get("NTHREADS", "8")))
from app.ai_v3 import HFClassifier, Desklib, chunks, windows
from app.extract import split_sentences
ap = argparse.ArgumentParser(); ap.add_argument("split"); ap.add_argument("model"); ap.add_argument("--windows", action="store_true")
ap.add_argument("--limit", type=int, default=100000); ap.add_argument("--sources", default=""); a = ap.parse_args()
D = B / "data/eval"; O = D / "v3raw"; O.mkdir(exist_ok=True)
SPECS = {"fk": (HFClassifier, {"hf_id": "fakespot-ai/roberta-base-ai-text-detection-v1", "ai_label": "AI"}),
         "dk": (Desklib, {"hf_id": "desklib/ai-text-detector-v1.01"})}
M = SPECS[a.model][0](SPECS[a.model][1])
def v2_groups(sents, max_words, max_sents=None):
    groups, cur, n = [], [], 0
    for i, s in enumerate(sents):
        cur.append(i); n += len(s.split())
        if n >= max_words or (max_sents and len(cur) >= max_sents): groups.append(cur); cur, n = [], 0
    if cur:
        if groups and max_sents and n < 25: groups[-1] += cur
        else: groups.append(cur)
    return groups
rows = [json.loads(l) for l in open(D / f"{a.split}.jsonl")][:a.limit]
out = O / f"{a.split}.{a.model}{'.win' if a.windows else ''}.jsonl"
done = sum(1 for _ in open(out)) if out.exists() else 0
t0 = time.time(); words = 0
with open(out, "a") as f:
    for i, r in enumerate(rows):
        if i < done: continue
        if a.sources and r.get("source") not in a.sources.split(","): f.write("null\n"); continue
        sents = r.get("sentences") or [s[2] for s in split_sentences(r["text"])]
        def L(groups): return M.logits([" ".join(sents[j] for j in g) for g in groups])
        rec = {"n": len(sents), "w": [len(s.split()) for s in sents]}
        if a.windows:
            wi = windows(sents); rec["win3"] = [[v, g] for v, g in zip(L(wi), wi)]
            if a.model == "fk":
                g2 = v2_groups(sents, 90, 3); rec["v2win"] = [[v, g] for v, g in zip(L(g2), g2)]
        else:
            ch = chunks(sents); rec["chunks"] = [[v, g] for v, g in zip(L(ch), ch)]
            if a.model == "fk":
                g2 = v2_groups(sents, 300); rec["v2chunks"] = [[v, g] for v, g in zip(L(g2), g2)]
        words += sum(rec["w"]); f.write(json.dumps(rec) + "\n"); f.flush()
print(a.split, a.model, a.windows, "done", round(time.time() - t0), "s", round((time.time() - t0) / max(words, 1) * 1000, 2), "s/1000w", flush=True)
