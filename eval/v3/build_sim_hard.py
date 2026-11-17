"""Harder plagiarism set for v3 (PAN-style construction, Wikipedia CC BY-SA sources from data/simeval/corpus.jsonl).
Obfuscations: llm_paraphrase (passage rewritten by Qwen2.5-0.5B-Instruct "in your own words"), llm_condense
(passage condensed to fewer sentences), mosaic (sentences from 2 sources interleaved, lightly edited).
Filler = sentences from corpus docs that are never put in the library. 20% clean docs.
-> data/simeval/hard_{dev,test}.jsonl, hard_sources.json"""
import json, random, re, pathlib, sys, os
B = pathlib.Path(__file__).resolve().parents[2]; sys.path.insert(0, str(B))
import torch; torch.set_num_threads(int(os.environ.get("NTHREADS", "4")))
from transformers import AutoTokenizer, AutoModelForCausalLM
from app.extract import split_sentences
R = random.Random(77); D = B / "data/simeval"
corpus = [json.loads(l) for l in open(D / "corpus.jsonl")]
old = [json.loads(l) for l in open(D / "susp_dev.jsonl")] + [json.loads(l) for l in open(D / "susp_test.jsonl")]
old_need = {l["src"] for d in old for l in d["labels"] if l["src"] is not None}
pool = [i for i in range(len(corpus)) if i not in old_need]; R.shuffle(pool)
filler_pool, src_pool = pool[:400], pool[400:]
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct"); m = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", torch_dtype=torch.float32).eval()
def llm(instr, text):
    msgs = [{"role": "user", "content": f"{instr}\n\n{text}"}]
    x = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True)
    with torch.no_grad():
        y = m.generate(**x, max_new_tokens=220, do_sample=True, temperature=0.7, top_p=0.9)
    out = tok.decode(y[0, x["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    return re.sub(r"\s+", " ", out)
def light(s):
    w = s.split(); w = [x for x in w if R.random() > 0.12 or len(w) < 6]
    if len(w) > 4: i = R.randrange(len(w) - 1); w[i], w[i+1] = w[i+1], w[i]
    return " ".join(w)
def sents(t): return [x[2] for x in split_sentences(t)]
out_path = D / "hard_all.jsonl"; done = sum(1 for _ in open(out_path)) if out_path.exists() else 0
sources = set(); si = 0
with open(out_path, "a") as f:
    for n in range(160):
        fill = sents(corpus[filler_pool[2 * n]]["text"]) + sents(corpus[filler_pool[2 * n + 1]]["text"])
        fill = fill[R.randrange(0, 5):][:R.randint(8, 14)]
        items = [(s, None, "clean") for s in fill]
        plan = []
        if R.random() >= 0.2:
            for _ in range(R.randint(1, 2)):
                plan.append((R.choice(["llm_paraphrase", "llm_paraphrase", "llm_condense", "mosaic"]), src_pool[si], src_pool[si + 1])); si += 2
        if n < done:  # keep RNG/source sequence identical on resume
            for ob, a, b in plan: sources.add(a); sources.add(b) if ob == "mosaic" else None
            continue
        for ob, a, b in plan:
            sa = sents(corpus[a]["text"]); st = R.randrange(0, max(1, len(sa) - 4)); pas = sa[st: st + R.randint(2, 4)]
            if ob == "mosaic":
                sb = sents(corpus[b]["text"]); st2 = R.randrange(0, max(1, len(sb) - 3)); pb = sb[st2: st2 + 2]
                new = []
                for k in range(max(len(pas), len(pb))):
                    if k < len(pas): new.append((light(pas[k]), a))
                    if k < len(pb): new.append((light(pb[k]), b))
                sources.update([a, b])
            else:
                instr = ("Rewrite the following passage in your own words. Keep the meaning and facts but change the wording and "
                         "sentence structure. Output only the rewritten passage.") if ob == "llm_paraphrase" else \
                        ("Condense the following passage into one or two sentences in your own words. Output only the result.")
                new = [(s, a) for s in sents(llm(instr, " ".join(pas)))]; sources.add(a)
            pos = R.randrange(0, len(items) + 1); items[pos:pos] = [(s, src, ob) for s, src in new]
        text = " ".join(s for s, _, _ in items)
        f.write(json.dumps({"id": n, "text": text, "sentences": [s for s, _, _ in items], "labels": [{"src": s, "type": t} for _, s, t in items]}) + "\n"); f.flush()
        print(n, flush=True)
docs = [json.loads(l) for l in open(out_path)]
for name, part in [("dev", docs[:60]), ("test", docs[60:])]:
    with open(D / f"hard_{name}.jsonl", "w") as f:
        for d in part: f.write(json.dumps(d) + "\n")
json.dump(sorted({l["src"] for d in docs for l in d["labels"] if l["src"] is not None}), open(D / "hard_sources.json", "w"))
print("done", len(docs))
