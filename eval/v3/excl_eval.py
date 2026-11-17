"""Measured quote / bibliography exclusion test (synthetic, deterministic, seed 7). Pure text, no models.
Builds 300 documents. Each has: 3 plain source sentences (must NOT be excluded), 2 quoted source sentences in a
random quoting style (must be excluded), and optionally a reference list in a random style (must be excluded).
Usage: excl_eval.py [module:function]  (default app.similarity:exclusion_flags) -> prints/returns metrics"""
import json, random, re, sys, pathlib, importlib
B = pathlib.Path(__file__).resolve().parents[2]; sys.path.insert(0, str(B))
from app.extract import split_sentences
import os; R = random.Random(int(os.environ.get("SEED", "7")))
corpus = [json.loads(l) for l in open(B / "data/simeval/corpus.jsonl")][:3000]
pool = [s[2] for c in corpus for s in split_sentences(c["text"])[:8] if 12 <= len(s[2].split()) <= 35 and '"' not in s[2] and "'" not in s[2]]
NAMES = ["Smith", "Moyo", "Chen", "García", "Okafor", "Dube", "Nguyen", "Brown"]
def quoted(s):
    a, y = R.choice(NAMES), R.randint(1995, 2024); body = s.rstrip(".")
    style = R.choice(["straight", "curly", "single_curly", "single", "long_lead", "block", "block_gt", "mid_sentence"])
    if style == "straight": return style, f'As {a} ({y}) wrote, "{body}." (p. {R.randint(2,300)})', False
    if style == "curly": return style, f"According to {a} ({y}), “{body}.”", False
    if style == "single_curly": return style, f"{a} ({y}) states that ‘{body}’.", False
    if style == "single": return style, f"{a} notes: '{body}.' ({a}, {y})", False
    if style == "long_lead": return style, f"In a widely cited discussion of the background to this topic in the regional literature, {a} ({y}, p. {R.randint(2,90)}) observes that \"{body}.\"", False
    if style == "block": return style, f"{a} ({y}) explains:\n\n    {body}. ({a}, {y}, p. {R.randint(2,90)})\n", True
    if style == "block_gt": return style, f"{a} ({y}) explains:\n\n> {body}.\n", True
    return style, f"The claim that \"{body[:1].lower() + body[1:]}\" is supported by {a} ({y}).", False
def refs():
    style = R.choice(["References", "REFERENCES", "Reference List", "Works Cited", "Bibliography", "Literature Cited", "Sources", "numbered_noheader", "References:"])
    items = [f"{R.choice(NAMES)}, {R.choice('ABCDJKM')}. ({R.randint(1990,2024)}). {R.choice(pool)[:80].rstrip('.')}. Journal of {R.choice(['Education','Science','History'])}, {R.randint(1,40)}({R.randint(1,4)}), {R.randint(1,200)}-{R.randint(201,400)}." for _ in range(R.randint(3, 7))]
    if style == "numbered_noheader": return style, "\n".join(f"[{k+1}] {x}" for k, x in enumerate(items))
    return style, style + "\n" + "\n".join(items)
docs = []
for n in range(300):
    plain = R.sample(pool, 3); q = [quoted(s) for s in R.sample(pool, 2)]
    parts = [("plain", p) for p in plain[:2]] + [("quote:" + st, t) for st, t, _ in q] + [("plain", plain[2])]
    body = "This essay discusses the topic in some detail and sets out the main arguments in turn. " + " ".join(t if "\n" not in t else "\n" + t + "\n" for _, t in parts) + "\nIn conclusion, these points show why the topic matters for students today and in the future."
    ref = refs() if R.random() < 0.7 else None
    text = body + ("\n\n" + ref[1] if ref else "")
    docs.append({"text": text, "parts": parts, "ref": ref})
def evaluate(fn):
    st = {"plain_wrongly_excluded": [0, 0], "quote_excluded": {}, "ref_excluded": {}}
    for d in docs:
        sents = split_sentences(d["text"]); flags = fn(d["text"], sents)
        def flag_of(fragment):   # any sentence overlapping the fragment's core
            core = fragment.strip().split("\n")[-1].strip()[5:60] if "\n" in fragment.strip() else fragment[5:60]
            hit = [f for (s, e, t), f in zip(sents, flags) if core[:40] in t or t[:40] in fragment]
            return any(hit) if hit else None
        for kind, t in d["parts"]:
            f = flag_of(t)
            if f is None: continue
            if kind == "plain": st["plain_wrongly_excluded"][0] += 1; st["plain_wrongly_excluded"][1] += f
            else: v = st["quote_excluded"].setdefault(kind[6:], [0, 0]); v[0] += 1; v[1] += f
        if d["ref"]:
            start = d["text"].rfind(d["ref"][1]); lines = [(s, f) for (s, e, t), f in zip(sents, flags) if s >= start + (len(d["ref"][1].split("\n")[0]) if not d["ref"][0].startswith("numbered") else 0)]
            v = st["ref_excluded"].setdefault(d["ref"][0], [0, 0]); v[0] += len(lines); v[1] += sum(bool(f) for _, f in lines)
        else:
            tail = [f for (s, e, t), f in zip(sents, flags) if t.startswith("In conclusion")]
            st["plain_wrongly_excluded"][0] += len(tail); st["plain_wrongly_excluded"][1] += sum(bool(f) for f in tail)
    agg = lambda dct: round(sum(v[1] for v in dct.values()) / max(1, sum(v[0] for v in dct.values())), 3)
    return {"plain_false_exclusion": round(st["plain_wrongly_excluded"][1] / st["plain_wrongly_excluded"][0], 4), "n_plain": st["plain_wrongly_excluded"][0],
            "quote_exclusion_rate": agg(st["quote_excluded"]), "quote_by_style": {k: f"{v[1]}/{v[0]}" for k, v in sorted(st["quote_excluded"].items())},
            "ref_line_exclusion_rate": agg(st["ref_excluded"]), "ref_by_style": {k: f"{v[1]}/{v[0]}" for k, v in sorted(st["ref_excluded"].items())}}
if __name__ == "__main__":
    mod, fn = (sys.argv[1] if len(sys.argv) > 1 else "app.similarity:exclusion_flags").split(":")
    print(json.dumps(evaluate(getattr(importlib.import_module(mod), fn)), indent=1))
