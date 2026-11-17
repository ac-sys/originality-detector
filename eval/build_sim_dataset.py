"""Synthetic plagiarism evaluation set (PAN-style construction), from Wikipedia (CC BY-SA).
corpus.jsonl : 12,000 source articles (first 400 words)  -> indexed library
susp_{dev,test}.jsonl : suspicious documents = held-out filler sentences + inserted passages from
   corpus docs with obfuscation: verbatim | light (word deletions/swaps) | backtranslation (en->de->en,
   Helsinki-NLP opus-mt) | shuffle (light edit + sentence reorder). Some docs also contain a properly
   quoted passage and a References section (should be excluded, not flagged). 20% of docs are clean.
Each sentence is labelled with its true source doc index (or null)."""
import json, random, re, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from datasets import load_dataset
from app.extract import split_sentences
R = random.Random(5); D = pathlib.Path(__file__).resolve().parent.parent / "data" / "simeval"
arts = []
for x in load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True):
    t = " ".join(re.sub(r"\s+", " ", x["text"]).split()[:400])
    if len(t.split()) >= 200: arts.append({"title": x["title"], "text": t})
    if len(arts) >= 12600: break
corpus, held = arts[:12000], arts[12000:]
with open(D / "corpus.jsonl", "w") as f:
    for a in corpus: f.write(json.dumps(a) + "\n")
from transformers import MarianMTModel, MarianTokenizer
import torch; torch.set_num_threads(4)
mt = {}
def bt(sents):
    out = sents
    for a, b in [("en", "de"), ("de", "en")]:
        k = f"Helsinki-NLP/opus-mt-{a}-{b}"
        if k not in mt: mt[k] = (MarianTokenizer.from_pretrained(k), MarianMTModel.from_pretrained(k).eval())
        tok, m = mt[k]
        with torch.no_grad():
            x = tok(out, return_tensors="pt", padding=True, truncation=True, max_length=256)
            out = tok.batch_decode(m.generate(**x, num_beams=2, max_new_tokens=200), skip_special_tokens=True)
    return out
def light(s):
    w = s.split()
    w = [x for x in w if R.random() > 0.12 or len(w) < 6]
    if len(w) > 4:
        i = R.randrange(len(w) - 1); w[i], w[i+1] = w[i+1], w[i]
    return " ".join(w)
def sents_of(t): return [x[2] for x in split_sentences(t)]
docs = []
for n in range(300):
    filler = sents_of(held[n % len(held)]["text"]) + sents_of(held[(n * 7 + 3) % len(held)]["text"])
    filler = filler[R.randrange(0, 5):][: R.randint(8, 16)]
    items = [(s, None, "clean") for s in filler]
    if R.random() >= 0.2:
        for _ in range(R.randint(1, 3)):
            src = R.randrange(len(corpus)); ss = sents_of(corpus[src]["text"])
            st = R.randrange(0, max(1, len(ss) - 4)); pas = ss[st: st + R.randint(2, 4)]
            ob = R.choice(["verbatim", "light", "backtranslation", "shuffle"])
            if ob == "light": pas = [light(s) for s in pas]
            elif ob == "backtranslation": pas = bt(pas)
            elif ob == "shuffle": pas = [light(s) for s in pas]; R.shuffle(pas)
            pos = R.randrange(0, len(items) + 1)
            items[pos:pos] = [(s, src, ob) for s in pas]
    if R.random() < 0.3:   # properly quoted, cited passage -> should be excluded
        src = R.randrange(len(corpus)); q = sents_of(corpus[src]["text"])[1]
        items.insert(R.randrange(len(items) + 1), (f'As one source puts it, "{q}" (Wikipedia, 2023).', None, "quoted"))
    text_sents = [s for s, _, _ in items]; text = " ".join(text_sents); labels = [{"src": src, "type": ob} for _, src, ob in items]
    if R.random() < 0.3:
        refs = [f"{corpus[R.randrange(len(corpus))]['title']}. Wikipedia, The Free Encyclopedia. Retrieved 2023." for _ in range(3)]
        text += "\nReferences\n" + "\n".join(refs)
    docs.append({"id": n, "text": text, "labels": labels, "sentences": text_sents})
for name, part in [("dev", docs[:100]), ("test", docs[100:])]:
    with open(D / f"susp_{name}.jsonl", "w") as f:
        for d in part: f.write(json.dumps(d) + "\n")
print("corpus", len(corpus), "docs", len(docs), "plag sentences", sum(1 for d in docs for l in d["labels"] if l["src"] is not None))
