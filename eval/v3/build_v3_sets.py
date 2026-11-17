"""v3 evaluation sets (all open data, none used for training any model here):
  esl_calib.jsonl : 300 extra ELLIPSE + 150 extra ICNALE essays NOT in esl.jsonl -> used ONLY to set ESL-aware
                    human thresholds; esl.jsonl stays fully held-out.
  academic.jsonl  : Liang et al. (2023) repo: CollegeEssay_real_70, CS224N_real_145 (human), CollegeEssay/CS224N
                    gpt3 + gpt3PromptEng (AI; PromptEng = prompted to sound more human/literary), TOEFL_gpt4polished
                    (human text polished by GPT-4: reported separately, no ground-truth 'right' answer).
  mixed.jsonl     : 150 synthetic mixed documents from the held-out TEST split (MAGE test): human sentences with
                    one inserted AI passage (or none for 20%), per-sentence labels."""
import json, random, re, pathlib, sys
B = pathlib.Path(__file__).resolve().parents[2]; sys.path.insert(0, str(B))
from datasets import load_dataset
from app.extract import split_sentences
R = random.Random(31); D = B / "data/eval"
def clean(t, maxw=600):
    t = re.sub(r"\s+", " ", str(t)).strip(); w = t.split(); return " ".join(w[:maxw]) if len(w) >= 80 else None
used = {json.loads(l)["text"] for l in open(D / "esl.jsonl")}
out = []
ell = list(load_dataset("tcapelle/feedback-prize-english-language-learning-fluency", split="train")); R.shuffle(ell)
for x in ell:
    t = clean(x["full_text"])
    if t and t not in used and sum(o["source"] == "esl_ellipse" for o in out) < 300: out.append({"text": t, "label": 0, "source": "esl_ellipse"}); used.add(t)
ic = list(load_dataset("srky/ICNALE_writing_score", split="train")); R.shuffle(ic)
for x in ic:
    t = clean(x["OriginalText"])
    if t and t not in used and sum(o["source"] == "esl_icnale" for o in out) < 150: out.append({"text": t, "label": 0, "source": "esl_icnale"}); used.add(t)
with open(D / "esl_calib.jsonl", "w") as f:
    for o in out: f.write(json.dumps(o) + "\n")
print("esl_calib", len(out))
base = pathlib.Path("/workspace/esl_src/Data_and_Results"); acad = []
for sub, d, lab in [("Human_Data", "CollegeEssay_real_70", 0), ("Human_Data", "CS224N_real_145", 0), ("Human_Data", "TOEFL_gpt4polished_91", -1),
                    ("GPT_Data", "CollegeEssay_gpt3_31", 1), ("GPT_Data", "CollegeEssay_gpt3PromptEng_31", 1),
                    ("GPT_Data", "CS224N_gpt3_145", 1), ("GPT_Data", "CS224N_gpt3PromptEng_145", 1)]:
    for x in json.load(open(base / sub / d / "data.json")):
        t = re.sub(r"\s+", " ", x["document"]).strip()
        if len(t.split()) >= 40: acad.append({"text": t, "label": lab, "source": d})
with open(D / "academic.jsonl", "w") as f:
    for o in acad: f.write(json.dumps(o) + "\n")
print("academic", len(acad))
test = [json.loads(l) for l in open(D / "test.jsonl")]
hum = [r for r in test if r["label"] == 0 and r["source"].startswith("mage")]
ai = [r for r in test if r["label"] == 1 and r["source"].startswith("mage")]
mixed = []
for n in range(150):
    h = [s[2] for s in split_sentences(hum[n % len(hum)]["text"])]
    items = [(s, 0) for s in h[:R.randint(6, 12)]]
    if R.random() >= 0.2:
        a = [s[2] for s in split_sentences(ai[(n * 3) % len(ai)]["text"])]
        st = R.randrange(0, max(1, len(a) - 4)); pas = a[st: st + R.randint(3, 6)]
        pos = R.randrange(0, len(items) + 1); items[pos:pos] = [(s, 1) for s in pas]
    mixed.append({"text": " ".join(s for s, _ in items), "sent_labels": [l for _, l in items], "sentences": [s for s, _ in items]})
with open(D / "mixed.jsonl", "w") as f:
    for o in mixed: f.write(json.dumps(o) + "\n")
print("mixed", len(mixed), "ai sents", sum(sum(m["sent_labels"]) for m in mixed), "total", sum(len(m["sent_labels"]) for m in mixed))
# mixed_dev: same construction from the CALIB split (MAGE valid) -> used to tune highlighting; mixed.jsonl stays held-out
calib = [json.loads(l) for l in open(D / "calib.jsonl")]
hum = [r for r in calib if r["label"] == 0 and r["source"].startswith("mage")]; ai = [r for r in calib if r["label"] == 1 and r["source"].startswith("mage")]
md = []
for n in range(120):
    h = [s[2] for s in split_sentences(hum[n % len(hum)]["text"])]; items = [(s, 0) for s in h[:R.randint(6, 12)]]
    if R.random() >= 0.2:
        a = [s[2] for s in split_sentences(ai[(n * 3) % len(ai)]["text"])]; st = R.randrange(0, max(1, len(a) - 4)); pas = a[st: st + R.randint(3, 6)]
        pos = R.randrange(0, len(items) + 1); items[pos:pos] = [(s, 1) for s in pas]
    md.append({"text": " ".join(s for s, _ in items), "sent_labels": [l for _, l in items], "sentences": [s for s, _ in items]})
with open(D / "mixed_dev.jsonl", "w") as f:
    for o in md: f.write(json.dumps(o) + "\n")
print("mixed_dev", len(md))
