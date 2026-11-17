"""Builds the v2 evaluation set (all open data):
  MAGE (Li et al. 2024, Apache-2.0): train/valid/test + OOD GPT-4 and GPT-4-paraphrased test sets
  RAID (Dugan et al. 2024, MIT): first ~250k rows of train.csv streamed, stratified sample by model/attack
  ESL human essays (evaluation only): TOEFL (Liang et al. 2023, MIT repo), ELLIPSE/Feedback-Prize ELL
  (tcapelle/feedback-prize-english-language-learning-fluency), ICNALE learner essays (srky/ICNALE_writing_score)
  Native student essays: Hewlett ASAP subset from the Liang et al. repo.
Splits: train (fit meta-classifier / fine-tune), calib (calibration + thresholds), test (held-out report).
RAID rows are split by source_id so the same prompt never appears in two splits."""
import json, random, re, collections, pathlib, hashlib
from datasets import load_dataset
R = random.Random(13); OUT = pathlib.Path(__file__).resolve().parent.parent / "data" / "eval"; OUT.mkdir(parents=True, exist_ok=True)
def clean(t, maxw=400):
    t = re.sub(r"\s+", " ", str(t)).strip(); w = t.split()
    return " ".join(w[:maxw]) if len(w) >= 80 else None
def mage(split, n_per_label, name=None):
    rows = {0: [], 1: []}
    ds = load_dataset("yaful/MAGE", data_files={"x": f"{split}.csv"}, split="x", streaming=True).shuffle(seed=13, buffer_size=20000)
    for x in ds:
        t = clean(x["text"]);
        if not t: continue
        y = 0 if str(x["label"]) == "1" else 1   # MAGE: 1 = human
        if len(rows[y]) < n_per_label:
            src = x["src"]; rows[y].append({"text": t, "label": y, "source": name or f"mage_{split}", "domain": src.split("_")[0], "generator": "human" if y == 0 else src.split("_", 1)[-1]})
        if all(len(v) >= n_per_label for v in rows.values()): break
    return rows[0] + rows[1]
def mage_ood(file, name):
    ds = load_dataset("yaful/MAGE", data_files={"x": file}, split="x")
    out = []
    for x in ds:
        t = clean(x["text"])
        if t: y = 0 if str(x["label"]) == "1" else 1; out.append({"text": t, "label": y, "source": name, "domain": "ood", "generator": "human" if y == 0 else name})
    R.shuffle(out); return out
def raid(max_rows=250000):
    buckets = collections.defaultdict(list)
    ds = load_dataset("liamdugan/raid", "raid", split="train", streaming=True)
    for i, x in enumerate(ds):
        if i >= max_rows: break
        t = clean(x["generation"])
        if not t: continue
        key = (x["model"], x["attack"]); b = buckets[key]
        item = {"text": t, "label": 0 if x["model"] == "human" else 1, "source": "raid", "domain": x["domain"], "generator": x["model"],
                "attack": x["attack"], "decoding": x["decoding"], "group": x["source_id"]}
        if len(b) < 60: b.append(item)
        elif R.random() < 0.02: b[R.randrange(60)] = item
    items = [it for b in buckets.values() for it in b]
    print("RAID buckets:", {f"{k[0]}|{k[1]}": len(v) for k, v in buckets.items()})
    return items
def esl():
    out = []
    base = pathlib.Path("/workspace/esl_src/Data_and_Results/Human_Data")
    for d, name in [("TOEFL_real_91", "esl_toefl"), ("HewlettStudentEssay_real_88", "native_hewlett")]:
        for x in json.load(open(base / d / "data.json")):
            t = clean(x["document"], 600) or re.sub(r"\s+", " ", x["document"]).strip()
            out.append({"text": t, "label": 0, "source": name, "domain": "student_essay", "generator": "human"})
    ell = [x for x in load_dataset("tcapelle/feedback-prize-english-language-learning-fluency", split="train")]
    R.shuffle(ell)
    for x in ell[:250]:
        t = clean(x["full_text"], 600)
        if t: out.append({"text": t, "label": 0, "source": "esl_ellipse", "domain": "student_essay", "generator": "human"})
    ic = [x for x in load_dataset("srky/ICNALE_writing_score", split="train")]; R.shuffle(ic)
    seen = set()
    for x in ic:
        t = clean(x["OriginalText"], 600)
        if t and t not in seen and sum(s.startswith("esl_icnale") for s in [o["source"] for o in out]) < 250:
            seen.add(t); out.append({"text": t, "label": 0, "source": "esl_icnale", "domain": "student_essay", "generator": "human"})
    for x in json.load(open(base.parent / "GPT_Data" / "TOEFL_gpt4polished_91" / "data.json")) if (base.parent / "GPT_Data" / "TOEFL_gpt4polished_91").exists() else []:
        pass
    return out
def dump(name, rows):
    with open(OUT / f"{name}.jsonl", "w") as f:
        for r in rows: f.write(json.dumps(r) + "\n")
    c = collections.Counter((r["source"], r["label"]) for r in rows); print(name, len(rows), dict(c))
if __name__ == "__main__":
    rd = raid(); groups = sorted({r["group"] for r in rd}); R.shuffle(groups)
    gsplit = {g: ("train" if i % 10 < 5 else "calib" if i % 10 < 7 else "test") for i, g in enumerate(groups)}
    train = mage("train", 600) + [r for r in rd if gsplit[r["group"]] == "train"]
    calib = mage("valid", 200) + [r for r in rd if gsplit[r["group"]] == "calib"]
    test = mage("test", 300) + mage_ood("test_ood_set_gpt.csv", "mage_ood_gpt4")[:150] + mage_ood("test_ood_set_gpt_para.csv", "mage_ood_gpt4_para")[:150] \
           + [r for r in rd if gsplit[r["group"]] == "test"]
    for n, r in [("train", train), ("calib", calib), ("test", test), ("esl", esl())]:
        R.shuffle(r); dump(n, r)
