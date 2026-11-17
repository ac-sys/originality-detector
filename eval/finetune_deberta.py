"""Fine-tune microsoft/deberta-v3-small (MIT) on the v2 train split (CPU). Saves to models/local/deberta-small-detector-v2."""
import json, pathlib, random, time, torch, sys
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
torch.set_num_threads(int(__import__("os").environ.get("NTHREADS","8"))); torch.manual_seed(0); random.seed(0)
B = pathlib.Path(__file__).resolve().parent.parent; OUT = B / "models" / "local" / "deberta-small-detector-v2"
rows = [json.loads(l) for l in open(B / "data/eval/train.jsonl")]
# balance: RAID is AI-heavy; downsample AI to keep classes near 50/50
hum = [r for r in rows if r["label"] == 0]; ai = [r for r in rows if r["label"] == 1]; random.shuffle(ai)
rows = hum + ai[:int(len(hum) * 1.3)]; random.shuffle(rows)
tok = AutoTokenizer.from_pretrained("microsoft/deberta-v3-small")
m = AutoModelForSequenceClassification.from_pretrained("microsoft/deberta-v3-small", dtype=torch.float32, num_labels=2, id2label={0: "human", 1: "ai"}, label2id={"human": 0, "ai": 1})
EPOCHS, BS, ML = 1, 16, 192
opt = torch.optim.AdamW(m.parameters(), lr=3e-5, weight_decay=0.01)
steps = EPOCHS * (len(rows) // BS); sch = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
m.train(); t0 = time.time(); st = 0
for ep in range(EPOCHS):
    random.shuffle(rows)
    for i in range(0, len(rows) - BS + 1, BS):
        b = rows[i:i+BS]; x = tok([r["text"] for r in b], truncation=True, max_length=ML, padding=True, return_tensors="pt")
        loss = m(**x, labels=torch.tensor([r["label"] for r in b])).loss
        loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); sch.step(); opt.zero_grad(); st += 1
        if st % 10 == 0: print(f"ep{ep} step {st}/{steps} loss {loss.item():.3f} {time.time()-t0:.0f}s", flush=True)
OUT.mkdir(parents=True, exist_ok=True); m.save_pretrained(OUT); tok.save_pretrained(OUT)
json.dump({"train_rows": len(rows), "epochs": EPOCHS, "max_len": ML, "seconds": round(time.time() - t0)}, open(OUT / "train_info.json", "w"))
print("saved", OUT)
