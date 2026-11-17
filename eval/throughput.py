"""Scale tests: (1) index the 12,000-doc Wikipedia corpus into a fresh library and time it, then time
checks against it; (2) build a 100+ page DOCX and PDF (~55k words) with planted plagiarism and run it
through the full job pipeline (extraction + AI detection + similarity). Results -> data/throughput.json"""
import os, sys, json, time, pathlib, random, io, tempfile
os.environ["DETECTOR_LIGHT"] = "0"
B = pathlib.Path(__file__).resolve().parent.parent; sys.path.insert(0, str(B))
tmp = tempfile.mkdtemp(); os.environ["DETECTOR_DATA"] = tmp
from app.similarity import Corpus
from app.extract import split_sentences, extract_text, normalise
R = random.Random(3); out = {}
corpus = [json.loads(l) for l in open(B / "data/simeval/corpus.jsonl")]
c = Corpus(pathlib.Path(tmp) / "corpus.db"); t0 = time.time()
for i, a in enumerate(corpus):
    c.add(a["title"], a["text"], split_sentences(a["text"]), source="wiki", save=False)
    if i % 2000 == 0: print("indexed", i, round(time.time() - t0), flush=True)
c.save_index(); ti = time.time() - t0
out["corpus"] = {"docs": len(corpus), "sentences": c.index().ntotal, "words": sum(len(a["text"].split()) for a in corpus),
                 "index_seconds": round(ti, 1), "docs_per_second": round(len(corpus) / ti, 1),
                 "db_mb": round(os.path.getsize(c.path) / 1e6, 1), "faiss_mb": round(os.path.getsize(c.index_path) / 1e6, 1)}
print(out["corpus"], flush=True)
# big document: 140 unseen articles + 30 verbatim/edited passages from the corpus
from datasets import load_dataset
import re, docx
held = []
for x in load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True).skip(20000):
    if len(x["text"].split()) > 300: held.append(re.sub(r"[ \t]+", " ", x["text"]))
    if len(held) >= 140: break
paras = []
for h in held: paras += [p for p in h.split("\n") if len(p.split()) > 20]
planted = []
for k in range(30):
    src = R.randrange(len(corpus)); ss = [s[2] for s in split_sentences(corpus[src]["text"])][2:5]
    planted.append(src); paras.insert(R.randrange(len(paras)), " ".join(ss))
words = sum(len(p.split()) for p in paras)
d = docx.Document()
for p in paras: d.add_paragraph(p)
bio = io.BytesIO(); d.save(bio); data_docx = bio.getvalue()
from reportlab.lib.pagesizes import A4; from reportlab.platypus import SimpleDocTemplate, Paragraph; from reportlab.lib.styles import getSampleStyleSheet
from xml.sax.saxutils import escape
pb = io.BytesIO(); st = getSampleStyleSheet()
SimpleDocTemplate(pb, pagesize=A4).build([Paragraph(escape(p), st["BodyText"]) for p in paras]); data_pdf = pb.getvalue()
from pypdf import PdfReader
pages = len(PdfReader(io.BytesIO(data_pdf)).pages)
t0 = time.time(); txt_pdf = extract_text("big.pdf", data_pdf); t_pdf = time.time() - t0
t0 = time.time(); txt_docx = extract_text("big.docx", data_docx); t_docx = time.time() - t0
out["big_document"] = {"words": words, "pdf_pages": pages, "pdf_extract_s": round(t_pdf, 1), "docx_extract_s": round(t_docx, 1)}
# full pipeline via JobManager
import app.main as M
M.corpus = c; M.jobs.corpus = c
t0 = time.time(); jid = M.jobs.submit("big.docx", data_docx, "big", add_to_library=False)
while M.jobs.jobs[jid]["status"] not in ("done", "error"): time.sleep(2)
tt = time.time() - t0; rep = M.jobs.report(jid)
found = {m["doc"] - 1 for s in rep["sentences"] for m in [s["exact"] or s["paraphrase"]] if m}
out["big_document"].update({"pipeline_seconds": round(tt, 1), "s_per_1000_words": round(tt / words * 1000, 2), "status": M.jobs.jobs[jid]["status"],
    "planted_sources": 30, "planted_sources_found": len(set(planted) & found), "similarity_percent": rep["similarity"]["percent"],
    "ai_probability": rep["ai"]["probability"]})
print(out, flush=True)
json.dump(out, open(B / "data/throughput.json", "w"), indent=2)
