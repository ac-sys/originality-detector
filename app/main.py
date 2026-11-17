import json, os, threading, time
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from .config import BASE, REPORTS, DATA, REMOTE_MANIFEST_URL
from .registry import Registry
from .ai_detect import Ensemble
from .similarity import Corpus
from .extract import extract_text, normalise, split_sentences
from .jobs import JobManager
from .pdf_report import build_pdf

app = FastAPI(title="Originality Detector MVP", version="0.1.0")
registry = Registry(); ensemble = Ensemble(registry); corpus = Corpus()
jobs = JobManager(corpus, ensemble, workers=int(os.environ.get("DETECTOR_WORKERS", "2")))
APPEALS = DATA / "appeals.json"
MAX_BYTES = 200 * 1024 * 1024

@app.get("/", response_class=HTMLResponse)
def index(): return (BASE / "static" / "index.html").read_text()

@app.get("/api/health")
def health(): return {"ok": True, "library_docs": corpus.count(), "models": ensemble.versions(), "similarity": corpus.p.get("id")}

@app.post("/api/submit")
async def submit(file: UploadFile = File(...), title: str = Form(""), add_to_library: bool = Form(True)):
    data = await file.read()
    if len(data) > MAX_BYTES: raise HTTPException(413, "File too large")
    if not file.filename.lower().endswith((".pdf", ".docx", ".txt", ".md")): raise HTTPException(400, "Use PDF, DOCX or TXT")
    return {"job_id": jobs.submit(file.filename, data, title or file.filename, add_to_library)}

@app.get("/api/jobs/{jid}")
def job(jid: str):
    if jid not in jobs.jobs: raise HTTPException(404)
    return jobs.jobs[jid]

@app.get("/api/reports/{jid}")
def report(jid: str):
    r = jobs.report(jid)
    if not r: raise HTTPException(404)
    return r

@app.get("/api/reports/{jid}/pdf")
def report_pdf(jid: str):
    r = jobs.report(jid)
    if not r: raise HTTPException(404)
    path = build_pdf(r, REPORTS / f"{jid}.pdf")
    return FileResponse(path, media_type="application/pdf", filename=f"report-{jid}.pdf")

@app.post("/api/library")
async def add_reference(file: UploadFile = File(...), title: str = Form(""), source: str = Form("reference")):
    text = normalise(extract_text(file.filename, await file.read()))
    return {"doc_id": corpus.add(title or file.filename, text, split_sentences(text), source=source)}

@app.get("/api/library")
def library():
    return [{"id": i, "title": t, "source": s} for i, (t, s) in corpus.titles().items()]

@app.post("/api/appeals/{jid}")
def appeal(jid: str, payload: dict):
    """Student/staff appeal against a report: stored for human review (evidence-first workflow)."""
    if not jobs.report(jid): raise HTTPException(404)
    items = json.loads(APPEALS.read_text()) if APPEALS.exists() else []
    items.append({"report": jid, "time": time.time(), "status": "open", "by": payload.get("by", ""), "reason": payload.get("reason", "")})
    APPEALS.write_text(json.dumps(items, indent=2)); return {"ok": True, "open_appeals": sum(i["status"] == "open" for i in items)}

@app.get("/api/appeals")
def appeals(): return json.loads(APPEALS.read_text()) if APPEALS.exists() else []

# ---- model registry / updates ----
@app.get("/api/registry")
def reg(): return registry.manifest

@app.get("/api/calibration")
def calibration():
    p = DATA / "eval_history.json"
    return json.loads(p.read_text()) if p.exists() else []

@app.post("/api/registry/check")
def reg_check(url: str = ""): return registry.check(url or None)

@app.post("/api/registry/update")
def reg_update(url: str = ""):
    """Stage new detector versions from the remote manifest, activate the remote's choice, hot-swap.
    (In production: run eval/evaluate.py on the staged model first and only activate if it beats the
    current model on the benchmark - see RESEARCH_AND_DESIGN.md.)"""
    remote = registry.fetch_remote(url or None)
    if remote is None: raise HTTPException(400, "No remote manifest configured (DETECTOR_MANIFEST_URL)")
    added = registry.stage(remote)
    registry.activate(active=remote.get("active"), calibration=remote.get("calibration"), reason=f"update from remote: {added}")
    ensemble.reload(); _reload_similarity(); return {"added": added, "active": ensemble.versions()}

@app.post("/api/registry/rollback")
def reg_rollback():
    try: prev = registry.rollback()
    except ValueError as e: raise HTTPException(400, str(e))
    ensemble.reload(); _reload_similarity(); return {"active": ensemble.versions(), "similarity": registry.manifest.get("similarity", {}).get("id")}

def _reload_similarity():
    """Apply similarity params from the manifest; re-embeds the library if the embedding model changed."""
    import app.similarity as sm
    sm.P = sm.sim_params(); old = corpus.p["embed_model"]; corpus.p = dict(sm.P)
    if corpus.p["embed_model"] != old:
        corpus._index = None; corpus.reindex()

def _scheduler():
    while True:
        time.sleep(24 * 3600)
        try:
            c = registry.check()
            if c.get("updates") or c.get("calibration_update"):
                print("[registry] updates available:", [d["id"] for d in c["updates"]])
        except Exception as e: print("[registry] check failed", e)
if REMOTE_MANIFEST_URL: threading.Thread(target=_scheduler, daemon=True).start()
