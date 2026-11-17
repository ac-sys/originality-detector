import json, os, threading, time
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from .config import BASE, REPORTS, DATA, REMOTE_MANIFEST_URL
from .registry import Registry
from .ai_detect import Ensemble
from .similarity import Corpus, fingerprints, Embedder
from .extract import extract_text, normalise, split_sentences
from .jobs import JobManager
from .pdf_report import build_pdf
from .rules import RuleEngine, compile_english_rule
from .websources import PROVIDERS

app = FastAPI(title="Originality Detector MVP", version="0.3.2")
registry = Registry(); ensemble = Ensemble(registry); corpus = Corpus(); rule_engine = RuleEngine()
jobs = JobManager(corpus, ensemble, workers=int(os.environ.get("DETECTOR_WORKERS", "2")), rules=rule_engine)
APPEALS = DATA / "appeals.json"
MAX_BYTES = 200 * 1024 * 1024

@app.get("/", response_class=HTMLResponse)
def index(): return (BASE / "static" / "index.html").read_text(encoding="utf-8")

@app.get("/api/health")
def health():
    return {
        "ok": True,
        "library_docs": corpus.count(),
        "models": ensemble.versions(),
        "similarity": corpus.p.get("id"),
        "active_rules": sum(1 for r in rule_engine.list_rules() if r.get("enabled", True)),
        "web_providers": list(PROVIDERS.keys()),
    }

@app.get("/api/providers")
def get_providers():
    return {
        "similarity_providers": [
            {"id": "wikipedia", "name": "Wikipedia", "free": True},
            {"id": "openalex", "name": "OpenAlex (250M+ Works)", "free": True},
            {"id": "crossref", "name": "CrossRef (150M+ DOIs)", "free": True},
            {"id": "semanticscholar", "name": "Semantic Scholar (200M+ Papers)", "free": True},
            {"id": "arxiv", "name": "arXiv Preprints", "free": True},
            {"id": "europepmc", "name": "Europe PMC (Biomedical/Science)", "free": True},
            {"id": "brave", "name": "Brave Web Search", "free": False, "configured": bool(os.environ.get("BRAVE_API_KEY"))},
        ],
        "online_ai_providers": {
            "sapling": bool(os.environ.get("SAPLING_API_KEY")),
            "gptzero": bool(os.environ.get("GPTZERO_API_KEY")),
            "llm": bool(os.environ.get("ONLINE_LLM_URL")),
        },
    }

@app.post("/api/submit")
async def submit(
    file: UploadFile = File(None),
    text: str = Form(""),
    title: str = Form(""),
    add_to_library: bool = Form(True),
    web_sources: str = Form(""),
    verify_refs: bool = Form(False),
    online_ai: bool = Form(False),
):
    if file is not None and file.filename:
        data = await file.read()
        fname = file.filename
        if len(data) > MAX_BYTES: raise HTTPException(413, "File too large")
        if not fname.lower().endswith((".pdf", ".docx", ".txt", ".md")): raise HTTPException(400, "Use PDF, DOCX, TXT or MD")
    elif text.strip():
        data = text.encode("utf-8")
        fname = (title.strip() or "pasted-text") + ".txt"
    else:
        raise HTTPException(400, "Provide a file or pasted text")
    ws = [p.strip() for p in web_sources.split(",") if p.strip()] if web_sources else None
    return {
        "job_id": jobs.submit(
            fname,
            data,
            title or fname,
            add_to_library,
            web_sources=ws,
            verify_refs=verify_refs,
            online_ai=online_ai,
        )
    }

@app.post("/api/batch")
async def submit_batch(files: list[UploadFile] = File(...), add_to_library: bool = Form(True)):
    """Stage 3: Batch upload & cross-student cohort collusion check."""
    if not files:
        raise HTTPException(400, "No files uploaded")
    docs = []
    job_ids = []
    for f in files:
        if not f.filename.lower().endswith((".pdf", ".docx", ".txt", ".md")):
            continue
        raw = await f.read()
        txt = normalise(extract_text(f.filename, raw))
        sents = split_sentences(txt)
        fps = {h for h, _, _ in fingerprints(txt)}
        docs.append({"filename": f.filename, "sents": sents, "fps": fps})
        jid = jobs.submit(f.filename, raw, f.filename, add_to_library)
        job_ids.append({"filename": f.filename, "job_id": jid})

    pairs = []
    for i in range(len(docs)):
        for j in range(i + 1, len(docs)):
            a, b = docs[i], docs[j]
            union_len = max(1, min(len(a["fps"]), len(b["fps"])))
            overlap = round(100.0 * len(a["fps"] & b["fps"]) / union_len, 1) if a["fps"] and b["fps"] else 0.0
            pairs.append({"doc_a": a["filename"], "doc_b": b["filename"], "fingerprint_overlap_percent": overlap})
    pairs.sort(key=lambda x: -x["fingerprint_overlap_percent"])
    return {"jobs": job_ids, "collusion_pairs": pairs}

@app.get("/api/jobs/{jid}")
def job(jid: str):
    if jid in jobs.jobs: return jobs.jobs[jid]
    if jobs.report(jid): return {"id": jid, "status": "done", "progress": 1.0}
    raise HTTPException(404)

@app.get("/api/reports")
def list_reports(limit: int = 50):
    return jobs.list_reports(limit=limit)

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

@app.delete("/api/library/{doc_id}")
def delete_library_doc(doc_id: int):
    if not corpus.delete(doc_id): raise HTTPException(404, "Document not found")
    return {"ok": True, "deleted": doc_id}

@app.post("/api/appeals/{jid}")
def appeal(jid: str, payload: dict):
    """Student/staff appeal against a report: stored for human review (evidence-first workflow)."""
    if not jobs.report(jid): raise HTTPException(404)
    items = json.loads(APPEALS.read_text(encoding="utf-8")) if APPEALS.exists() else []
    items.append({"report": jid, "time": time.time(), "status": "open", "by": payload.get("by", ""), "reason": payload.get("reason", "")})
    APPEALS.write_text(json.dumps(items, indent=2), encoding="utf-8"); return {"ok": True, "open_appeals": sum(i["status"] == "open" for i in items)}

@app.get("/api/appeals")
def appeals(): return json.loads(APPEALS.read_text(encoding="utf-8")) if APPEALS.exists() else []

@app.patch("/api/appeals/{idx}")
def update_appeal(idx: int, payload: dict):
    items = json.loads(APPEALS.read_text(encoding="utf-8")) if APPEALS.exists() else []
    if idx < 0 or idx >= len(items): raise HTTPException(404, "Appeal not found")
    if "status" in payload: items[idx]["status"] = payload["status"]
    if "resolution" in payload: items[idx]["resolution"] = payload["resolution"]
    APPEALS.write_text(json.dumps(items, indent=2), encoding="utf-8")
    return {"ok": True, "appeal": items[idx]}

# ---- plain-English AI rules ----
@app.get("/api/rules")
def list_rules():
    return rule_engine.list_rules()

@app.post("/api/rules")
def create_rule(payload: dict):
    try:
        return rule_engine.add_rule(
            instruction=payload.get("instruction", ""),
            effect=payload.get("effect"),
            weight=payload.get("weight"),
            enabled=payload.get("enabled", True),
        )
    except ValueError as e:
        raise HTTPException(400, str(e))

@app.patch("/api/rules/{rule_id}")
def patch_rule(rule_id: str, payload: dict):
    r = rule_engine.update_rule(rule_id, payload)
    if not r: raise HTTPException(404, "Rule not found")
    return r

@app.delete("/api/rules/{rule_id}")
def remove_rule(rule_id: str):
    if not rule_engine.delete_rule(rule_id): raise HTTPException(404, "Rule not found")
    return {"ok": True, "deleted": rule_id}

@app.post("/api/rules/test")
def test_rules_on_text(payload: dict):
    txt = normalise(payload.get("text", ""))
    sents = split_sentences(txt)
    if not sents:
        return {"sentences": [], "rules_applied": []}
    fake_ai = {"probability": 0.5, "share_flagged": 0.0, "sentence_probs": [0.5] * len(sents), "sentence_flagged": [False] * len(sents), "verdict": "no_strong_signal"}
    embs = Embedder.encode([t for _, _, t in sents])
    hits = rule_engine.evaluate(txt, sents, fake_ai, embeddings=embs)
    return {
        "ai": fake_ai,
        "sentences": [{"text": t, "ai_prob": fake_ai["sentence_probs"][i], "ai_flag": fake_ai["sentence_flagged"][i], "rule_hits": hits[i]} for i, (_, _, t) in enumerate(sents)],
    }

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
