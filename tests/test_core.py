import io, json, time, os
import docx
from app.extract import extract_text, split_sentences, normalise
from app.similarity import fingerprints, Corpus
from app.registry import Registry
from app.config import MANIFEST, DATA

A = ("The quick brown fox jumps over the lazy dog near the river bank. Students must cite every source they use in "
     "their assignments. Universities in Zimbabwe are adopting digital learning platforms quickly.")
def test_split():
    s = split_sentences(A); assert len(s) == 3 and A[s[1][0]:s[1][1]].startswith("Students")
def test_docx_and_txt():
    d = docx.Document(); d.add_paragraph(A); b = io.BytesIO(); d.save(b)
    assert "lazy dog" in extract_text("x.docx", b.getvalue()); assert extract_text("x.txt", A.encode()) == A
def test_pdf():
    from reportlab.pdfgen import canvas
    b = io.BytesIO(); c = canvas.Canvas(b); c.drawString(50, 800, "Hello plagiarism world sentence."); c.save()
    assert "plagiarism" in extract_text("x.pdf", b.getvalue())
def test_winnowing_robust_to_case_and_punct():
    f1 = {h for h, _, _ in fingerprints(A)}; f2 = {h for h, _, _ in fingerprints(A.upper().replace(".", ";"))}
    assert f1 and f1 == f2
def test_corpus_exact_match(tmp_path):
    c = Corpus(tmp_path / "c.db"); c.add("source1", A, split_sentences(A))
    q = "My essay says: the quick brown fox jumps over the lazy dog near the river bank. Then other unrelated words follow here."
    r = c.check(q, split_sentences(q))
    assert r[0]["exact"] and r[0]["exact"]["title"] == "source1"; assert r[1]["exact"] is None
def test_registry_activate_rollback():
    r = Registry(MANIFEST); before = dict(r.manifest["active"])
    r.activate(active={"classifier": "x@2"}, reason="t"); assert r.manifest["active"]["classifier"] == "x@2"
    r.rollback(); assert r.manifest["active"] == before
def test_registry_check_remote(tmp_path):
    r = Registry(MANIFEST); remote = json.loads(MANIFEST.read_text())
    remote["detectors"].append({"id": "new@2.0.0", "slot": "classifier", "type": "hf_sequence_classifier", "hf_id": "x"})
    p = tmp_path / "m.json"; p.write_text(json.dumps(remote))
    c = r.check("file://" + str(p)); assert [d["id"] for d in c["updates"]] == ["new@2.0.0"]
def test_api_flow():
    from fastapi.testclient import TestClient
    from app.main import app
    cl = TestClient(app)
    j = cl.post("/api/submit", files={"file": ("a.txt", A.encode())}, data={"title": "first"}).json()["job_id"]
    for _ in range(50):
        if cl.get(f"/api/jobs/{j}").json()["status"] in ("done", "error"): break
        time.sleep(0.1)
    assert cl.get(f"/api/jobs/{j}").json()["status"] == "done"
    j2 = cl.post("/api/submit", files={"file": ("b.txt", A.encode())}).json()["job_id"]
    for _ in range(50):
        if cl.get(f"/api/jobs/{j2}").json()["status"] == "done": break
        time.sleep(0.1)
    rep = cl.get(f"/api/reports/{j2}").json()
    assert rep["similarity"]["exact_percent"] == 100.0 and rep["similarity"]["sources"][0]["title"] == "first"
    assert "not proof" in rep["ai"]["disclaimer"]
    pdf = cl.get(f"/api/reports/{j2}/pdf"); assert pdf.content[:4] == b"%PDF"
    assert cl.post(f"/api/appeals/{j2}", json={"reason": "I wrote it"}).json()["ok"]
    big = (A + "\n") * 400   # ~1200 sentences -> multiple blocks
    j3 = cl.post("/api/submit", files={"file": ("big.txt", big.encode())}, data={"add_to_library": "false"}).json()["job_id"]
    for _ in range(200):
        if cl.get(f"/api/jobs/{j3}").json()["status"] == "done": break
        time.sleep(0.1)
    assert len(cl.get(f"/api/reports/{j3}").json()["sentences"]) == 1200

def test_exclusions():
    from app.similarity import exclusion_flags
    t = 'Intro text here that is long enough. As Smith said, "students must always cite every source they use in assignments" (Smith, 2020). More text.\nReferences\nSmith, J. (2020). Academic writing. Harare Press.'
    s = split_sentences(t); f = exclusion_flags(t, s)
    assert "quotation" in f and f[-1] == "bibliography" and f[0] is None

def test_v2_summarise_thresholds():
    from app.ai_detect import Ensemble
    cal = [h["calibration"] for h in json.loads(MANIFEST.read_text())["history"] if h["calibration"].get("type") == "threshold_v2"][-1]; assert cal["type"] == "threshold_v2"  # v2 kept for rollback
    e = Ensemble.__new__(Ensemble); e.cal = cal; e.registry = Registry(MANIFEST)
    sents = split_sentences(A)
    hi = e.summarise([9.0] * 3, [9.0] * 3, cal, sents); lo = e.summarise([-9.0] * 3, [-9.0] * 3, cal, sents)
    assert hi["verdict"] == "likely_ai_strict" and lo["verdict"] == "no_strong_signal" and hi["probability"] > lo["probability"]
    assert all(hi["sentence_flagged"]) and not any(lo["sentence_flagged"])

def test_rollback_restores_v1_and_similarity():
    r = Registry(MANIFEST); v2 = json.loads(MANIFEST.read_text())
    assert v2["calibration"]["type"] == "threshold_v3" and v2["similarity"].get("version") == 3
    assert v2["similarity"]["id"].endswith("3.2.0") and "condense" in v2["similarity"]
    r.rollback(); m = json.loads(MANIFEST.read_text())   # v3.2 -> v3.1
    assert m["similarity"]["id"].endswith("3.1.0") and "condense" not in m["similarity"]
    r.rollback(); m = json.loads(MANIFEST.read_text())   # v3.1 -> v3 (similarity back to v2 matcher)
    assert m["calibration"]["type"] == "threshold_v3" and "version" not in m["similarity"] and m["similarity"]["para_threshold"] == 0.92
    r.rollback(); m = json.loads(MANIFEST.read_text())   # v3 -> v2
    assert m["calibration"]["type"] == "threshold_v2" and "classifier2" not in m["active"]
    r.rollback(); m = json.loads(MANIFEST.read_text())   # v2 -> v1
    assert m["active"]["classifier"] == "hc3-roberta@1.0.0" and m["similarity"]["embed_model"].endswith("MiniLM-L6-v2")
    MANIFEST.write_text(json.dumps(v2))

def test_web_query_selection():
    from app.websources import pick_queries, check_web
    s = split_sentences(A * 3); assert len(pick_queries(s, 2)) <= 2
    assert check_web(s, providers=[])["enabled"] is False

def test_library_delete_and_paste_submit(tmp_path):
    c = Corpus(tmp_path / "del.db")
    did = c.add("temp_doc", A, split_sentences(A))
    assert c.count() == 1 and c.delete(did) is True and c.count() == 0
    assert c.delete(99999) is False
    from fastapi.testclient import TestClient
    from app.main import app
    cl = TestClient(app)
    j = cl.post("/api/submit", data={"text": A, "title": "pasted-doc", "add_to_library": "false"}).json()["job_id"]
    for _ in range(50):
        if cl.get(f"/api/jobs/{j}").json()["status"] == "done": break
        time.sleep(0.1)
    rlist = cl.get("/api/reports").json()
    assert any(r["id"] == j for r in rlist)
    assert cl.patch("/api/appeals/0", json={"status": "resolved"}).json()["appeal"]["status"] == "resolved"
