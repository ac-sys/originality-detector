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

def test_english_ai_rules_and_batch_collusion(tmp_path):
    from app.rules import RuleEngine, compile_english_rule
    from app.websources import extract_reference_entries, PROVIDERS
    assert {"wikipedia", "openalex", "crossref", "semanticscholar", "arxiv", "europepmc"} <= set(PROVIDERS.keys())
    c1 = compile_english_rule("Flag sentences containing 'delve into' or 'rich tapestry'")
    assert c1["effect"] == "increase_ai" and "delve into" in c1["phrases"]
    c2 = compile_english_rule("Do not flag sentences with code or statistical formulas under 40 words")
    assert c2["effect"] == "exempt" and "code_or_math" in c2["structural"] and c2["max_words"] == 40
    eng = RuleEngine(tmp_path / "rules.json")
    added = eng.add_rule("Flag sentences containing 'quantum synergy'")
    sample = "This essay explores quantum synergy in modern business. In our fieldwork, we measured p < 0.05 across samples."
    sents = split_sentences(sample)
    fake_ai = {"probability": 0.4, "share_flagged": 0.0, "sentence_probs": [0.4, 0.4], "sentence_flagged": [False, False], "verdict": "no_strong_signal"}
    hits = eng.evaluate(sample, sents, fake_ai)
    assert any(h["id"] == added["id"] for h in hits[0])
    assert fake_ai["sentence_probs"][0] > 0.4
    assert fake_ai["sentence_probs"][1] <= 0.05  # exempted by p < 0.05 statistical rule
    refs = extract_reference_entries("Body text here.\nReferences\nSmith, J. (2021). Deep learning in education. Journal of AI, 12(3), 45-60.")
    assert len(refs) == 1 and "Smith" in refs[0]
    from fastapi.testclient import TestClient
    from app.main import app
    cl = TestClient(app)
    r_create = cl.post("/api/rules", json={"instruction": "Flag sentences containing 'synthetic buzzword'"}).json()
    assert r_create["id"].startswith("rule-")
    r_test = cl.post("/api/rules/test", json={"text": "This contains synthetic buzzword in a sentence."}).json()
    assert len(r_test["sentences"][0]["rule_hits"]) >= 1
    assert cl.delete(f"/api/rules/{r_create['id']}").json()["ok"] is True
    b_res = cl.post("/api/batch", files=[("files", ("s1.txt", A.encode())), ("files", ("s2.txt", A.encode()))], data={"add_to_library": "false"}).json()
    assert len(b_res["collusion_pairs"]) == 1 and b_res["collusion_pairs"][0]["fingerprint_overlap_percent"] == 100.0
