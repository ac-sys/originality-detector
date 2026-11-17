"""Runs the real models (slow). Enabled with RUN_MODEL_TESTS=1."""
import os, pytest
pytestmark = pytest.mark.skipif(os.environ.get("RUN_MODEL_TESTS") != "1", reason="set RUN_MODEL_TESTS=1")
def test_real_ensemble_and_paraphrase(tmp_path):
    import app.config as cfg; cfg.LIGHT_MODE = False
    import app.ai_detect as ad, app.similarity as sm; ad.LIGHT_MODE = False; sm.LIGHT_MODE = False
    from app.registry import Registry; from app.extract import split_sentences
    e = ad.Ensemble(Registry(cfg.MANIFEST))
    t = "Climate change is one of the most pressing issues of our time. It affects ecosystems, economies, and communities around the world. Addressing it requires coordinated global action."
    r = e.analyse(t, split_sentences(t)); assert 0 < r["probability"] < 1 and len(r["sentence_probs"]) == 3 and "verdict" in r
    c = sm.Corpus(tmp_path / "c.db"); src = "Lecturers should give students clear guidance about using artificial intelligence tools in coursework."
    c.add("src", src, split_sentences(src))
    q = "Students ought to receive clear direction from lecturers on how AI tools may be used in their coursework."
    res = c.check(q, split_sentences(q)); assert res[0]["paraphrase"] and res[0]["exact"] is None
