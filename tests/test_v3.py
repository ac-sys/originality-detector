"""v3 unit tests (run in light mode, no model weights needed)."""
from app.ai_v3 import chunks, windows, smooth_flags, block_features, summarise_v3, fuse

class Fake:
    """Deterministic fake detector: logit = +8 if the text contains 'AIWORD', else -6."""
    def logits(self, texts): return [8.0 if "AIWORD" in t else -6.0 for t in texts]

CAL = {"type": "threshold_v3", "slots": {"fk": "classifier", "dk": "classifier2"}, "chunk_words": 300, "min_tail": 120,
       "window_sents": 3, "window_max_words": 90,
       "doc": {"bias": 0.0, "weights": {"fk": 0.5, "dk": 0.5}, "thresholds": {"fpr_1pct": 3.0, "fpr_2pct": 2.0},
               "isotonic": [[-5, 5], [0.01, 0.99]]},
       "win": {"bias": 0.0, "weights": {"fk": 1.0}, "thresholds": {"fpr_1pct": 4.0, "fpr_2pct": 3.0},
               "isotonic": [[-5, 5], [0.01, 0.99]], "use": "fpr_2pct", "min_run": 2}}

def test_chunks_merge_short_tail():
    s = ["w " * 100] * 3 + ["w " * 50]
    assert chunks(s) == [[0, 1, 2, 3]]
    s = ["w " * 100] * 3 + ["w " * 150]
    assert chunks(s) == [[0, 1, 2], [3]]

def test_sliding_windows_cover_every_sentence():
    s = ["a b c"] * 7
    w = windows(s)
    assert len(w) == 5 and w[0] == [0, 1, 2] and w[-1] == [4, 5, 6]
    assert sorted({i for g in w for i in g}) == list(range(7))
    assert windows(["x"]) == [[0]]

def test_smooth_flags_drops_isolated():
    assert smooth_flags([5, -1, 5, 5, -1], 0, 2) == [False, False, True, True, False]
    assert smooth_flags([5, -1], 0, 1) == [True, False]

def test_fuse_clips():
    assert fuse({"bias": 1, "weights": {"a": 2}}, {"a": 100}) == 31

def test_block_features_and_summary_localise_ai_segment():
    human = [f"Human sentence number {i} about my weekend." for i in range(8)]
    ai = [f"AIWORD generated sentence {i} delves into the tapestry." for i in range(6)]
    sents = human + ai
    dets = {"fk": Fake(), "dk": Fake()}
    A, B = block_features(dets, CAL, sents)
    out = summarise_v3(A, B, CAL, sents)
    flags = out["sentence_flagged"]
    assert all(flags[8:]) and not any(flags[:6])
    assert out["verdict"] in ("likely_ai_strict", "possible_ai", "no_strong_signal")
    assert 0 < out["share_flagged"] < 1

def test_all_human_not_flagged():
    sents = [f"Plain human sentence {i}." for i in range(10)]
    A, B = block_features({"fk": Fake(), "dk": Fake()}, CAL, sents)
    out = summarise_v3(A, B, CAL, sents)
    assert out["verdict"] == "no_strong_signal" and not any(out["sentence_flagged"])

def test_ensemble_v3_light_mode_neutral():
    from app.ai_detect import Ensemble
    class Reg:
        manifest = {"active": {}, "calibration": CAL}
    e = Ensemble(Reg())
    sents = [(0, 5, "Hello there."), (6, 12, "Another one.")]
    a, b, cal = e.features("Hello there. Another one.", sents)
    out = e.summarise(a, b, cal, sents)
    assert out["verdict"] == "no_strong_signal" and "disclaimer" in out

def test_exclusion_v31_styles():
    from app.similarity import exclusion_flags_v31
    from app.extract import split_sentences
    t = ("Plain opening sentence about the topic that is long enough to count here. According to Moyo (2016), “The innermost layer of the cortex produces androgens in many target organs.” "
         "Dube notes that ‘single curly quotes are also a common way to mark a direct quotation’. Another plain sentence follows with no quotation at all in it.\n\n"
         "    This is an indented block quotation that runs for well over eight words in total. (Chen, 2020, p. 4)\n\n"
         "Closing plain sentence of the body text appears here.\n"
         "[1] Smith, J. (2020). A study of things. Journal of Stuff, 3(2), 1-10.\n[2] Chen, L. (2019). Another study. Science, 1(1), 2-3.\n[3] Dube, K. (2018). Third. History, 2(2), 5-9.")
    s = split_sentences(t); f = exclusion_flags_v31(t, s)
    got = {x[2][:20]: y for x, y in zip(s, f)}
    assert got["Plain opening senten"] is None and got["Another plain senten"] is None and got["Closing plain senten"] is None
    assert got["According to Moyo (2"] == "quotation" and got["Dube notes that ‘sin"] == "quotation"
    assert any(v == "quotation" for k, v in got.items() if k.startswith("This is an indented"))
    assert all(v == "bibliography" for k, v in got.items() if k.startswith("["))

def test_sentence_split_curly_quotes():
    from app.extract import split_sentences
    assert len(split_sentences("He said “it works.” Then he left.")) == 2

def test_quote_pairing_v32_no_desync():
    # v3.1 paired a short quote's closing mark with the next quote far away and excluded plain sentences in between
    from app.similarity import exclusion_flags_v31
    from app.extract import split_sentences
    t = ('He was married to "Sallie" with whom he had sixteen children. He was an acknowledged leader of the community for many years. '
         'A double-7" gatefold edition was released in the same year. The song sold an estimated thirteen million downloads by then. '
         'She said "this is a properly quoted sentence of reasonable length" and left.')
    s = split_sentences(t); f = exclusion_flags_v31(t, s)
    assert f[:4] == [None] * 4 and f[4] == "quotation"

def test_entity_tokens():
    from app.similarity import entity_tokens
    assert {"alessandria", "italy", "1872"} <= set(entity_tokens("On August 5, 1872, Alessandria, Italy, was the site"))
