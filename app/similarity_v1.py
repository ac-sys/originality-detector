"""Similarity: winnowing fingerprints (Schleimer et al. 2003) for verbatim/near-verbatim copying,
plus sentence embeddings (MiniLM) for paraphrase. Corpus lives in SQLite and grows with submissions."""
import hashlib, re, sqlite3, threading, json, time
import numpy as np
from .config import DB_PATH, LIGHT_MODE, WEB_SEARCH_API_KEY
K, W = 5, 4           # k-gram of 5 words, window 4
PARA_THRESHOLD = 0.82 # cosine for paraphrase match
_lock = threading.Lock()

def _words(text):
    return [(m.group().lower(), m.start(), m.end()) for m in re.finditer(r"[A-Za-z0-9']+", text)]
def fingerprints(text):
    """Returns list of (hash, char_start, char_end) selected by winnowing."""
    w = _words(text)
    if len(w) < K: return []
    hs = []
    for i in range(len(w)-K+1):
        h = int.from_bytes(hashlib.blake2b(" ".join(x[0] for x in w[i:i+K]).encode(), digest_size=8).digest(), "big") >> 1
        hs.append((h, w[i][1], w[i+K-1][2]))
    if len(hs) <= W: return [min(hs)]
    sel, last = [], None
    for i in range(len(hs)-W+1):
        win = hs[i:i+W]; j = min(range(W), key=lambda t: (win[t][0], -t)); idx = i+j
        if idx != last: sel.append(hs[idx]); last = idx
    return sel

class Embedder:
    _m = None
    @classmethod
    def get(cls):
        if LIGHT_MODE: return None
        if cls._m is None:
            from sentence_transformers import SentenceTransformer
            cls._m = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
        return cls._m
    @classmethod
    def encode(cls, sents):
        m = cls.get()
        if m is None or not sents: return np.zeros((len(sents), 384), dtype=np.float32)
        return m.encode(sents, batch_size=64, normalize_embeddings=True, show_progress_bar=False).astype(np.float32)

class Corpus:
    def __init__(self, path=DB_PATH):
        self.db = sqlite3.connect(str(path), check_same_thread=False)
        self.db.executescript("""
        CREATE TABLE IF NOT EXISTS docs(id INTEGER PRIMARY KEY, title TEXT, source TEXT, added REAL, text TEXT);
        CREATE TABLE IF NOT EXISTS fp(h INTEGER, doc INTEGER, s INTEGER, e INTEGER);
        CREATE INDEX IF NOT EXISTS fp_h ON fp(h);
        CREATE TABLE IF NOT EXISTS sents(doc INTEGER, s INTEGER, e INTEGER, txt TEXT, emb BLOB);""")
        self._emb_cache = None
    def count(self):
        return self.db.execute("SELECT COUNT(*) FROM docs").fetchone()[0]
    def add(self, title, text, sentences, source="submission", embeddings=None):
        with _lock:
            cur = self.db.execute("INSERT INTO docs(title,source,added,text) VALUES(?,?,?,?)", (title, source, time.time(), text))
            d = cur.lastrowid
            self.db.executemany("INSERT INTO fp VALUES(?,?,?,?)", [(h, d, s, e) for h, s, e in fingerprints(text)])
            if embeddings is None: embeddings = Embedder.encode([x[2] for x in sentences])
            self.db.executemany("INSERT INTO sents VALUES(?,?,?,?,?)",
                [(d, s, e, t, embeddings[i].tobytes()) for i, (s, e, t) in enumerate(sentences)])
            self.db.commit(); self._emb_cache = None
            return d
    def _emb_matrix(self):
        if self._emb_cache is None:
            rows = self.db.execute("SELECT doc,txt,emb FROM sents").fetchall()
            M = np.frombuffer(b"".join(r[2] for r in rows), dtype=np.float32).reshape(-1, 384) if rows else np.zeros((0, 384), np.float32)
            self._emb_cache = (M, [(r[0], r[1]) for r in rows])
        return self._emb_cache
    def titles(self):
        return {r[0]: (r[1], r[2]) for r in self.db.execute("SELECT id,title,source FROM docs")}
    def check(self, text, sentences, embeddings=None, exclude_doc=None):
        """Per-sentence evidence: exact (winnowing) and paraphrase (embedding) matches."""
        titles = self.titles()
        res = [{"exact": None, "paraphrase": None} for _ in sentences]
        # exact
        fps = fingerprints(text)
        hits = {}
        for h, s, e in fps:
            for d, ds, de in self.db.execute("SELECT doc,s,e FROM fp WHERE h=?", (h,)):
                if d == exclude_doc: continue
                hits.setdefault((s, e), []).append(d)
        for i, (ss, se, _) in enumerate(sentences):
            docs = [d for (s, e), ds in hits.items() if s < se and e > ss for d in ds]
            n_fp = sum(1 for h, s, e in fps if s >= ss and e <= se) or 1
            if docs:
                best = max(set(docs), key=docs.count)
                cov = min(1.0, docs.count(best) / n_fp)
                if cov >= 0.3:
                    res[i]["exact"] = {"doc": best, "title": titles.get(best, ("?",))[0], "coverage": round(cov, 2)}
        # paraphrase
        M, meta = self._emb_matrix()
        if len(M) and not LIGHT_MODE:
            E = embeddings if embeddings is not None else Embedder.encode([x[2] for x in sentences])
            sims = E @ M.T
            for i in range(len(sentences)):
                row = sims[i]
                if exclude_doc is not None:
                    row = np.where(np.array([m[0] for m in meta]) == exclude_doc, -1, row)
                j = int(np.argmax(row)); sc = float(row[j])
                if sc >= PARA_THRESHOLD and len(sentences[i][2].split()) >= 6:
                    res[i]["paraphrase"] = {"doc": meta[j][0], "title": titles.get(meta[j][0], ("?",))[0],
                                            "score": round(sc, 3), "matched_text": meta[j][1][:300]}
        return res

def web_check(sentences):
    """Optional web-source check. Stub: enabled only when WEB_SEARCH_API_KEY is set.
    Plug in a search API (e.g. Bing/Brave/Google CSE) here: query distinctive sentences, fetch
    top results, fingerprint them and add to the corpus with source='web'."""
    if not WEB_SEARCH_API_KEY:
        return {"enabled": False, "note": "Web check disabled: set WEB_SEARCH_API_KEY to enable."}
    return {"enabled": True, "note": "Web provider not yet implemented in MVP.", "matches": []}
