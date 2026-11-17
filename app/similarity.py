"""Similarity v2/v3 (v3 = params version>=3, see _paraphrase_v3).
- Exact / near-verbatim: winnowing fingerprints (Schleimer et al. 2003) over word k-grams, with a
  document-frequency whitelist (very common k-grams are ignored, like MOSS) and source-span alignment.
- Paraphrase: sentence embeddings (model chosen via the registry manifest) in a FAISS index.
- Quote and bibliography exclusion; optional web-source providers (see app/websources.py).
The library (SQLite + FAISS file) grows with every submission. Parameters come from manifest["similarity"]."""
import hashlib, re, sqlite3, threading, json, time, os
import numpy as np
from .config import DB_PATH, LIGHT_MODE, DATA, MANIFEST
_lock = threading.RLock()
def sim_params():
    try: s = json.loads(MANIFEST.read_text()).get("similarity", {})
    except Exception: s = {}
    d = {"id": "minilm-w5@1.0.0", "embed_model": "sentence-transformers/all-MiniLM-L6-v2", "dim": 384, "query_prefix": "",
         "k": 5, "w": 4, "para_threshold": 0.82, "exact_coverage": 0.3, "df_max_frac": 0.01, "df_min": 20, "min_words": 6}
    d.update(s); return d
P = sim_params()

def _words(text):
    return [(m.group().lower(), m.start(), m.end()) for m in re.finditer(r"[A-Za-z0-9']+", text)]
def fingerprints(text, k=None, w=None):
    """Winnowing: list of (hash, char_start, char_end)."""
    k = k or P["k"]; w = w or P["w"]; ws = _words(text)
    if len(ws) < k: return []
    hs = []
    for i in range(len(ws) - k + 1):
        h = int.from_bytes(hashlib.blake2b(" ".join(x[0] for x in ws[i:i+k]).encode(), digest_size=8).digest(), "big") >> 1
        hs.append((h, ws[i][1], ws[i+k-1][2]))
    if len(hs) <= w: return [min(hs)]
    sel, last = [], None
    for i in range(len(hs) - w + 1):
        win = hs[i:i+w]; j = min(range(w), key=lambda t: (win[t][0], -t)); idx = i + j
        if idx != last: sel.append(hs[idx]); last = idx
    return sel

class _HFEmbedder:
    """Pure transformers + torch sentence embedder (mean pooling + L2 norm).
    Avoids importing sentence_transformers/datasets/pandas (which triggers Windows App Control / Cython DLL errors)."""
    def __init__(self, name):
        import torch
        from transformers import AutoTokenizer, AutoModel
        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            self.tok = AutoTokenizer.from_pretrained(name, local_files_only=True)
            self.model = AutoModel.from_pretrained(name, local_files_only=True).to(self.device).eval()
        except Exception:
            self.tok = AutoTokenizer.from_pretrained(name)
            self.model = AutoModel.from_pretrained(name).to(self.device).eval()
    def encode(self, texts, batch_size=64):
        torch = self.torch
        out = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                x = self.tok(batch, padding=True, truncation=True, max_length=512, return_tensors="pt").to(self.device)
                h = self.model(**x).last_hidden_state
                m = x["attention_mask"].unsqueeze(-1).float()
                pooled = (h * m).sum(1) / m.sum(1).clamp_min(1e-9)
                normed = torch.nn.functional.normalize(pooled, p=2, dim=1)
                out.append(normed.cpu().numpy().astype(np.float32))
        return np.vstack(out) if out else np.zeros((0, P["dim"]), dtype=np.float32)

class Embedder:
    _m = {}
    @classmethod
    def get(cls, name=None):
        if LIGHT_MODE: return None
        name = name or P["embed_model"]
        with _lock:
            if name not in cls._m:
                cls._m[name] = _HFEmbedder(name)
            return cls._m[name]
    @classmethod
    def encode(cls, sents, name=None, prefix=None):
        m = cls.get(name); dim = P["dim"]
        if m is None or not sents: return np.zeros((len(sents), dim), dtype=np.float32)
        pre = P["query_prefix"] if prefix is None else prefix
        return m.encode([pre + s for s in sents], batch_size=64)

# ---------- exclusions ----------
_BIB = re.compile(r"^\s*(references|bibliography|works cited|reference list|sources)\s*:?\s*$", re.I)
_QUOTE = re.compile(r"[\"“”]([^\"“”]{20,})[\"“”]")
def exclusion_flags(text, sentences, exclude_quotes=True, exclude_bibliography=True):
    flags = [None] * len(sentences)
    if exclude_bibliography:
        bib = None
        for m in re.finditer(r"[^\n]+", text):
            if _BIB.match(m.group()) and m.start() > len(text) * 0.3: bib = m.start()
        if bib is not None:
            for i, (s, e, t) in enumerate(sentences):
                if s >= bib: flags[i] = "bibliography"
    if exclude_quotes:
        for i, (s, e, t) in enumerate(sentences):
            if flags[i]: continue
            q = sum(len(m.group(1)) for m in _QUOTE.finditer(t))
            if q >= 0.6 * len(t): flags[i] = "quotation"
    return flags

# ---- v3.1 exclusion (measured in eval/v3/excl_eval.py) ----
_BIB31 = re.compile(r"^\s*(references?( list| cited)?|bibliography|works cited|literature cited|sources( cited)?|cited works|reference list)\s*:?\s*$", re.I)
_REFLINE = re.compile(r"^\s*(\[\d{1,3}\]|\d{1,3}[.)])\s+.*\b(1[89]|20)\d\d\b|^\s*[A-Z][\w'’\-]+,\s+(?:[A-Z]\.\s?)+.*\((1[89]|20)\d\d[a-z]?\)")
# Quotes are paired left-to-right (no minimum inside the regex, so a short quote such as "Sallie" consumes its own
# closing mark instead of pairing with the next quote far away); only quotes of >= 20 chars are excluded (v3.2 fix).
_Q31 = [re.compile(r'(?:^|(?<=[\s(\[\u2014-]))"(?=\S)([^"\n]*?)(?<=\S)"(?=[\s.,;:!?)\]\u2014-]|$)', re.M), re.compile(r"“([^“”\n]*)”"), re.compile(r"«([^«»\n]*)»"), re.compile(r"‘([^‘’\n]*)’(?!\w)"),
        re.compile(r"(?<![\w])'([^'\n]*?)'(?![\w])")]
_BLOCK = re.compile(r"^(?:[ \t]{4,}|\t|>\s?)(\S.{30,})$", re.M)
_CITE = re.compile(r"\([^()]{0,60}\b(1[89]|20)\d\d[^()]{0,30}\)|\(p+\.\s*\d+[^()]*\)")
def exclusion_flags_v31(text, sentences, exclude_quotes=True, exclude_bibliography=True):
    flags = [None] * len(sentences)
    if exclude_bibliography:
        lines = [(m.start(), m.group()) for m in re.finditer(r"[^\n]+", text)]
        bib = None
        for st, ln in lines:
            if _BIB31.match(ln) and st > len(text) * 0.3: bib = st
        if bib is None:   # header-less reference list: trailing run of >=3 reference-looking lines
            run = []
            for st, ln in reversed(lines):
                if _REFLINE.match(ln): run.append(st)
                elif run: break
            if len(run) >= 3 and run[-1] > len(text) * 0.3: bib = run[-1]
        if bib is not None:
            for i, (s, e, t) in enumerate(sentences):
                if s >= bib: flags[i] = "bibliography"
    if exclude_quotes:
        mask = bytearray(len(text))
        for rx in _Q31:
            for m in rx.finditer(text):
                if 20 <= len(m.group(1)) <= 600: mask[m.start(1):m.end(1)] = b"\x01" * (m.end(1) - m.start(1))
        for m in _BLOCK.finditer(text):
            if len(m.group(1).split()) >= 8: mask[m.start(1):m.end(1)] = b"\x01" * (m.end(1) - m.start(1))
        for i, (s, e, t) in enumerate(sentences):
            if flags[i] or e <= s: continue
            body = len(_CITE.sub("", t).strip()) or 1
            if sum(mask[s:e]) >= 0.5 * body: flags[i] = "quotation"
    return flags

class Corpus:
    def __init__(self, path=DB_PATH, params=None):
        self.p = dict(params or P); self.path = str(path)
        self.db = sqlite3.connect(self.path, check_same_thread=False)
        self.db.executescript("""
        PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL;
        CREATE TABLE IF NOT EXISTS docs(id INTEGER PRIMARY KEY, title TEXT, source TEXT, added REAL, text TEXT);
        CREATE TABLE IF NOT EXISTS fp(h INTEGER, doc INTEGER, s INTEGER, e INTEGER);
        CREATE INDEX IF NOT EXISTS fp_h ON fp(h);
        CREATE TABLE IF NOT EXISTS fpdf(h INTEGER PRIMARY KEY, df INTEGER);
        CREATE TABLE IF NOT EXISTS sents(id INTEGER PRIMARY KEY, doc INTEGER, s INTEGER, e INTEGER, txt TEXT);
        CREATE TABLE IF NOT EXISTS meta(k TEXT PRIMARY KEY, v TEXT);""")
        self.index_path = self.path + ".faiss"; self._index = None
    # ---- vector index ----
    def index(self):
        import faiss
        if self._index is None:
            meta = dict(self.db.execute("SELECT k,v FROM meta").fetchall())
            if os.path.exists(self.index_path) and meta.get("embed_model") == self.p["embed_model"]:
                self._index = faiss.read_index(self.index_path)
            else:
                self._index = faiss.IndexIDMap2(faiss.IndexFlatIP(self.p["dim"]))
                if self.count() and not LIGHT_MODE: self.reindex()
        return self._index
    def reindex(self, batch=4096):
        """Re-embed the whole library (used when the registry activates a new embedding model)."""
        import faiss
        idx = faiss.IndexIDMap2(faiss.IndexFlatIP(self.p["dim"]))
        cur = self.db.execute("SELECT id, txt FROM sents")
        while True:
            rows = cur.fetchmany(batch)
            if not rows: break
            idx.add_with_ids(Embedder.encode([r[1] for r in rows], self.p["embed_model"], self.p["query_prefix"]), np.array([r[0] for r in rows], dtype=np.int64))
        self._index = idx; self.save_index()
    def save_index(self):
        import faiss
        if self._index is not None:
            faiss.write_index(self._index, self.index_path)
            self.db.execute("INSERT OR REPLACE INTO meta VALUES('embed_model',?)", (self.p["embed_model"],)); self.db.commit()
    def count(self): return self.db.execute("SELECT COUNT(*) FROM docs").fetchone()[0]
    def add(self, title, text, sentences, source="submission", embeddings=None, save=True):
        with _lock:
            d = self.db.execute("INSERT INTO docs(title,source,added,text) VALUES(?,?,?,?)", (title, source, time.time(), text)).lastrowid
            fps = fingerprints(text, self.p["k"], self.p["w"])
            self.db.executemany("INSERT INTO fp VALUES(?,?,?,?)", [(h, d, s, e) for h, s, e in fps])
            self.db.executemany("INSERT INTO fpdf VALUES(?,1) ON CONFLICT(h) DO UPDATE SET df=df+1", [(h,) for h in {h for h, _, _ in fps}])
            first = self.db.execute("SELECT COALESCE(MAX(id),0) FROM sents").fetchone()[0] + 1
            self.db.executemany("INSERT INTO sents(id,doc,s,e,txt) VALUES(?,?,?,?,?)", [(first + i, d, s, e, t) for i, (s, e, t) in enumerate(sentences)])
            self.db.commit()
            if not LIGHT_MODE and sentences:
                if embeddings is None: embeddings = Embedder.encode([x[2] for x in sentences], self.p["embed_model"], self.p["query_prefix"])
                self.index().add_with_ids(np.ascontiguousarray(embeddings, dtype=np.float32), np.arange(first, first + len(sentences), dtype=np.int64))
                if save: self.save_index()
            return d
    def titles(self):
        return {r[0]: (r[1], r[2]) for r in self.db.execute("SELECT id,title,source FROM docs")}
    def delete(self, doc_id: int):
        """Remove a document and its sentences/fingerprints/vectors from the library."""
        with _lock:
            row = self.db.execute("SELECT id FROM docs WHERE id=?", (doc_id,)).fetchone()
            if not row: return False
            sids = [r[0] for r in self.db.execute("SELECT id FROM sents WHERE doc=?", (doc_id,)).fetchall()]
            hs = {r[0] for r in self.db.execute("SELECT DISTINCT h FROM fp WHERE doc=?", (doc_id,)).fetchall()}
            for h in hs:
                self.db.execute("UPDATE fpdf SET df = MAX(0, df - 1) WHERE h=?", (h,))
            self.db.execute("DELETE FROM fpdf WHERE df <= 0")
            self.db.execute("DELETE FROM fp WHERE doc=?", (doc_id,))
            self.db.execute("DELETE FROM sents WHERE doc=?", (doc_id,))
            self.db.execute("DELETE FROM docs WHERE id=?", (doc_id,))
            self.db.commit()
            if sids and not LIGHT_MODE:
                try:
                    self.index().remove_ids(np.array(sids, dtype=np.int64))
                    self.save_index()
                except Exception:
                    self._index = None
            return True
    def _doc_text(self, d):
        r = self.db.execute("SELECT text FROM docs WHERE id=?", (d,)).fetchone(); return r[0] if r else ""
    def check(self, text, sentences, embeddings=None, exclude_doc=None, exclude_quotes=True, exclude_bibliography=True, paraphrase=True):
        p = self.p; n_docs = max(1, self.count())
        df_cap = max(p["df_min"], int(p["df_max_frac"] * n_docs))
        titles = {}
        def title(d):
            if d not in titles:
                r = self.db.execute("SELECT title, source FROM docs WHERE id=?", (d,)).fetchone(); titles[d] = r or ("?", "?")
            return titles[d]
        xf = exclusion_flags_v31 if p.get("exclusion") == "v31" else exclusion_flags
        res = [{"exact": None, "paraphrase": None, "excluded": f} for f in xf(text, sentences, exclude_quotes, exclude_bibliography)]
        fps = fingerprints(text, p["k"], p["w"])
        uniq = list({h for h, _, _ in fps})
        common = set()
        for i in range(0, len(uniq), 900):
            q = uniq[i:i+900]
            common |= {h for h, df in self.db.execute(f"SELECT h, df FROM fpdf WHERE h IN ({','.join('?'*len(q))}) AND df > ?", (*q, df_cap))}
        hitmap = {}
        for i in range(0, len(uniq), 900):
            q = [h for h in uniq[i:i+900] if h not in common]
            if not q: continue
            for h, d, s, e in self.db.execute(f"SELECT h, doc, s, e FROM fp WHERE h IN ({','.join('?'*len(q))})", q):
                if d != exclude_doc: hitmap.setdefault(h, []).append((d, s, e))
        # sentence -> fingerprints inside it
        starts = np.array([s for s, _, _ in sentences]); ends = np.array([e for _, e, _ in sentences])
        per = [[] for _ in sentences]
        for h, s, e in fps:
            i = int(np.searchsorted(starts, s, side="right") - 1)
            if 0 <= i < len(sentences) and s < ends[i]: per[i].append(h)
        for i, hs in enumerate(per):
            if not hs: continue
            cnt = {}
            spans = {}
            for h in hs:
                for d, s, e in hitmap.get(h, []):
                    cnt[d] = cnt.get(d, 0) + 1; spans.setdefault(d, []).append((s, e))
            if not cnt: continue
            best = max(cnt, key=cnt.get); cov = min(1.0, cnt[best] / len(hs))
            if cov >= p["exact_coverage"] and (cnt[best] >= 2 or len(hs) == 1):
                sp = spans[best]; s0, e0 = min(x[0] for x in sp), max(x[1] for x in sp)
                src = self._doc_text(best)
                res[i]["exact"] = {"doc": best, "title": title(best)[0], "coverage": round(cov, 2), "source_span": [s0, e0],
                                   "matched_text": src[max(0, s0):min(len(src), e0, s0 + 400)]}
        if paraphrase and not LIGHT_MODE and self.count():
            idx = self.index()
            if idx.ntotal:
                E = embeddings if embeddings is not None else Embedder.encode([x[2] for x in sentences], p["embed_model"], p["query_prefix"])
                if p.get("version", 2) >= 3: self._paraphrase_v3(res, sentences, E, idx, exclude_doc, title)
                else: self._paraphrase_v2(res, sentences, E, idx, exclude_doc, title)
        return res
    def _sent_info(self, ids):
        info = {}; ids = list(ids)
        for i0 in range(0, len(ids), 900):
            q = ids[i0:i0+900]
            for sid, d, t in self.db.execute(f"SELECT id, doc, txt FROM sents WHERE id IN ({','.join('?'*len(q))})", q): info[sid] = (d, t)
        return info
    def _paraphrase_v2(self, res, sentences, E, idx, exclude_doc, title):
        p = self.p
        kk = 4 if exclude_doc is not None else 1
        D, I = idx.search(np.ascontiguousarray(E, dtype=np.float32), kk)
        info = self._sent_info({int(j) for row in I for j in row if j >= 0})
        for i in range(len(sentences)):
            if res[i]["exact"] or len(sentences[i][2].split()) < p["min_words"]: continue
            for sc, sid in zip(D[i], I[i]):
                if sid < 0 or int(sid) not in info: continue
                d, t = info[int(sid)]
                if d == exclude_doc: continue
                if sc >= p["para_threshold"]:
                    res[i]["paraphrase"] = {"doc": d, "title": title(d)[0], "score": round(float(sc), 3), "matched_text": t[:400]}
                break
    def _paraphrase_v3(self, res, sentences, E, idx, exclude_doc, title):
        """v3: (1) high-confidence sentence hits (cos >= para_threshold); (2) source-document candidate selection:
        a library doc that already explains >= 1 exact/high hit, or >= seed_min sentences at cos >= seed_threshold,
        becomes a candidate; (3) dense alignment of every unmatched sentence (and adjacent-sentence pairs, for
        condensed/merged sentences) against ALL sentences of the candidate docs with a lower threshold
        (align_threshold). This is the PAN seeding -> extension idea; it recovers heavier paraphrase, condensation
        and mosaic plagiarism without lowering the global threshold."""
        p = self.p; K = int(p.get("topk", 5)); n = len(sentences)
        Ef = np.ascontiguousarray(E, dtype=np.float32)
        D, I = idx.search(Ef, K)
        info = self._sent_info({int(j) for row in I for j in row if j >= 0})
        ok = [len(sentences[i][2].split()) >= p["min_words"] and not res[i].get("excluded") for i in range(n)]
        seeds = {}
        for i in range(n):
            if res[i]["exact"]: seeds[res[i]["exact"]["doc"]] = seeds.get(res[i]["exact"]["doc"], 0) + 2; continue
            if not ok[i]: continue
            for sc, sid in zip(D[i], I[i]):
                if sid < 0 or int(sid) not in info: continue
                d, t = info[int(sid)]
                if d == exclude_doc: continue
                if sc >= p["para_threshold"] and res[i]["paraphrase"] is None:
                    res[i]["paraphrase"] = {"doc": d, "title": title(d)[0], "score": round(float(sc), 3), "matched_text": t[:400], "stage": "direct"}
                    seeds[d] = seeds.get(d, 0) + 2
                elif sc >= p.get("seed_threshold", 0.87):
                    seeds[d] = seeds.get(d, 0) + 1
        cands = [d for d, c in sorted(seeds.items(), key=lambda x: -x[1]) if c >= p.get("seed_min", 2)][: int(p.get("max_cand_docs", 8))]
        top1 = []
        for i in range(n):
            t1 = None
            for sc, sid in zip(D[i], I[i]):
                if sid >= 0 and int(sid) in info and info[int(sid)][0] != exclude_doc: t1 = (float(sc), info[int(sid)][0], info[int(sid)][1]); break
            top1.append(t1)
        if cands: self._align_v3(res, sentences, Ef, idx, cands, title, ok)
        if p.get("condense"): self._condense_v32(res, sentences, Ef, idx, top1, title, ok)
        return res
    def _cand_rows(self, idx, docs, pairs=True):
        p = self.p; rows = []
        for d in docs: rows += [(sid, d, t) for sid, t in self.db.execute("SELECT id, txt FROM sents WHERE doc=? ORDER BY id", (d,))]
        if not rows: return rows, None
        S = np.vstack([idx.reconstruct(int(r[0])) for r in rows]).astype(np.float32)
        if pairs:
            pr = [(rows[k][0], rows[k][1], rows[k][2] + " " + rows[k + 1][2]) for k in range(len(rows) - 1) if rows[k][1] == rows[k + 1][1]]
            if pr: S = np.vstack([S, Embedder.encode([x[2] for x in pr], p["embed_model"], p["query_prefix"])]); rows = rows + pr
        return rows, S
    def _condense_v32(self, res, sentences, Ef, idx, top1, title, ok):
        """v3.2 condense/summary stage, only for still-unmatched sentences, gated on strong evidence:
        top-1 library doc d with cos >= c_top, best single/adjacent-pair alignment to d >= c_sim, AND either
        (a) >= c_nent entity/number tokens of the sentence of which >= c_ent are present in d (facts survive summarising), or
        (b) a neighbouring submission sentence also has d as top-1 with cos >= r_top (passage-level, many-to-one)."""
        c = self.p["condense"]; n = len(sentences)
        todo = [i for i in range(n) if ok[i] and not res[i]["exact"] and not res[i]["paraphrase"] and top1[i]
                and top1[i][0] >= c["c_top"] and len(sentences[i][2].split()) >= c.get("c_nw", 6)]
        if not todo: return
        docs = sorted({top1[i][1] for i in todo}); rows, S = self._cand_rows(idx, docs)
        if not rows: return
        owner = np.array([r[1] for r in rows]); dtok = {}
        for i in todo:
            d = top1[i][1]; m = owner == d
            if not m.any(): continue
            sims = Ef[i] @ S[m].T; j = int(np.argmax(sims)); sc = float(sims[j])
            if sc < c["c_sim"]: continue
            if d not in dtok: dtok[d] = set(entity_tokens(self._doc_text(d), all_tokens=True))
            e = set(entity_tokens(sentences[i][2]))
            ent_ok = len(e) >= c["c_nent"] and len(e & dtok[d]) / max(1, len(e)) >= c["c_ent"]
            run = any(0 <= j2 < n and top1[j2] and top1[j2][1] == d and top1[j2][0] >= c["r_top"] for j2 in (i - 1, i + 1))
            if ent_ok or run:
                t = [r for r, mm in zip(rows, m) if mm][j][2]
                res[i]["paraphrase"] = {"doc": d, "title": title(d)[0], "score": round(sc, 3), "matched_text": t[:400], "stage": "condense"}
    def _align_v3(self, res, sentences, Ef, idx, cands, title, ok):
        p = self.p; n = len(sentences)
        rows, S = self._cand_rows(idx, cands, p.get("align_pairs", True))
        if not rows: return
        sim = Ef @ S.T
        thr = p.get("align_threshold", 0.86)
        for i in range(n):
            if res[i]["exact"] or res[i]["paraphrase"] or not ok[i]: continue
            j = int(np.argmax(sim[i])); sc = float(sim[i, j])
            if sc >= thr:
                sid, d, t = rows[j]
                res[i]["paraphrase"] = {"doc": d, "title": title(d)[0], "score": round(sc, 3), "matched_text": t[:400], "stage": "aligned"}
        return res

def web_check(sentences, text=None, providers=None, embeddings=None):
    from .websources import check_web
    return check_web(sentences, providers=providers, embeddings=embeddings)

_ENT = re.compile(r"\b(?:[A-Z][\w'\-]*[a-z0-9][\w'\-]*|[A-Z]{2,}|\d[\d.,]*\d|\d)\b")
_STOPCAP = {"The", "A", "An", "In", "On", "At", "This", "That", "These", "Those", "It", "He", "She", "They", "We", "His", "Her",
            "Its", "Their", "As", "After", "Before", "During", "While", "However", "Although", "When", "Also", "And", "But", "Of", "For", "By", "With"}
def entity_tokens(text, all_tokens=False):
    """Capitalised (non-stopword) words and numbers: names, places, dates, figures. Used as the lexical gate of the
    v3.2 condense stage: condensed/summarised plagiarism keeps the facts even when wording and structure change."""
    return [m.group().lower().strip(".,") for m in _ENT.finditer(text) if m.group() not in _STOPCAP]
