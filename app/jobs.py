"""Background job queue: large documents are split into blocks and processed by a worker pool,
with progress reported per block. Reports are persisted as JSON."""
import json, threading, time, uuid, traceback
from concurrent.futures import ThreadPoolExecutor
from .config import REPORTS
from .extract import extract_text, normalise, split_sentences
from .similarity import Embedder, web_check
BLOCK = 120  # sentences per block
class JobManager:
    def __init__(self, corpus, ensemble, workers=2):
        self.corpus, self.ens = corpus, ensemble
        self.pool = ThreadPoolExecutor(workers); self.jobs = {}; self.lock = threading.Lock()
    def submit(self, filename, data, title=None, add_to_library=True, web_sources=None):
        jid = uuid.uuid4().hex[:12]
        self.jobs[jid] = {"id": jid, "status": "queued", "progress": 0.0, "filename": filename, "created": time.time()}
        self.pool.submit(self._run, jid, filename, data, title or filename, add_to_library, web_sources)
        return jid
    def _set(self, jid, **kw):
        with self.lock: self.jobs[jid].update(kw)
    def _run(self, jid, filename, data, title, add, web_sources=None):
        try:
            self._set(jid, status="extracting")
            text = normalise(extract_text(filename, data))
            sents = split_sentences(text)
            if not sents: raise ValueError("No text could be extracted from the file")
            self._set(jid, status="processing", sentences=len(sents))
            n_blocks = (len(sents) + BLOCK - 1) // BLOCK
            lp, cl, embs = [], [], []
            import numpy as np
            cal = None
            for b in range(n_blocks):
                blk = sents[b*BLOCK:(b+1)*BLOCK]
                off = blk[0][0]; sub = text[off:blk[-1][1]]
                rel = [(s-off, e-off, t) for s, e, t in blk]
                a, c, cal = self.ens.features(sub, rel)
                lp += a; cl += c; embs.append(Embedder.encode([t for _, _, t in blk]))
                self._set(jid, progress=round(0.9*(b+1)/n_blocks, 3))
            E = np.vstack(embs)
            ai = self.ens.summarise(lp, cl, cal, sents)
            sim = self.corpus.check(text, sents, embeddings=E)
            words = [len(t.split()) for _, _, t in sents]; tot = sum(words) or 1
            for r in sim:
                if r.get("excluded"): r["exact"] = r["paraphrase"] = None
            exact_w = sum(w for w, r in zip(words, sim) if r["exact"])
            para_w = sum(w for w, r in zip(words, sim) if r["paraphrase"] and not r["exact"])
            excl_w = sum(w for w, r in zip(words, sim) if r.get("excluded"))
            sources = {}
            for w, r in zip(words, sim):
                m = r["exact"] or r["paraphrase"]
                if m:
                    src = sources.setdefault(m["doc"], {"doc": m["doc"], "title": m["title"], "words": 0})
                    src["words"] += w
            srcs = sorted(sources.values(), key=lambda x: -x["words"])
            for s in srcs: s["percent"] = round(100*s["words"]/tot, 1)
            report = {"id": jid, "title": title, "filename": filename, "created": time.time(),
                      "words": tot, "similarity": {"percent": round(100*(exact_w+para_w)/tot, 1),
                      "exact_percent": round(100*exact_w/tot, 1), "paraphrase_percent": round(100*para_w/tot, 1),
                      "excluded_percent": round(100*excl_w/tot, 1), "sources": srcs, "web": web_check(sents, providers=web_sources)},
                      "ai": {k: v for k, v in ai.items() if k not in ("sentence_probs", "sentence_flagged")},
                      "sentences": [{"text": t, "ai_prob": ai["sentence_probs"][i],
                                     "ai_flag": bool(ai["sentence_flagged"][i]) if "sentence_flagged" in ai else ai["sentence_probs"][i] >= 0.5,
                                     "excluded": sim[i].get("excluded"), "exact": sim[i]["exact"],
                                     "paraphrase": sim[i]["paraphrase"]} for i, (_, _, t) in enumerate(sents)]}
            if add:
                report["library_doc_id"] = self.corpus.add(title, text, sents, embeddings=E)
            (REPORTS / f"{jid}.json").write_text(json.dumps(report), encoding="utf-8")
            self._set(jid, status="done", progress=1.0)
        except Exception as e:
            traceback.print_exc(); self._set(jid, status="error", error=str(e))
    def report(self, jid):
        p = REPORTS / f"{jid}.json"
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
    def list_reports(self, limit=50):
        out = []
        files = sorted(REPORTS.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:limit]
        for p in files:
            try:
                r = json.loads(p.read_text(encoding="utf-8"))
                out.append({
                    "id": r.get("id", p.stem),
                    "title": r.get("title", p.stem),
                    "filename": r.get("filename", ""),
                    "created": r.get("created", p.stat().st_mtime),
                    "words": r.get("words", 0),
                    "similarity_percent": r.get("similarity", {}).get("percent", 0),
                    "ai_probability": r.get("ai", {}).get("probability"),
                    "ai_verdict": r.get("ai", {}).get("verdict", "no_strong_signal"),
                })
            except Exception:
                continue
        return out
