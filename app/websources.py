"""Web-source check (opt-in, because it sends short snippets of the submission to third parties).
Providers (enable with WEB_SOURCES env, comma-separated):
  wikipedia - MediaWiki search + extracts API (free, no key)        https://www.mediawiki.org/wiki/API:Search
  openalex  - OpenAlex works search, scholarly abstracts (free, no key) https://docs.openalex.org/
  brave     - Brave Search API (needs BRAVE_API_KEY; free tier subject to Brave's terms)
Each provider returns candidate (url, title, text); candidates are compared with winnowing + embeddings."""
import os, re, json, time, urllib.parse, urllib.request
import numpy as np
UA = {"User-Agent": "OriginalityDetector/2.0 (university plagiarism checker; contact: admin@localhost)"}
def _get(url, headers=None, timeout=15):
    req = urllib.request.Request(url, headers={**UA, **(headers or {})})
    with urllib.request.urlopen(req, timeout=timeout) as r: return json.loads(r.read())
def wikipedia(query, n=3):
    q = urllib.parse.urlencode({"action": "query", "list": "search", "srsearch": query, "srlimit": n, "format": "json"})
    hits = _get("https://en.wikipedia.org/w/api.php?" + q)["query"]["search"]
    if not hits: return []
    titles = "|".join(h["title"] for h in hits)
    q = urllib.parse.urlencode({"action": "query", "prop": "extracts", "explaintext": 1, "titles": titles, "format": "json", "exlimit": n})
    pages = _get("https://en.wikipedia.org/w/api.php?" + q)["query"]["pages"].values()
    return [{"url": "https://en.wikipedia.org/wiki/" + urllib.parse.quote(p["title"].replace(" ", "_")), "title": p["title"], "text": p.get("extract", "")[:60000]} for p in pages]
def openalex(query, n=3):
    q = urllib.parse.urlencode({"search": query, "per-page": n})
    out = []
    for w in _get("https://api.openalex.org/works?" + q).get("results", []):
        inv = w.get("abstract_inverted_index") or {}
        pos = sorted((p, word) for word, ps in inv.items() for p in ps)
        txt = " ".join(word for _, word in pos)
        if txt: out.append({"url": w.get("doi") or w.get("id"), "title": w.get("title") or "", "text": txt})
    return out
def brave(query, n=3):
    key = os.environ.get("BRAVE_API_KEY")
    if not key: return []
    q = urllib.parse.urlencode({"q": query, "count": n})
    res = _get("https://api.search.brave.com/res/v1/web/search?" + q, {"X-Subscription-Token": key, "Accept": "application/json"})
    return [{"url": r["url"], "title": r.get("title", ""), "text": re.sub("<[^>]+>", "", r.get("description", "") + " " + " ".join(r.get("extra_snippets", [])))} for r in res.get("web", {}).get("results", [])]
PROVIDERS = {"wikipedia": wikipedia, "openalex": openalex, "brave": brave}

def pick_queries(sentences, n=6):
    cands = [(i, t) for i, (_, _, t) in enumerate(sentences) if 12 <= len(t.split()) <= 60]
    cands.sort(key=lambda x: -len(set(re.findall(r"[a-z]{6,}", x[1].lower()))))
    step = max(1, len(cands) // n)
    return [c for c in cands[::step]][:n]

def check_web(sentences, providers=None, max_queries=None):
    names = providers or [p for p in os.environ.get("WEB_SOURCES", "").split(",") if p.strip()]
    if not names:
        return {"enabled": False, "note": "Web check off. Set WEB_SOURCES=wikipedia,openalex (free) and/or brave (BRAVE_API_KEY) to enable."}
    from .similarity import fingerprints, Embedder
    from .extract import split_sentences
    qs = pick_queries(sentences, max_queries or int(os.environ.get("WEB_MAX_QUERIES", "6")))
    cands, errors = {}, []
    for i, t in qs:
        query = " ".join(t.split()[:14])
        for name in names:
            try:
                for c in PROVIDERS[name.strip()](query): cands.setdefault(c["url"], c | {"provider": name})
            except Exception as e: errors.append(f"{name}: {e}")
            time.sleep(0.2)
    matches = []
    if cands:
        sent_fp = [{h for h, _, _ in fingerprints(t)} for _, _, t in sentences]
        S = Embedder.encode([t for _, _, t in sentences])
        for url, c in cands.items():
            src_s = split_sentences(c["text"])[:2000]
            if not src_s: continue
            src_fp = {h for h, _, _ in fingerprints(c["text"])}
            E = Embedder.encode([t for _, _, t in src_s]); sims = S @ E.T if len(E) else np.zeros((len(sentences), 0))
            for i in range(len(sentences)):
                ov = len(sent_fp[i] & src_fp) / max(1, len(sent_fp[i]))
                j = int(np.argmax(sims[i])) if sims.shape[1] else -1
                sc = float(sims[i, j]) if j >= 0 else 0
                if ov >= 0.5 or sc >= 0.85:
                    matches.append({"sentence": i, "url": url, "title": c["title"], "provider": c["provider"],
                                    "type": "exact" if ov >= 0.5 else "paraphrase", "score": round(max(ov, sc), 3), "matched_text": src_s[j][2][:300] if j >= 0 else ""})
    best = {}
    for m in matches:
        if m["sentence"] not in best or m["score"] > best[m["sentence"]]["score"]: best[m["sentence"]] = m
    return {"enabled": True, "providers": names, "queries": len(qs), "candidates": len(cands), "matches": list(best.values()), "errors": errors[:5]}
