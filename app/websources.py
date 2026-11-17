"""Web-source similarity, citation verification, and online AI verification connectors (Stage 2 & Stage 3).
Free keyless providers:
  - wikipedia        : MediaWiki search + extracts API (free, no key)
  - openalex         : OpenAlex scholarly works & abstracts (free, no key)
  - crossref         : CrossRef 150M+ DOIs, titles & abstracts (free, no key)
  - semanticscholar  : Semantic Scholar 200M+ academic papers (free, no key)
  - arxiv            : arXiv open-access preprints (free, no key)
  - europepmc        : Europe PMC biomedical & science abstracts (free, no key)
Optional keyed providers:
  - brave            : Brave Web Search API (BRAVE_API_KEY)
Citation / Fake-Reference Verification:
  - verify_citations : Checks bibliography entries against CrossRef + OpenAlex to spot hallucinated AI references.
Online AI / LLM Second-Opinion Connectors:
  - check_online_ai  : Queries Sapling AI, GPTZero, or an OpenAI/Gemini/Ollama-compatible LLM endpoint."""
import os, re, json, time, urllib.parse, urllib.request, xml.etree.ElementTree as ET
import numpy as np

UA = {"User-Agent": "OriginalityDetector/3.2 (university originality & citation checker; mailto:admin@localhost)"}


def _get(url, headers=None, timeout=12):
    req = urllib.request.Request(url, headers={**UA, **(headers or {})})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def _get_raw(url, headers=None, timeout=12):
    req = urllib.request.Request(url, headers={**UA, **(headers or {})})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", errors="ignore")


def wikipedia(query, n=3):
    q = urllib.parse.urlencode({"action": "query", "list": "search", "srsearch": query, "srlimit": n, "format": "json"})
    hits = _get("https://en.wikipedia.org/w/api.php?" + q)["query"]["search"]
    if not hits:
        return []
    titles = "|".join(h["title"] for h in hits)
    q = urllib.parse.urlencode({"action": "query", "prop": "extracts", "explaintext": 1, "titles": titles, "format": "json", "exlimit": n})
    pages = _get("https://en.wikipedia.org/w/api.php?" + q)["query"]["pages"].values()
    return [
        {
            "url": "https://en.wikipedia.org/wiki/" + urllib.parse.quote(p["title"].replace(" ", "_")),
            "title": p["title"],
            "text": p.get("extract", "")[:60000],
        }
        for p in pages
        if p.get("extract")
    ]


def openalex(query, n=3):
    q = urllib.parse.urlencode({"search": query, "per-page": n})
    out = []
    for w in _get("https://api.openalex.org/works?" + q).get("results", []):
        inv = w.get("abstract_inverted_index") or {}
        pos = sorted((p, word) for word, ps in inv.items() for p in ps)
        txt = " ".join(word for _, word in pos)
        if txt:
            out.append({"url": w.get("doi") or w.get("id"), "title": w.get("title") or "", "text": txt})
    return out


def crossref(query, n=3):
    q = urllib.parse.urlencode({"query.bibliographic": query, "rows": n})
    out = []
    for it in _get("https://api.crossref.org/works?" + q).get("message", {}).get("items", []):
        title = " ".join(it.get("title") or [])
        abstr = re.sub(r"<[^>]+>", " ", it.get("abstract") or "").strip()
        txt = (title + ". " + abstr).strip()
        if len(txt.split()) >= 8:
            doi = it.get("DOI", "")
            url = f"https://doi.org/{doi}" if doi else (it.get("URL") or f"https://api.crossref.org/works/{doi}")
            out.append({"url": url, "title": title or doi, "text": txt})
    return out


def semanticscholar(query, n=3):
    q = urllib.parse.urlencode({"query": query, "limit": n, "fields": "title,abstract,url,externalIds"})
    out = []
    for p in _get("https://api.semanticscholar.org/graph/v1/paper/search?" + q).get("data", []):
        title = p.get("title") or ""
        abstr = p.get("abstract") or ""
        txt = f"{title}. {abstr}".strip()
        if len(txt.split()) >= 8:
            url = p.get("url") or f"https://www.semanticscholar.org/paper/{p.get('paperId', '')}"
            out.append({"url": url, "title": title, "text": txt})
    return out


def arxiv(query, n=3):
    q = urllib.parse.urlencode({"search_query": f"all:{query}", " start": 0, "max_results": n})
    xml_text = _get_raw("http://export.arxiv.org/api/query?" + q)
    root = ET.fromstring(xml_text)
    ns = {"a": "http://www.w3.org/2005/Atom"}
    out = []
    for entry in root.findall("a:entry", ns):
        title = (entry.findtext("a:title", "", ns) or "").strip().replace("\n", " ")
        summary = (entry.findtext("a:summary", "", ns) or "").strip().replace("\n", " ")
        url = (entry.findtext("a:id", "", ns) or "").strip()
        if summary:
            out.append({"url": url, "title": title, "text": f"{title}. {summary}"})
    return out


def europepmc(query, n=3):
    q = urllib.parse.urlencode({"query": query, "format": "json", "pageSize": n, "resultType": "core"})
    out = []
    for r in _get("https://www.ebi.ac.uk/europepmc/webservices/rest/search?" + q).get("resultList", {}).get("result", []):
        title = r.get("title") or ""
        abstr = r.get("abstractText") or ""
        txt = f"{title}. {abstr}".strip()
        if len(txt.split()) >= 8:
            doi = r.get("doi")
            url = f"https://doi.org/{doi}" if doi else f"https://europepmc.org/article/{r.get('source','MED')}/{r.get('id','')}"
            out.append({"url": url, "title": title, "text": txt})
    return out


def brave(query, n=3):
    key = os.environ.get("BRAVE_API_KEY")
    if not key:
        return []
    q = urllib.parse.urlencode({"q": query, "count": n})
    res = _get("https://api.search.brave.com/res/v1/web/search?" + q, {"X-Subscription-Token": key, "Accept": "application/json"})
    return [
        {
            "url": r["url"],
            "title": r.get("title", ""),
            "text": re.sub("<[^>]+>", "", r.get("description", "") + " " + " ".join(r.get("extra_snippets", []))),
        }
        for r in res.get("web", {}).get("results", [])
    ]


PROVIDERS = {
    "wikipedia": wikipedia,
    "openalex": openalex,
    "crossref": crossref,
    "semanticscholar": semanticscholar,
    "arxiv": arxiv,
    "europepmc": europepmc,
    "brave": brave,
}


def pick_queries(sentences, n=6):
    cands = [(i, t) for i, (_, _, t) in enumerate(sentences) if 12 <= len(t.split()) <= 60]
    cands.sort(key=lambda x: -len(set(re.findall(r"[a-z]{6,}", x[1].lower()))))
    step = max(1, len(cands) // n)
    return [c for c in cands[::step]][:n]


def check_web(sentences, providers=None, max_queries=None, embeddings=None):
    names = providers if providers is not None else [p for p in os.environ.get("WEB_SOURCES", "").split(",") if p.strip()]
    if not names:
        return {
            "enabled": False,
            "note": "Web check off. Enable Wikipedia, OpenAlex, CrossRef, Semantic Scholar, arXiv, or EuropePMC in the UI or via WEB_SOURCES.",
        }
    from .similarity import fingerprints, Embedder
    from .extract import split_sentences

    qs = pick_queries(sentences, max_queries or int(os.environ.get("WEB_MAX_QUERIES", "6")))
    cands, errors = {}, []
    for i, t in qs:
        query = " ".join(t.split()[:14])
        for name in names:
            pname = name.strip().lower()
            if pname not in PROVIDERS:
                continue
            try:
                for c in PROVIDERS[pname](query):
                    if c.get("url"):
                        cands.setdefault(c["url"], c | {"provider": pname})
            except Exception as e:
                errors.append(f"{pname}: {e}")
            time.sleep(0.15)
    matches = []
    if cands:
        sent_fp = [{h for h, _, _ in fingerprints(t)} for _, _, t in sentences]
        S = embeddings if (embeddings is not None and len(embeddings) == len(sentences)) else Embedder.encode([t for _, _, t in sentences])
        for url, c in cands.items():
            src_s = split_sentences(c["text"])[:2000]
            if not src_s:
                continue
            src_fp = {h for h, _, _ in fingerprints(c["text"])}
            E = Embedder.encode([t for _, _, t in src_s])
            sims = S @ E.T if len(E) else np.zeros((len(sentences), 0))
            for i in range(len(sentences)):
                ov = len(sent_fp[i] & src_fp) / max(1, len(sent_fp[i]))
                j = int(np.argmax(sims[i])) if sims.shape[1] else -1
                sc = float(sims[i, j]) if j >= 0 else 0
                if ov >= 0.5 or sc >= 0.85:
                    matches.append(
                        {
                            "sentence": i,
                            "url": url,
                            "title": c["title"],
                            "provider": c["provider"],
                            "type": "exact" if ov >= 0.5 else "paraphrase",
                            "score": round(max(ov, sc), 3),
                            "matched_text": src_s[j][2][:300] if j >= 0 else "",
                        }
                    )
    best = {}
    for m in matches:
        if m["sentence"] not in best or m["score"] > best[m["sentence"]]["score"]:
            best[m["sentence"]] = m
    return {
        "enabled": True,
        "providers": names,
        "queries": len(qs),
        "candidates": len(cands),
        "matches": list(best.values()),
        "errors": errors[:5],
    }


# ---------- Stage 2B: Automatic Citation & Reference Verification ----------
_BIB_HEADER = re.compile(r"^\s*(references?( list| cited)?|bibliography|works cited|literature cited|sources)\s*:?\s*$", re.I)
_YEAR_RX = re.compile(r"\b((?:19|20)\d\d)[a-z]?\b")


def extract_reference_entries(text: str, max_refs: int = 12):
    """Extract reference list entries from the bibliography section of a document."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    bib_start = None
    for idx, ln in enumerate(lines):
        if _BIB_HEADER.match(ln) and idx >= len(lines) * 0.2:
            bib_start = idx + 1
            break
    ref_lines = lines[bib_start:] if bib_start is not None else []
    if not ref_lines:
        # Look for trailing lines containing a publication year and author-like pattern
        trailing = []
        for ln in reversed(lines[-40:]):
            if _YEAR_RX.search(ln) and len(ln.split()) >= 5:
                trailing.append(ln)
            elif len(trailing) >= 2:
                break
        ref_lines = list(reversed(trailing))

    entries = []
    for ln in ref_lines:
        if len(ln.split()) >= 5 and _YEAR_RX.search(ln):
            entries.append(ln[:350])
        if len(entries) >= max_refs:
            break
    return entries


def verify_citations(text: str, enabled: bool = False, max_refs: int = 10):
    """Verify bibliography references against CrossRef and OpenAlex to detect fabricated/hallucinated AI citations."""
    refs = extract_reference_entries(text, max_refs=max_refs)
    if not enabled:
        return {"enabled": False, "extracted_count": len(refs), "verified": [], "unverified": []}
    if not refs:
        return {"enabled": True, "extracted_count": 0, "verified": [], "unverified": [], "note": "No bibliography entries detected."}

    verified, unverified = [], []
    for ref in refs:
        clean_q = re.sub(r"https?://\S+|doi:\S+|\[\d+\]", " ", ref).strip()
        words = [w.lower() for w in re.findall(r"[A-Za-z]{4,}", clean_q) if w.lower() not in {"available", "retrieved", "journal", "press", "university", "volume", "issue", "pages"}]
        matched_item = None
        try:
            q = urllib.parse.urlencode({"query.bibliographic": clean_q[:180], "rows": 2})
            items = _get("https://api.crossref.org/works?" + q, timeout=10).get("message", {}).get("items", [])
            for it in items:
                title = " ".join(it.get("title") or [])
                authors = " ".join(f"{a.get('family','')} {a.get('given','')}" for a in (it.get("author") or []))
                cand_blob = f"{title} {authors}".lower()
                overlap = sum(1 for w in set(words[:12]) if w in cand_blob)
                if overlap >= max(2, min(4, len(set(words[:12])) // 2)):
                    doi = it.get("DOI", "")
                    matched_item = {
                        "reference": ref,
                        "matched_title": title,
                        "doi": f"https://doi.org/{doi}" if doi else (it.get("URL") or ""),
                        "source": "crossref",
                    }
                    break
        except Exception:
            pass

        if not matched_item:
            try:
                q = urllib.parse.urlencode({"search": clean_q[:150], "per-page": 2})
                for w in _get("https://api.openalex.org/works?" + q, timeout=10).get("results", []):
                    title = w.get("title") or ""
                    overlap = sum(1 for kw in set(words[:10]) if kw in title.lower())
                    if overlap >= max(2, min(4, len(set(words[:10])) // 2)):
                        matched_item = {
                            "reference": ref,
                            "matched_title": title,
                            "doi": w.get("doi") or w.get("id") or "",
                            "source": "openalex",
                        }
                        break
            except Exception:
                pass

        if matched_item:
            verified.append(matched_item)
        else:
            unverified.append({"reference": ref, "warning": "Not found in CrossRef or OpenAlex — verify against hallucinated AI citation"})
        time.sleep(0.1)

    return {
        "enabled": True,
        "extracted_count": len(refs),
        "verified_count": len(verified),
        "unverified_count": len(unverified),
        "verified": verified,
        "unverified": unverified,
    }


# ---------- Stage 3: Online AI Second-Opinion / LLM Rubric Connector ----------
def check_online_ai(text: str, rules: list = None, enabled: bool = False):
    """Optional online AI detector & LLM rubric second-opinion connector.
    Supports:
      - SAPLING_API_KEY (https://api.sapling.ai/api/v1/aidetect)
      - GPTZERO_API_KEY (https://api.gptzero.me/v2/predict/text)
      - ONLINE_LLM_URL + ONLINE_LLM_KEY + ONLINE_LLM_MODEL (OpenAI-compatible / Ollama chat completions)"""
    if not enabled:
        return {"enabled": False}

    results = []
    snippet = text[:8000]

    sapling_key = os.environ.get("SAPLING_API_KEY", "").strip()
    if sapling_key:
        try:
            req = urllib.request.Request(
                "https://api.sapling.ai/api/v1/aidetect",
                data=json.dumps({"key": sapling_key, "text": snippet}).encode("utf-8"),
                headers={**UA, "Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=15) as r:
                resp = json.loads(r.read())
                results.append({"provider": "sapling", "score": round(float(resp.get("score", 0.0)), 3)})
        except Exception as e:
            results.append({"provider": "sapling", "error": str(e)})

    gptzero_key = os.environ.get("GPTZERO_API_KEY", "").strip()
    if gptzero_key:
        try:
            req = urllib.request.Request(
                "https://api.gptzero.me/v2/predict/text",
                data=json.dumps({"document": snippet}).encode("utf-8"),
                headers={**UA, "Content-Type": "application/json", "x-api-key": gptzero_key},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=15) as r:
                resp = json.loads(r.read())
                docs = resp.get("documents") or [{}]
                prob = docs[0].get("completely_generated_prob", 0.0)
                results.append({"provider": "gptzero", "score": round(float(prob), 3)})
        except Exception as e:
            results.append({"provider": "gptzero", "error": str(e)})

    llm_url = os.environ.get("ONLINE_LLM_URL", "").strip()
    if llm_url:
        try:
            llm_key = os.environ.get("ONLINE_LLM_KEY", "").strip()
            llm_model = os.environ.get("ONLINE_LLM_MODEL", "gpt-4o-mini").strip()
            rules_txt = "\n".join(f"- {r.get('instruction','')}" for r in (rules or []) if r.get("enabled", True))
            prompt = (
                "Evaluate the following academic submission for AI-writing patterns and compliance with these custom English rules:\n"
                f"{rules_txt or '(Use standard academic originality criteria)'}\n\n"
                'Respond in JSON with keys: {"ai_probability": float (0..1), "summary": string}.\n\nText:\n'
                + snippet[:4000]
            )
            headers = {**UA, "Content-Type": "application/json"}
            if llm_key:
                headers["Authorization"] = f"Bearer {llm_key}"
            req = urllib.request.Request(
                llm_url,
                data=json.dumps({"model": llm_model, "messages": [{"role": "user", "content": prompt}], "temperature": 0.0}).encode("utf-8"),
                headers=headers,
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=20) as r:
                resp = json.loads(r.read())
                content = resp.get("choices", [{}])[0].get("message", {}).get("content", "")
                results.append({"provider": f"llm ({llm_model})", "response": content[:500]})
        except Exception as e:
            results.append({"provider": "llm", "error": str(e)})

    return {
        "enabled": True,
        "configured_providers": [k for k, v in [("sapling", sapling_key), ("gptzero", gptzero_key), ("llm", llm_url)] if v],
        "results": results,
    }
