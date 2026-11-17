"""OPT-IN: add free academic / encyclopedic text to the local similarity library.
Sources (all free, public APIs, no key): OpenAlex abstracts, arXiv abstracts, Wikipedia article extracts.
Only abstracts / public extracts are stored. Respect the APIs' rate limits; set a contact email for OpenAlex's polite pool.
Usage:
  python scripts/ingest_sources.py --openalex "climate change adaptation" --n 200 --mailto you@uni.ac.zw
  python scripts/ingest_sources.py --arxiv "large language models" --n 100
  python scripts/ingest_sources.py --wikipedia "Great Zimbabwe" "Photosynthesis"
  add --dry-run to fetch and count without writing; --out F.jsonl saves what was fetched; --from-jsonl F.jsonl ingests a saved batch."""
import argparse, json, sys, time, pathlib, urllib.parse, urllib.request, xml.etree.ElementTree as ET
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
UA = {"User-Agent": "detector-library-ingest/3.1 (academic integrity tool; self-hosted)"}
def get(url):
    for k in range(3):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=30) as r: return r.read()
        except Exception as e:
            if k == 2: raise
            time.sleep(2 * (k + 1))
def openalex(q, n, mailto=None):
    out, cursor = [], "*"
    while len(out) < n and cursor:
        u = "https://api.openalex.org/works?" + urllib.parse.urlencode({"search": q, "filter": "has_abstract:true", "per-page": 100, "cursor": cursor, **({"mailto": mailto} if mailto else {})})
        d = json.loads(get(u)); cursor = d["meta"].get("next_cursor")
        for w in d["results"]:
            inv = w.get("abstract_inverted_index") or {}
            pos = sorted((p, t) for t, ps in inv.items() for p in ps); text = " ".join(t for _, t in pos)
            if len(text.split()) >= 40: out.append((w.get("display_name") or w["id"], text, "openalex:" + w["id"].rsplit("/", 1)[-1]))
        if not d["results"]: break
        time.sleep(0.2)
    return out[:n]
def arxiv(q, n):
    out, start = [], 0; ns = {"a": "http://www.w3.org/2005/Atom"}
    while len(out) < n:
        u = "https://export.arxiv.org/api/query?" + urllib.parse.urlencode({"search_query": " AND ".join(f"all:{w}" for w in q.split()), "start": start, "max_results": min(100, n - len(out))})
        es = ET.fromstring(get(u)).findall("a:entry", ns)
        if not es: break
        for e in es:
            out.append((" ".join(e.find("a:title", ns).text.split()), " ".join(e.find("a:summary", ns).text.split()), "arxiv:" + e.find("a:id", ns).text.rsplit("/", 1)[-1]))
        start += len(es); time.sleep(3)   # arXiv asks for >= 3 s between calls
    return out[:n]
def wikipedia(titles, lang="en"):
    out = []
    for t in titles:
        u = f"https://{lang}.wikipedia.org/w/api.php?" + urllib.parse.urlencode({"action": "query", "prop": "extracts", "explaintext": 1, "titles": t, "format": "json", "redirects": 1})
        for pg in json.loads(get(u))["query"]["pages"].values():
            if pg.get("extract"): out.append((pg["title"], pg["extract"], f"wikipedia:{lang}:{pg['title']}"))
        time.sleep(0.5)
    return out
if __name__ == "__main__":
    a = argparse.ArgumentParser(); a.add_argument("--openalex"); a.add_argument("--arxiv"); a.add_argument("--wikipedia", nargs="*")
    a.add_argument("--n", type=int, default=100); a.add_argument("--mailto"); a.add_argument("--dry-run", action="store_true")
    a.add_argument("--out", help="also save fetched docs as JSONL"); a.add_argument("--from-jsonl", dest="src", help="ingest docs from a saved JSONL instead of fetching"); a = a.parse_args()
    docs = [tuple(json.loads(l)) for l in open(a.src)] if a.src else (openalex(a.openalex, a.n, a.mailto) if a.openalex else []) + (arxiv(a.arxiv, a.n) if a.arxiv else []) + (wikipedia(a.wikipedia) if a.wikipedia else [])
    print(f"fetched {len(docs)} documents, {sum(len(d[1].split()) for d in docs)} words")
    if a.out:
        with open(a.out, "a") as f:
            for d in docs: f.write(json.dumps(list(d)) + "\n")
    if a.dry_run or not docs: sys.exit()
    from app.similarity import Corpus; from app.extract import split_sentences, normalise
    c = Corpus(); have = {v[0] for v in c.titles().values()}
    added = 0
    for title, text, src in docs:
        t = f"{title} [{src}]"
        if t in have: continue
        text = normalise(text); c.add(t, text, split_sentences(text), source=src.split(":")[0], save=False); added += 1
    c.save_index(); print("added", added, "library docs now", c.count())
