import io, re
def extract_text(filename: str, data: bytes) -> str:
    name = filename.lower()
    if name.endswith(".pdf"):
        from pypdf import PdfReader
        r = PdfReader(io.BytesIO(data))
        return "\n".join((p.extract_text() or "") for p in r.pages)
    if name.endswith(".docx"):
        import docx
        d = docx.Document(io.BytesIO(data))
        parts = [p.text for p in d.paragraphs]
        for t in d.tables:
            for row in t.rows:
                parts.append(" | ".join(c.text for c in row.cells))
        return "\n".join(parts)
    if name.endswith(".txt") or name.endswith(".md"):
        for enc in ("utf-8", "latin-1"):
            try: return data.decode(enc)
            except UnicodeDecodeError: pass
    raise ValueError("Unsupported file type (use PDF, DOCX or TXT)")

_SENT = re.compile(r"(?<=[.!?])[\"')\]”’»]*\s+(?=[A-Z0-9\"'(\[“‘«])")
def split_sentences(text: str):
    """Returns list of (start, end, sentence) spans over normalised text."""
    out = []
    for para in re.finditer(r"[^\n]+", text):
        p, off, last = para.group(), para.start(), 0
        for m in _SENT.finditer(p):
            s = p[last:m.start()+1].strip()
            if s: out.append((off+last, off+m.start()+1, s))
            last = m.end()
        s = p[last:].strip()
        if s: out.append((off+last, off+len(p), s))
    return out
def normalise(text: str) -> str:
    text = text.replace("\r", "")
    text = re.sub(r"[ \t\f\v]+", " ", text)
    text = re.sub(r"-\n(?=[a-z])", "", text)     # PDF hyphenation
    return re.sub(r"\n{2,}", "\n", text).strip()
