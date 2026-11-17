from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from xml.sax.saxutils import escape
import datetime
def build_pdf(report, path):
    st = getSampleStyleSheet(); body = ParagraphStyle("b", parent=st["BodyText"], fontSize=9, leading=12)
    el = [Paragraph("Originality Report", st["Title"]),
          Paragraph(escape(f"{report['title']}  |  {report['words']} words  |  "
                    f"{datetime.datetime.fromtimestamp(report['created']):%Y-%m-%d %H:%M}"), body), Spacer(1, 8)]
    ai, sim = report["ai"], report["similarity"]
    p = ai.get("probability")
    rows = [["Similarity (internal library)", f"{sim['percent']}%  (exact {sim['exact_percent']}%, paraphrase {sim['paraphrase_percent']}%)"],
            ["AI-writing likelihood", "n/a" if p is None else f"{round(p*100)}% ± {round(ai['uncertainty']*100)} pts" + ("  (LOW CONFIDENCE: short text)" if ai.get("low_confidence") else "")],
            ["AI verdict", escape(ai.get("verdict_text", "-"))],
            ["Share of text flagged as AI-like", f"{round(ai.get('share_flagged', 0)*100)}%"],
            ["Models", escape(", ".join(f"{k}: {v}" for k, v in ai.get("models", {}).items()))]]
    t = Table(rows, colWidths=[170, 330]); t.setStyle(TableStyle([("GRID", (0,0), (-1,-1), .3, colors.grey), ("FONTSIZE", (0,0), (-1,-1), 9)]))
    el += [t, Spacer(1, 6), Paragraph("<i>" + escape(ai["disclaimer"]) + "</i>", body), Spacer(1, 8)]
    if ai.get("rules_applied"):
        el.append(Paragraph("Custom English AI Rules Triggered", st["Heading3"]))
        for r in ai["rules_applied"]:
            el.append(Paragraph(escape(f"[{r['effect']}] {r['instruction']} ({r['sentences_matched']} sentences)"), body))
    cites = sim.get("citations") or {}
    if cites.get("enabled") and cites.get("unverified"):
        el.append(Paragraph("Unverified / Potential Hallucinated Citations", st["Heading3"]))
        for u in cites["unverified"]:
            el.append(Paragraph(escape(f"• {u['reference']}"), body))
    if sim["sources"]:
        el.append(Paragraph("Matched sources", st["Heading3"]))
        for s in sim["sources"]: el.append(Paragraph(escape(f"#{s['doc']} {s['title']} — {s['percent']}% of words"), body))
    el.append(Paragraph("Sentence-level evidence", st["Heading3"]))
    el.append(Paragraph("Key: <font backColor='#ffd6d6'>red</font> = exact match, <font backColor='#ffe9b3'>amber</font> = paraphrase match, "
                        "<font backColor='#dcd6ff'>purple</font> = AI-like passage (~3-sentence window above the 2% FPR threshold). Quoted text and bibliography are excluded from similarity.", body))
    for s in report["sentences"]:
        txt = escape(s["text"]); note = ""
        if s["exact"]: bg = "#ffd6d6"; note = f" [exact: #{s['exact']['doc']} source: \"{s['exact'].get('matched_text','')[:120]}\"]"
        elif s["paraphrase"]: bg = "#ffe9b3"; note = f" [paraphrase {s['paraphrase']['score']}: #{s['paraphrase']['doc']}]"
        elif s.get("excluded"): bg = None; note = f" [excluded: {s['excluded']}]"
        elif s.get("ai_flag", s["ai_prob"] >= 0.5): bg = "#dcd6ff"; note = f" [AI {s['ai_prob']:.2f}]"
        else: bg = None
        el.append(Paragraph((f"<font backColor='{bg}'>{txt}</font>" if bg else txt) + f"<font size=7 color='grey'>{escape(note)}</font>", body))
    SimpleDocTemplate(str(path), pagesize=A4, leftMargin=40, rightMargin=40).build(el)
    return path
