import re, docx
d = docx.Document(); code = False
for line in open("RESEARCH_AND_DESIGN.md"):
    l = line.rstrip("\n")
    if l.startswith("```"): code = not code; continue
    if code: p = d.add_paragraph(l); p.style = "No Spacing"; [setattr(r.font, "name", "Courier New") for r in p.runs]; continue
    if not l.strip() or re.match(r"^\|[-| ]+\|$", l): continue
    l2 = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1 (\2)", l).replace("**", "").replace("*", "")
    if l.startswith("#"): n = len(l) - len(l.lstrip("#")); d.add_heading(l2.lstrip("# "), level=min(n, 3))
    elif l.startswith("|"): d.add_paragraph(" | ".join(c.strip() for c in l2.strip("|").split("|")))
    elif re.match(r"^\s*- ", l): d.add_paragraph(l2.strip()[2:], style="List Bullet")
    elif re.match(r"^\s*\d+\. ", l): d.add_paragraph(re.sub(r"^\s*\d+\. ", "", l2), style="List Number")
    else: d.add_paragraph(l2)
d.save("RESEARCH_AND_DESIGN.docx"); print("docx ok")
