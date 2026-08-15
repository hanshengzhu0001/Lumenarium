from pathlib import Path
import re

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.platypus import (
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs" / "SCENEBA_MATHEMATICAL_FORMULATION.md"
OUTPUT = ROOT / "docs" / "SCENEBA_MATHEMATICAL_FORMULATION.pdf"


def esc(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
styles = getSampleStyleSheet()
body = ParagraphStyle(
    "BodyCN",
    parent=styles["BodyText"],
    fontName="STSong-Light",
    fontSize=9.2,
    leading=13.3,
    spaceAfter=4,
)
title = ParagraphStyle(
    "TitleCN",
    parent=styles["Title"],
    fontName="STSong-Light",
    fontSize=20,
    leading=25,
    alignment=TA_CENTER,
    textColor=colors.HexColor("#172554"),
    spaceAfter=10,
)
h1 = ParagraphStyle(
    "H1CN",
    parent=styles["Heading1"],
    fontName="STSong-Light",
    fontSize=14,
    leading=18,
    textColor=colors.HexColor("#1d4ed8"),
    spaceBefore=10,
    spaceAfter=6,
)
h2 = ParagraphStyle(
    "H2CN",
    parent=styles["Heading2"],
    fontName="STSong-Light",
    fontSize=11,
    leading=15,
    textColor=colors.HexColor("#334155"),
    spaceBefore=7,
    spaceAfter=4,
)
eq = ParagraphStyle(
    "Equation",
    parent=body,
    fontName="STSong-Light",
    fontSize=9.4,
    leading=14,
    leftIndent=9 * mm,
    rightIndent=7 * mm,
    backColor=colors.HexColor("#f8fafc"),
    borderColor=colors.HexColor("#cbd5e1"),
    borderWidth=0.5,
    borderPadding=5,
    spaceBefore=4,
    spaceAfter=6,
)
bullet = ParagraphStyle(
    "BulletCN",
    parent=body,
    leftIndent=6 * mm,
    firstLineIndent=-3 * mm,
)
caption = ParagraphStyle(
    "Caption",
    parent=body,
    fontSize=8,
    leading=10,
    textColor=colors.HexColor("#64748b"),
    alignment=TA_CENTER,
)


def inline_markup(s: str) -> str:
    s = esc(s)
    s = re.sub(r"`([^`]+)`", r"<font name='Courier'>\1</font>", s)
    s = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", s)
    return s


def equation_text(lines):
    s = " ".join(x.strip() for x in lines)
    s = s.replace(r"\qquad", "    ").replace(r"\quad", "  ")
    s = s.replace(r"\,", " ").replace(r"\;", " ")
    replacements = {
        r"\mathbb R": "ℝ", r"\mathbb": "", r"\in": "∈", r"\subset": "⊂",
        r"\propto": "∝", r"\sum": "Σ", r"\prod": "Π", r"\min": "min",
        r"\max": "max", r"\arg\min": "arg min", r"\approx": "≈",
        r"\nabla": "∇", r"\lambda": "λ", r"\alpha": "α",
        r"\gamma": "γ", r"\rho": "ρ", r"\psi": "ψ", r"\Omega": "Ω",
        r"\Sigma": "Σ", r"\Pi": "Π", r"\top": "T", r"\star": "*",
        r"\ldots": "…", r"\left": "", r"\right": "", r"\big": "",
        r"\mathrm": "", r"\operatorname": "", r"\mathbf": "",
        r"\begin{bmatrix}": "[", r"\end{bmatrix}": "]",
        r"\begin{aligned}": "", r"\end{aligned}": "",
        r"\\": "  ", "&": " ", "{": "", "}": "",
    }
    for a, b in replacements.items():
        s = s.replace(a, b)
    s = s.replace("^", "<super>").replace("_", "<sub>")
    # Avoid malformed rich-text tags from arbitrary LaTeX subscripts.
    s = s.replace("<super>", "^").replace("<sub>", "_")
    return esc(s)


def parse():
    lines = SOURCE.read_text(encoding="utf-8").splitlines()
    story = []
    i = 0
    table_count = 0
    while i < len(lines):
        line = lines[i].rstrip()
        if not line:
            story.append(Spacer(1, 2.5 * mm))
            i += 1
            continue
        if line.startswith("# "):
            story.append(Paragraph(inline_markup(line[2:]), title))
            story.append(Paragraph(
                "A mathematically explicit design specification for the Lumenarium hybrid inference extension",
                caption,
            ))
            i += 1
            continue
        if line.startswith("## "):
            story.append(Paragraph(inline_markup(line[3:]), h1))
            i += 1
            continue
        if line.startswith("### "):
            story.append(Paragraph(inline_markup(line[4:]), h2))
            i += 1
            continue
        if line == r"\[":
            block = []
            i += 1
            while i < len(lines) and lines[i].strip() != r"\]":
                block.append(lines[i])
                i += 1
            story.append(Paragraph(equation_text(block), eq))
            i += 1
            continue
        if line.startswith("|"):
            raw = []
            while i < len(lines) and lines[i].startswith("|"):
                raw.append(lines[i])
                i += 1
            rows = [
                [inline_markup(c.strip()) for c in r.strip("|").split("|")]
                for r in raw
                if not re.match(r"^\|\s*:?-+", r)
            ]
            data = [[Paragraph(c, body) for c in row] for row in rows]
            table = Table(data, repeatRows=1, hAlign="LEFT")
            table.setStyle(TableStyle([
                ("FONTNAME", (0, 0), (-1, -1), "STSong-Light"),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dbeafe")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#1e3a8a")),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#94a3b8")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]))
            table_count += 1
            story.append(KeepTogether([table, Spacer(1, 2 * mm)]))
            continue
        if re.match(r"^\d+\.\s", line) or line.startswith("- "):
            marker = "•" if line.startswith("- ") else line.split(".", 1)[0] + "."
            text = line[2:] if line.startswith("- ") else line.split(".", 1)[1].strip()
            story.append(Paragraph(f"{marker} {inline_markup(text)}", bullet))
            i += 1
            continue
        paragraph = [line]
        i += 1
        while i < len(lines):
            nxt = lines[i].rstrip()
            if (not nxt or nxt.startswith(("#", "|", "- ", r"\["))
                    or re.match(r"^\d+\.\s", nxt)):
                break
            paragraph.append(nxt)
            i += 1
        story.append(Paragraph(inline_markup(" ".join(paragraph)), body))
    return story


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("STSong-Light", 8)
    canvas.setFillColor(colors.HexColor("#64748b"))
    canvas.drawString(18 * mm, 11 * mm, "SceneBA mathematical formulation · v0.1")
    canvas.drawRightString(192 * mm, 11 * mm, f"{doc.page}")
    canvas.restoreState()


def main():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(OUTPUT),
        pagesize=A4,
        rightMargin=17 * mm,
        leftMargin=17 * mm,
        topMargin=16 * mm,
        bottomMargin=17 * mm,
        title="SceneBA Mathematical Formulation",
        author="Lumenarium Research",
        subject="Uncertainty-aware discrete-continuous scene bundle adjustment",
    )
    doc.build(parse(), onFirstPage=footer, onLaterPages=footer)
    print(OUTPUT)


if __name__ == "__main__":
    main()
