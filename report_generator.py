import io
from datetime import datetime, timezone

import matplotlib
matplotlib.use("Agg")  # no display backend needed, just render to image
import matplotlib.pyplot as plt

from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak,
)


PASS_LABELS = {"Excellent", "Good"}
NEEDS_IMPROVEMENT_LABELS = {"Needs Improvement"}
FAIL_LABELS = {"Poor"}


# =====================================================
# Styles
# =====================================================

def _build_styles():
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="ReportTitle", parent=styles["Title"], fontSize=22, spaceAfter=6,
    ))
    styles.add(ParagraphStyle(
        name="ReportSubtitle", parent=styles["Normal"], fontSize=10,
        textColor=colors.grey, spaceAfter=18,
    ))
    styles.add(ParagraphStyle(
        name="SectionHeading", parent=styles["Heading2"], spaceBefore=18, spaceAfter=8,
    ))
    styles.add(ParagraphStyle(
        name="BodySmall", parent=styles["Normal"], fontSize=9, leading=12,
    ))
    styles.add(ParagraphStyle(
        name="HeaderSmall", parent=styles["Normal"], fontSize=8, leading=10,
        textColor=colors.white, fontName="Helvetica-Bold",
    ))
    styles.add(ParagraphStyle(
        name="Recommendation", parent=styles["Normal"], fontSize=10, leading=14,
        spaceAfter=6, leftIndent=10,
    ))
    return styles


# =====================================================
# Chart — dimension-wise average scores
# =====================================================

def _build_dimension_chart_image(aggregate: dict) -> Image:
    dimensions = ["Accuracy", "Relevance", "Completeness", "Overall"]
    values = [
        aggregate.get("avg_accuracy", 0) * 100,
        aggregate.get("avg_relevance", 0) * 100,
        aggregate.get("avg_completeness", 0) * 100,
        aggregate.get("avg_overall_score", 0) * 100,
    ]

    fig, ax = plt.subplots(figsize=(6, 3))
    bars = ax.bar(dimensions, values, color=["#6b5b95", "#6b5b95", "#6b5b95", "#4a3f70"])
    ax.set_ylim(0, 100)
    ax.set_ylabel("Average Score (%)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 2, f"{value:.0f}", ha="center", fontsize=9)

    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    buf.seek(0)

    return Image(buf, width=5.5 * inch, height=2.75 * inch)


# =====================================================
# Improvement recommendations — auto-generated from score patterns
# =====================================================

def _build_recommendations(aggregate: dict, results: list[dict]) -> list[str]:
    recommendations = []

    if aggregate.get("avg_accuracy", 1) < 0.6:
        recommendations.append(
            "Accuracy scores are below average across this batch. Review responses against "
            "their reference answers or retrieved source content, and consider whether the "
            "knowledge base needs broader topic coverage."
        )

    if aggregate.get("avg_relevance", 1) < 0.6:
        recommendations.append(
            "Relevance scores are below average. Responses may be drifting from the specific "
            "question asked — review prompt design or response generation for on-topic focus."
        )

    if aggregate.get("avg_completeness", 1) < 0.6:
        recommendations.append(
            "Completeness scores are below average. Several responses likely address only part "
            "of multi-part questions — review flagged missing aspects in individual results."
        )

    total_flagged = aggregate.get("total_flagged_hallucinations", 0)
    if total_flagged > 0:
        recommendations.append(
            f"{total_flagged} hallucinated claim(s) were flagged across this batch. Review the "
            f"'Hallucinated Responses' section below for the specific unsupported statements."
        )

    fail_count = sum(1 for r in results if r.get("verdict_label") == "Poor")
    if fail_count > 0:
        recommendations.append(
            f"{fail_count} response(s) received a 'Poor' verdict. These represent the highest-priority "
            f"cases for review, since they combine low scores across multiple dimensions."
        )

    if not recommendations:
        recommendations.append(
            "No significant quality issues were detected in this batch. All dimension averages "
            "are within acceptable ranges and no responses received a 'Poor' verdict."
        )

    return recommendations


# =====================================================
# Table builders
# =====================================================

def _truncate(text: str, max_len: int = 90) -> str:
    text = text or ""
    return text if len(text) <= max_len else text[:max_len].rstrip() + "..."


def _build_results_table(results: list[dict], styles) -> Table:
    header_labels = ["Question", "Answer", "Verdict", "Overall", "Accur.", "Relev.", "Complete.", "Halluc."]
    header = [Paragraph(h, styles["HeaderSmall"]) for h in header_labels]
    data = [header]

    for r in results:
        data.append([
            Paragraph(_truncate(r.get("question", ""), 70), styles["BodySmall"]),
            Paragraph(_truncate(r.get("answer", ""), 70), styles["BodySmall"]),
            Paragraph(r.get("verdict_label", ""), styles["BodySmall"]),
            f"{r.get('overall_score', 0) * 100:.0f}%",
            f"{r.get('accuracy', 0) * 100:.0f}%",
            f"{r.get('relevance', 0) * 100:.0f}%",
            f"{r.get('completeness', 0) * 100:.0f}%",
            r.get("hallucination_risk", ""),
        ])

    table = Table(data, colWidths=[1.25 * inch, 1.25 * inch, 1.0 * inch, 0.55 * inch, 0.55 * inch, 0.55 * inch, 0.75 * inch, 0.6 * inch], repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4a3f70")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f5f3fa")]),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    return table


def _build_hallucinated_table(results: list[dict], styles):
    flagged_rows = [r for r in results if r.get("hallucination_flagged_statements")]

    if not flagged_rows:
        return Paragraph("No hallucinated claims were flagged in this batch.", styles["BodySmall"])

    header = ["Question", "Flagged Claim(s)"]
    data = [header]

    for r in flagged_rows:
        claims_text = "<br/>&bull; ".join(r["hallucination_flagged_statements"])
        data.append([
            Paragraph(_truncate(r.get("question", ""), 60), styles["BodySmall"]),
            Paragraph("&bull; " + claims_text, styles["BodySmall"]),
        ])

    table = Table(data, colWidths=[2 * inch, 4.3 * inch], repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#a34a4a")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#fbeeee")]),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    return table


def _build_verdict_summary_table(results: list[dict]) -> Table:
    pass_count = sum(1 for r in results if r.get("verdict_label") in PASS_LABELS)
    needs_improvement_count = sum(1 for r in results if r.get("verdict_label") in NEEDS_IMPROVEMENT_LABELS)
    fail_count = sum(1 for r in results if r.get("verdict_label") in FAIL_LABELS)

    data = [
        ["Pass", "Needs Improvement", "Fail"],
        [str(pass_count), str(needs_improvement_count), str(fail_count)],
    ]

    table = Table(data, colWidths=[1.9 * inch, 1.9 * inch, 1.9 * inch])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4a3f70")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), 11),
        ("FONTSIZE", (0, 1), (-1, 1), 16),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("BACKGROUND", (0, 1), (0, 1), colors.HexColor("#e6f5ec")),
        ("BACKGROUND", (1, 1), (1, 1), colors.HexColor("#fdf3e0")),
        ("BACKGROUND", (2, 1), (2, 1), colors.HexColor("#fbeeee")),
    ]))
    return table


# =====================================================
# Public entry point
# =====================================================

def generate_report_pdf(results: list[dict], aggregate: dict) -> bytes:
    """
    Builds the full PDF report and returns it as raw bytes, ready to be
    streamed back as a file download.
    """
    styles = _build_styles()
    buf = io.BytesIO()

    doc = SimpleDocTemplate(
        buf, pagesize=letter,
        topMargin=0.6 * inch, bottomMargin=0.6 * inch,
        leftMargin=0.6 * inch, rightMargin=0.6 * inch,
    )

    story = []

    # --- 1. Project details ---
    story.append(Paragraph("AI Response Quality Evaluator", styles["ReportTitle"]))
    generated_at = datetime.now(timezone.utc).strftime("%B %d, %Y at %H:%M UTC")
    story.append(Paragraph(
        f"Evaluation Report &bull; Generated {generated_at} &bull; {len(results)} response(s) evaluated",
        styles["ReportSubtitle"],
    ))

    # --- 2. Batch summary ---
    story.append(Paragraph("Batch Summary", styles["SectionHeading"]))
    story.append(_build_verdict_summary_table(results))
    story.append(Spacer(1, 12))
    story.append(Paragraph(
        f"Average Accuracy: {aggregate.get('avg_accuracy', 0) * 100:.0f}% &nbsp;&nbsp;|&nbsp;&nbsp; "
        f"Average Relevance: {aggregate.get('avg_relevance', 0) * 100:.0f}% &nbsp;&nbsp;|&nbsp;&nbsp; "
        f"Average Completeness: {aggregate.get('avg_completeness', 0) * 100:.0f}% &nbsp;&nbsp;|&nbsp;&nbsp; "
        f"Average Overall: {aggregate.get('avg_overall_score', 0) * 100:.0f}%",
        styles["Normal"],
    ))

    # --- 3. Dimension-wise scores chart ---
    story.append(Paragraph("Dimension-Wise Scores", styles["SectionHeading"]))
    story.append(_build_dimension_chart_image(aggregate))

    story.append(PageBreak())

    # --- 4. Individual evaluation results ---
    story.append(Paragraph("Individual Evaluation Results", styles["SectionHeading"]))
    story.append(_build_results_table(results, styles))

    story.append(PageBreak())

    # --- 5. Hallucinated responses ---
    story.append(Paragraph("Hallucinated Responses", styles["SectionHeading"]))
    story.append(_build_hallucinated_table(results, styles))
    story.append(Spacer(1, 16))

    # --- 6. Overall verdicts (distribution) ---
    story.append(Paragraph("Verdict Distribution", styles["SectionHeading"]))
    verdict_counts: dict = {}
    for r in results:
        label = r.get("verdict_label", "Unknown")
        verdict_counts[label] = verdict_counts.get(label, 0) + 1
    verdict_text = " &nbsp;&nbsp;|&nbsp;&nbsp; ".join(f"{label}: {count}" for label, count in verdict_counts.items())
    story.append(Paragraph(verdict_text, styles["Normal"]))
    story.append(Spacer(1, 16))

    # --- 7. Improvement recommendations ---
    story.append(Paragraph("Improvement Recommendations", styles["SectionHeading"]))
    for i, rec in enumerate(_build_recommendations(aggregate, results), start=1):
        story.append(Paragraph(f"{i}. {rec}", styles["Recommendation"]))

    doc.build(story)

    buf.seek(0)
    return buf.read()