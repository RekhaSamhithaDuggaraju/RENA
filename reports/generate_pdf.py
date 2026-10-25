"""
RENA — audit report PDF generator. Takes a real result dict from
core.pipeline.run_audit() and renders it as a professional report.
No placeholder numbers: every value in the PDF is read from the audit dict.
"""
from __future__ import annotations

import datetime
import json
from xml.sax.saxutils import escape
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, HRFlowable, Preformatted
)

LAVENDER = colors.HexColor("#6E56CF")
LAVENDER_DARK = colors.HexColor("#3A2E6E")
INK = colors.HexColor("#1A1523")
MUTED = colors.HexColor("#6B7280")

VERDICT_COLORS = {
    "Likely Forgotten": colors.HexColor("#15803D"),
    "Not Forgotten": colors.HexColor("#B91C1C"),
    "Ambiguous": colors.HexColor("#B45309"),
    "Inconclusive": colors.HexColor("#6B7280"),
}


def _styles():
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle("RenaTitle", parent=styles["Title"], textColor=LAVENDER_DARK,
                               fontSize=26, spaceAfter=4))
    styles.add(ParagraphStyle("RenaSubtitle", parent=styles["Normal"], textColor=MUTED,
                               fontSize=11, spaceAfter=20))
    styles.add(ParagraphStyle("RenaH2", parent=styles["Heading2"], textColor=LAVENDER_DARK,
                               spaceBefore=18, spaceAfter=8))
    styles.add(ParagraphStyle("RenaBody", parent=styles["Normal"], fontSize=10, leading=14))
    styles.add(ParagraphStyle("RenaSmall", parent=styles["Normal"], fontSize=8.5,
                               textColor=MUTED, leading=12))
    return styles


def generate_report(audit: dict, dataset_summary: dict, output_path: str,
                     audit_id: str = "RENA-AUDIT-0001"):
    styles = _styles()
    doc = SimpleDocTemplate(output_path, pagesize=letter,
                             topMargin=0.75 * inch, bottomMargin=0.75 * inch,
                             leftMargin=0.75 * inch, rightMargin=0.75 * inch)
    story = []

    # --- Cover / header -----------------------------------------------
    story.append(Paragraph("RENA", styles["RenaTitle"]))
    story.append(Paragraph("Reliable Evaluation for Neural Unlearning Audits", styles["RenaSubtitle"]))
    story.append(HRFlowable(width="100%", color=LAVENDER, thickness=1.2))
    story.append(Spacer(1, 14))

    meta_rows = [
        ["Audit ID", audit_id],
        ["Algorithm Audited", audit["algorithm"]],
        ["Generated", datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")],
        ["Dataset", f"{dataset_summary['n_features']}-feature, "
                    f"{dataset_summary['n_classes']}-class classification"],
        ["Forget Set Size", f"{dataset_summary['n_forget']} samples"],
        ["Retain Set Size", f"{dataset_summary['n_retain']} samples"],
    ]
    if isinstance(dataset_summary.get("forget_fraction"), (int, float)):
        meta_rows.append(["Forget Fraction", f"{dataset_summary['forget_fraction']:.1%}"])
    if dataset_summary.get("split_seed") is not None:
        meta_rows.append(["Data Split Seed", str(dataset_summary["split_seed"])])
    meta_table = Table(meta_rows, colWidths=[1.8 * inch, 4.2 * inch])
    meta_table.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), 9.5),
        ("TEXTCOLOR", (0, 0), (0, -1), MUTED),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("LINEBELOW", (0, 0), (-1, -1), 0.4, colors.HexColor("#E5E7EB")),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 20))

    # --- Executive summary / verdict -----------------------------------
    story.append(Paragraph("Executive Summary", styles["RenaH2"]))
    verdict = audit["consensus"]["verdict"]
    verdict_color = VERDICT_COLORS.get(verdict, INK)
    verdict_para_style = ParagraphStyle("Verdict", parent=styles["RenaBody"],
                                         fontSize=16, textColor=verdict_color,
                                         fontName="Helvetica-Bold")
    story.append(Paragraph(f"Consensus Verdict: {verdict}", verdict_para_style))
    story.append(Paragraph(
        f"Consensus score: <b>{audit['consensus']['consensus_score']:.2f}</b> / 1.00 "
        f"&nbsp;&nbsp;|&nbsp;&nbsp; Inter-module agreement: "
        f"<b>{audit['consensus']['agreement']:.2f}</b> / 1.00",
        styles["RenaBody"]
    ))
    story.append(Spacer(1, 8))
    for finding in audit["diagnosis"]["findings"]:
        story.append(Paragraph(f"&bull; {finding}", styles["RenaBody"]))
    story.append(Spacer(1, 10))

    # --- Model utility ----------------------------------------------
    story.append(Paragraph("Model Utility", styles["RenaH2"]))
    acc = audit["accuracy"]
    util_table = Table([
        ["Split", "Accuracy"],
        ["Retain set", f"{acc['retain']*100:.1f}%"],
        ["Test set", f"{acc['test']*100:.1f}%"],
        ["Forget set", f"{acc['forget']*100:.1f}%"],
        ["Original model (test set, pre-unlearning)", f"{acc['original_model_test']*100:.1f}%"],
    ], colWidths=[3.8 * inch, 2.2 * inch])
    util_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), LAVENDER),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9.5),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F5F3FF")]),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(util_table)
    story.append(Spacer(1, 10))

    # --- Verification module breakdown -------------------------------
    story.append(Paragraph("Verification Module Results", styles["RenaH2"]))
    rows = [["Module", "Forget Score", "Reliability", "Raw Statistic"]]
    for m in audit["module_results"]:
        rows.append([
            m["module_name"], f"{m['forget_score']:.2f}", f"{m['reliability']:.2f}",
            f"{m['raw_statistic']:.4f}"
        ])
    mod_table = Table(rows, colWidths=[2.6 * inch, 1.2 * inch, 1.2 * inch, 1.2 * inch])
    mod_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), LAVENDER_DARK),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F5F3FF")]),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(mod_table)
    story.append(Spacer(1, 6))
    for m in audit["module_results"]:
        story.append(Paragraph(f"<b>{m['module_name']}</b>: {m['reliability_reason']}",
                                styles["RenaSmall"]))
    story.append(Spacer(1, 10))

    # --- Consensus contribution breakdown -----------------------------
    story.append(Paragraph("Consensus Weighting", styles["RenaH2"]))
    contrib_rows = [["Module", "Weight in Consensus Score"]]
    for c in audit["consensus"]["per_module_contribution"]:
        contrib_rows.append([c["module"], f"{c['weight_pct']:.1f}%"])
    contrib_table = Table(contrib_rows, colWidths=[3.8 * inch, 2.2 * inch])
    contrib_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), LAVENDER),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9.5),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F5F3FF")]),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(contrib_table)

    if audit.get("method_trace"):
        story.append(Paragraph("Method-Specific Diagnostics", styles["RenaH2"]))
        trace_text = json.dumps(audit["method_trace"], indent=2, sort_keys=True, default=str)
        trace_style = ParagraphStyle("Trace", parent=styles["RenaSmall"], fontName="Courier",
                                     fontSize=7, leading=9, backColor=colors.HexColor("#F5F3FF"),
                                     borderColor=colors.HexColor("#E5E7EB"), borderWidth=0.5,
                                     borderPadding=8, spaceBefore=4, spaceAfter=8)
        story.append(Preformatted(escape(trace_text[:12000]), trace_style, maxLineLength=95))

    # --- Recommendations ------------------------------------------------
    story.append(Paragraph("Recommendations", styles["RenaH2"]))
    for rec in audit["diagnosis"]["recommendations"]:
        story.append(Paragraph(f"&bull; {rec}", styles["RenaBody"]))

    story.append(PageBreak())
    story.append(Paragraph("Methodology & Limitations", styles["RenaH2"]))
    story.append(Paragraph(
        "This audit reports statistical evidence of forgetting across four independently "
        "computed verification modules, combined via a reliability-weighted consensus "
        "engine that forces an 'Inconclusive' verdict whenever the modules disagree beyond "
        "a calibrated threshold, rather than averaging disagreement into false confidence. "
        "This is not a formal or cryptographic guarantee of data deletion — it is evidence, "
        "not proof. Verification power decreases with small forget-set sizes and with "
        "datasets/models where baseline accuracy is very high, since there is less room for "
        "a detectable statistical signal. Reliability discounts applied to specific modules "
        "for this audit are listed above and should be read alongside the verdict. The dataset "
        "summary records split provenance when available; split seed does not override fixed "
        "method-specific optimizer seeds.",
        styles["RenaBody"]
    ))

    doc.build(story)
    return output_path
