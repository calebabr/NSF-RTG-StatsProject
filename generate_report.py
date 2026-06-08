#!/usr/bin/env python3
"""Generate a PDF summary report of the KTH-TIPS2 SPD Geometry Analysis results."""

import os
import csv
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.colors import HexColor
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle,
    PageBreak, KeepTogether
)
from reportlab.lib import colors

FIG_DIR = "figures"
RES_DIR = "results"
OUTPUT = "KTH_TIPS2_Analysis_Report.pdf"

WIDTH, HEIGHT = letter
MARGIN = 0.75 * inch

def load_csv(path):
    with open(path, newline="") as f:
        return list(csv.reader(f))

def build_report():
    doc = SimpleDocTemplate(
        OUTPUT, pagesize=letter,
        leftMargin=MARGIN, rightMargin=MARGIN,
        topMargin=MARGIN, bottomMargin=MARGIN,
    )

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        "ReportTitle", parent=styles["Title"], fontSize=20, spaceAfter=6,
        textColor=HexColor("#1a1a2e"),
    ))
    styles.add(ParagraphStyle(
        "Subtitle", parent=styles["Normal"], fontSize=11, spaceAfter=18,
        alignment=TA_CENTER, textColor=HexColor("#555555"),
    ))
    styles.add(ParagraphStyle(
        "SectionHead", parent=styles["Heading1"], fontSize=14, spaceBefore=18,
        spaceAfter=8, textColor=HexColor("#16213e"),
    ))
    styles.add(ParagraphStyle(
        "SubHead", parent=styles["Heading2"], fontSize=12, spaceBefore=12,
        spaceAfter=6, textColor=HexColor("#1a1a2e"),
    ))
    styles.add(ParagraphStyle(
        "Body", parent=styles["Normal"], fontSize=10, leading=14,
        spaceAfter=8,
    ))
    styles.add(ParagraphStyle(
        "Caption", parent=styles["Normal"], fontSize=9, alignment=TA_CENTER,
        textColor=HexColor("#444444"), spaceAfter=12, spaceBefore=4,
    ))

    story = []
    avail_w = WIDTH - 2 * MARGIN

    # ── Title page ───────────────────────────────────────────────
    story.append(Spacer(1, 1.5 * inch))
    story.append(Paragraph("KTH-TIPS2 SPD Geometry Analysis", styles["ReportTitle"]))
    story.append(Paragraph("Summary of Results", styles["Subtitle"]))
    story.append(Spacer(1, 0.3 * inch))
    story.append(Paragraph(
        "This report summarizes the analysis of SPD (Symmetric Positive Definite) manifold "
        "geometries applied to image covariance descriptors from the KTH-TIPS2-b texture dataset. "
        "Three Riemannian metrics --Affine Invariant (AI), Log-Euclidean (LogE), and "
        "Bures-Wasserstein (BW) --are compared across visualization, feature preprocessing, "
        "and downstream classification tasks.",
        styles["Body"],
    ))
    story.append(Spacer(1, 0.15 * inch))
    story.append(Paragraph(
        "<b>Dataset:</b> KTH-TIPS2-b  -- 11 material texture classes, 22x22 covariance matrices "
        "(253-dimensional tangent features after projection). Subset used: <b>sample_a</b> (n ~ 1,188).",
        styles["Body"],
    ))
    story.append(Paragraph(
        "<b>Pipeline overview:</b> Raw images -> covariance matrices -> SPD geometry selection "
        "-> barycenter + tangent projection -> optional standardization -> "
        "optional dimension reduction (PCA / UMAP) -> classification.",
        styles["Body"],
    ))

    # ── Task 1 ───────────────────────────────────────────────────
    story.append(PageBreak())
    story.append(Paragraph("Task 1: Manifold UMAP vs Tangent-Space UMAP", styles["SectionHead"]))
    story.append(Paragraph(
        "Two UMAP embedding strategies are compared for each geometry: "
        "<b>Manifold UMAP</b> operates on precomputed pairwise SPD distances, while "
        "<b>Tangent UMAP</b> first projects matrices into tangent space and then applies "
        "standard Euclidean UMAP. Neighborhood overlap (k=15), trustworthiness, and "
        "Spearman rank correlation between distance matrices quantify agreement.",
        styles["Body"],
    ))

    # Task 1 figure
    fig1 = os.path.join(FIG_DIR, "task1_umap_grid.png")
    if os.path.exists(fig1):
        img = Image(fig1, width=avail_w, height=avail_w * 1.25)
        img.hAlign = "CENTER"
        story.append(img)
        story.append(Paragraph(
            "Figure 1: UMAP embeddings colored by material class. Left column: Manifold UMAP; "
            "Right column: Tangent UMAP. Rows: BW, AI, LogE (top to bottom).",
            styles["Caption"],
        ))

    # Task 1 table
    story.append(Paragraph("Quantitative Comparison Metrics", styles["SubHead"]))
    t1_data = load_csv(os.path.join(RES_DIR, "task1_metrics.csv"))
    header = ["Metric", "Neighborhood\nOverlap", "Trustworthiness\n(Manifold)", "Trustworthiness\n(Tangent)", "Spearman rho", "p-value"]
    rows = [header]
    for row in t1_data[1:]:
        rows.append([row[0], row[1], row[2], row[3], row[4], row[5]])

    t = Table(rows, hAlign="CENTER")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), HexColor("#16213e")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, HexColor("#f0f0f0")]),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(t)
    story.append(Spacer(1, 0.15 * inch))

    story.append(Paragraph("<b>Key findings:</b>", styles["Body"]))
    story.append(Paragraph(
        "• <b>LogE</b> shows the highest agreement between manifold and tangent embeddings "
        "(overlap = 0.939, Spearman rho = 0.996), indicating its tangent projection "
        "preserves manifold structure almost perfectly.",
        styles["Body"],
    ))
    story.append(Paragraph(
        "• <b>BW</b> shows strong agreement (overlap = 0.854, rho = 0.943), "
        "with both embeddings achieving comparable trustworthiness (~0.980).",
        styles["Body"],
    ))
    story.append(Paragraph(
        "• <b>AI</b> has markedly lower neighborhood overlap (0.478) and Spearman correlation (0.712), "
        "suggesting that tangent projection under the AI metric distorts local neighborhood structure "
        "more substantially than the other geometries.",
        styles["Body"],
    ))
    story.append(Paragraph(
        "• All six embeddings achieve high trustworthiness (>= 0.979), confirming UMAP "
        "faithfully preserves local neighborhoods regardless of input representation.",
        styles["Body"],
    ))

    # ── Task 2 ───────────────────────────────────────────────────
    story.append(PageBreak())
    story.append(Paragraph("Task 2: Raw vs Standardized Tangent Features", styles["SectionHead"]))
    story.append(Paragraph(
        "This task investigates the effect of z-score standardization on tangent-space features "
        "across three questions: (Q1) how feature variance distributions change, "
        "(Q2) how PCA explained variance is affected, and (Q3) whether standardization "
        "improves classification accuracy.",
        styles["Body"],
    ))

    # Q1 figure
    story.append(Paragraph("Q1: Feature Standard Deviations", styles["SubHead"]))
    fig2a = os.path.join(FIG_DIR, "task2_feature_stds_boxplot.png")
    if os.path.exists(fig2a):
        img = Image(fig2a, width=avail_w * 0.85, height=avail_w * 0.42)
        img.hAlign = "CENTER"
        story.append(img)
        story.append(Paragraph(
            "Figure 2: Distribution of per-feature standard deviations before and after standardization (log scale).",
            styles["Caption"],
        ))

    story.append(Paragraph(
        "Raw feature variances span several orders of magnitude, especially for AI and BW. "
        "Standardization collapses all features to unit variance, equalizing their influence "
        "on distance-based methods.",
        styles["Body"],
    ))

    # Q2 figure
    story.append(Paragraph("Q2: PCA Cumulative Variance Explained", styles["SubHead"]))
    fig2b = os.path.join(FIG_DIR, "task2_pca_cve.png")
    if os.path.exists(fig2b):
        img = Image(fig2b, width=avail_w * 0.92, height=avail_w * 0.31)
        img.hAlign = "CENTER"
        story.append(img)
        story.append(Paragraph(
            "Figure 3: Cumulative variance explained by PCA components. Solid = raw, dashed = standardized.",
            styles["Caption"],
        ))

    story.append(Paragraph(
        "For raw features, a small number of PCA components capture most of the variance "
        "(dominated by high-variance features). After standardization, variance is spread more "
        "evenly, requiring more components to reach 95%. This means PCA on raw features is "
        "driven by scale rather than discriminative power.",
        styles["Body"],
    ))

    # Q3 figure + table
    story.append(Paragraph("Q3: Classification Accuracy (KNN, k=5)", styles["SubHead"]))
    fig2c = os.path.join(FIG_DIR, "task2_classification_barplot.png")
    if os.path.exists(fig2c):
        img = Image(fig2c, width=avail_w * 0.75, height=avail_w * 0.47)
        img.hAlign = "CENTER"
        story.append(img)
        story.append(Paragraph(
            "Figure 4: Mean 5-fold CV accuracy with error bars (KNN k=5), raw vs standardized.",
            styles["Caption"],
        ))

    t2_data = load_csv(os.path.join(RES_DIR, "task2_standardization_accuracy.csv"))
    header2 = ["Metric", "Standardized", "Accuracy (mean)", "Accuracy (std)"]
    rows2 = [header2]
    for row in t2_data[1:]:
        std_label = "Yes" if row[1] == "True" else "No"
        rows2.append([row[0], std_label, row[2], row[3]])

    t2 = Table(rows2, hAlign="CENTER")
    t2.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), HexColor("#16213e")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, HexColor("#f0f0f0")]),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(t2)
    story.append(Spacer(1, 0.15 * inch))

    story.append(Paragraph(
        "• Standardization improves AI accuracy substantially (95.5% -> 97.9%), "
        "consistent with AI’s wide raw variance spread.",
        styles["Body"],
    ))
    story.append(Paragraph(
        "• BW sees a modest gain (97.0% -> 97.4%).",
        styles["Body"],
    ))
    story.append(Paragraph(
        "• LogE is slightly hurt by standardization (99.5% -> 99.1%), suggesting its "
        "raw feature scales are already well-balanced for KNN.",
        styles["Body"],
    ))

    # ── Task 3 ───────────────────────────────────────────────────
    story.append(PageBreak())
    story.append(Paragraph("Task 3: Downstream Classification  -- Nested Cross-Validation", styles["SectionHead"]))
    story.append(Paragraph(
        "A rigorous nested CV design (5 outer / 3 inner folds) evaluates 54 combinations: "
        "3 geometries x 3 pipelines x 2 standardization settings x 3 classifiers. "
        "Pipeline A uses full 253-dim tangent features, Pipeline B applies PCA, and "
        "Pipeline C applies UMAP for dimension reduction. Latent dimensions q in {5, 10, 20} "
        "are tuned in the inner loop along with classifier hyperparameters.",
        styles["Body"],
    ))

    # Task 3 heatmap
    fig3 = os.path.join(FIG_DIR, "task3_accuracy_heatmap.png")
    if os.path.exists(fig3):
        img = Image(fig3, width=avail_w * 0.7, height=avail_w * 0.47)
        img.hAlign = "CENTER"
        story.append(img)
        story.append(Paragraph(
            "Figure 5: Best accuracy per geometry x pipeline (max over standardization, classifier, and q).",
            styles["Caption"],
        ))

    # Top 10 results table
    story.append(Paragraph("Top 10 Configurations by Accuracy", styles["SubHead"]))
    t3_data = load_csv(os.path.join(RES_DIR, "task3_classification_results.csv"))
    t3_rows = []
    for row in t3_data[1:]:
        t3_rows.append({
            "metric": row[0], "pipeline": row[1],
            "std": "Yes" if row[2] == "True" else "No",
            "clf": row[3], "acc": float(row[4]), "acc_std": float(row[5]),
            "prec": float(row[6]), "recall": float(row[7]),
            "auc": float(row[8]), "runtime": float(row[9]),
        })
    t3_rows.sort(key=lambda r: r["acc"], reverse=True)

    header3 = ["Metric", "Pipeline", "Std", "Classifier", "Accuracy", "± Std", "AUC"]
    top10 = [header3]
    for r in t3_rows[:10]:
        top10.append([
            r["metric"], r["pipeline"], r["std"], r["clf"],
            f"{r['acc']:.4f}", f"{r['acc_std']:.4f}", f"{r['auc']:.4f}",
        ])

    t3 = Table(top10, hAlign="CENTER")
    t3.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), HexColor("#16213e")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, HexColor("#f0f0f0")]),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(t3)
    story.append(Spacer(1, 0.15 * inch))

    story.append(Paragraph("<b>Key findings:</b>", styles["Body"]))
    story.append(Paragraph(
        "• <b>LogE dominates</b>: the top configurations are all LogE with Pipeline A "
        "(full features), achieving 99.8–99.9% accuracy and perfect AUC (1.000). "
        "LogE + standardized logistic regression and SVM both reach 99.9%.",
        styles["Body"],
    ))
    story.append(Paragraph(
        "• <b>Pipeline A (no dimension reduction) consistently outperforms B and C</b> across "
        "all geometries. The 253-dimensional tangent features contain sufficient discriminative "
        "information without needing reduction.",
        styles["Body"],
    ))
    story.append(Paragraph(
        "• <b>Pipeline C (UMAP reduction) consistently underperforms</b>, likely because "
        "UMAP’s nonlinear mapping discards information useful for supervised classification, "
        "despite being good for visualization.",
        styles["Body"],
    ))
    story.append(Paragraph(
        "• <b>SVM and logistic regression outperform KNN</b> across all settings, with SVM "
        "generally achieving the highest accuracy per geometry-pipeline combination.",
        styles["Body"],
    ))
    story.append(Paragraph(
        "• <b>Runtime</b>: LogE is dramatically faster than BW and AI (e.g., 4s vs 650s per "
        "Pipeline A combination), making it the most practical choice in addition to being "
        "the most accurate.",
        styles["Body"],
    ))

    # ── Conclusions ──────────────────────────────────────────────
    story.append(Spacer(1, 0.25 * inch))
    story.append(Paragraph("Conclusions", styles["SectionHead"]))
    story.append(Paragraph(
        "1. <b>Log-Euclidean geometry is the clear winner</b> for this dataset, achieving the "
        "highest classification accuracy (99.9%), the best manifold–tangent agreement "
        "(Task 1), and the fastest computation time. Its tangent projection preserves "
        "virtually all discriminative structure from the manifold.",
        styles["Body"],
    ))
    story.append(Paragraph(
        "2. <b>Tangent projection is a reliable proxy for manifold distances</b>, especially "
        "under LogE and BW geometries. The AI metric shows more distortion in tangent space, "
        "which may explain its slightly lower classification performance.",
        styles["Body"],
    ))
    story.append(Paragraph(
        "3. <b>Standardization helps when raw feature scales are imbalanced</b> (AI benefits most), "
        "but can slightly hurt when scales are already balanced (LogE). The choice should be "
        "geometry-dependent.",
        styles["Body"],
    ))
    story.append(Paragraph(
        "4. <b>Dimension reduction is unnecessary for this feature set.</b> Full 253-dim tangent "
        "features with a simple SVM or logistic regression achieve near-perfect classification. "
        "PCA and UMAP reduction reduce accuracy, with UMAP showing the largest drop.",
        styles["Body"],
    ))

    doc.build(story)
    print(f"Report saved to: {os.path.abspath(OUTPUT)}")


if __name__ == "__main__":
    build_report()
