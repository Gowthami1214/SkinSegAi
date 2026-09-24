"""
SkinGenAI - Publication-Grade PDF Documentation Generator
Builds a complete, styled technical report using ReportLab Platypus.
"""

from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
)
from reportlab.pdfgen import canvas


class NumberedCanvas(canvas.Canvas):
    """Adds running headers and footers with dynamic total page count."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        if self._pageNumber == 1:
            return

        self.saveState()
        self.setFont("Helvetica-Bold", 8)
        self.setFillColor(colors.HexColor("#64748b"))

        # Running Header
        self.drawString(54, 755, "SkinSegAI — Few-Shot Skin Lesion Segmentation (MobileNetV3-UNet)")
        self.setStrokeColor(colors.HexColor("#e2e8f0"))
        self.setLineWidth(0.5)
        self.line(54, 747, 558, 747)

        # Running Footer
        self.setFont("Helvetica", 8)
        self.drawString(54, 38, "Confidential — Academic Research Prototype & Benchmark Report")
        page_str = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(558, 38, page_str)
        self.line(54, 50, 558, 50)

        self.restoreState()


def build_pdf(filename="SkinSegAI_Documentation.pdf"):
    pdf_path = Path(filename).resolve()
    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54,
    )

    PRIMARY = colors.HexColor("#0f172a")      # Deep Slate/Navy
    ACCENT = colors.HexColor("#0284c7")       # Medical Blue
    TEAL = colors.HexColor("#0d9488")         # Clinical Teal
    DARK_TEXT = colors.HexColor("#1e293b")    # Charcoal
    BG_CARD = colors.HexColor("#f8fafc")      # Off-white card
    BORDER = colors.HexColor("#cbd5e1")       # Subtle border

    # Typography Styles
    title_style = ParagraphStyle(
        "CoverTitle",
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        textColor=PRIMARY,
        spaceAfter=6,
    )
    subtitle_style = ParagraphStyle(
        "CoverSubtitle",
        fontName="Helvetica",
        fontSize=10.5,
        leading=14,
        textColor=ACCENT,
        spaceAfter=10,
    )
    h1_style = ParagraphStyle(
        "Heading1_Custom",
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=16,
        textColor=PRIMARY,
        spaceBefore=10,
        spaceAfter=6,
        keepWithNext=True,
    )
    h2_style = ParagraphStyle(
        "Heading2_Custom",
        fontName="Helvetica-Bold",
        fontSize=10,
        leading=14,
        textColor=ACCENT,
        spaceBefore=8,
        spaceAfter=4,
        keepWithNext=True,
    )
    body_style = ParagraphStyle(
        "Body_Custom",
        fontName="Helvetica",
        fontSize=8.5,
        leading=12,
        textColor=DARK_TEXT,
        spaceAfter=5,
    )
    bullet_style = ParagraphStyle(
        "Bullet_Custom",
        fontName="Helvetica",
        fontSize=8.5,
        leading=12,
        textColor=DARK_TEXT,
        leftIndent=12,
        spaceAfter=3,
    )
    callout_style = ParagraphStyle(
        "Callout_Custom",
        fontName="Helvetica-Oblique",
        fontSize=8,
        leading=11.5,
        textColor=PRIMARY,
    )
    table_cell = ParagraphStyle(
        "TableCell",
        fontName="Helvetica",
        fontSize=7.5,
        leading=10,
        textColor=DARK_TEXT,
    )
    table_cell_bold = ParagraphStyle(
        "TableCellBold",
        fontName="Helvetica-Bold",
        fontSize=7.5,
        leading=10,
        textColor=DARK_TEXT,
    )
    table_cell_header = ParagraphStyle(
        "TableCellHeader",
        fontName="Helvetica-Bold",
        fontSize=8,
        leading=10.5,
        textColor=colors.white,
    )

    story = []

    # Title Banner
    story.append(Paragraph("SkinSegAI: Technical Project Documentation", title_style))
    story.append(Paragraph("Few-Shot Medical Image Lesion Segmentation Using Transfer Learning with MobileNetV3-UNet", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=TEAL, spaceBefore=0, spaceAfter=8))

    # 1. Executive Summary Table
    story.append(Paragraph("1. Executive Summary & Project Specifications", h1_style))
    meta_data = [
        [Paragraph("Project Title", table_cell_bold), Paragraph("Few-Shot Medical Image Lesion Segmentation with Transfer Learning", table_cell)],
        [Paragraph("System Identifier", table_cell_bold), Paragraph("<b>SkinSegAI</b> (Interactive Clinical Research Dashboard)", table_cell)],
        [Paragraph("Model Architecture", table_cell_bold), Paragraph("MobileNetV3-Small (Encoder) + 4-Stage U-Net (Decoder with Skip Connections)", table_cell)],
        [Paragraph("Core Technologies", table_cell_bold), Paragraph("PyTorch 2.11, Torchvision 0.26, Streamlit 1.55, Plotly, NumPy, Pillow", table_cell)],
        [Paragraph("Dataset Benchmark", table_cell_bold), Paragraph("ISIC 2018 Task 1: Lesion Boundary Segmentation (Dermatoscopic RGB + Binary Masks)", table_cell)],
        [Paragraph("Evaluated Splits", table_cell_bold), Paragraph("<b>50, 100, 250, and 500 images</b> (Low-Data / Few-Shot Regimes)", table_cell)],
        [Paragraph("Best Validation Metrics", table_cell_bold), Paragraph("<b>Dice: 87.07% | IoU: 77.52%</b> (at 500 images, Epoch 9)", table_cell)],
        [Paragraph("Remote Repository", table_cell_bold), Paragraph("https://github.com/Gowthami1214/SkinSegAi", table_cell)],
    ]
    t_meta = Table(meta_data, colWidths=[120, 384])
    t_meta.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), BG_CARD),
        ('GRID', (0, 0), (-1, -1), 0.5, BORDER),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
    ]))
    story.append(t_meta)
    story.append(Spacer(1, 6))

    # 2. Clinical Background & Motivation
    story.append(Paragraph("2. Clinical Background & Research Motivation", h1_style))
    story.append(Paragraph(
        "Malignant melanoma represents the most fatal form of skin cancer, yet early surgical excision yields a 5-year survival rate exceeding 99%. "
        "Clinical assessment relies on dermoscopy and the <b>ABCD criteria</b> (Asymmetry, Border irregularity, Color variegation, Diameter). "
        "Automated lesion boundary segmentation is the foundational first step: without an exact border, computerized systems cannot compute asymmetry, "
        "border irregularity, or pigment network distribution.",
        body_style
    ))
    story.append(Paragraph(
        "<b>The Data Scarcity Dilemma:</b> Standard deep learning segmentation models require thousands of pixel-level annotations. "
        "Manual pixel contouring by certified dermatologists takes 3–8 minutes per image, creating an insurmountable bottleneck for rare conditions or resource-limited health systems. "
        "SkinGenAI investigates: <i>What is the minimum labeled training set needed to produce clinically viable segmentation masks using transfer learning?</i>",
        body_style
    ))
    story.append(Spacer(1, 6))

    # 3. Architecture & Pipeline
    story.append(Paragraph("3. Deep Learning Architecture: MobileNetV3-UNet", h1_style))
    story.append(Paragraph(
        "The model integrates a lightweight <b>MobileNetV3-Small</b> backbone (ImageNet-pretrained) with a symmetrical <b>U-Net decoder</b> "
        "via multi-scale skip connections. This design pairs high parameter efficiency (~1.5M parameters) with spatial boundary reconstruction.",
        body_style
    ))

    arch_data = [
        [Paragraph("Pipeline Stage", table_cell_header), Paragraph("Feature Dim", table_cell_header), Paragraph("Channels", table_cell_header), Paragraph("Architectural Role", table_cell_header)],
        [Paragraph("Input Dermoscopy", table_cell), Paragraph("224 × 224", table_cell), Paragraph("3 (RGB)", table_cell), Paragraph("Standardized normalized dermoscopic photo", table_cell)],
        [Paragraph("Encoder Stage 0 (Skip 4)", table_cell), Paragraph("112 × 112", table_cell), Paragraph("16", table_cell), Paragraph("Fine hair, edge, and contrast transitions", table_cell)],
        [Paragraph("Encoder Stage 1 (Skip 3)", table_cell), Paragraph("56 × 56", table_cell), Paragraph("16", table_cell), Paragraph("Local pigment networks and texture maps", table_cell)],
        [Paragraph("Encoder Stage 2 (Skip 2)", table_cell), Paragraph("28 × 28", table_cell), Paragraph("24", table_cell), Paragraph("Regional lesion symmetry features", table_cell)],
        [Paragraph("Encoder Stage 8 (Skip 1)", table_cell), Paragraph("14 × 14", table_cell), Paragraph("48", table_cell), Paragraph("Lesion global shape context", table_cell)],
        [Paragraph("Bottleneck Bridge", table_cell), Paragraph("7 × 7", table_cell), Paragraph("576", table_cell), Paragraph("Highest semantic abstraction", table_cell)],
        [Paragraph("Decoder UpConv 1–4", table_cell), Paragraph("7→14→28→56→112", table_cell), Paragraph("48, 24, 16, 16", table_cell), Paragraph("ConvTranspose2d upsampling + Skip fusion", table_cell)],
        [Paragraph("Final Projection", table_cell), Paragraph("224 × 224", table_cell), Paragraph("1 logit", table_cell), Paragraph("Sigmoid → Binary thresholding at θ=0.50", table_cell)],
    ]
    t_arch = Table(arch_data, colWidths=[105, 80, 65, 254])
    t_arch.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), PRIMARY),
        ('GRID', (0, 0), (-1, -1), 0.5, BORDER),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, BG_CARD]),
        ('TOPPADDING', (0, 0), (-1, -1), 2.5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2.5),
    ]))
    story.append(t_arch)
    story.append(Spacer(1, 6))

    # 4. Loss Functions & Metrics
    story.append(Paragraph("4. Optimization & Validation Metrics", h1_style))
    story.append(Paragraph(
        "<b>Hybrid Loss:</b> Dermoscopy images exhibit severe foreground/background class imbalance (lesions cover only 15–30% of canvas area). "
        "SkinGenAI trains with <b>L_total = L_BCE + L_Dice</b>, where BCE stabilizes per-pixel probability convergence and Dice Loss directly penalizes missed lesion areas.",
        body_style
    ))

    metric_data = [
        [Paragraph("Metric", table_cell_header), Paragraph("Formula", table_cell_header), Paragraph("Clinical Interpretation", table_cell_header)],
        [
            Paragraph("<b>Dice Similarity (DSC)</b>", table_cell),
            Paragraph("<b>2·TP / (2·TP + FP + FN)</b>", table_cell),
            Paragraph("Gold standard ISIC metric. Measures contour completeness and overlap harmonic mean.", table_cell)
        ],
        [
            Paragraph("<b>IoU (Jaccard Index)</b>", table_cell),
            Paragraph("<b>TP / (TP + FP + FN)</b>", table_cell),
            Paragraph("Strict overlap area divided by total union area. Mathematically equal to Dice / (2 - Dice).", table_cell)
        ],
        [
            Paragraph("<b>Precision</b>", table_cell),
            Paragraph("<b>TP / (TP + FP)</b>", table_cell),
            Paragraph("Measures over-segmentation. High precision ensures minimal healthy skin is excised.", table_cell)
        ],
        [
            Paragraph("<b>Recall (Sensitivity)</b>", table_cell),
            Paragraph("<b>TP / (TP + FN)</b>", table_cell),
            Paragraph("Critical oncology metric. High recall guarantees malignant tumor cells are not missed.", table_cell)
        ],
    ]
    t_metric = Table(metric_data, colWidths=[100, 120, 284])
    t_metric.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), TEAL),
        ('GRID', (0, 0), (-1, -1), 0.5, BORDER),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, BG_CARD]),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
    ]))
    story.append(t_metric)
    story.append(Spacer(1, 6))

    # 5. Benchmark Results
    story.append(Paragraph("5. Empirical Benchmark Results across Few-Shot Configurations", h1_style))
    bench_data = [
        [Paragraph("Configuration", table_cell_header), Paragraph("Training Samples", table_cell_header), Paragraph("Best Epoch", table_cell_header), Paragraph("Validation Dice", table_cell_header), Paragraph("Validation IoU", table_cell_header), Paragraph("Empirical Progression", table_cell_header)],
        [Paragraph("Config 1", table_cell_bold), Paragraph("50 images", table_cell), Paragraph("15", table_cell), Paragraph("<b>0.5104</b> (51.0%)", table_cell), Paragraph("0.3621 (36.2%)", table_cell), Paragraph("Baseline: Identifies dark centroids; rough borders", table_cell)],
        [Paragraph("Config 2", table_cell_bold), Paragraph("100 images", table_cell), Paragraph("15", table_cell), Paragraph("<b>0.5765</b> (57.6%)", table_cell), Paragraph("0.4106 (41.1%)", table_cell), Paragraph("+6.6% Gain: Learns radial symmetry & artifact suppression", table_cell)],
        [Paragraph("Config 3", table_cell_bold), Paragraph("250 images", table_cell), Paragraph("15", table_cell), Paragraph("<b>0.8246</b> (82.5%)", table_cell), Paragraph("0.7081 (70.8%)", table_cell), Paragraph("<b>Critical Inflection Point (+24.8% jump)</b>", table_cell)],
        [Paragraph("Config 4", table_cell_bold), Paragraph("500 images", table_cell), Paragraph("9", table_cell), Paragraph("<b>0.8707</b> (87.1%)", table_cell), Paragraph("0.7752 (77.5%)", table_cell), Paragraph("Near-supervised SOTA: Rapid convergence by epoch 9", table_cell)],
    ]
    t_bench = Table(bench_data, colWidths=[60, 75, 50, 85, 85, 149])
    t_bench.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), PRIMARY),
        ('GRID', (0, 0), (-1, -1), 0.5, BORDER),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, BG_CARD]),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
    ]))
    story.append(t_bench)
    story.append(Spacer(1, 6))

    # Callout
    callout_data = [[
        Paragraph(
            "<b>Key Research Finding:</b> An exponential performance jump occurs between <b>100 and 250 images</b> (Dice increases from 57.65% to 82.46%). "
            "At 250 images, the dataset provides sufficient pigment diversity for MobileNetV3's multi-scale skip connections to fully engage, "
            "proving that <b>250 samples represent the critical viability threshold</b> for clinical transfer learning in dermoscopy.",
            callout_style
        )
    ]]
    t_callout = Table(callout_data, colWidths=[504])
    t_callout.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#f0fdfa")),
        ('BOX', (0, 0), (-1, -1), 1, TEAL),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
    ]))
    story.append(t_callout)
    story.append(Spacer(1, 6))

    # 6. Streamlit Dashboard
    story.append(Paragraph("6. Interactive Streamlit Dashboard System", h1_style))
    dash_items = [
        "<b>Executive Dashboard:</b> Status badge, 4-step pipeline visualizer, live validation score cards, and color-coded Dice progress bars.",
        "<b>Experiment Lab:</b> Hyperparameter inspection, Plotly learning curves (Data Size vs Dice/IoU), and data insights.",
        "<b>Segmentation Studio:</b> Sample ISIC library or upload, dynamic threshold slider (θ: 0.10–0.90), 4-panel visualizer (Original, Heatmap, Mask, Overlay), mask export button, and side-by-side comparative panel across all 4 models.",
        "<b>Model Comparison:</b> Score cards, interactive <b>Dice Coefficient Matrix Heatmap</b>, side-by-side grouped bar charts, and radar fingerprints.",
        "<b>Test Evaluation:</b> Discovers image-mask pairs in sample_data/ to compute live Dice, IoU, Precision, and Recall on the fly.",
        "<b>About Project:</b> Technical methodology, architectural diagrams, and clinical disclaimers.",
    ]
    for item in dash_items:
        story.append(Paragraph(f"• {item}", bullet_style))

    story.append(Spacer(1, 6))

    # 7. Deployment & Clinical Disclaimer
    story.append(Paragraph("7. Cloud Deployment & Academic Disclaimer", h1_style))
    story.append(Paragraph(
        "<b>Continuous Cloud Deployment:</b> The codebase is deployed on <b>Streamlit Community Cloud</b> connected to "
        "GitHub repository <code>Gowthami1214/SkinGenAI</code> on branch <code>main</code>. "
        "Dependencies in <code>requirements.txt</code> utilize PyTorch's CPU extra-index-url, reducing cloud image build sizes from 2.5GB to ~170MB.",
        body_style
    ))
    story.append(Spacer(1, 3))

    disclaimer_data = [[
        Paragraph(
            "<b>Clinical Disclaimer:</b> SkinGenAI is an academic research prototype designed to study few-shot transfer learning under data scarcity. "
            "Predictions are not certified by the FDA, CE, or medical regulatory bodies, and must never be used for primary diagnostic purposes without a licensed dermatologist.",
            callout_style
        )
    ]]
    t_disclaimer = Table(disclaimer_data, colWidths=[504])
    t_disclaimer.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#fffbeb")),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor("#d97706")),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
    ]))
    story.append(t_disclaimer)

    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"PDF successfully generated at: {pdf_path}")
    return pdf_path


if __name__ == "__main__":
    build_pdf()
