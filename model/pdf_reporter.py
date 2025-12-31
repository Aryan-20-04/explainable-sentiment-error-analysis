import os
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from textwrap import wrap
from datetime import datetime

def generate_error_report(
    index: int,
    text: str,
    true_label: int,
    pred_label: int,
    confidence: dict,
    error_type: str,
    output_dir: str = "reports"
):
    """
    Generates a standalone PDF report for a single misclassification.

    Parameters:
    - index: sample index
    - text: review text
    - true_label: 0 or 1
    - pred_label: 0 or 1
    - confidence: dict with POSITIVE / NEGATIVE scores
    - error_type: categorized error string
    - output_dir: folder to save PDFs
    """

    os.makedirs(output_dir, exist_ok=True)

    filename = os.path.join(output_dir, f"error_report_{index}.pdf")
    c = canvas.Canvas(filename, pagesize=A4)
    width, height = A4

    y = height - 40

    # Title
    c.setFont("Helvetica-Bold", 15)
    c.drawString(40, y, "Transformer Sentiment Error Analysis Report")

    y -= 30
    c.setFont("Helvetica", 10)
    c.drawString(40, y, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Metadata
    y -= 30
    c.setFont("Helvetica-Bold", 11)
    c.drawString(40, y, "Prediction Summary")

    y -= 18
    c.setFont("Helvetica", 10)
    c.drawString(40, y, f"True Label      : {'POSITIVE' if true_label else 'NEGATIVE'}")
    y -= 14
    c.drawString(40, y, f"Predicted Label : {'POSITIVE' if pred_label else 'NEGATIVE'}")
    y -= 14
    c.drawString(40, y, f"Confidence      : {confidence}")
    y -= 14
    c.drawString(40, y, f"Error Category  : {error_type}")

    # Review text
    y -= 30
    c.setFont("Helvetica-Bold", 11)
    c.drawString(40, y, "Review Text")

    y -= 18
    c.setFont("Helvetica", 9)

    for line in wrap(text, 95):
        if y < 40:
            c.showPage()
            y = height - 40
            c.setFont("Helvetica", 9)
        c.drawString(40, y, line)
        y -= 12

    # Footer
    y -= 20
    c.setFont("Helvetica-Oblique", 8)
    c.drawString(
        40,
        y,
        "This report explains why the model made an incorrect prediction using explainable AI techniques."
    )

    c.save()
    return filename
