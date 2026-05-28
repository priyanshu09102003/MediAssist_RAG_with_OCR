import io
from datetime import datetime
from typing import Optional

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm, mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table,
    TableStyle, HRFlowable, KeepTogether,
)

import config


# ── Color palette ─────────────────────────────────────────────────────────────

PRIMARY_COLOR   = colors.HexColor("#1a6b4a")    # deep medical green
ACCENT_COLOR    = colors.HexColor("#2ecc71")    # lighter green
DANGER_COLOR    = colors.HexColor("#e74c3c")    # red for warnings
LIGHT_BG        = colors.HexColor("#f0faf5")    # pale green background
HEADER_BG       = colors.HexColor("#1a6b4a")
TABLE_HEADER_BG = colors.HexColor("#2e7d52")
TABLE_ALT_ROW   = colors.HexColor("#eaf6ef")
BORDER_COLOR    = colors.HexColor("#a8d5b8")
DARK_TEXT       = colors.HexColor("#1a1a2e")
MUTED_TEXT      = colors.HexColor("#6b7280")


# ── Style definitions ─────────────────────────────────────────────────────────

def _build_styles():
    base = getSampleStyleSheet()

    styles = {
        "clinic_name": ParagraphStyle(
            "clinic_name",
            fontName="Helvetica-Bold",
            fontSize=20,
            textColor=colors.white,
            alignment=TA_CENTER,
            spaceAfter=2,
        ),
        "clinic_tagline": ParagraphStyle(
            "clinic_tagline",
            fontName="Helvetica",
            fontSize=9,
            textColor=colors.HexColor("#c8f0da"),
            alignment=TA_CENTER,
        ),
        "section_header": ParagraphStyle(
            "section_header",
            fontName="Helvetica-Bold",
            fontSize=11,
            textColor=PRIMARY_COLOR,
            spaceBefore=10,
            spaceAfter=4,
        ),
        "field_label": ParagraphStyle(
            "field_label",
            fontName="Helvetica-Bold",
            fontSize=9,
            textColor=MUTED_TEXT,
        ),
        "field_value": ParagraphStyle(
            "field_value",
            fontName="Helvetica",
            fontSize=10,
            textColor=DARK_TEXT,
        ),
        "diagnosis": ParagraphStyle(
            "diagnosis",
            fontName="Helvetica-Bold",
            fontSize=11,
            textColor=PRIMARY_COLOR,
            spaceBefore=4,
            spaceAfter=4,
        ),
        "rx_symbol": ParagraphStyle(
            "rx_symbol",
            fontName="Helvetica-Bold",
            fontSize=28,
            textColor=PRIMARY_COLOR,
        ),
        "med_name": ParagraphStyle(
            "med_name",
            fontName="Helvetica-Bold",
            fontSize=10,
            textColor=DARK_TEXT,
        ),
        "med_detail": ParagraphStyle(
            "med_detail",
            fontName="Helvetica",
            fontSize=9,
            textColor=MUTED_TEXT,
        ),
        "advice_text": ParagraphStyle(
            "advice_text",
            fontName="Helvetica",
            fontSize=10,
            textColor=DARK_TEXT,
            leading=16,
            spaceBefore=4,
        ),
        "disclaimer": ParagraphStyle(
            "disclaimer",
            fontName="Helvetica-Oblique",
            fontSize=8,
            textColor=MUTED_TEXT,
            alignment=TA_CENTER,
            leading=12,
        ),
        "footer": ParagraphStyle(
            "footer",
            fontName="Helvetica",
            fontSize=8,
            textColor=MUTED_TEXT,
            alignment=TA_CENTER,
        ),
    }
    return styles


# ── Main PDF generator 

def generate_prescription_pdf(
    patient_name: str,
    patient_age: int,
    patient_gender: str,
    diagnosis: str,
    medications: list[dict],
    advice: str = "",
    follow_up: str = "",
    session_id: int = 0,
    blood_group: str = "",
    allergies: list[str] = None,
    doctor_note: str = "",
) -> bytes:

    buf    = io.BytesIO()
    doc    = SimpleDocTemplate(
        buf,
        pagesize=A4,
        topMargin=1.5 * cm,
        bottomMargin=1.5 * cm,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
    )
    styles  = _build_styles()
    story   = []
    W       = A4[0] - 4 * cm   # usable width

    date_str  = datetime.now().strftime("%d %B %Y")
    time_str  = datetime.now().strftime("%I:%M %p")
    ref_no    = f"MA-{session_id:05d}-{datetime.now().strftime('%Y%m%d')}"

    # ── Header 
    header_data = [[
        Paragraph(config.PRESCRIPTION_CLINIC_NAME, styles["clinic_name"]),
    ]]
    header_table = Table(header_data, colWidths=[W])
    header_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), HEADER_BG),
        ("TOPPADDING",    (0, 0), (-1, -1), 14),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
        ("LEFTPADDING",   (0, 0), (-1, -1), 16),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 16),
        ("ROUNDEDCORNERS", [6]),
    ]))
    story.append(header_table)
    story.append(Spacer(1, 4 * mm))

    # Subtitle bar
    sub_data = [[
        Paragraph("AI-Assisted Medical Consultation Summary", ParagraphStyle(
            "sub", fontName="Helvetica-Oblique", fontSize=9,
            textColor=PRIMARY_COLOR, alignment=TA_LEFT)),
        Paragraph(f"Ref: {ref_no}", ParagraphStyle(
            "ref", fontName="Helvetica", fontSize=8,
            textColor=MUTED_TEXT, alignment=TA_RIGHT)),
    ]]
    sub_table = Table(sub_data, colWidths=[W * 0.6, W * 0.4])
    sub_table.setStyle(TableStyle([
        ("LINEBELOW", (0, 0), (-1, -1), 0.5, BORDER_COLOR),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(sub_table)
    story.append(Spacer(1, 4 * mm))

    # ── Patient info + Date side by side ──────────────────────────────────────
    allergies_text = ", ".join(allergies) if allergies else "None known"

    patient_info = [
        [Paragraph("PATIENT DETAILS", styles["field_label"]),
         Paragraph("DATE & TIME", styles["field_label"])],
        [Paragraph(f"<b>{patient_name}</b>", styles["field_value"]),
         Paragraph(date_str, styles["field_value"])],
        [Paragraph(
            f"Age: {patient_age} yrs   Gender: {patient_gender.title()}   "
            f"Blood Group: {blood_group or 'N/A'}",
            styles["field_value"]),
         Paragraph(time_str, styles["field_value"])],
        [Paragraph(f"Allergies: {allergies_text}",
                   ParagraphStyle("allergy", fontName="Helvetica", fontSize=9,
                                  textColor=DANGER_COLOR if allergies else MUTED_TEXT)),
         Paragraph("", styles["field_value"])],
    ]

    info_table = Table(patient_info, colWidths=[W * 0.65, W * 0.35])
    info_table.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), LIGHT_BG),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING",   (0, 0), (-1, -1), 10),
        ("LINEBELOW",     (0, -1), (-1, -1), 0.5, BORDER_COLOR),
    ]))
    story.append(info_table)
    story.append(Spacer(1, 5 * mm))

    # ── Diagnosis ──────────────────────────────────────────────────────────────
    story.append(Paragraph("DIAGNOSIS", styles["field_label"]))
    story.append(Spacer(1, 1 * mm))

    diag_table = Table(
        [[Paragraph(diagnosis or "See consultation notes", styles["diagnosis"])]],
        colWidths=[W],
    )
    diag_table.setStyle(TableStyle([
        ("BACKGROUND",  (0, 0), (-1, -1), LIGHT_BG),
        ("LINERIGHT",   (0, 0), (0, 0),  3, ACCENT_COLOR),
        ("LEFTPADDING", (0, 0), (-1, -1), 12),
        ("TOPPADDING",  (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(diag_table)
    story.append(Spacer(1, 5 * mm))

    # ── Rx Medications ────────────────────────────────────────────────────────
    story.append(Paragraph("℞", styles["rx_symbol"]))
    story.append(Spacer(1, 2 * mm))

    if medications:
        # Table header
        med_header = [
            Paragraph("MEDICATION", ParagraphStyle(
                "mh", fontName="Helvetica-Bold", fontSize=9, textColor=colors.white)),
            Paragraph("DOSE", ParagraphStyle(
                "mh", fontName="Helvetica-Bold", fontSize=9, textColor=colors.white)),
            Paragraph("FREQUENCY", ParagraphStyle(
                "mh", fontName="Helvetica-Bold", fontSize=9, textColor=colors.white)),
            Paragraph("DURATION", ParagraphStyle(
                "mh", fontName="Helvetica-Bold", fontSize=9, textColor=colors.white)),
            Paragraph("NOTES", ParagraphStyle(
                "mh", fontName="Helvetica-Bold", fontSize=9, textColor=colors.white)),
        ]

        med_rows = [med_header]
        for i, med in enumerate(medications):
            row_bg = TABLE_ALT_ROW if i % 2 == 0 else colors.white
            row = [
                Paragraph(med.get("name", ""),      styles["med_name"]),
                Paragraph(med.get("dose", ""),      styles["med_detail"]),
                Paragraph(med.get("frequency", ""), styles["med_detail"]),
                Paragraph(med.get("duration", ""),  styles["med_detail"]),
                Paragraph(med.get("notes", ""),     styles["med_detail"]),
            ]
            med_rows.append(row)

        col_widths = [W * 0.30, W * 0.12, W * 0.20, W * 0.15, W * 0.23]
        med_table = Table(med_rows, colWidths=col_widths, repeatRows=1)
        med_table.setStyle(TableStyle([
            # Header
            ("BACKGROUND",    (0, 0), (-1, 0),  TABLE_HEADER_BG),
            ("TOPPADDING",    (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("LEFTPADDING",   (0, 0), (-1, -1), 8),
            ("RIGHTPADDING",  (0, 0), (-1, -1), 8),
            # Grid
            ("GRID",          (0, 0), (-1, -1), 0.25, BORDER_COLOR),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [TABLE_ALT_ROW, colors.white]),
            ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ]))
        story.append(med_table)
    else:
        story.append(Paragraph("No medications prescribed in this session.",
                                styles["advice_text"]))

    story.append(Spacer(1, 6 * mm))

    # ── Advice ────────────────────────────────────────────────────────────────
    if advice:
        story.append(HRFlowable(width=W, thickness=0.5, color=BORDER_COLOR))
        story.append(Spacer(1, 3 * mm))
        story.append(Paragraph("ADVICE & INSTRUCTIONS", styles["field_label"]))
        story.append(Spacer(1, 2 * mm))
        story.append(Paragraph(advice, styles["advice_text"]))
        story.append(Spacer(1, 4 * mm))

    # ── Follow-up ─────────────────────────────────────────────────────────────
    if follow_up:
        story.append(Paragraph("FOLLOW-UP", styles["field_label"]))
        story.append(Spacer(1, 2 * mm))
        fu_table = Table(
            [[Paragraph(follow_up, styles["advice_text"])]],
            colWidths=[W],
        )
        fu_table.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (-1, -1), colors.HexColor("#fff8e6")),
            ("LEFTPADDING",   (0, 0), (-1, -1), 10),
            ("TOPPADDING",    (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ("LINELEFT",      (0, 0), (0, 0), 3, colors.HexColor("#f39c12")),
        ]))
        story.append(fu_table)
        story.append(Spacer(1, 5 * mm))

    # ── Doctor note ───────────────────────────────────────────────────────────
    if doctor_note:
        story.append(Paragraph("ADDITIONAL NOTES", styles["field_label"]))
        story.append(Spacer(1, 2 * mm))
        story.append(Paragraph(doctor_note, styles["advice_text"]))
        story.append(Spacer(1, 5 * mm))

    # ── Signature area ────────────────────────────────────────────────────────
    story.append(Spacer(1, 8 * mm))
    sig_data = [[
        Paragraph("Patient's Signature: ___________________",
                  ParagraphStyle("sig", fontName="Helvetica", fontSize=9, textColor=MUTED_TEXT)),
        Paragraph(
            f"<b>{config.PRESCRIPTION_CLINIC_NAME}</b><br/>"
            f"AI Medical Assistant<br/>"
            f"<font size='8' color='grey'>Generated: {date_str} {time_str}</font>",
            ParagraphStyle("sig2", fontName="Helvetica", fontSize=9,
                           textColor=DARK_TEXT, alignment=TA_RIGHT)),
    ]]
    sig_table = Table(sig_data, colWidths=[W * 0.5, W * 0.5])
    sig_table.setStyle(TableStyle([
        ("LINEABOVE", (0, 0), (-1, -1), 0.5, BORDER_COLOR),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(sig_table)
    story.append(Spacer(1, 4 * mm))

    # ── Disclaimer ────────────────────────────────────────────────────────────
    story.append(HRFlowable(width=W, thickness=0.5, color=BORDER_COLOR))
    story.append(Spacer(1, 2 * mm))
    story.append(Paragraph(config.PRESCRIPTION_DISCLAIMER, styles["disclaimer"]))
    story.append(Spacer(1, 2 * mm))
    story.append(Paragraph(
        f"MediAssist AI  •  Ref: {ref_no}  •  {date_str}",
        styles["footer"],
    ))

    # ── Build PDF
    doc.build(story)
    buf.seek(0)
    return buf.read()



if __name__ == "__main__":
    print("Testing PDF generator...")

    pdf_bytes = generate_prescription_pdf(
        patient_name="Priya Sharma",
        patient_age=32,
        patient_gender="female",
        diagnosis="Viral Fever with Tension Headache",
        medications=[
            {"name": "Paracetamol 500mg", "dose": "1 tablet",
             "frequency": "Every 6 hours", "duration": "3 days",
             "notes": "After food"},
            {"name": "ORS Sachet", "dose": "1 sachet in 1L water",
             "frequency": "Twice daily", "duration": "3 days",
             "notes": "Stay hydrated"},
            {"name": "Cetirizine 10mg", "dose": "1 tablet",
             "frequency": "Once at night", "duration": "3 days",
             "notes": "May cause drowsiness"},
        ],
        advice=(
            "Rest well. Drink plenty of fluids (water, coconut water, ORS). "
            "Avoid cold foods and drinks. Eat light meals. "
            "Sponge with lukewarm water if fever is above 102°F."
        ),
        follow_up="Visit a doctor if fever persists beyond 3 days or exceeds 103°F.",
        session_id=42,
        blood_group="B+",
        allergies=["penicillin"],
    )

    out_path = "data/test_prescription.pdf"
    with open(out_path, "wb") as f:
        f.write(pdf_bytes)

    print(f"  ✅ PDF generated: {out_path} ({len(pdf_bytes):,} bytes)")
    print("  Open the file to verify the layout looks correct.")