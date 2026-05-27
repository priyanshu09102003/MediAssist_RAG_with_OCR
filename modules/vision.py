import base64
import io
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import fitz                          
from PIL import Image
import pytesseract

from google import genai
from google.genai import types

import config


# ── Configure Gemini client 

_client = genai.Client(api_key=config.GEMINI_API_KEY)


# ── Result dataclasses 

@dataclass
class ImageAnalysisResult:
    description: str              # what Gemini sees in the image
    visible_symptoms: list[str] = field(default_factory=list)
    affected_area: str = ""
    severity_hint: str = "mild"   # visual severity estimate
    possible_conditions: list[str] = field(default_factory=list)
    recommendations: str = ""
    needs_urgent_care: bool = False
    raw_response: str = ""


@dataclass
class LabReportResult:
    extracted_text: str = ""
    parameters: list[dict] = field(default_factory=list)
    # [{"name": "Hemoglobin", "value": "9.2", "unit": "g/dL",
    #   "reference": "12-16", "status": "LOW", "flag": True}]
    abnormal_flags: list[str] = field(default_factory=list)
    ai_summary: str = ""
    report_type: str = ""          # "Blood Test" | "Urine" | "X-Ray" | etc.
    critical_values: list[str] = field(default_factory=list)


# ── Image helpers 

ALLOWED_MIME_TYPES = {
    ".jpg":  "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png":  "image/png",
    ".webp": "image/webp",
    ".gif":  "image/gif",
}

MAX_IMAGE_SIZE_MB  = 10
MAX_DIMENSION_PX   = 1600  


def validate_and_preprocess(
    image_bytes: bytes,
    mime_type: str = "image/jpeg",
    max_dimension: int = MAX_DIMENSION_PX,
) -> tuple[bytes, str]:
    """
    Validate image size and resize if needed.
    Returns (processed_bytes, mime_type).
    Raises ValueError for invalid images.
    """
    if len(image_bytes) > MAX_IMAGE_SIZE_MB * 1024 * 1024:
        raise ValueError(f"Image too large (max {MAX_IMAGE_SIZE_MB} MB)")

    try:
        img = Image.open(io.BytesIO(image_bytes))
    except Exception:
        raise ValueError("Invalid or corrupted image file")

    # Convert RGBA / palette to RGB
    if img.mode in ("RGBA", "P", "LA"):
        img = img.convert("RGB")
        mime_type = "image/jpeg"

    # Resize if too large
    w, h = img.size
    if max(w, h) > max_dimension:
        ratio = max_dimension / max(w, h)
        img = img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)

    buf = io.BytesIO()
    fmt = "JPEG" if "jpeg" in mime_type else "PNG"
    img.save(buf, format=fmt, quality=85)
    return buf.getvalue(), mime_type


def image_to_base64(image_bytes: bytes) -> str:
    return base64.b64encode(image_bytes).decode("utf-8")


def pdf_to_images(pdf_bytes: bytes) -> list[bytes]:
    """Convert PDF pages to JPEG image bytes list (for lab report PDFs)."""
    images = []
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        for page in doc:
            pix = page.get_pixmap(dpi=200)
            images.append(pix.tobytes("jpeg"))
        doc.close()
    except Exception as e:
        print(f"  ⚠️  PDF to image error: {e}")
    return images


# ── Patient Image Analysis 

VISION_PROMPT = """You are a clinical AI assistant analyzing a patient-submitted medical image.
The patient has described their concern as: "{user_query}"

Please analyze this image and provide:

1. DESCRIPTION: What do you observe in the image? (visible symptoms, affected area, appearance)
2. AFFECTED AREA: Which body part or region is shown?
3. VISIBLE SYMPTOMS: List specific observable signs (redness, swelling, rash pattern, wound type, discharge, etc.)
4. POSSIBLE CONDITIONS: Based on visual appearance only, what conditions might this suggest? (2-4 possibilities)
5. SEVERITY VISUAL ESTIMATE: MILD | MODERATE | SEVERE | EMERGENCY
6. RECOMMENDATIONS: What immediate actions should the patient take?
7. URGENT CARE NEEDED: YES or NO

Important notes:
- Be specific and clinical in your observation
- State clearly that visual assessment is NOT a diagnosis
- Always recommend professional medical evaluation
- If the image shows a medical emergency (severe wound, signs of stroke, etc.), say URGENT CARE: YES immediately

Format your response as:
DESCRIPTION: ...
AFFECTED_AREA: ...
VISIBLE_SYMPTOMS: symptom1 | symptom2 | symptom3
POSSIBLE_CONDITIONS: condition1 | condition2 | condition3
SEVERITY: MILD/MODERATE/SEVERE/EMERGENCY
RECOMMENDATIONS: ...
URGENT_CARE: YES/NO
"""


def analyze_patient_image(
    image_bytes: bytes,
    mime_type: str,
    user_query: str = "Please analyze my condition",
) -> ImageAnalysisResult:
    
    try:
        # Preprocess
        processed_bytes, mime_type = validate_and_preprocess(image_bytes, mime_type)

        # Build prompt
        prompt = VISION_PROMPT.format(user_query=user_query)

        # Call Gemini Vision
        response = _client.models.generate_content(
            model=config.GEMINI_VISION_MODEL,
            contents=[
                types.Part.from_bytes(data=processed_bytes, mime_type=mime_type),
                types.Part.from_text(text=prompt),
            ],
        )

        raw = response.text or ""
        return _parse_vision_response(raw)

    except Exception as e:
        return ImageAnalysisResult(
            description=f"Image analysis failed: {str(e)}",
            raw_response=str(e),
        )


def _parse_vision_response(raw: str) -> ImageAnalysisResult:
    

    def extract(label: str) -> str:
        match = re.search(rf"{label}:\s*(.+?)(?=\n[A-Z_]+:|$)", raw,
                          re.IGNORECASE | re.DOTALL)
        return match.group(1).strip() if match else ""

    description  = extract("DESCRIPTION")
    affected     = extract("AFFECTED_AREA")
    symptoms_raw = extract("VISIBLE_SYMPTOMS")
    conditions_r = extract("POSSIBLE_CONDITIONS")
    severity_raw = extract("SEVERITY")
    recommend    = extract("RECOMMENDATIONS")
    urgent_raw   = extract("URGENT_CARE")

    symptoms   = [s.strip() for s in symptoms_raw.split("|") if s.strip()]
    conditions = [c.strip() for c in conditions_r.split("|") if c.strip()]

    severity_map = {
        "EMERGENCY": "emergency",
        "SEVERE":    "severe",
        "MODERATE":  "moderate",
        "MILD":      "mild",
    }
    severity = "mild"
    for key, val in severity_map.items():
        if key in severity_raw.upper():
            severity = val
            break

    urgent = "YES" in urgent_raw.upper()

    return ImageAnalysisResult(
        description=description or raw[:500],
        visible_symptoms=symptoms,
        affected_area=affected,
        severity_hint=severity,
        possible_conditions=conditions,
        recommendations=recommend,
        needs_urgent_care=urgent,
        raw_response=raw,
    )


def image_result_to_text(result: ImageAnalysisResult) -> str:
    
    parts = [f"Visual Assessment of Patient Image:"]
    if result.description:
        parts.append(f"Observation: {result.description}")
    if result.affected_area:
        parts.append(f"Affected Area: {result.affected_area}")
    if result.visible_symptoms:
        parts.append(f"Visible Symptoms: {', '.join(result.visible_symptoms)}")
    if result.possible_conditions:
        parts.append(f"Possible Conditions (visual only): {', '.join(result.possible_conditions)}")
    parts.append(f"Visual Severity Estimate: {result.severity_hint.upper()}")
    if result.recommendations:
        parts.append(f"Recommendations: {result.recommendations}")
    if result.needs_urgent_care:
        parts.append("⚠️ URGENT CARE INDICATED based on visual assessment.")
    return "\n".join(parts)


# ── Lab Report OCR + Analysis 

LAB_REPORT_PROMPT = """You are a medical lab report analyzer. 
Analyze this lab report and extract all information in a structured format.

Extract:
1. REPORT_TYPE: What type of report is this? (Complete Blood Count, Liver Function Test, etc.)
2. PARAMETERS: For each test parameter, extract name, value, unit, reference range, and if it's normal/high/low
3. ABNORMAL: List all parameters that are outside normal range
4. CRITICAL: List any critically abnormal values that need immediate attention
5. SUMMARY: Plain-language explanation of what these results mean for the patient

Format as JSON:
{{
  "report_type": "...",
  "parameters": [
    {{"name": "Hemoglobin", "value": "9.2", "unit": "g/dL", "reference": "12.0-16.0", "status": "LOW", "flag": true}},
    ...
  ],
  "abnormal_flags": ["Hemoglobin is LOW at 9.2 g/dL (normal: 12-16)", ...],
  "critical_values": ["..."],
  "summary": "Plain English summary of the report for the patient..."
}}

Respond ONLY with the JSON. No other text.
"""


def analyze_lab_report(
    file_bytes: bytes,
    file_extension: str,
) -> LabReportResult:
    """
    Analyze a lab report file (PDF, image) using Gemini Vision + OCR.

    Args:
        file_bytes     : raw file bytes
        file_extension : '.pdf', '.jpg', '.png', etc.

    Returns:
        LabReportResult with extracted parameters and AI summary
    """
    ext = file_extension.lower()

    # Get image bytes to send to Gemini
    if ext == ".pdf":
        page_images = pdf_to_images(file_bytes)
        if not page_images:
            return LabReportResult(ai_summary="Could not extract pages from PDF.")
        # Use first 3 pages (most lab reports fit in 1-2 pages)
        images_to_analyze = page_images[:3]
    elif ext in ALLOWED_MIME_TYPES:
        images_to_analyze = [file_bytes]
    else:
        return LabReportResult(ai_summary=f"Unsupported file type: {ext}")

    # Fallback OCR text using pytesseract
    ocr_text = _extract_ocr_text(images_to_analyze)

    # Send to Gemini Vision for structured extraction
    combined_result = _gemini_analyze_lab(images_to_analyze, ocr_text)
    return combined_result


def _extract_ocr_text(image_list: list[bytes]) -> str:
    """Extract raw text from images using pytesseract as fallback."""
    texts = []
    for img_bytes in image_list:
        try:
            img = Image.open(io.BytesIO(img_bytes))
            text = pytesseract.image_to_string(img, lang="eng")
            texts.append(text.strip())
        except Exception:
            pass
    return "\n\n".join(texts)


def _gemini_analyze_lab(
    image_list: list[bytes],
    ocr_fallback: str = "",
) -> LabReportResult:
    """Send lab report images to Gemini for structured extraction."""
    try:
        contents = []

        # Add all page images
        for img_bytes in image_list:
            try:
                proc_bytes, _ = validate_and_preprocess(img_bytes, "image/jpeg")
                contents.append(
                    types.Part.from_bytes(data=proc_bytes, mime_type="image/jpeg")
                )
            except Exception:
                pass

        # Add prompt + OCR fallback text
        prompt_text = LAB_REPORT_PROMPT
        if ocr_fallback:
            prompt_text += f"\n\nAdditional OCR text extracted:\n{ocr_fallback[:2000]}"

        contents.append(types.Part.from_text(text=prompt_text))

        response = _client.models.generate_content(
            model=config.GEMINI_VISION_MODEL,
            contents=contents,
        )

        raw = response.text or ""
        # Strip markdown code fences if present
        raw = re.sub(r"```json|```", "", raw).strip()

        data = json.loads(raw)
        return LabReportResult(
            extracted_text=ocr_fallback,
            parameters=data.get("parameters", []),
            abnormal_flags=data.get("abnormal_flags", []),
            ai_summary=data.get("summary", ""),
            report_type=data.get("report_type", "Lab Report"),
            critical_values=data.get("critical_values", []),
        )

    except json.JSONDecodeError:
        # Gemini returned non-JSON — use raw response as summary
        return LabReportResult(
            extracted_text=ocr_fallback,
            ai_summary=response.text if 'response' in dir() else "Analysis failed.",
        )
    except Exception as e:
        return LabReportResult(
            extracted_text=ocr_fallback,
            ai_summary=f"Lab report analysis failed: {str(e)}",
        )


# ── Testing 
if __name__ == "__main__":
    print("Vision module loaded successfully.")
    print("Functions available:")
    print("  • analyze_patient_image(image_bytes, mime_type, user_query)")
    print("  • analyze_lab_report(file_bytes, file_extension)")
    print("  • image_result_to_text(result)")
    print("\nNote: Full test requires an actual image file.")
    print("These functions will be called from the Streamlit UI.")