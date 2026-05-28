import re
from dataclasses import dataclass, field
from typing import Optional

import config


# ── Result dataclass

@dataclass
class TriageResult:
    severity: str                        
    color: str   = "#2ecc71"
    icon: str    = "🟢"
    label: str   = "Mild"
    message: str = ""
    emergency_contacts: list[dict] = field(default_factory=list)
    specialist_hint: str = ""
    rule_triggered: str = ""             


# ── Emergency keyword sets

EMERGENCY_KEYWORDS = {
    # Cardiac
    "chest pain", "chest tightness", "heart attack", "cardiac arrest",
    "left arm pain", "jaw pain with chest",
    # Neurological
    "stroke", "face drooping", "sudden numbness", "sudden confusion",
    "sudden severe headache", "worst headache of my life",
    "can't speak", "cannot speak", "slurred speech",
    # Respiratory
    "can't breathe", "cannot breathe", "difficulty breathing",
    "choking", "stopped breathing", "respiratory arrest",
    # Bleeding
    "severe bleeding", "not stopping bleeding", "vomiting blood",
    "blood in stool", "coughing up blood",
    # Consciousness
    "unconscious", "fainted", "passed out", "unresponsive",
    "seizure", "convulsion",
    # Allergic
    "anaphylaxis", "throat swelling", "tongue swelling",
    "severe allergic",
    # Other emergencies
    "overdose", "poisoning", "suicide", "self harm",
    "attempted suicide", "want to die",
}

SEVERE_KEYWORDS = {
    "high fever", "fever above 103", "fever above 104",
    "severe chest pain", "severe abdominal pain",
    "difficulty swallowing", "stiff neck with fever",
    "confusion", "disoriented", "altered consciousness",
    "blood in urine", "blood in vomit",
    "severe dehydration", "not urinating", "no urine",
    "shortness of breath", "labored breathing",
    "severe headache", "sudden vision loss",
    "extreme fatigue", "unable to walk",
    "severe rash spreading fast",
}

MODERATE_KEYWORDS = {
    "fever", "persistent vomiting", "vomiting since",
    "diarrhea", "abdominal pain", "ear pain",
    "sore throat", "cough", "body ache", "joint pain",
    "rash", "skin irritation", "swelling",
    "headache", "dizziness", "nausea",
    "urinary pain", "burning urination",
    "back pain", "chest discomfort",
}

# ── Specialist hints per emergency type 

SPECIALIST_HINTS = {
    "cardiac":       "Cardiologist",
    "neurological":  "Neurologist",
    "respiratory":   "Pulmonologist",
    "dermatology":   "Dermatologist",
    "eyes":          "Ophthalmologist",
    "ears":          "ENT Specialist",
    "mental health": "Psychiatrist",
    "diabetes":      "Endocrinologist",
    "kidney":        "Nephrologist",
    "stomach":       "Gastroenterologist",
    "bone":          "Orthopedic Surgeon",
    "children":      "Pediatrician",
    "women":         "Gynecologist",
    "cancer":        "Oncologist",
}



SEVERITY_CONFIG = {
    "emergency": {
        "color": "#e74c3c", "icon": "🔴", "label": "EMERGENCY",
        "message": (
            "⚠️ **This appears to be a medical emergency.**\n"
            "Please call **112** or go to the nearest Emergency Room **immediately**.\n"
            "Do not wait. Do not drive yourself if possible."
        ),
    },
    "severe": {
        "color": "#e67e22", "icon": "🟠", "label": "Severe",
        "message": (
            "Your symptoms appear **severe** and require prompt medical attention.\n"
            "Please visit a doctor or urgent care **today**."
        ),
    },
    "moderate": {
        "color": "#f39c12", "icon": "🟡", "label": "Moderate",
        "message": (
            "Your symptoms are **moderate**. Monitor closely.\n"
            "See a doctor within the next 24–48 hours if symptoms persist or worsen."
        ),
    },
    "mild": {
        "color": "#2ecc71", "icon": "🟢", "label": "Mild",
        "message": (
            "Your symptoms appear **mild**.\n"
            "Home care and rest may help. See a doctor if symptoms worsen or last more than 3 days."
        ),
    },
}


# ── Rule-based fast triage

def triage_fast(text: str) -> TriageResult:
    text_lower = text.lower()

    # Check emergency first
    for kw in EMERGENCY_KEYWORDS:
        if kw in text_lower:
            cfg = SEVERITY_CONFIG["emergency"]
            return TriageResult(
                severity="emergency",
                color=cfg["color"],
                icon=cfg["icon"],
                label=cfg["label"],
                message=cfg["message"],
                emergency_contacts=config.EMERGENCY_CONTACTS,
                rule_triggered=kw,
            )

    # Check severe
    for kw in SEVERE_KEYWORDS:
        if kw in text_lower:
            cfg = SEVERITY_CONFIG["severe"]
            return TriageResult(
                severity="severe",
                color=cfg["color"],
                icon=cfg["icon"],
                label=cfg["label"],
                message=cfg["message"],
                rule_triggered=kw,
            )

    # Check moderate
    for kw in MODERATE_KEYWORDS:
        if kw in text_lower:
            cfg = SEVERITY_CONFIG["moderate"]
            return TriageResult(
                severity="moderate",
                color=cfg["color"],
                icon=cfg["icon"],
                label=cfg["label"],
                message=cfg["message"],
                rule_triggered=kw,
            )

    # Default: mild
    cfg = SEVERITY_CONFIG["mild"]
    return TriageResult(
        severity="mild",
        color=cfg["color"],
        icon=cfg["icon"],
        label=cfg["label"],
        message=cfg["message"],
    )


def triage_from_llm_response(severity_str: str) -> TriageResult:
    
    level = severity_str.lower()
    if level not in SEVERITY_CONFIG:
        level = "mild"

    cfg = SEVERITY_CONFIG[level]
    contacts = config.EMERGENCY_CONTACTS if level == "emergency" else []

    return TriageResult(
        severity=level,
        color=cfg["color"],
        icon=cfg["icon"],
        label=cfg["label"],
        message=cfg["message"],
        emergency_contacts=contacts,
    )


def get_specialist_hint(symptoms: str) -> Optional[str]:
    """Suggest a specialist based on symptom keywords."""
    text = symptoms.lower()

    mapping = {
        "Cardiologist":       ["chest", "heart", "palpitation", "cardiac"],
        "Neurologist":        ["headache", "migraine", "seizure", "numbness", "stroke"],
        "Dermatologist":      ["rash", "skin", "acne", "eczema", "psoriasis", "wound"],
        "Ophthalmologist":    ["eye", "vision", "blurry", "blind"],
        "ENT Specialist":     ["ear", "nose", "throat", "sinus", "hearing"],
        "Psychiatrist":       ["anxiety", "depression", "stress", "mental", "sleep"],
        "Endocrinologist":    ["diabetes", "thyroid", "sugar", "hormone"],
        "Gastroenterologist": ["stomach", "abdomen", "liver", "bowel", "digestion"],
        "Pulmonologist":      ["lung", "breath", "asthma", "cough", "respiratory"],
        "Orthopedic Surgeon": ["bone", "fracture", "joint", "spine", "back pain"],
        "Nephrologist":       ["kidney", "urine", "renal"],
        "Gynecologist":       ["menstrual", "period", "pregnancy", "uterus", "ovary"],
        "Pediatrician":       ["child", "baby", "infant", "toddler", "kid"],
        "Oncologist":         ["cancer", "tumor", "lump", "malignant"],
        "Urologist":          ["urinary", "prostate", "bladder", "penis", "testis"],
    }

    for specialist, keywords in mapping.items():
        if any(kw in text for kw in keywords):
            return specialist

    return None


# ── Quick test 
if __name__ == "__main__":
    print("Testing triage module...\n")

    test_cases = [
        ("I have chest pain and left arm numbness",           "emergency"),
        ("High fever 104F with stiff neck and confusion",     "severe"),
        ("I have a headache and fever since yesterday",       "moderate"),
        ("Mild cold and runny nose for 2 days",               "mild"),
        ("I cannot breathe, throat is swelling",              "emergency"),
    ]

    all_pass = True
    for text, expected in test_cases:
        result = triage_fast(text)
        status = "✅" if result.severity == expected else "❌"
        if result.severity != expected:
            all_pass = False
        print(f"  {status} [{result.icon} {result.label}] \"{text[:45]}\"")
        if result.severity != expected:
            print(f"       Expected: {expected}, Got: {result.severity}")

    print(f"\n{'All triage tests passed.' if all_pass else 'Some tests failed.'}")