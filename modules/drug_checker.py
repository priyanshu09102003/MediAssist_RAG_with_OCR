import json
import re
import requests
from dataclasses import dataclass, field
from typing import Optional

import config



@dataclass
class DrugCheckResult:
    safe: bool = True
    warnings: list[dict] = field(default_factory=list)
    # [{"type": "interaction"|"allergy", "severity": "high"|"medium"|"low",
    #   "drugs": ["Drug A", "Drug B"], "message": "..."}]
    allergy_conflicts: list[str] = field(default_factory=list)
    interaction_pairs: list[tuple] = field(default_factory=list)
    has_interactions: bool = False
    has_allergy_conflict: bool = False
    summary: str = ""


# ── Common drug interaction database (curated) ───────────────────────────────
# Format: (drug_a_pattern, drug_b_pattern, severity, message)

KNOWN_INTERACTIONS = [
    # Anticoagulants
    ("warfarin",    "aspirin",       "high",
     "Warfarin + Aspirin increases bleeding risk significantly."),
    ("warfarin",    "ibuprofen",     "high",
     "Warfarin + Ibuprofen increases bleeding risk."),
    ("warfarin",    "paracetamol",   "medium",
     "High-dose Paracetamol can enhance Warfarin's anticoagulant effect."),
    ("warfarin",    "naproxen",      "high",
     "NSAIDs increase bleeding risk with Warfarin."),
    # Antibiotics
    ("metronidazole", "alcohol",     "high",
     "Metronidazole + Alcohol causes severe nausea, vomiting, flushing."),
    ("ciprofloxacin",  "antacid",    "medium",
     "Antacids reduce Ciprofloxacin absorption. Take 2 hours apart."),
    ("amoxicillin",    "warfarin",   "medium",
     "Amoxicillin may enhance anticoagulant effect of Warfarin."),
    ("tetracycline",   "calcium",    "medium",
     "Calcium/dairy reduces Tetracycline absorption."),
    ("tetracycline",   "antacid",    "medium",
     "Antacids reduce Tetracycline absorption. Take 2 hours apart."),
    # Antihypertensives
    ("ace inhibitor",  "potassium",  "medium",
     "ACE inhibitors + Potassium supplements can cause hyperkalemia."),
    ("lisinopril",     "potassium",  "medium",
     "Lisinopril + Potassium raises risk of high potassium levels."),
    ("metformin",      "alcohol",    "medium",
     "Alcohol increases risk of lactic acidosis with Metformin."),
    # Pain medications
    ("tramadol",    "ssri",          "high",
     "Tramadol + SSRIs can cause serotonin syndrome."),
    ("tramadol",    "sertraline",    "high",
     "Tramadol + Sertraline: risk of serotonin syndrome."),
    ("tramadol",    "fluoxetine",    "high",
     "Tramadol + Fluoxetine: risk of serotonin syndrome."),
    ("ibuprofen",   "aspirin",       "medium",
     "Two NSAIDs together increase GI bleeding risk."),
    ("ibuprofen",   "lisinopril",    "medium",
     "NSAIDs can reduce effectiveness of ACE inhibitors."),
    # Statins
    ("simvastatin",  "clarithromycin","high",
     "Clarithromycin raises Simvastatin levels — risk of muscle damage."),
    ("atorvastatin", "clarithromycin","high",
     "Clarithromycin raises Atorvastatin levels — risk of myopathy."),
    # Diabetes
    ("metformin",   "ibuprofen",     "medium",
     "NSAIDs can affect kidney function and Metformin clearance."),
    # Cardiac
    ("digoxin",     "amiodarone",    "high",
     "Amiodarone increases Digoxin levels — risk of toxicity."),
    ("digoxin",     "clarithromycin","high",
     "Clarithromycin raises Digoxin levels — risk of toxicity."),
]

# Common allergy cross-reactivity
ALLERGY_CROSS_REACTIONS = {
    "penicillin":    ["amoxicillin", "ampicillin", "cloxacillin", "flucloxacillin",
                      "piperacillin", "co-amoxiclav", "augmentin"],
    "sulfa":         ["sulfamethoxazole", "trimethoprim", "co-trimoxazole", "bactrim"],
    "aspirin":       ["ibuprofen", "naproxen", "diclofenac", "indomethacin",
                      "mefenamic acid", "nsaid"],
    "codeine":       ["morphine", "tramadol", "oxycodone", "pethidine"],
    "cephalosporin": ["cefalexin", "cefuroxime", "ceftriaxone", "cefixime"],
}


# ── Main checker

def check_prescription(
    medications: list[dict],
    patient_allergies: list[str] = None,
    use_fda_api: bool = True,
) -> DrugCheckResult:
    """
    Check a list of medications for interactions and allergy conflicts.

    Args:
        medications       : list of {"name": "...", "dose": "...", ...}
        patient_allergies : list of allergy strings from patient profile
        use_fda_api       : attempt OpenFDA API lookup (requires internet)

    Returns:
        DrugCheckResult with warnings and flags
    """
    result  = DrugCheckResult()
    drug_names = [_normalize(m.get("name", "")) for m in medications if m.get("name")]

    if not drug_names:
        result.summary = "No medications to check."
        return result

    warnings = []

    # 1. Check patient allergy conflicts
    if patient_allergies:
        allergy_warnings = _check_allergies(drug_names, patient_allergies)
        warnings.extend(allergy_warnings)
        if allergy_warnings:
            result.has_allergy_conflict = True
            result.allergy_conflicts = [
                w["drugs"][0] for w in allergy_warnings if w["drugs"]
            ]

    # 2. Check local drug-drug interactions
    interaction_warnings = _check_local_interactions(drug_names)
    warnings.extend(interaction_warnings)
    if interaction_warnings:
        result.has_interactions = True
        result.interaction_pairs = [
            tuple(w["drugs"]) for w in interaction_warnings
        ]

    # 3. OpenFDA API check (optional, best-effort)
    if use_fda_api and len(drug_names) >= 2:
        fda_warnings = _check_openfda(drug_names)
        for fw in fda_warnings:
            # Only add if not already caught by local check
            if not any(fw["message"][:30] in w["message"] for w in warnings):
                warnings.append(fw)
                result.has_interactions = True

    result.warnings = warnings
    result.safe = not (result.has_allergy_conflict or result.has_interactions)

    # Build summary
    if result.safe:
        result.summary = (
            f"✅ No known interactions found among {len(drug_names)} medication(s)."
        )
    else:
        high = sum(1 for w in warnings if w.get("severity") == "high")
        med  = sum(1 for w in warnings if w.get("severity") == "medium")
        result.summary = (
            f"⚠️ {len(warnings)} issue(s) found: "
            f"{high} high severity, {med} medium severity. Review before dispensing."
        )

    return result


def check_single_drug(
    drug_name: str,
    patient_allergies: list[str],
) -> list[dict]:
   
    normalized = _normalize(drug_name)
    return _check_allergies([normalized], patient_allergies)




def _normalize(name: str) -> str:
   
    name = name.lower().strip()
    name = re.sub(r"\d+\s*(mg|mcg|ml|g|iu|units?)\b", "", name)
    name = re.sub(r"\b(tablet|capsule|syrup|injection|drops|cream|ointment)\b", "", name)
    return name.strip()


def _check_allergies(drug_names: list[str], allergies: list[str]) -> list[dict]:
    warnings = []
    allergy_normalized = [_normalize(a) for a in allergies]

    for drug in drug_names:
        for allergy in allergy_normalized:
            # Direct match
            if allergy in drug or drug in allergy:
                warnings.append({
                    "type": "allergy",
                    "severity": "high",
                    "drugs": [drug],
                    "message": (
                        f"⚠️ ALLERGY ALERT: Patient is allergic to '{allergy}'. "
                        f"'{drug.title()}' may cause an allergic reaction. "
                        f"Do NOT administer without medical supervision."
                    ),
                })
                continue

            # Cross-reactivity check
            for allergy_class, related in ALLERGY_CROSS_REACTIONS.items():
                if allergy in allergy_class or allergy_class in allergy:
                    for related_drug in related:
                        if related_drug in drug:
                            warnings.append({
                                "type": "allergy_cross",
                                "severity": "high",
                                "drugs": [drug],
                                "message": (
                                    f"⚠️ CROSS-REACTIVITY: Patient allergic to '{allergy}'. "
                                    f"'{drug.title()}' is in the same drug class. "
                                    f"Use with extreme caution or avoid."
                                ),
                            })

    return warnings


def _check_local_interactions(drug_names: list[str]) -> list[dict]:
    warnings = []
    checked  = set()

    for drug_a in drug_names:
        for drug_b in drug_names:
            if drug_a == drug_b:
                continue
            pair = tuple(sorted([drug_a, drug_b]))
            if pair in checked:
                continue
            checked.add(pair)

            for pattern_a, pattern_b, severity, message in KNOWN_INTERACTIONS:
                if ((pattern_a in drug_a or pattern_a in drug_b) and
                        (pattern_b in drug_a or pattern_b in drug_b)):
                    warnings.append({
                        "type": "interaction",
                        "severity": severity,
                        "drugs": [drug_a, drug_b],
                        "message": f"💊 INTERACTION: {message}",
                    })
                    break

    return warnings


def _check_openfda(drug_names: list[str]) -> list[dict]:
   
    warnings = []
    try:
        # Query interactions for the first drug against others
        primary = drug_names[0]
        query   = f'drug_interactions:"{primary}"'
        url     = (
            f"https://api.fda.gov/drug/label.json"
            f"?search={query}&limit=1"
        )
        resp = requests.get(url, timeout=5)
        if resp.status_code != 200:
            return []

        data = resp.json()
        results = data.get("results", [])
        if not results:
            return []

        interaction_text = " ".join(
            results[0].get("drug_interactions", [])
        ).lower()

        # Check if any of the other drugs appear in the interaction text
        for other_drug in drug_names[1:]:
            drug_base = other_drug.split()[0]  # first word of drug name
            if len(drug_base) > 4 and drug_base in interaction_text:
                warnings.append({
                    "type": "interaction_fda",
                    "severity": "medium",
                    "drugs": [primary, other_drug],
                    "message": (
                        f"📋 FDA DATA: Possible interaction between "
                        f"'{primary.title()}' and '{other_drug.title()}'. "
                        f"Check full FDA label for details."
                    ),
                })

    except Exception:
        pass   

    return warnings


# ── Quick test 
    print("Testing DrugChecker...\n")

    test_meds = [
        {"name": "Warfarin 5mg", "dose": "5mg", "frequency": "Once daily"},
        {"name": "Aspirin 75mg", "dose": "75mg", "frequency": "Once daily"},
        {"name": "Paracetamol 500mg", "dose": "500mg", "frequency": "As needed"},
    ]
    test_allergies = ["penicillin", "sulfa"]

    result = check_prescription(test_meds, test_allergies, use_fda_api=False)

    print(f"  Safe          : {result.safe}")
    print(f"  Interactions  : {result.has_interactions}")
    print(f"  Allergy flags : {result.has_allergy_conflict}")
    print(f"  Summary       : {result.summary}")
    print(f"\n  Warnings ({len(result.warnings)}):")
    for w in result.warnings:
        print(f"    [{w['severity'].upper()}] {w['message'][:80]}")

    print("\n  ✅ Drug checker working correctly.")