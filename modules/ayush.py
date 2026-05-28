
from dataclasses import dataclass, field
from typing import Optional

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

import config


# ── Result dataclass

@dataclass
class AYUSHResult:
    condition: str
    ayurvedic: list[dict] = field(default_factory=list)
    # [{"remedy": "Tulsi tea", "how_to": "Boil 5-6 tulsi leaves...",
    #   "benefit": "Antiviral, reduces fever", "system": "Ayurveda"}]
    yoga: list[dict] = field(default_factory=list)
    # [{"name": "Anulom Vilom", "duration": "5-10 min", "benefit": "..."}]
    diet_do: list[str] = field(default_factory=list)      # what to eat
    diet_avoid: list[str] = field(default_factory=list)   # what to avoid
    lifestyle: list[str] = field(default_factory=list)
    disclaimer: str = ""
    source: str = "local"   # "local" | "gemini"


# ── Curated AYUSH Knowledge Base

AYUSH_DB: dict[str, dict] = {

    "fever": {
        "ayurvedic": [
            {"remedy": "Tulsi (Holy Basil) Tea",
             "how_to": "Boil 8–10 fresh tulsi leaves in 2 cups water for 5 min. Add ginger and honey. Drink 2–3 times daily.",
             "benefit": "Antiviral, antipyretic, boosts immunity",
             "system": "Ayurveda"},
            {"remedy": "Giloy (Guduchi) Kadha",
             "how_to": "Boil 1 inch giloy stem in water for 10 min. Add black pepper and honey. Drink once daily.",
             "benefit": "Powerful immunomodulator, reduces fever and fatigue",
             "system": "Ayurveda"},
            {"remedy": "Ginger-Honey-Lemon Water",
             "how_to": "Mix 1 tsp ginger juice + 1 tsp honey + juice of half lemon in warm water. Drink 3× daily.",
             "benefit": "Anti-inflammatory, promotes sweating to break fever",
             "system": "Home Remedy"},
            {"remedy": "Saptaparna (Alstonia) Bark",
             "how_to": "Available as Sudarshan Churna — take 1 tsp with warm water twice daily.",
             "benefit": "Classical Ayurvedic antipyretic",
             "system": "Ayurveda"},
        ],
        "yoga": [
            {"name": "Shavasana (Corpse Pose)", "duration": "15–20 min",
             "benefit": "Complete rest, reduces body temperature, calms nervous system"},
            {"name": "Nadi Shodhana Pranayama", "duration": "5 min",
             "benefit": "Balances body heat, calms Pitta dosha"},
        ],
        "diet_do": [
            "Drink plenty of warm water, herbal teas, and coconut water",
            "Eat light — khichdi, moong dal soup, rice gruel (kanji)",
            "Include turmeric milk (haldi doodh) at bedtime",
            "Fresh pomegranate juice for hydration",
        ],
        "diet_avoid": [
            "Avoid heavy, oily, fried foods",
            "Avoid cold drinks, ice cream, cold water",
            "Avoid non-vegetarian food during fever",
            "Avoid spicy and sour foods",
        ],
        "lifestyle": [
            "Rest completely — avoid exertion",
            "Sponge forehead with lukewarm (not cold) water",
            "Keep room cool with natural ventilation",
            "Steam inhalation with eucalyptus drops for congestion",
        ],
    },

    "cold and cough": {
        "ayurvedic": [
            {"remedy": "Sitopaladi Churna",
             "how_to": "Take ½ tsp with honey twice daily. Available at Ayurvedic pharmacies.",
             "benefit": "Classical Ayurvedic remedy for cough, cold, and respiratory issues",
             "system": "Ayurveda"},
            {"remedy": "Turmeric Milk (Haldi Doodh / Golden Milk)",
             "how_to": "Mix ½ tsp turmeric + pinch of black pepper in 1 glass warm milk. Drink at bedtime.",
             "benefit": "Anti-inflammatory, antiviral, soothes throat",
             "system": "Ayurveda"},
            {"remedy": "Ginger-Tulsi-Honey Decoction",
             "how_to": "Boil ginger + tulsi leaves for 5 min. Add honey when cool. Take 3× daily.",
             "benefit": "Expectorant, relieves congestion, antiviral",
             "system": "Home Remedy"},
            {"remedy": "Trikatu Churna",
             "how_to": "¼ tsp with honey before meals. Contains dry ginger, black pepper, long pepper.",
             "benefit": "Clears Kapha, relieves congestion and cough",
             "system": "Ayurveda"},
        ],
        "yoga": [
            {"name": "Bhastrika Pranayama (Bellows Breath)", "duration": "3–5 min",
             "benefit": "Clears respiratory passages, strengthens lungs"},
            {"name": "Kapalbhati", "duration": "5 min",
             "benefit": "Clears sinuses, expels Kapha from chest"},
            {"name": "Ustrasana (Camel Pose)", "duration": "3–5 breaths",
             "benefit": "Opens chest, improves lung capacity"},
        ],
        "diet_do": [
            "Warm soups — tomato soup, chicken soup, dal soup",
            "Tulsi-ginger-honey tea throughout the day",
            "Steam inhalation 2× daily with ajwain (carom seeds) or eucalyptus",
            "Gargle with warm salt water 3× daily",
            "Eat light warm foods — khichdi, daliya",
        ],
        "diet_avoid": [
            "Cold water, cold drinks, ice cream, curd at night",
            "Bananas and citrus fruits during acute phase",
            "Fried, oily, and heavy foods",
            "Refrigerated foods",
        ],
        "lifestyle": [
            "Keep head and chest warm — avoid cold drafts",
            "Steam inhalation twice daily",
            "Nasya therapy — 2 drops sesame oil in each nostril at bedtime",
            "Avoid air conditioning directly on body",
        ],
    },

    "headache": {
        "ayurvedic": [
            {"remedy": "Peppermint Oil Massage",
             "how_to": "Dilute peppermint oil with coconut oil. Massage gently on temples and forehead in circular motion.",
             "benefit": "Cooling, analgesic effect on tension headaches",
             "system": "Home Remedy"},
            {"remedy": "Brahmi (Bacopa) Ghrita",
             "how_to": "¼ tsp Brahmi ghee with warm milk at bedtime.",
             "benefit": "Reduces stress-related and migraine headaches, calms Vata",
             "system": "Ayurveda"},
            {"remedy": "Shirashooladi Vajra Rasa",
             "how_to": "Classical Ayurvedic tablet for headache — consult Ayurvedic practitioner for dose.",
             "benefit": "Treats chronic headache and migraine",
             "system": "Ayurveda"},
            {"remedy": "Ginger-Lemon Tea",
             "how_to": "Boil grated ginger in water, add lemon juice and honey. Drink when headache starts.",
             "benefit": "Anti-inflammatory, relieves tension headache",
             "system": "Home Remedy"},
        ],
        "yoga": [
            {"name": "Anulom Vilom (Alternate Nostril Breathing)", "duration": "10 min",
             "benefit": "Balances nervous system, reduces tension headache"},
            {"name": "Sheetali Pranayama (Cooling Breath)", "duration": "5 min",
             "benefit": "Cools the body and mind, relieves Pitta headache"},
            {"name": "Balasana (Child's Pose)", "duration": "2–3 min",
             "benefit": "Releases neck and shoulder tension causing headache"},
            {"name": "Neck Stretches", "duration": "5 min",
             "benefit": "Relieves cervicogenic (neck-related) headache"},
        ],
        "diet_do": [
            "Stay well hydrated — dehydration is a common cause",
            "Eat magnesium-rich foods: nuts, seeds, leafy greens",
            "Regular meal times — skipping meals triggers headache",
            "Warm ginger tea",
        ],
        "diet_avoid": [
            "Caffeine (or sudden caffeine withdrawal)",
            "Processed foods with MSG and preservatives",
            "Alcohol and red wine",
            "Strong cheese and fermented foods if prone to migraines",
        ],
        "lifestyle": [
            "Sleep at fixed times — irregular sleep triggers migraines",
            "Reduce screen time and take eye breaks every 20 minutes",
            "Apply cold or warm compress on forehead based on type",
            "Practice stress management — headache is often stress-driven",
        ],
    },

    "diabetes": {
        "ayurvedic": [
            {"remedy": "Karela (Bitter Gourd) Juice",
             "how_to": "Extract juice of 1 bitter gourd. Drink 30 ml on empty stomach every morning.",
             "benefit": "Natural hypoglycemic, mimics insulin action",
             "system": "Ayurveda"},
            {"remedy": "Vijaysar (Indian Kino) Water",
             "how_to": "Soak a Vijaysar wooden tumbler in water overnight. Drink the water in morning.",
             "benefit": "Clinically studied — reduces blood sugar, improves insulin sensitivity",
             "system": "Ayurveda"},
            {"remedy": "Methi (Fenugreek) Seeds",
             "how_to": "Soak 1 tsp methi seeds overnight. Eat seeds and drink water in morning.",
             "benefit": "High soluble fiber slows glucose absorption, lowers HbA1c",
             "system": "Home Remedy"},
            {"remedy": "Jamun (Black Plum) Seed Powder",
             "how_to": "Dry and powder jamun seeds. Take ½ tsp with water twice daily.",
             "benefit": "Jamboline in seeds inhibits starch conversion to sugar",
             "system": "Ayurveda"},
        ],
        "yoga": [
            {"name": "Mandukasana (Frog Pose)", "duration": "3×30 sec",
             "benefit": "Massages pancreas, stimulates insulin production"},
            {"name": "Dhanurasana (Bow Pose)", "duration": "3×20 sec",
             "benefit": "Stimulates pancreatic function"},
            {"name": "Pranayama (Kapalbhati + Anulom Vilom)", "duration": "15 min daily",
             "benefit": "Reduces stress hormones that raise blood sugar"},
            {"name": "Brisk Walking", "duration": "30–45 min daily",
             "benefit": "Most effective — lowers blood glucose naturally"},
        ],
        "diet_do": [
            "Eat high-fiber foods: whole grains, vegetables, dal",
            "Include bitter foods: karela, methi leaves, neem",
            "Small frequent meals — avoid long gaps",
            "Cinnamon (dalchini) in food — improves insulin sensitivity",
        ],
        "diet_avoid": [
            "White rice, maida, refined flour products",
            "Sugary drinks, fruit juices, sweets",
            "Potatoes, yam in large quantities",
            "Alcohol and processed foods",
        ],
        "lifestyle": [
            "Monitor blood sugar regularly",
            "Never skip prescribed medicines — AYUSH is complementary only",
            "Manage stress — cortisol directly raises blood sugar",
            "Maintain healthy weight — even 5% weight loss improves sugar control",
        ],
    },

    "hypertension": {
        "ayurvedic": [
            {"remedy": "Sarpagandha (Rauwolfia) Tablets",
             "how_to": "Available as Sarpagandha Vati — take as per Ayurvedic doctor's advice.",
             "benefit": "Well-studied Ayurvedic antihypertensive, contains reserpine",
             "system": "Ayurveda"},
            {"remedy": "Arjuna Bark Decoction",
             "how_to": "Boil 1 tsp Arjuna bark powder in 1 cup milk + 1 cup water. Reduce to 1 cup. Drink daily.",
             "benefit": "Strengthens heart muscle, reduces blood pressure",
             "system": "Ayurveda"},
            {"remedy": "Garlic (Lahsun)",
             "how_to": "Eat 2 raw garlic cloves on empty stomach daily. Or take garlic supplement.",
             "benefit": "Allicin reduces blood pressure, anti-atherosclerotic",
             "system": "Home Remedy"},
        ],
        "yoga": [
            {"name": "Shavasana", "duration": "20 min",
             "benefit": "Lowers blood pressure through deep relaxation"},
            {"name": "Bhramari Pranayama (Humming Bee Breath)", "duration": "10 min",
             "benefit": "Significantly reduces BP — medically studied"},
            {"name": "Anulom Vilom", "duration": "10–15 min",
             "benefit": "Balances autonomic nervous system, lowers BP"},
        ],
        "diet_do": [
            "DASH diet: fruits, vegetables, low-fat dairy, whole grains",
            "Reduce sodium — use sendha namak (rock salt) instead",
            "Potassium-rich foods: banana, coconut water, dal",
            "Flaxseeds and omega-3 rich foods",
        ],
        "diet_avoid": [
            "High-sodium foods: pickles, papad, processed foods",
            "Alcohol and smoking",
            "Caffeine in excess",
            "Red meat and saturated fats",
        ],
        "lifestyle": [
            "Check BP daily at same time",
            "Never stop prescribed medication without doctor advice",
            "Reduce stress — practice meditation 20 min daily",
            "Maintain healthy weight — every kg lost reduces BP by ~1 mmHg",
        ],
    },

    "indigestion": {
        "ayurvedic": [
            {"remedy": "Ajwain (Carom) with Black Salt",
             "how_to": "Mix ½ tsp ajwain + pinch of black salt. Chew and swallow with warm water after meals.",
             "benefit": "Instant relief from bloating, gas, indigestion",
             "system": "Home Remedy"},
            {"remedy": "Triphala Churna",
             "how_to": "Take 1 tsp Triphala powder with warm water at bedtime.",
             "benefit": "Regulates digestion, mild laxative, detoxifying",
             "system": "Ayurveda"},
            {"remedy": "Hingvastak Churna",
             "how_to": "¼ tsp with first bite of food. Contains asafoetida (hing), ginger, pepper.",
             "benefit": "Stimulates digestive fire (Agni), relieves gas and bloating",
             "system": "Ayurveda"},
        ],
        "yoga": [
            {"name": "Pawanmuktasana (Wind-Relieving Pose)", "duration": "1 min each side",
             "benefit": "Releases trapped gas, relieves bloating"},
            {"name": "Vajrasana (Thunderbolt Pose)", "duration": "5–10 min after meals",
             "benefit": "Only yoga pose safe to do after eating — aids digestion"},
            {"name": "Trikonasana (Triangle Pose)", "duration": "5 breaths each side",
             "benefit": "Stimulates digestive organs"},
        ],
        "diet_do": [
            "Eat slowly and chew food thoroughly",
            "Drink warm water with meals — not cold water",
            "Ginger tea or jeera (cumin) water before meals",
            "Eat at regular times",
        ],
        "diet_avoid": [
            "Overeating — eat only until 75% full",
            "Lying down immediately after meals",
            "Carbonated drinks",
            "Very spicy, oily, or fried food",
        ],
        "lifestyle": [
            "Walk 10–15 min after meals",
            "Eat dinner at least 2 hours before sleeping",
            "Reduce stress — gut-brain axis directly affects digestion",
        ],
    },

    "skin rash": {
        "ayurvedic": [
            {"remedy": "Neem (Azadirachta) Paste",
             "how_to": "Grind fresh neem leaves into paste. Apply on affected area for 20 min, wash off.",
             "benefit": "Antibacterial, antifungal, anti-inflammatory",
             "system": "Ayurveda"},
            {"remedy": "Aloe Vera Gel",
             "how_to": "Apply fresh aloe vera gel directly on rash 2–3 times daily.",
             "benefit": "Cooling, anti-inflammatory, promotes skin healing",
             "system": "Home Remedy"},
            {"remedy": "Turmeric + Coconut Oil Paste",
             "how_to": "Mix ½ tsp turmeric with coconut oil. Apply on rash, leave 15 min.",
             "benefit": "Antibacterial, reduces inflammation and itching",
             "system": "Ayurveda"},
            {"remedy": "Manjistha (Indian Madder) Capsules",
             "how_to": "500mg twice daily with water. Available at Ayurvedic stores.",
             "benefit": "Blood purifier, treats chronic skin conditions",
             "system": "Ayurveda"},
        ],
        "yoga": [
            {"name": "Anulom Vilom + Kapalbhati", "duration": "15 min daily",
             "benefit": "Blood purification, reduces skin inflammation from within"},
        ],
        "diet_do": [
            "Drink 8–10 glasses of water daily",
            "Neem tea or Neem capsules",
            "Turmeric milk at bedtime",
            "Fresh fruits and vegetables rich in antioxidants",
        ],
        "diet_avoid": [
            "Spicy, oily, and fried foods — aggravate Pitta",
            "Seafood if prone to skin allergies",
            "Alcohol and processed food",
            "Dairy products during acute rash (if suspected allergy)",
        ],
        "lifestyle": [
            "Keep skin clean and dry",
            "Wear loose cotton clothing",
            "Avoid synthetic fabrics and harsh detergents",
            "Do not scratch — it worsens inflammation and risk of infection",
        ],
    },

    "joint pain": {
        "ayurvedic": [
            {"remedy": "Shallaki (Boswellia) Capsules",
             "how_to": "400mg capsule twice daily with meals. Well-studied for arthritis.",
             "benefit": "Clinically proven anti-inflammatory for joints",
             "system": "Ayurveda"},
            {"remedy": "Mahanarayan Oil Massage",
             "how_to": "Warm the oil slightly. Massage gently on affected joint for 10–15 min daily.",
             "benefit": "Reduces pain, stiffness, and swelling in joints",
             "system": "Ayurveda"},
            {"remedy": "Guggul (Commiphora) Tablets",
             "how_to": "500mg twice daily. Available as Yograj Guggul or Triphala Guggul.",
             "benefit": "Anti-inflammatory, reduces uric acid, treats Vata disorders",
             "system": "Ayurveda"},
            {"remedy": "Turmeric + Ginger Decoction",
             "how_to": "Boil ½ tsp turmeric + grated ginger in water. Drink twice daily.",
             "benefit": "Natural COX-2 inhibitor — reduces joint inflammation",
             "system": "Home Remedy"},
        ],
        "yoga": [
            {"name": "Sukshma Vyayama (Joint Exercises)", "duration": "10 min",
             "benefit": "Gentle movements to lubricate joints, reduce stiffness"},
            {"name": "Tadasana (Mountain Pose)", "duration": "1 min",
             "benefit": "Improves posture, reduces spinal joint pressure"},
            {"name": "Viparita Karani (Legs Up Wall)", "duration": "10 min",
             "benefit": "Reduces inflammation in knee and hip joints"},
        ],
        "diet_do": [
            "Anti-inflammatory foods: walnuts, flaxseeds, fatty fish",
            "Turmeric and ginger in daily cooking",
            "Calcium-rich foods: ragi, sesame seeds, dairy",
            "Vitamin D: get 20 min of morning sunlight daily",
        ],
        "diet_avoid": [
            "Sour foods at night: curd, tamarind, pickles — aggravate Vata",
            "Cold and refrigerated foods",
            "Excess salt — causes water retention, worsens swelling",
            "Nightshade vegetables (tomato, potato, brinjal) in Vata arthritis",
        ],
        "lifestyle": [
            "Hot compress for chronic pain, cold compress for acute swelling",
            "Maintain healthy weight — every extra kg = 4 kg pressure on knee",
            "Swim or cycle — low-impact exercise preserves joint health",
            "Avoid sitting on floor for long periods",
        ],
    },

    "anxiety": {
        "ayurvedic": [
            {"remedy": "Ashwagandha (Withania somnifera)",
             "how_to": "300–500mg standardized extract capsule at bedtime. Or ½ tsp powder in warm milk.",
             "benefit": "Adaptogen — reduces cortisol by 30%, clinically proven for anxiety",
             "system": "Ayurveda"},
            {"remedy": "Brahmi (Bacopa monnieri)",
             "how_to": "300mg capsule twice daily, or 1 tsp Brahmi churna with warm milk.",
             "benefit": "Calms nervous system, improves memory, reduces anxiety",
             "system": "Ayurveda"},
            {"remedy": "Jatamansi (Nardostachys) Powder",
             "how_to": "½ tsp with warm water at bedtime.",
             "benefit": "Natural anxiolytic and sleep aid, balances Vata",
             "system": "Ayurveda"},
        ],
        "yoga": [
            {"name": "Nadi Shodhana Pranayama", "duration": "10–15 min",
             "benefit": "Activates parasympathetic nervous system, reduces anxiety immediately"},
            {"name": "Yoga Nidra (Yogic Sleep)", "duration": "20–30 min",
             "benefit": "Deep relaxation equivalent to 4 hours sleep, powerful for anxiety"},
            {"name": "Shavasana", "duration": "15–20 min",
             "benefit": "Complete mental and physical relaxation"},
        ],
        "diet_do": [
            "Magnesium-rich foods: dark chocolate, pumpkin seeds, spinach",
            "Warm milk with ashwagandha and nutmeg at bedtime",
            "Regular meals — blood sugar drops worsen anxiety",
            "Herbal teas: chamomile, lavender, brahmi tea",
        ],
        "diet_avoid": [
            "Excess caffeine — worsens anxiety symptoms",
            "Alcohol — short-term relief but worsens anxiety long-term",
            "Sugar and processed foods — cause blood sugar swings",
            "Skipping meals",
        ],
        "lifestyle": [
            "Practice 4-7-8 breathing when anxiety spikes",
            "Regular sleep schedule — anxiety and sleep are deeply connected",
            "Digital detox 1 hour before bed",
            "Journaling or talking to someone trusted",
        ],
    },

    "anaemia": {
        "ayurvedic": [
            {"remedy": "Punarnava Mandur Tablets",
             "how_to": "Classical Ayurvedic iron supplement — 2 tablets twice daily with meals.",
             "benefit": "Iron supplementation with digestive herbs for better absorption",
             "system": "Ayurveda"},
            {"remedy": "Pomegranate Juice + Jaggery",
             "how_to": "Drink 1 glass fresh pomegranate juice daily. Add small piece of jaggery.",
             "benefit": "Rich in iron, Vitamin C aids absorption, jaggery is bioavailable iron",
             "system": "Home Remedy"},
            {"remedy": "Moringa (Drumstick) Leaves",
             "how_to": "Add moringa leaves to dal or eat as sabzi. Or take moringa powder ½ tsp daily.",
             "benefit": "Highest plant-based iron content, also has Vitamin C for absorption",
             "system": "Home Remedy"},
        ],
        "yoga": [
            {"name": "Sarvangasana (Shoulder Stand)", "duration": "1–3 min",
             "benefit": "Stimulates thyroid, improves blood circulation"},
            {"name": "Pranayama", "duration": "15 min",
             "benefit": "Improves oxygenation of blood"},
        ],
        "diet_do": [
            "Iron-rich foods: beetroot, spinach, methi, jaggery, sesame seeds",
            "Vitamin C with iron: eat lemon/amla with iron-rich foods",
            "Moringa leaves, pomegranate, black dates",
            "Cook in iron kadai — food absorbs iron from cookware",
        ],
        "diet_avoid": [
            "Tea/coffee immediately after meals — tannins block iron absorption",
            "Calcium supplements at same time as iron",
            "Excess dairy with iron-rich meals",
        ],
        "lifestyle": [
            "Get morning sunlight for Vitamin D (aids iron metabolism)",
            "Deworm every 6 months (common cause of anaemia in India)",
            "Regular blood tests to monitor Hb levels",
        ],
    },
}

# ── Keyword matcher

CONDITION_KEYWORDS: dict[str, list[str]] = {
    "fever":          ["fever", "bukhaar", "temperature", "pyrexia", "viral"],
    "cold and cough": ["cold", "cough", "khansi", "sardi", "runny nose", "congestion", "flu", "influenza"],
    "headache":       ["headache", "sir dard", "migraine", "head pain", "sir mein dard"],
    "diabetes":       ["diabetes", "sugar", "blood sugar", "hyperglycemia", "madhumeh"],
    "hypertension":   ["hypertension", "high blood pressure", "bp high", "blood pressure"],
    "indigestion":    ["indigestion", "acidity", "gas", "bloating", "constipation", "stomach", "acrid"],
    "skin rash":      ["rash", "itching", "skin", "eczema", "allergy", "urticaria", "khujli"],
    "joint pain":     ["joint pain", "arthritis", "knee pain", "back pain", "jodo mein dard"],
    "anxiety":        ["anxiety", "stress", "depression", "tension", "mental", "sleep", "insomnia"],
    "anaemia":        ["anaemia", "anemia", "low hemoglobin", "weakness", "fatigue", "pallor"],
}

DISCLAIMER = (
    "⚠️ **AYUSH Disclaimer**: These suggestions are traditional/complementary remedies "
    "for general wellness support. They are NOT a replacement for prescribed medical treatment. "
    "Always consult a qualified doctor before stopping or changing any medication. "
    "For serious conditions, follow your allopathic treatment first."
)


# ── Main functions 

def get_ayush_suggestions(
    symptoms: str,
    diagnosis: str = "",
) -> Optional[AYUSHResult]:
    """
    Get AYUSH/Ayurvedic suggestions for a given symptom/diagnosis.

    First checks local curated DB (instant, no API).
    Falls back to Gemini if condition not in local DB.

    Args:
        symptoms  : patient's symptom description
        diagnosis : AI-diagnosed condition (optional, improves matching)

    Returns:
        AYUSHResult or None if no relevant suggestions found
    """
    combined = f"{symptoms} {diagnosis}".lower()

    # Try local DB first
    matched_key = _match_condition(combined)
    if matched_key:
        data = AYUSH_DB[matched_key]
        return AYUSHResult(
            condition=matched_key.title(),
            ayurvedic=data.get("ayurvedic", []),
            yoga=data.get("yoga", []),
            diet_do=data.get("diet_do", []),
            diet_avoid=data.get("diet_avoid", []),
            lifestyle=data.get("lifestyle", []),
            disclaimer=DISCLAIMER,
            source="local",
        )

    # Fallback to Gemini for unlisted conditions
    return _get_gemini_ayush(symptoms, diagnosis)


def _match_condition(text: str) -> Optional[str]:
    
    best_match = None
    best_score = 0

    for condition, keywords in CONDITION_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in text)
        if score > best_score:
            best_score = score
            best_match = condition

    return best_match if best_score > 0 else None


def _get_gemini_ayush(symptoms: str, diagnosis: str) -> Optional[AYUSHResult]:
    
    try:
        llm = ChatGoogleGenerativeAI(
            model=config.GEMINI_MODEL,
            google_api_key=config.GEMINI_API_KEY,
            temperature=0.4,
        )

        prompt = f"""You are an expert in Ayurveda, Yoga, and traditional Indian medicine (AYUSH).

A patient has the following condition:
Symptoms: {symptoms}
Diagnosis: {diagnosis or 'not specified'}

Please provide practical AYUSH/Ayurvedic suggestions in this EXACT format:

CONDITION: [condition name]

AYURVEDIC_REMEDIES:
1. [Remedy name] | [How to prepare/use] | [Benefit] | [System: Ayurveda/Yoga/Homeopathy]
2. [Remedy name] | [How to prepare/use] | [Benefit] | [System]

YOGA:
1. [Pose/Pranayama name] | [Duration] | [Benefit]
2. [Pose/Pranayama name] | [Duration] | [Benefit]

DIET_DO:
- [what to eat/drink]
- [what to eat/drink]

DIET_AVOID:
- [what to avoid]
- [what to avoid]

LIFESTYLE:
- [lifestyle tip]
- [lifestyle tip]

Keep suggestions practical, safe, and specific to Indian context.
Include only evidence-based or traditionally well-established remedies.
"""

        response = llm.invoke([HumanMessage(content=prompt)])
        return _parse_gemini_ayush(response.content, symptoms)

    except Exception:
        return None


def _parse_gemini_ayush(text: str, symptoms: str) -> AYUSHResult:
    """Parse Gemini's AYUSH response into structured AYUSHResult."""
    import re

    def extract_section(label: str) -> str:
        match = re.search(
            rf"{label}:\s*(.*?)(?=\n[A-Z_]+:|$)", text,
            re.DOTALL | re.IGNORECASE
        )
        return match.group(1).strip() if match else ""

    # Condition
    cond_match = re.search(r"CONDITION:\s*(.+)", text)
    condition  = cond_match.group(1).strip() if cond_match else symptoms[:30]

    # Ayurvedic remedies
    ayurvedic = []
    rem_text = extract_section("AYURVEDIC_REMEDIES")
    for line in rem_text.splitlines():
        line = re.sub(r"^\d+\.\s*", "", line).strip()
        parts = [p.strip() for p in line.split("|")]
        if len(parts) >= 3:
            ayurvedic.append({
                "remedy":  parts[0],
                "how_to":  parts[1],
                "benefit": parts[2],
                "system":  parts[3] if len(parts) > 3 else "Ayurveda",
            })

    # Yoga
    yoga = []
    yoga_text = extract_section("YOGA")
    for line in yoga_text.splitlines():
        line = re.sub(r"^\d+\.\s*", "", line).strip()
        parts = [p.strip() for p in line.split("|")]
        if len(parts) >= 2:
            yoga.append({
                "name":     parts[0],
                "duration": parts[1] if len(parts) > 1 else "",
                "benefit":  parts[2] if len(parts) > 2 else "",
            })

    # Lists
    def parse_list(label: str) -> list[str]:
        section = extract_section(label)
        return [
            re.sub(r"^[-•*]\s*", "", line).strip()
            for line in section.splitlines()
            if line.strip() and line.strip() not in ("", "-")
        ]

    return AYUSHResult(
        condition=condition,
        ayurvedic=ayurvedic,
        yoga=yoga,
        diet_do=parse_list("DIET_DO"),
        diet_avoid=parse_list("DIET_AVOID"),
        lifestyle=parse_list("LIFESTYLE"),
        disclaimer=DISCLAIMER,
        source="gemini",
    )


# ── Quick test 
if __name__ == "__main__":
    print("Testing AYUSH module...\n")

    test_cases = [
        ("I have fever and body ache since yesterday", "Viral Fever"),
        ("headache and stress at work",                "Tension Headache"),
        ("blood sugar is high, 210 fasting",           "Type 2 Diabetes"),
        ("knee joint pain and stiffness in morning",   "Osteoarthritis"),
    ]

    for symptoms, diagnosis in test_cases:
        result = get_ayush_suggestions(symptoms, diagnosis)
        if result:
            print(f"✅ {result.condition} [{result.source}]")
            print(f"   Remedies : {len(result.ayurvedic)}")
            print(f"   Yoga     : {len(result.yoga)}")
            print(f"   Diet tips: {len(result.diet_do)} do / {len(result.diet_avoid)} avoid")
            if result.ayurvedic:
                r = result.ayurvedic[0]
                print(f"   Top remedy: {r['remedy']} — {r['benefit']}")
            print()
        else:
            print(f"⚠️  No suggestions found for: {symptoms}\n")

    print("AYUSH module ready.")