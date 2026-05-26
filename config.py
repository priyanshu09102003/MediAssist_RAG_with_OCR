import os
from pathlib import Path
from dotenv import load_dotenv

# ── Load .env file 
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")


# ── Helper
def _require(key: str) -> str:
    
    value = os.getenv(key)
    if not value or value.startswith("your_"):
        raise EnvironmentError(
            f"[config] Missing or unset environment variable: '{key}'. "
            f"Please fill it in your .env file."
        )
    return value


# ── App 
APP_NAME        = os.getenv("APP_NAME", "MediAssist")
APP_VERSION     = os.getenv("APP_VERSION", "1.0.0")
DEBUG           = os.getenv("DEBUG", "False").lower() == "true"

# ── API Keys 
GEMINI_API_KEY  = _require("GEMINI_API_KEY")


GCP_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "")
if GCP_CREDENTIALS:
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = GCP_CREDENTIALS

# ── Paths 
SQLITE_DB_PATH  = BASE_DIR / os.getenv("SQLITE_DB_PATH", "data/healthcare.db")
CHROMA_DB_PATH  = str(BASE_DIR / os.getenv("CHROMA_DB_PATH", "data/chroma_db"))
MEDICAL_KB_PATH = BASE_DIR / os.getenv("MEDICAL_KB_PATH", "data/medical_kb")

# Ensure directories exist at import time
SQLITE_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
Path(CHROMA_DB_PATH).mkdir(parents=True, exist_ok=True)
MEDICAL_KB_PATH.mkdir(parents=True, exist_ok=True)

# ── Gemini Models 
GEMINI_MODEL        = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_VISION_MODEL = os.getenv("GEMINI_VISION_MODEL", "gemini-2.5-flash")
GEMINI_MAX_TOKENS   = int(os.getenv("GEMINI_MAX_TOKENS", "2048"))
GEMINI_TEMPERATURE  = float(os.getenv("GEMINI_TEMPERATURE", "0.3"))

# ── RAG / Retrieval 
CHUNK_SIZE      = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP   = int(os.getenv("CHUNK_OVERLAP", "100"))
RETRIEVER_TOP_K = int(os.getenv("RETRIEVER_TOP_K", "5"))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "models/text-embedding-004")

# ── Language 
DEFAULT_LANGUAGE = os.getenv("DEFAULT_LANGUAGE", "en")

SUPPORTED_LANGUAGES = {
    "en": "English",
    "hi": "Hindi",
    "bn": "Bengali",
    "ta": "Tamil",
    "te": "Telugu",
}

# STT language codes for Google Cloud Speech
STT_LANGUAGE_CODES = {
    "en": "en-IN",
    "hi": "hi-IN",
    "bn": "bn-IN",
    "ta": "ta-IN",
    "te": "te-IN",
}

# ── Triage Severity Levels 
SEVERITY_LEVELS = {
    "mild":      {"color": "#2ecc71", "label": "Mild",      "icon": "🟢"},
    "moderate":  {"color": "#f39c12", "label": "Moderate",  "icon": "🟡"},
    "severe":    {"color": "#e67e22", "label": "Severe",    "icon": "🟠"},
    "emergency": {"color": "#e74c3c", "label": "Emergency", "icon": "🔴"},
}

# Emergency contacts (India)
EMERGENCY_CONTACTS = {
    "National Emergency": "112",
    "Ambulance":          "108",
    "Police":             "100",
    "Fire":               "101",
}

# ── Prescription Defaults
PRESCRIPTION_CLINIC_NAME    = "MediAssist AI Clinic"
PRESCRIPTION_DISCLAIMER     = (
    "This prescription is AI-generated for informational purposes only. "
    "Please consult a licensed medical professional before taking any medication."
)

# ── Quick sanity-check 
if __name__ == "__main__":
    print(f"✅ App          : {APP_NAME} v{APP_VERSION}")
    print(f"✅ Gemini Model : {GEMINI_MODEL}")
    print(f"✅ DB Path      : {SQLITE_DB_PATH}")
    print(f"✅ ChromaDB     : {CHROMA_DB_PATH}")
    print(f"✅ Medical KB   : {MEDICAL_KB_PATH}")
    print(f"✅ Chunk Size   : {CHUNK_SIZE} | Overlap: {CHUNK_OVERLAP} | Top-K: {RETRIEVER_TOP_K}")
    print("Config loaded successfully.")