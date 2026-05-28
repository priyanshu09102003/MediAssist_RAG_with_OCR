"""
core/llm_chain.py
-----------------
The brain of MediAssist AI.

Builds a full RAG chain:
    User Query
        → Retrieve relevant medical KB chunks (ChromaDB)
        → Build rich prompt (patient profile + vitals + history + KB context)
        → Send to Gemini 1.5 Pro
        → Return structured MedicalResponse

Also handles:
    - Triage severity classification
    - Differential diagnosis extraction
    - Prescription JSON extraction
    - Language-aware responses (EN/HI)
"""

import json
import re
from dataclasses import dataclass, field
from typing import Optional

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from core.vector_store import VectorStore
from core.memory import SessionMemory, PatientContextBuilder
import config


# ── Response dataclass ────────────────────────────────────────────────────────

@dataclass
class MedicalResponse:
    answer: str                          # main response text
    severity: str = "mild"               # mild | moderate | severe | emergency
    differential: list[dict] = field(default_factory=list)
    # [{"condition": "Dengue", "likelihood": "High", "reason": "..."}]
    prescription: Optional[dict] = None
    # {"diagnosis": "...", "medications": [...], "advice": "...", "follow_up": "..."}
    sources: list[str] = field(default_factory=list)
    language: str = "en"
    needs_specialist: Optional[str] = None   # e.g. "Cardiologist"
    emergency_alert: bool = False


# ── System Prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are MediAssist AI, an expert clinical assistant trained on WHO, MedlinePlus, and global medical guidelines. You assist patients in understanding their symptoms, possible diagnoses, and treatment options.

ROLE & BEHAVIOR:
- Act like an experienced, empathetic general physician taking a detailed history
- Always ask clarifying questions if information is insufficient (one question at a time)
- Collect: onset, duration, severity (1-10), associated symptoms, past history, medications, allergies
- Never dismiss a symptom — always take the patient seriously
- Be warm, clear, and avoid unnecessary medical jargon
- If the patient writes in Hindi, respond entirely in Hindi
- If the patient writes in English, respond in English
- For mixed language (Hinglish), match their style

RESPONSE STRUCTURE (always follow this order):
1. Acknowledge the patient's concern with empathy
2. Ask 1-2 focused clarifying questions if needed (or skip if enough info)
3. Provide your clinical assessment
4. List possible conditions (differential diagnosis) if applicable
5. Recommend treatment / home care / medication if appropriate
6. State when to seek emergency care
7. Suggest specialist referral if condition warrants it

SAFETY RULES:
- Always include: "Please consult a licensed doctor for final diagnosis and treatment."
- For EMERGENCY symptoms (chest pain, stroke signs, severe bleeding, difficulty breathing, loss of consciousness) — immediately direct to emergency: "Call 112 NOW. Go to nearest ER immediately."
- Never recommend controlled substances or opioids
- Flag drug allergies before any medication suggestion

TRIAGE CLASSIFICATION (include in every response after assessment):
Classify as exactly one of: MILD | MODERATE | SEVERE | EMERGENCY
Format: [TRIAGE: MILD] or [TRIAGE: EMERGENCY] etc.

DIFFERENTIAL DIAGNOSIS (when applicable):
Format as JSON block after your response:
```differential
[
  {{"condition": "Condition Name", "likelihood": "High/Medium/Low", "reason": "Brief reason"}},
  {{"condition": "Condition Name", "likelihood": "Medium", "reason": "Brief reason"}}
]
```

PRESCRIPTION (only when treatment is clearly indicated):
Format as JSON block:
```prescription
{{
  "diagnosis": "Primary diagnosis",
  "medications": [
    {{"name": "Medicine Name", "dose": "dose", "frequency": "how often", "duration": "how long", "notes": "with food / avoid alcohol etc"}},
  ],
  "advice": "General advice, diet, rest, lifestyle",
  "follow_up": "When to come back or see a doctor"
}}
```

SPECIALIST REFERRAL (when needed):
Format: [REFER: Cardiologist] or [REFER: Dermatologist] etc.

PATIENT CONTEXT:
{patient_context}

MEDICAL KNOWLEDGE BASE:
{rag_context}

CONVERSATION HISTORY:
{chat_history_text}
"""


# ── MedicalChain ──────────────────────────────────────────────────────────────

class MedicalChain:
    """
    Main RAG chain for medical consultation.

    Usage:
        chain = MedicalChain()
        response = chain.run(
            query="I have fever and headache since 2 days",
            memory=session_memory,
            patient_context=context_builder.build(),
        )
        print(response.answer)
        print(response.severity)
    """

    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model=config.GEMINI_MODEL,
            google_api_key=config.GEMINI_API_KEY,
            temperature=config.GEMINI_TEMPERATURE,
            max_output_tokens=config.GEMINI_MAX_TOKENS,
        )
        self.vector_store = VectorStore()

    # ── Main entry point ──────────────────────────────────────────────────────

    def run(
        self,
        query: str,
        memory: SessionMemory,
        patient_context: str = "",
        language: str = "en",
        image_description: str = "",    # pre-analyzed image text from vision module
    ) -> MedicalResponse:
        """
        Run the full RAG pipeline for one user turn.

        Args:
            query             : user's text query
            memory            : active SessionMemory object
            patient_context   : built by PatientContextBuilder.build()
            language          : 'en' or 'hi'
            image_description : text output from vision analysis (if any)

        Returns:
            MedicalResponse with answer, severity, differential, prescription
        """

        # 1. Combine query with image description if present
        full_query = query
        if image_description:
            full_query = (
                f"{query}\n\n"
                f"[Image Analysis Result]: {image_description}"
            )

        # 2. Retrieve relevant medical KB chunks
        rag_context, sources = self._retrieve_context(full_query)

        # 3. Explicit language instruction — always set, overrides conversation history
        if language == "hi":
            lang_instruction = (
                "\n\n🔴 LANGUAGE OVERRIDE — MANDATORY: "
                "The patient's CURRENT message is in Hindi. "
                "You MUST respond ENTIRELY in Hindi only. "
                "Do NOT use English even if previous messages were in English."
            )
        else:
            lang_instruction = (
                "\n\n🔴 LANGUAGE OVERRIDE — MANDATORY: "
                "The patient's CURRENT message is in English. "
                "You MUST respond ENTIRELY in English only. "
                "Do NOT use Hindi even if previous messages were in Hindi."
            )

        # 4. Build system prompt
        system_content = SYSTEM_PROMPT.format(
            patient_context=patient_context or "Patient profile not yet created.",
            rag_context=rag_context,
            chat_history_text=memory.get_history_text(),
        ) + lang_instruction

        # 5. Build message list for Gemini
        messages = [
            SystemMessage(content=system_content),
            *memory.get_langchain_messages(),
            HumanMessage(content=full_query),
        ]

        # 6. Call Gemini
        try:
            response = self.llm.invoke(messages)
            raw_text = response.content
        except Exception as e:
            raw_text = (
                f"I'm sorry, I encountered a technical issue. Please try again.\n"
                f"Error: {str(e)}"
            )

        # 7. Parse structured fields from response
        severity     = self._extract_severity(raw_text)
        differential = self._extract_differential(raw_text)
        prescription = self._extract_prescription(raw_text)
        specialist   = self._extract_specialist(raw_text)
        clean_answer = self._clean_answer(raw_text)

        # 8. Update memory
        memory.add_user_message(query, persist=False)   # persisted by caller
        memory.add_assistant_message(clean_answer, persist=False)

        return MedicalResponse(
            answer=clean_answer,
            severity=severity,
            differential=differential,
            prescription=prescription,
            sources=sources,
            language=language,
            needs_specialist=specialist,
            emergency_alert=(severity == "emergency"),
        )

    # ── Vision-aware query ─────────────────────────────────────────────────────

    def run_with_image(
        self,
        query: str,
        image_bytes: bytes,
        image_mime: str,
        memory: SessionMemory,
        patient_context: str = "",
        language: str = "en",
    ) -> MedicalResponse:
        """
        Send image directly to Gemini Vision alongside the text query.
        Used when the patient uploads a photo of their condition.
        """
        import base64

        # Retrieve RAG context based on text query
        rag_context, sources = self._retrieve_context(query)

        lang_instruction = ""
        if language == "hi":
            lang_instruction = (
                "\n\n🔴 LANGUAGE OVERRIDE — MANDATORY: "
                "Respond ENTIRELY in Hindi only."
            )
        else:
            lang_instruction = (
                "\n\n🔴 LANGUAGE OVERRIDE — MANDATORY: "
                "Respond ENTIRELY in English only."
            )

        system_content = SYSTEM_PROMPT.format(
            patient_context=patient_context or "Patient profile not yet created.",
            rag_context=rag_context,
            chat_history_text=memory.get_history_text(),
        ) + lang_instruction

        # Build multimodal message with image
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")

        multimodal_content = [
            {
                "type": "text",
                "text": (
                    f"{query}\n\n"
                    "Please analyze the image provided and describe what you observe "
                    "(visible symptoms, affected area, skin condition, rash, wound, etc.) "
                    "as part of your clinical assessment."
                ),
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{image_mime};base64,{image_b64}"
                },
            },
        ]

        messages = [
            SystemMessage(content=system_content),
            *memory.get_langchain_messages(),
            HumanMessage(content=multimodal_content),
        ]

        try:
            response = self.llm.invoke(messages)
            raw_text = response.content
        except Exception as e:
            raw_text = f"Error analyzing image: {str(e)}"

        severity     = self._extract_severity(raw_text)
        differential = self._extract_differential(raw_text)
        prescription = self._extract_prescription(raw_text)
        specialist   = self._extract_specialist(raw_text)
        clean_answer = self._clean_answer(raw_text)

        memory.add_assistant_message(clean_answer, persist=False)

        return MedicalResponse(
            answer=clean_answer,
            severity=severity,
            differential=differential,
            prescription=prescription,
            sources=sources,
            language=language,
            needs_specialist=specialist,
            emergency_alert=(severity == "emergency"),
        )

    # ── Triage-only quick call ────────────────────────────────────────────────

    def classify_triage(self, symptoms: str) -> str:
        """
        Quick triage classification without full RAG.
        Returns: 'mild' | 'moderate' | 'severe' | 'emergency'
        """
        prompt = f"""
You are a triage nurse. Based on the symptoms below, classify severity.
Reply with ONLY one word: MILD, MODERATE, SEVERE, or EMERGENCY.

Symptoms: {symptoms}

Classification:"""
        try:
            resp = self.llm.invoke([HumanMessage(content=prompt)])
            text = resp.content.strip().upper()
            for level in ("EMERGENCY", "SEVERE", "MODERATE", "MILD"):
                if level in text:
                    return level.lower()
        except Exception:
            pass
        return "mild"

    # ── RAG Retrieval ─────────────────────────────────────────────────────────

    def _retrieve_context(self, query: str) -> tuple[str, list[str]]:
        """Search ChromaDB and return formatted context + source names."""
        try:
            docs_with_scores = self.vector_store.search_with_scores(
                query, k=config.RETRIEVER_TOP_K
            )
            if not docs_with_scores:
                return "No relevant medical information found.", []

            parts   = []
            sources = []
            for i, (doc, score) in enumerate(docs_with_scores, 1):
                source = doc.metadata.get("source_name", "Medical Reference")
                cat    = doc.metadata.get("category", "general")
                parts.append(
                    f"[Ref {i} | {source} | {cat} | relevance: {score:.2f}]\n"
                    f"{doc.page_content}"
                )
                if source not in sources:
                    sources.append(source)

            return "\n\n---\n\n".join(parts), sources

        except Exception as e:
            return f"Knowledge base unavailable: {e}", []

    # ── Parsers ───────────────────────────────────────────────────────────────

    def _extract_severity(self, text: str) -> str:
        match = re.search(r"\[TRIAGE:\s*(MILD|MODERATE|SEVERE|EMERGENCY)\]",
                          text, re.IGNORECASE)
        if match:
            return match.group(1).lower()
        # Fallback: keyword scan
        text_lower = text.lower()
        if any(w in text_lower for w in ["emergency", "call 112", "go to er", "ambulance"]):
            return "emergency"
        if any(w in text_lower for w in ["severe", "critical", "serious", "urgent"]):
            return "severe"
        if any(w in text_lower for w in ["moderate", "concerning", "monitor"]):
            return "moderate"
        return "mild"

    def _extract_differential(self, text: str) -> list[dict]:
        match = re.search(
            r"```differential\s*(.*?)```", text, re.DOTALL | re.IGNORECASE
        )
        if not match:
            return []
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            return []

    def _extract_prescription(self, text: str) -> Optional[dict]:
        match = re.search(
            r"```prescription\s*(.*?)```", text, re.DOTALL | re.IGNORECASE
        )
        if not match:
            return None
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            return None

    def _extract_specialist(self, text: str) -> Optional[str]:
        match = re.search(r"\[REFER:\s*([^\]]+)\]", text, re.IGNORECASE)
        return match.group(1).strip() if match else None

    def _clean_answer(self, text: str) -> str:
        """Remove structured JSON blocks and triage/refer tags from display text."""
        text = re.sub(r"```differential.*?```", "", text, flags=re.DOTALL)
        text = re.sub(r"```prescription.*?```", "", text, flags=re.DOTALL)
        text = re.sub(r"\[TRIAGE:\s*\w+\]", "", text)
        text = re.sub(r"\[REFER:\s*[^\]]+\]", "", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()


# ── Module-level singleton ────────────────────────────────────────────────────
# from core.llm_chain import medical_chain
medical_chain = MedicalChain()



if __name__ == "__main__":
    print("Testing MedicalChain...")

    # Create a test patient + session
    pid = db.create_patient("Test User", 30, "male", "B+",
                             allergies=["penicillin"],
                             chronic_conditions=[])
    sid = db.create_session(pid, chief_complaint="Fever test")

    from core.database import db
    from core.memory import SessionMemory, PatientContextBuilder

    memory  = SessionMemory(patient_id=pid, session_id=sid)
    context = PatientContextBuilder(patient_id=pid).build()

    chain    = MedicalChain()
    response = chain.run(
        query="I have had a fever of 102°F and a bad headache for the past 2 days. I also feel very tired.",
        memory=memory,
        patient_context=context,
        language="en",
    )

    print(f"\n✅ Response received ({len(response.answer)} chars)")
    print(f"   Severity     : {response.severity}")
    print(f"   Differential : {len(response.differential)} conditions")
    print(f"   Prescription : {'Yes' if response.prescription else 'No'}")
    print(f"   Specialist   : {response.needs_specialist or 'None'}")
    print(f"   Sources      : {response.sources}")
    print(f"\n--- Answer Preview ---\n{response.answer[:500]}...")
    print("\nMedicalChain working correctly.")