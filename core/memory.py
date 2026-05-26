
import json
from dataclasses import dataclass, field
from typing import Optional
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage

from core.database import db
import config


# ── Chat Message 

@dataclass
class ChatMessage:
    role: str          #For the role of the chat
    content: str
    input_type: str = "text"   
    image_path: Optional[str] = None


# ── SessionMemory

class SessionMemory:

    MAX_HISTORY_TURNS = 20 

    def __init__(
        self,
        patient_id: int,
        session_id: int,
        language: str = "en",
        family_member_id: Optional[int] = None,
    ):
        self.patient_id       = patient_id
        self.session_id       = session_id
        self.language         = language
        self.family_member_id = family_member_id
        self._history: list[ChatMessage] = []

        # Load any existing messages if session was resumed
        self._load_existing_messages()

    def _load_existing_messages(self):
        """Reload messages from DB if resuming an existing session."""
        msgs = db.get_messages(self.session_id)
        for m in msgs:
            if m["role"] in ("user", "assistant"):
                self._history.append(ChatMessage(
                    role=m["role"],
                    content=m["content"],
                    input_type=m.get("input_type", "text"),
                    image_path=m.get("image_path"),
                ))

    def add_user_message(
        self,
        content: str,
        input_type: str = "text",
        image_path: Optional[str] = None,
        persist: bool = True,
    ):
        msg = ChatMessage(role="user", content=content,
                          input_type=input_type, image_path=image_path)
        self._history.append(msg)
        if persist:
            db.add_message(self.session_id, "user", content,
                           input_type=input_type, image_path=image_path)

    def add_assistant_message(self, content: str, persist: bool = True):
        msg = ChatMessage(role="assistant", content=content)
        self._history.append(msg)
        if persist:
            db.add_message(self.session_id, "assistant", content)

    def get_langchain_messages(self) -> list[BaseMessage]:
        """
        Return the last MAX_HISTORY_TURNS as LangChain message objects
        for direct use in ChatGoogleGenerativeAI.
        """
        recent = self._history[-self.MAX_HISTORY_TURNS * 2:]
        messages = []
        for msg in recent:
            if msg.role == "user":
                messages.append(HumanMessage(content=msg.content))
            elif msg.role == "assistant":
                messages.append(AIMessage(content=msg.content))
        return messages

    def get_history_text(self) -> str:
        """Plain-text version of recent history for prompt injection."""
        recent = self._history[-10:]
        lines = []
        for msg in recent:
            tag = "Patient" if msg.role == "user" else "Assistant"
            lines.append(f"{tag}: {msg.content}")
        return "\n".join(lines) if lines else "No conversation yet."

    def is_empty(self) -> bool:
        return len(self._history) == 0

    def turn_count(self) -> int:
        return len([m for m in self._history if m.role == "user"])

    def clear(self):
        self._history = []


# ── Patient Context Builder 

class PatientContextBuilder:
  
    def __init__(self, patient_id: int, family_member_id: Optional[int] = None):
        self.patient_id       = patient_id
        self.family_member_id = family_member_id

    def build(self) -> str:
        
        parts = []

        # ── Active profile 
        if self.family_member_id:
            member = self._get_family_member()
            if member:
                parts.append(self._format_profile(member, is_family=True))
        else:
            patient = db.get_patient(self.patient_id)
            if patient:
                parts.append(self._format_profile(patient, is_family=False))

        # ── Latest vitals 
        vitals_summary = db.get_latest_vitals_summary(self.patient_id)
        if vitals_summary != "No vitals recorded.":
            parts.append(f"Recent Vitals: {vitals_summary}")

        # ── Past consultation history 
        history = db.get_session_history_text(self.patient_id, limit_sessions=3)
        if history != "No previous consultation history found.":
            parts.append(f"Past Consultation History:\n{history}")

        if not parts:
            return "Patient profile not yet created."

        return "\n\n".join(parts)

    def _format_profile(self, profile: dict, is_family: bool) -> str:
        label = "Family Member Profile" if is_family else "Patient Profile"

        allergies = profile.get("allergies", "[]")
        if isinstance(allergies, str):
            try:
                allergies = json.loads(allergies)
            except Exception:
                allergies = []

        conditions = profile.get("chronic_conditions", "[]")
        if isinstance(conditions, str):
            try:
                conditions = json.loads(conditions)
            except Exception:
                conditions = []

        relation = f" ({profile.get('relation', '')})" if is_family else ""

        lines = [
            f"[{label}]",
            f"Name       : {profile.get('name', 'N/A')}{relation}",
            f"Age/Gender : {profile.get('age', 'N/A')} / {profile.get('gender', 'N/A')}",
            f"Blood Group: {profile.get('blood_group') or 'Not recorded'}",
            f"Allergies  : {', '.join(allergies) if allergies else 'None known'}",
            f"Chronic Conditions: {', '.join(conditions) if conditions else 'None'}",
        ]
        return "\n".join(lines)

    def _get_family_member(self) -> Optional[dict]:
        members = db.get_family_members(self.patient_id)
        for m in members:
            if m["id"] == self.family_member_id:
                return m
        return None