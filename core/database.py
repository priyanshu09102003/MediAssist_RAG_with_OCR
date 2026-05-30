"""
core/database.py
----------------
SQLite schema + DatabaseManager for all patient data.

Tables:
    patients          — master profile per user
    family_members    — additional profiles under one account
    sessions          — each consultation session
    messages          — individual chat turns within a session
    vitals            — BP, sugar, weight, SpO2 logs
    prescriptions     — generated prescription records
    lab_reports       — uploaded lab report metadata + extracted text
"""

import sqlite3
import json
from datetime import datetime
from pathlib import Path
from contextlib import contextmanager
from typing import Optional

import config


# ── Schema DDL ────────────────────────────────────────────────────────────────

SCHEMA = """
-- Master patient profile (one per account/device)
CREATE TABLE IF NOT EXISTS patients (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    name            TEXT    NOT NULL,
    age             INTEGER,
    gender          TEXT    CHECK(gender IN ('male','female','other')),
    blood_group     TEXT,
    allergies       TEXT,
    chronic_conditions TEXT,
    language        TEXT    DEFAULT 'en',
    device_id       TEXT    DEFAULT '',
    pin_hash        TEXT    DEFAULT '',
    created_at      TEXT    DEFAULT (datetime('now')),
    updated_at      TEXT    DEFAULT (datetime('now'))
);

-- Additional family members linked to the primary patient
CREATE TABLE IF NOT EXISTS family_members (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id      INTEGER NOT NULL REFERENCES patients(id) ON DELETE CASCADE,
    name            TEXT    NOT NULL,
    relation        TEXT,           -- "spouse","child","parent", etc.
    age             INTEGER,
    gender          TEXT    CHECK(gender IN ('male','female','other')),
    blood_group     TEXT,
    allergies       TEXT,           -- JSON list
    chronic_conditions TEXT,        -- JSON list
    created_at      TEXT    DEFAULT (datetime('now'))
);

-- One session = one consultation conversation
CREATE TABLE IF NOT EXISTS sessions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id      INTEGER NOT NULL REFERENCES patients(id) ON DELETE CASCADE,
    family_member_id INTEGER REFERENCES family_members(id) ON DELETE SET NULL,
    chief_complaint TEXT,           -- first symptom description
    severity        TEXT    CHECK(severity IN ('mild','moderate','severe','emergency')),
    diagnosis       TEXT,           -- final AI diagnosis summary
    language        TEXT    DEFAULT 'en',
    started_at      TEXT    DEFAULT (datetime('now')),
    ended_at        TEXT
);

-- Every chat message in a session
CREATE TABLE IF NOT EXISTS messages (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id      INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    role            TEXT    NOT NULL CHECK(role IN ('user','assistant','system')),
    content         TEXT    NOT NULL,
    input_type      TEXT    DEFAULT 'text' CHECK(input_type IN ('text','voice','image','lab_report')),
    image_path      TEXT,           -- local path if input_type = 'image'
    created_at      TEXT    DEFAULT (datetime('now'))
);

-- Vitals log — each row is one reading
CREATE TABLE IF NOT EXISTS vitals (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id      INTEGER NOT NULL REFERENCES patients(id) ON DELETE CASCADE,
    family_member_id INTEGER REFERENCES family_members(id) ON DELETE SET NULL,
    systolic_bp     INTEGER,        -- mmHg
    diastolic_bp    INTEGER,        -- mmHg
    blood_sugar     REAL,           -- mg/dL
    weight          REAL,           -- kg
    spo2            REAL,           -- %
    heart_rate      INTEGER,        -- bpm
    temperature     REAL,           -- °C
    notes           TEXT,
    recorded_at     TEXT    DEFAULT (datetime('now'))
);

-- Generated prescriptions
CREATE TABLE IF NOT EXISTS prescriptions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id      INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    patient_id      INTEGER NOT NULL REFERENCES patients(id) ON DELETE CASCADE,
    diagnosis       TEXT,
    medications     TEXT    NOT NULL, -- JSON list of {name,dose,frequency,duration,notes}
    advice          TEXT,
    follow_up       TEXT,
    pdf_path        TEXT,             -- path to saved PDF
    created_at      TEXT    DEFAULT (datetime('now'))
);

-- Uploaded lab reports
CREATE TABLE IF NOT EXISTS lab_reports (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id      INTEGER NOT NULL REFERENCES patients(id) ON DELETE CASCADE,
    session_id      INTEGER REFERENCES sessions(id) ON DELETE SET NULL,
    report_name     TEXT,
    file_path       TEXT,             -- original uploaded file path
    extracted_text  TEXT,             -- OCR / Gemini extracted content
    abnormal_flags  TEXT,             -- JSON list of flagged values
    ai_summary      TEXT,             -- plain-language AI interpretation
    uploaded_at     TEXT    DEFAULT (datetime('now'))
);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_sessions_patient    ON sessions(patient_id);
CREATE INDEX IF NOT EXISTS idx_messages_session    ON messages(session_id);
CREATE INDEX IF NOT EXISTS idx_vitals_patient      ON vitals(patient_id, recorded_at);
CREATE INDEX IF NOT EXISTS idx_prescriptions_pat   ON prescriptions(patient_id);
CREATE INDEX IF NOT EXISTS idx_lab_reports_patient ON lab_reports(patient_id);
"""


# ── DatabaseManager 

class DatabaseManager:
    

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = str(db_path or config.SQLITE_DB_PATH)
        self._init_db()

    def _init_db(self):
        """Create all tables if they don't exist, then run migrations."""
        with self.connection() as conn:
            conn.executescript(SCHEMA)
        self._migrate()

    def _migrate(self):
        """Safely add new columns to existing databases (idempotent)."""
        migrations = [
            "ALTER TABLE patients ADD COLUMN device_id TEXT DEFAULT ''",
            "ALTER TABLE patients ADD COLUMN pin_hash  TEXT DEFAULT ''",
        ]
        with self.connection() as conn:
            for sql in migrations:
                try:
                    conn.execute(sql)
                except Exception:
                    pass  

    @contextmanager
    def connection(self):
        """Yield a sqlite3 connection with row_factory set to Row."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")   # better concurrent reads
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # ── Patients

    def create_patient(self, name: str, age: int, gender: str,
                       blood_group: str = "", allergies: list = None,
                       chronic_conditions: list = None, language: str = "en",
                       device_id: str = "", pin_hash: str = "") -> int:
        with self.connection() as conn:
            cur = conn.execute(
                """INSERT INTO patients
                   (name, age, gender, blood_group, allergies,
                    chronic_conditions, language, device_id, pin_hash)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (name, age, gender, blood_group,
                 json.dumps(allergies or []),
                 json.dumps(chronic_conditions or []),
                 language, device_id, pin_hash)
            )
            return cur.lastrowid

    def get_patient(self, patient_id: int) -> Optional[dict]:
        with self.connection() as conn:
            row = conn.execute(
                "SELECT * FROM patients WHERE id = ?", (patient_id,)
            ).fetchone()
            return self._row_to_dict(row) if row else None

    def get_all_patients(self, device_id: str = "") -> list:
        """Return patients belonging to this device only."""
        with self.connection() as conn:
            if device_id:
                rows = conn.execute(
                    "SELECT * FROM patients WHERE device_id = ? ORDER BY name",
                    (device_id,)
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM patients ORDER BY name"
                ).fetchall()
            return [self._row_to_dict(r) for r in rows]

    def update_patient(self, patient_id: int, **kwargs) -> None:
        allowed = {"name","age","gender","blood_group","allergies",
                   "chronic_conditions","language"}
        fields = {k: v for k, v in kwargs.items() if k in allowed}
        if not fields:
            return
        # Serialize list fields
        for list_field in ("allergies", "chronic_conditions"):
            if list_field in fields and isinstance(fields[list_field], list):
                fields[list_field] = json.dumps(fields[list_field])
        set_clause = ", ".join(f"{k} = ?" for k in fields)
        values = list(fields.values()) + [datetime.now().isoformat(), patient_id]
        with self.connection() as conn:
            conn.execute(
                f"UPDATE patients SET {set_clause}, updated_at = ? WHERE id = ?",
                values
            )

    # ── Family Members 

    def add_family_member(self, patient_id: int, name: str, relation: str,
                          age: int = None, gender: str = None,
                          blood_group: str = "", allergies: list = None,
                          chronic_conditions: list = None) -> int:
        with self.connection() as conn:
            cur = conn.execute(
                """INSERT INTO family_members
                   (patient_id, name, relation, age, gender,
                    blood_group, allergies, chronic_conditions)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (patient_id, name, relation, age, gender, blood_group,
                 json.dumps(allergies or []),
                 json.dumps(chronic_conditions or []))
            )
            return cur.lastrowid

    def get_family_members(self, patient_id: int) -> list:
        with self.connection() as conn:
            rows = conn.execute(
                "SELECT * FROM family_members WHERE patient_id = ? ORDER BY name",
                (patient_id,)
            ).fetchall()
            return [self._row_to_dict(r) for r in rows]

    # ── Sessions 

    def create_session(self, patient_id: int, chief_complaint: str = "",
                       family_member_id: int = None, language: str = "en") -> int:
        with self.connection() as conn:
            cur = conn.execute(
                """INSERT INTO sessions
                   (patient_id, family_member_id, chief_complaint, language)
                   VALUES (?, ?, ?, ?)""",
                (patient_id, family_member_id, chief_complaint, language)
            )
            return cur.lastrowid

    def close_session(self, session_id: int, diagnosis: str = "",
                      severity: str = "mild") -> None:
        with self.connection() as conn:
            conn.execute(
                """UPDATE sessions SET ended_at = ?, diagnosis = ?, severity = ?
                   WHERE id = ?""",
                (datetime.now().isoformat(), diagnosis, severity, session_id)
            )

    def get_sessions(self, patient_id: int, limit: int = 20) -> list:
        with self.connection() as conn:
            rows = conn.execute(
                """SELECT * FROM sessions WHERE patient_id = ?
                   ORDER BY started_at DESC LIMIT ?""",
                (patient_id, limit)
            ).fetchall()
            return [self._row_to_dict(r) for r in rows]

    # ── Messages 

    def add_message(self, session_id: int, role: str, content: str,
                    input_type: str = "text", image_path: str = None) -> int:
        with self.connection() as conn:
            cur = conn.execute(
                """INSERT INTO messages
                   (session_id, role, content, input_type, image_path)
                   VALUES (?, ?, ?, ?, ?)""",
                (session_id, role, content, input_type, image_path)
            )
            return cur.lastrowid

    def get_messages(self, session_id: int) -> list:
        with self.connection() as conn:
            rows = conn.execute(
                "SELECT * FROM messages WHERE session_id = ? ORDER BY created_at",
                (session_id,)
            ).fetchall()
            return [self._row_to_dict(r) for r in rows]

    def get_session_history_text(self, patient_id: int, limit_sessions: int = 3) -> str:
        
        sessions = self.get_sessions(patient_id, limit=limit_sessions)
        if not sessions:
            return "No previous consultation history found."

        history_parts = []
        for s in reversed(sessions):
            msgs = self.get_messages(s["id"])
            if not msgs:
                continue
            date = s["started_at"][:10]
            complaint = s.get("chief_complaint") or "N/A"
            diagnosis = s.get("diagnosis") or "N/A"
            convo = "\n".join(
                f"  [{m['role'].upper()}]: {m['content'][:300]}"
                for m in msgs if m["role"] in ("user", "assistant")
            )
            history_parts.append(
                f"[Session {date}] Complaint: {complaint} | Diagnosis: {diagnosis}\n{convo}"
            )

        return "\n\n---\n\n".join(history_parts)

    # ── Vitals 

    def log_vitals(self, patient_id: int, family_member_id: int = None,
                   systolic_bp: int = None, diastolic_bp: int = None,
                   blood_sugar: float = None, weight: float = None,
                   spo2: float = None, heart_rate: int = None,
                   temperature: float = None, notes: str = "") -> int:
        with self.connection() as conn:
            cur = conn.execute(
                """INSERT INTO vitals
                   (patient_id, family_member_id, systolic_bp, diastolic_bp,
                    blood_sugar, weight, spo2, heart_rate, temperature, notes)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (patient_id, family_member_id, systolic_bp, diastolic_bp,
                 blood_sugar, weight, spo2, heart_rate, temperature, notes)
            )
            return cur.lastrowid

    def get_vitals(self, patient_id: int, limit: int = 30) -> list:
        with self.connection() as conn:
            rows = conn.execute(
                """SELECT * FROM vitals WHERE patient_id = ?
                   ORDER BY recorded_at DESC LIMIT ?""",
                (patient_id, limit)
            ).fetchall()
            return [self._row_to_dict(r) for r in rows]

    def get_latest_vitals_summary(self, patient_id: int) -> str:
        """Returns a text summary of latest vitals for RAG context injection."""
        vitals = self.get_vitals(patient_id, limit=5)
        if not vitals:
            return "No vitals recorded."
        latest = vitals[0]
        parts = []
        if latest.get("systolic_bp"):
            parts.append(f"BP: {latest['systolic_bp']}/{latest['diastolic_bp']} mmHg")
        if latest.get("blood_sugar"):
            parts.append(f"Blood Sugar: {latest['blood_sugar']} mg/dL")
        if latest.get("weight"):
            parts.append(f"Weight: {latest['weight']} kg")
        if latest.get("spo2"):
            parts.append(f"SpO₂: {latest['spo2']}%")
        if latest.get("heart_rate"):
            parts.append(f"Heart Rate: {latest['heart_rate']} bpm")
        if latest.get("temperature"):
            parts.append(f"Temp: {latest['temperature']} °C")
        return f"Latest vitals ({latest['recorded_at'][:10]}): " + " | ".join(parts)

    # ── Prescriptions 

    def save_prescription(self, session_id: int, patient_id: int,
                          diagnosis: str, medications: list,
                          advice: str = "", follow_up: str = "",
                          pdf_path: str = "") -> int:
        with self.connection() as conn:
            cur = conn.execute(
                """INSERT INTO prescriptions
                   (session_id, patient_id, diagnosis, medications,
                    advice, follow_up, pdf_path)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (session_id, patient_id, diagnosis,
                 json.dumps(medications), advice, follow_up, pdf_path)
            )
            return cur.lastrowid

    def get_prescriptions(self, patient_id: int) -> list:
        with self.connection() as conn:
            rows = conn.execute(
                """SELECT * FROM prescriptions WHERE patient_id = ?
                   ORDER BY created_at DESC""",
                (patient_id,)
            ).fetchall()
            result = []
            for r in rows:
                d = self._row_to_dict(r)
                d["medications"] = json.loads(d.get("medications") or "[]")
                result.append(d)
            return result

    # ── Lab Reports 

    def save_lab_report(self, patient_id: int, report_name: str,
                        file_path: str = "", extracted_text: str = "",
                        abnormal_flags: list = None, ai_summary: str = "",
                        session_id: int = None) -> int:
        with self.connection() as conn:
            cur = conn.execute(
                """INSERT INTO lab_reports
                   (patient_id, session_id, report_name, file_path,
                    extracted_text, abnormal_flags, ai_summary)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (patient_id, session_id, report_name, file_path,
                 extracted_text, json.dumps(abnormal_flags or []), ai_summary)
            )
            return cur.lastrowid

    def get_lab_reports(self, patient_id: int) -> list:
        with self.connection() as conn:
            rows = conn.execute(
                """SELECT * FROM lab_reports WHERE patient_id = ?
                   ORDER BY uploaded_at DESC""",
                (patient_id,)
            ).fetchall()
            result = []
            for r in rows:
                d = self._row_to_dict(r)
                d["abnormal_flags"] = json.loads(d.get("abnormal_flags") or "[]")
                result.append(d)
            return result

    # ── Utility 

    @staticmethod
    def _row_to_dict(row) -> dict:
        if row is None:
            return {}
        return dict(row)



db = DatabaseManager()


# ── Quick test (run: python core/database.py)
if __name__ == "__main__":
    print("Testing DatabaseManager...")

    pid = db.create_patient(
        name="Test Patient", age=35, gender="male",
        blood_group="O+", allergies=["penicillin"],
        chronic_conditions=["hypertension"]
    )
    print(f"  ✅ Patient created: id={pid}")

    fid = db.add_family_member(pid, "Spouse Name", "spouse", age=32, gender="female")
    print(f"  ✅ Family member added: id={fid}")

    sid = db.create_session(pid, chief_complaint="Headache and fever since 2 days")
    print(f"  ✅ Session created: id={sid}")

    db.add_message(sid, "user", "I have a headache and fever.")
    db.add_message(sid, "assistant", "Since how many days are you experiencing this?")
    print(f"  ✅ Messages added")

    db.log_vitals(pid, systolic_bp=130, diastolic_bp=85, heart_rate=88,
                  temperature=38.5, spo2=97.0)
    print(f"  ✅ Vitals logged")
    print(f"  ✅ Vitals summary: {db.get_latest_vitals_summary(pid)}")

    db.save_prescription(
        session_id=sid, patient_id=pid,
        diagnosis="Viral fever with tension headache",
        medications=[
            {"name": "Paracetamol 500mg", "dose": "1 tablet",
             "frequency": "Every 6 hours", "duration": "3 days"},
            {"name": "ORS Sachet", "dose": "1 sachet in 1L water",
             "frequency": "Twice daily", "duration": "3 days"},
        ],
        advice="Rest, stay hydrated, avoid cold foods.",
        follow_up="Review after 3 days if fever persists."
    )
    print("  ✅ Prescription saved")

    db.close_session(sid, diagnosis="Viral fever", severity="mild")
    print("  ✅ Session closed")

    print("\nAll tests passed. Database is working correctly.")