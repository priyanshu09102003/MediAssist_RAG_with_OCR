"""
ui/chat_ui.py
-------------
Main consultation page — the heart of the MediAssist AI interface.

Features:
    - Chat interface (text / voice / image input)
    - Real-time triage badge
    - Differential diagnosis panel
    - AYUSH remedies toggle
    - Prescription generator + drug checker
    - Source references
    - New session / end session controls
"""

import io
import json
import streamlit as st
from datetime import datetime

from core.database import db
from core.llm_chain import MedicalChain, MedicalResponse
from core.memory import SessionMemory, PatientContextBuilder
from modules.triage import triage_fast, triage_from_llm_response
from modules.drug_checker import check_prescription
from modules.ayush import get_ayush_suggestions
from modules.vision import analyze_patient_image, image_result_to_text
from modules.voice import transcribe_audio, text_to_speech, detect_language
from utils.pdf_generator import generate_prescription_pdf
import config


# ── Lazy singleton for MedicalChain ──────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_chain():
    return MedicalChain()


# ── Session helpers ───────────────────────────────────────────────────────────

def ensure_session():
    """Create a new consultation session if one doesn't exist."""
    if not st.session_state.session_id:
        sid = db.create_session(
            patient_id=st.session_state.patient_id,
            family_member_id=st.session_state.family_member_id,
            language=st.session_state.language,
        )
        st.session_state.session_id = sid
        st.session_state.memory = SessionMemory(
            patient_id=st.session_state.patient_id,
            session_id=sid,
            language=st.session_state.language,
            family_member_id=st.session_state.family_member_id,
        )
        st.session_state.chat_messages = []
        st.session_state.triage_result = None
        st.session_state.last_response = None


def end_session(diagnosis: str = "", severity: str = "mild"):
    """Close current session and reset for next one."""
    if st.session_state.session_id:
        db.close_session(
            st.session_state.session_id,
            diagnosis=diagnosis,
            severity=severity,
        )
    st.session_state.session_id    = None
    st.session_state.memory        = None
    st.session_state.chat_messages = []
    st.session_state.triage_result = None
    st.session_state.last_response = None


# ── UI components ─────────────────────────────────────────────────────────────

def render_triage_badge(severity: str):
    cfg = config.SEVERITY_LEVELS.get(severity, config.SEVERITY_LEVELS["mild"])
    badge_cls = f"badge-{severity}"
    st.markdown(
        f'<span class="triage-badge {badge_cls}">'
        f'{cfg["icon"]} {cfg["label"]}</span>',
        unsafe_allow_html=True,
    )


def render_emergency_alert():
    contacts_html = " &nbsp;|&nbsp; ".join(
        f"<b>{k}</b>: {v}"
        for k, v in config.EMERGENCY_CONTACTS.items()
    )
    st.markdown(f"""
    <div class="emergency-alert">
        <div style='font-size:1.1rem; font-weight:700; color:#922b21'>
            🚨 MEDICAL EMERGENCY DETECTED
        </div>
        <div style='margin-top:4px; color:#7b241c; font-size:0.9rem'>
            Please call emergency services immediately.<br>
            {contacts_html}
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_chat_messages():
    """Render all chat messages in the conversation."""
    messages = st.session_state.chat_messages

    if not messages:
        st.markdown("""
        <div style='text-align:center; padding:3rem 1rem; color:#6b8f76'>
            <div style='font-size:2.5rem; margin-bottom:0.75rem'>🩺</div>
            <div style='font-size:1rem; font-weight:500'>
                Describe your symptoms or upload an image</div>
            <div style='font-size:0.85rem; margin-top:0.35rem'>
                You can type, speak (🎤), or upload a photo of your condition</div>
        </div>
        """, unsafe_allow_html=True)
        return

    for msg in messages:
        role    = msg["role"]
        content = msg["content"]
        meta    = msg.get("meta", {})

        if role == "user":
            with st.chat_message("user", avatar="👤"):
                # Show image thumbnail if input was image
                if meta.get("image_name"):
                    st.caption(f"📷 Image: {meta['image_name']}")
                if meta.get("voice_transcribed"):
                    st.caption("🎤 Voice input")
                st.markdown(content)

        elif role == "assistant":
            with st.chat_message("assistant", avatar="🏥"):
                # Triage badge inline
                if meta.get("severity"):
                    render_triage_badge(meta["severity"])

                st.markdown(content)

                # Differential diagnosis
                if meta.get("differential"):
                    with st.expander("🔬 Differential Diagnosis", expanded=False):
                        for item in meta["differential"]:
                            likelihood = item.get("likelihood", "")
                            color = {"High": "#e74c3c",
                                     "Medium": "#f39c12",
                                     "Low": "#2ecc71"}.get(likelihood, "#888")
                            st.markdown(
                                f"**{item.get('condition', '')}** "
                                f"<span style='color:{color}; font-size:0.8rem'>"
                                f"● {likelihood}</span>",
                                unsafe_allow_html=True,
                            )
                            st.caption(item.get("reason", ""))

                # AYUSH suggestions
                if meta.get("ayush") and st.session_state.show_ayush:
                    ayush = meta["ayush"]
                    with st.expander(
                        f"🌿 AYUSH / Ayurvedic Remedies — {ayush.condition}",
                        expanded=False,
                    ):
                        if ayush.ayurvedic:
                            st.markdown("**Ayurvedic Remedies**")
                            for r in ayush.ayurvedic:
                                st.markdown(f"🌱 **{r['remedy']}** — {r['benefit']}")
                                st.caption(f"How to use: {r['how_to']}")

                        if ayush.yoga:
                            st.markdown("**Yoga & Pranayama**")
                            for y in ayush.yoga:
                                st.markdown(
                                    f"🧘 **{y['name']}** ({y['duration']}) — {y['benefit']}"
                                )

                        if ayush.diet_do or ayush.diet_avoid:
                            col1, col2 = st.columns(2)
                            with col1:
                                st.markdown("**✅ Eat / Do**")
                                for d in ayush.diet_do:
                                    st.markdown(f"• {d}")
                            with col2:
                                st.markdown("**❌ Avoid**")
                                for d in ayush.diet_avoid:
                                    st.markdown(f"• {d}")

                        st.info(ayush.disclaimer)

                # Specialist referral
                if meta.get("specialist"):
                    st.warning(
                        f"👨‍⚕️ **Specialist Recommended**: "
                        f"Consider consulting a **{meta['specialist']}** "
                        f"for this condition."
                    )

                # Sources
                if meta.get("sources"):
                    with st.expander("📚 Medical References", expanded=False):
                        for src in meta["sources"]:
                            st.markdown(f"• {src}")

                # Prescription button
                if meta.get("prescription"):
                    rx = meta["prescription"]
                    with st.expander("💊 View Prescription", expanded=True):
                        st.markdown(f"**Diagnosis:** {rx.get('diagnosis', '')}")
                        meds = rx.get("medications", [])
                        if meds:
                            import pandas as pd
                            df = pd.DataFrame(meds)
                            st.dataframe(df, use_container_width=True, hide_index=True)
                        if rx.get("advice"):
                            st.info(f"📝 **Advice:** {rx['advice']}")
                        if rx.get("follow_up"):
                            st.warning(f"🔄 **Follow-up:** {rx['follow_up']}")

                        # Drug check before offering download
                        patient = st.session_state.patient_data
                        allergies = []
                        if patient:
                            try:
                                allergies = json.loads(
                                    patient.get("allergies", "[]")
                                )
                            except Exception:
                                pass

                        drug_result = check_prescription(
                            meds, allergies, use_fda_api=False
                        )

                        if not drug_result.safe:
                            for w in drug_result.warnings:
                                sev = w.get("severity", "medium")
                                if sev == "high":
                                    st.error(w["message"])
                                else:
                                    st.warning(w["message"])

                        # Generate and offer PDF download
                        if patient:
                            try:
                                allergies_list = json.loads(
                                    patient.get("allergies", "[]")
                                )
                            except Exception:
                                allergies_list = []

                            pdf_bytes = generate_prescription_pdf(
                                patient_name=patient.get("name", "Patient"),
                                patient_age=patient.get("age", 0),
                                patient_gender=patient.get("gender", ""),
                                diagnosis=rx.get("diagnosis", ""),
                                medications=meds,
                                advice=rx.get("advice", ""),
                                follow_up=rx.get("follow_up", ""),
                                session_id=st.session_state.session_id or 0,
                                blood_group=patient.get("blood_group", ""),
                                allergies=allergies_list,
                            )

                            # Save to DB
                            db.save_prescription(
                                session_id=st.session_state.session_id,
                                patient_id=st.session_state.patient_id,
                                diagnosis=rx.get("diagnosis", ""),
                                medications=meds,
                                advice=rx.get("advice", ""),
                                follow_up=rx.get("follow_up", ""),
                            )

                            fname = (
                                f"prescription_"
                                f"{patient.get('name','patient').replace(' ','_')}_"
                                f"{datetime.now().strftime('%Y%m%d')}.pdf"
                            )
                            st.download_button(
                                label="⬇️ Download Prescription PDF",
                                data=pdf_bytes,
                                file_name=fname,
                                mime="application/pdf",
                                use_container_width=True,
                            )


# ── Process user input ────────────────────────────────────────────────────────

def process_input(
    user_text: str,
    image_bytes: bytes = None,
    image_name: str = "",
    image_mime: str = "image/jpeg",
    voice_transcribed: bool = False,
):
    """Core function: send input through RAG chain and update chat."""
    if not user_text.strip() and not image_bytes:
        return

    chain   = get_chain()
    memory  = st.session_state.memory
    lang    = st.session_state.language

    # Auto-detect language from text and persist immediately
    if user_text:
        detected = detect_language(user_text)
        if detected in ("hi", "en"):
            lang = detected
            st.session_state.language = detected  # persist so next turn inherits it

    # Build patient context
    context = PatientContextBuilder(
        patient_id=st.session_state.patient_id,
        family_member_id=st.session_state.family_member_id,
    ).build()

    # Fast triage for immediate badge display
    fast_triage = triage_fast(user_text)

    # Add user message to display
    st.session_state.chat_messages.append({
        "role":    "user",
        "content": user_text or "(image uploaded)",
        "meta": {
            "image_name":       image_name,
            "voice_transcribed": voice_transcribed,
        },
    })

    # Persist user message to DB
    db.add_message(
        st.session_state.session_id,
        role="user",
        content=user_text or f"[Image: {image_name}]",
        input_type="image" if image_bytes else ("voice" if voice_transcribed else "text"),
    )

    # Update session chief complaint if first message
    if len(st.session_state.chat_messages) == 1 and user_text:
        db.close_session(
            st.session_state.session_id,
            diagnosis="",
            severity=fast_triage.severity,
        )
        # Re-open with complaint
        with db.connection() as conn:
            conn.execute(
                "UPDATE sessions SET chief_complaint=?, ended_at=NULL WHERE id=?",
                (user_text[:200], st.session_state.session_id),
            )

    # Call RAG chain
    with st.spinner("🩺 Analyzing..."):
        if image_bytes:
            response: MedicalResponse = chain.run_with_image(
                query=user_text or "Please analyze this image",
                image_bytes=image_bytes,
                image_mime=image_mime,
                memory=memory,
                patient_context=context,
                language=lang,
            )
        else:
            response: MedicalResponse = chain.run(
                query=user_text,
                memory=memory,
                patient_context=context,
                language=lang,
            )

    # Triage from LLM response
    triage = triage_from_llm_response(response.severity)
    st.session_state.triage_result = triage

    # AYUSH suggestions
    ayush_result = None
    if st.session_state.show_ayush:
        diagnosis_hint = ""
        if response.differential:
            diagnosis_hint = response.differential[0].get("condition", "")
        ayush_result = get_ayush_suggestions(user_text, diagnosis_hint)

    # Persist assistant message
    db.add_message(
        st.session_state.session_id,
        role="assistant",
        content=response.answer,
    )

    # Add to display messages
    st.session_state.chat_messages.append({
        "role":    "assistant",
        "content": response.answer,
        "meta": {
            "severity":     response.severity,
            "differential": response.differential,
            "prescription": response.prescription,
            "specialist":   response.needs_specialist,
            "sources":      response.sources,
            "ayush":        ayush_result,
        },
    })

    st.session_state.last_response = response

    # Auto-close session with diagnosis if prescription was generated
    if response.prescription:
        db.close_session(
            st.session_state.session_id,
            diagnosis=response.prescription.get("diagnosis", ""),
            severity=response.severity,
        )


# ── Main page renderer ────────────────────────────────────────────────────────

def render_consult_page():
    ensure_session()

    # ── Page header ───────────────────────────────────────────────────────────
    col_title, col_actions = st.columns([3, 1])
    with col_title:
        patient = st.session_state.patient_data
        name    = patient.get("name", "Patient") if patient else "Patient"
        st.markdown(
            f'<div class="page-header">🩺 Consultation</div>'
            f'<div class="page-subheader">AI-assisted clinical assessment for {name}</div>',
            unsafe_allow_html=True,
        )

    with col_actions:
        st.markdown("<div style='margin-top:1rem'></div>", unsafe_allow_html=True)
        if st.button("🔄 New Session", use_container_width=True):
            severity = ""
            if st.session_state.last_response:
                severity = st.session_state.last_response.severity
            end_session(severity=severity)
            st.rerun()

    # ── Emergency alert ───────────────────────────────────────────────────────
    if (st.session_state.triage_result and
            st.session_state.triage_result.severity == "emergency"):
        render_emergency_alert()

    # ── Chat display area ─────────────────────────────────────────────────────
    with st.container():
        render_chat_messages()

    st.markdown("<hr style='margin:0.5rem 0; border-color:#d1e8d8'>",
                unsafe_allow_html=True)

    # ── Input area ────────────────────────────────────────────────────────────
    # Input mode tabs
    tab_text, tab_voice, tab_image = st.tabs(["💬 Type", "🎤 Voice", "📷 Image"])

    # ── Text input ────────────────────────────────────────────────────────────
    with tab_text:
        with st.form("text_input_form", clear_on_submit=True):
            col_inp, col_btn = st.columns([5, 1])
            with col_inp:
                placeholder = (
                    "अपने लक्षण बताएं..." if st.session_state.language == "hi"
                    else "Describe how you are feeling... e.g. I have had a fever and headache for 2 days "
                )
                user_input = st.text_area(
                    "Message",
                    placeholder=placeholder,
                    height=80,
                    label_visibility="collapsed",
                )
            with col_btn:
                st.markdown("<div style='margin-top:1.5rem'></div>",
                            unsafe_allow_html=True)
                send = st.form_submit_button("Send →", use_container_width=True)

        if send and user_input.strip():
            process_input(user_input.strip())
            st.rerun()

    # ── Voice input ───────────────────────────────────────────────────────────
    with tab_voice:
        st.info(
            "🎤 Record your symptoms in **Hindi or English**. "
            "The AI will transcribe and respond in your language."
        )
        audio_input = st.audio_input("Click to record")

        if audio_input is not None:
            audio_bytes = audio_input.read()
            with st.spinner("🎤 Transcribing audio..."):
                result = transcribe_audio(
                    audio_bytes, language=st.session_state.language
                )

            if result.text:
                st.success(f"**Transcribed:** {result.text}")
                st.caption(f"Detected language: {result.language} | Method: {result.method}")
                if st.button("Send this message →", use_container_width=True,
                             key="send_voice"):
                    process_input(
                        result.text,
                        voice_transcribed=True,
                    )
                    st.rerun()
            else:
                st.warning(
                    f"Could not transcribe audio. "
                    f"{result.error or 'Please try again or type your message.'}"
                )

    # ── Image input ───────────────────────────────────────────────────────────
    with tab_image:
        st.info(
            "📷 Upload a photo of your condition (skin, wound, rash, eye, etc.) "
            "or a lab report for AI analysis."
        )

        upload_type = st.radio(
            "Upload type",
            ["Patient Image (wound/rash/skin)", "Lab Report (blood test/scan)"],
            horizontal=True,
            label_visibility="collapsed",
        )

        uploaded = st.file_uploader(
            "Upload image or lab report",
            type=["jpg", "jpeg", "png", "pdf"],
            label_visibility="collapsed",
        )

        if uploaded:
            # Preview
            if uploaded.type.startswith("image"):
                st.image(uploaded, caption="Preview", width=250)

            caption_input = st.text_input(
                "Describe your concern (optional)",
                placeholder="e.g. This rash appeared 2 days ago and is spreading...",
            )

            if st.button("🔍 Analyze & Consult", use_container_width=True,
                         key="send_image"):
                file_bytes = uploaded.read()

                if "Lab Report" in upload_type:
                    # Lab report analysis
                    from modules.vision import analyze_lab_report
                    with st.spinner("🔬 Analyzing lab report..."):
                        ext = f".{uploaded.name.split('.')[-1]}"
                        lab_result = analyze_lab_report(file_bytes, ext)

                    # Save to DB
                    db.save_lab_report(
                        patient_id=st.session_state.patient_id,
                        session_id=st.session_state.session_id,
                        report_name=uploaded.name,
                        extracted_text=lab_result.extracted_text,
                        abnormal_flags=lab_result.abnormal_flags,
                        ai_summary=lab_result.ai_summary,
                    )

                    # Show results
                    if lab_result.abnormal_flags:
                        st.error("⚠️ Abnormal values detected:")
                        for flag in lab_result.abnormal_flags:
                            st.markdown(f"• {flag}")

                    # Feed summary into chat
                    query = (
                        f"I uploaded my {lab_result.report_type or 'lab report'}. "
                        f"Summary: {lab_result.ai_summary[:300]}. "
                        f"{caption_input}"
                    )
                    process_input(query, image_name=uploaded.name)
                    st.rerun()

                else:
                    # Patient image
                    mime = uploaded.type or "image/jpeg"
                    query = caption_input or "Please analyze this image of my condition"
                    process_input(
                        query,
                        image_bytes=file_bytes,
                        image_name=uploaded.name,
                        image_mime=mime,
                    )
                    st.rerun()

    
    if not st.session_state.chat_messages:
        st.markdown(
            "<div style='color:#6b8f76; font-size:0.82rem; "
            "margin-top:0.75rem'>Quick start:</div>",
            unsafe_allow_html=True,
        )
        suggestions = [
            "I have a fever and headache since 2 days",
            "मुझे खांसी और सर्दी है",
            "My blood pressure is high",
            "I have a rash on my arm",
        ]
        cols = st.columns(len(suggestions))
        for col, suggestion in zip(cols, suggestions):
            with col:
                if st.button(
                    suggestion[:28] + ("..." if len(suggestion) > 28 else ""),
                    use_container_width=True,
                    key=f"sug_{suggestion[:10]}",
                ):
                    process_input(suggestion)
                    st.rerun()