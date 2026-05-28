
import streamlit as st
from core.database import db
import config


def render_history_page():
    st.markdown(
        '<div class="page-header">📋 Consultation History</div>'
        '<div class="page-subheader">Your past sessions and diagnoses</div>',
        unsafe_allow_html=True,
    )

    pid = st.session_state.patient_id
    sessions = db.get_sessions(pid, limit=50)

    if not sessions:
        st.info("No past consultations found. Start a new consultation from the Consult page.")
        return

    # Search / filter
    col1, col2 = st.columns([3, 1])
    with col1:
        search = st.text_input("🔍 Search by complaint or diagnosis",
                               placeholder="e.g. fever, headache...")
    with col2:
        severity_filter = st.selectbox(
            "Filter by severity",
            ["All", "Mild", "Moderate", "Severe", "Emergency"],
        )

    filtered = sessions
    if search:
        q = search.lower()
        filtered = [
            s for s in sessions
            if q in (s.get("chief_complaint") or "").lower()
            or q in (s.get("diagnosis") or "").lower()
        ]
    if severity_filter != "All":
        filtered = [s for s in filtered
                    if (s.get("severity") or "mild").lower() == severity_filter.lower()]

    st.markdown(f"**{len(filtered)} session(s) found**")
    st.markdown("---")

    for session in filtered:
        severity = session.get("severity") or "mild"
        cfg      = config.SEVERITY_LEVELS.get(severity, config.SEVERITY_LEVELS["mild"])
        date     = (session.get("started_at") or "")[:10]
        complaint = session.get("chief_complaint") or "No complaint recorded"
        diagnosis = session.get("diagnosis") or "Ongoing / No diagnosis"

        with st.expander(
            f"{cfg['icon']} {date}  —  {complaint[:60]}", expanded=False
        ):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**Date:** {date}")
                st.markdown(f"**Complaint:** {complaint}")
                st.markdown(f"**Diagnosis:** {diagnosis}")
            with col2:
                st.markdown(
                    f"**Severity:** "
                    f"<span style='color:{cfg['color']}; font-weight:600'>"
                    f"{cfg['icon']} {cfg['label']}</span>",
                    unsafe_allow_html=True,
                )
                st.markdown(f"**Language:** {session.get('language', 'en').upper()}")

            # Show messages
            msgs = db.get_messages(session["id"])
            if msgs:
                st.markdown("**Conversation:**")
                for m in msgs:
                    role_label = "👤 You" if m["role"] == "user" else "🏥 MediAssist"
                    st.markdown(
                        f"<div style='background:{'#f0faf5' if m['role']=='assistant' else '#fff'};"
                        f"border-left:3px solid {'#2ecc71' if m['role']=='assistant' else '#1a6b4a'};"
                        f"padding:8px 12px; margin:4px 0; border-radius:0 6px 6px 0; font-size:0.88rem'>"
                        f"<b>{role_label}:</b> {(m['content'] or '')[:400]}"
                        f"{'...' if len(m['content'] or '') > 400 else ''}</div>",
                        unsafe_allow_html=True,
                    )