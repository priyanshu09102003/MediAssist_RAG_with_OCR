
import json
import streamlit as st
from datetime import datetime
from core.database import db
from utils.pdf_generator import generate_prescription_pdf


def render_prescriptions_page():
    st.markdown(
        '<div class="page-header">💊 Prescriptions</div>'
        '<div class="page-subheader">All your AI-generated prescriptions</div>',
        unsafe_allow_html=True,
    )

    pid          = st.session_state.patient_id
    patient      = st.session_state.patient_data
    prescriptions = db.get_prescriptions(pid)

    if not prescriptions:
        st.info("No prescriptions yet. A prescription is generated during consultation "
                "when the AI determines treatment is needed.")
        return

    st.markdown(f"**{len(prescriptions)} prescription(s) on file**")

    for i, rx in enumerate(prescriptions):
        date      = (rx.get("created_at") or "")[:10]
        diagnosis = rx.get("diagnosis") or "See consultation"
        meds      = rx.get("medications") or []
        n_meds    = len(meds)

        with st.expander(f"💊 {date}  —  {diagnosis[:50]}", expanded=(i == 0)):
            col1, col2 = st.columns([2, 1])

            with col1:
                st.markdown(f"**Diagnosis:** {diagnosis}")
                if meds:
                    st.markdown("**Medications:**")
                    for m in meds:
                        st.markdown(
                            f"• **{m.get('name','')}** — "
                            f"{m.get('dose','')} | "
                            f"{m.get('frequency','')} | "
                            f"{m.get('duration','')}",
                        )
                        if m.get("notes"):
                            st.caption(f"  ℹ️ {m['notes']}")
                if rx.get("advice"):
                    st.info(f"📝 {rx['advice']}")
                if rx.get("follow_up"):
                    st.warning(f"🔄 {rx['follow_up']}")

            with col2:
                st.markdown(f"**Date:** {date}")
                st.markdown(f"**Medications:** {n_meds}")

                # Re-generate PDF for download
                if patient:
                    try:
                        allergies = json.loads(patient.get("allergies", "[]"))
                    except Exception:
                        allergies = []

                    pdf_bytes = generate_prescription_pdf(
                        patient_name=patient.get("name", "Patient"),
                        patient_age=patient.get("age", 0),
                        patient_gender=patient.get("gender", ""),
                        diagnosis=diagnosis,
                        medications=meds,
                        advice=rx.get("advice", ""),
                        follow_up=rx.get("follow_up", ""),
                        session_id=rx.get("session_id", 0),
                        blood_group=patient.get("blood_group", ""),
                        allergies=allergies,
                    )

                    fname = (
                        f"prescription_{patient.get('name','p').replace(' ','_')}"
                        f"_{date}.pdf"
                    )
                    st.download_button(
                        "⬇️ Download PDF",
                        data=pdf_bytes,
                        file_name=fname,
                        mime="application/pdf",
                        use_container_width=True,
                        key=f"dl_{rx.get('id', i)}",
                    )