import json
import streamlit as st
from core.database import db


def render_profile_page():
    st.markdown(
        '<div class="page-header">👤 My Profile</div>'
        '<div class="page-subheader">Your personal health profile</div>',
        unsafe_allow_html=True,
    )

    pid     = st.session_state.patient_id
    patient = db.get_patient(pid)
    if not patient:
        st.error("Profile not found.")
        return

    # Parse JSON fields
    try:
        allergies   = json.loads(patient.get("allergies", "[]"))
    except Exception:
        allergies   = []
    try:
        conditions  = json.loads(patient.get("chronic_conditions", "[]"))
    except Exception:
        conditions  = []

    tab_view, tab_edit = st.tabs(["👁 View", "✏️ Edit"])

    with tab_view:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"**Name:** {patient.get('name','')}")
            st.markdown(f"**Age:** {patient.get('age','')} years")
            st.markdown(f"**Gender:** {patient.get('gender','').title()}")
            st.markdown(f"**Blood Group:** {patient.get('blood_group') or 'Not set'}")
        with c2:
            st.markdown(f"**Language:** {'Hindi' if patient.get('language')=='hi' else 'English'}")
            st.markdown(f"**Member since:** {(patient.get('created_at') or '')[:10]}")

        if allergies:
            st.error(f"⚠️ Known Allergies: {', '.join(allergies)}")
        else:
            st.success("✅ No known allergies")

        if conditions:
            st.warning(f"🏥 Chronic Conditions: {', '.join(conditions)}")
        else:
            st.info("No chronic conditions recorded")

        # Stats
        st.markdown("---")
        st.markdown("#### 📊 Health Summary")
        sessions      = db.get_sessions(pid)
        prescriptions = db.get_prescriptions(pid)
        lab_reports   = db.get_lab_reports(pid)
        vitals        = db.get_vitals(pid, limit=1)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Consultations",  len(sessions))
        c2.metric("Prescriptions",  len(prescriptions))
        c3.metric("Lab Reports",    len(lab_reports))
        c4.metric("Vitals Logged",  len(db.get_vitals(pid, limit=100)))

        if vitals:
            st.markdown(f"**Latest Vitals:** {db.get_latest_vitals_summary(pid)}")

    with tab_edit:
        with st.form("edit_profile_form"):
            c1, c2 = st.columns(2)
            with c1:
                new_name = st.text_input("Name", value=patient.get("name", ""))
                new_age  = st.number_input("Age", 1, 120,
                                           value=int(patient.get("age") or 25))
            with c2:
                new_gender = st.selectbox(
                    "Gender",
                    ["male", "female", "other"],
                    index=["male","female","other"].index(
                        patient.get("gender","male")
                    ),
                )
                new_bg = st.selectbox(
                    "Blood Group",
                    ["", "A+","A-","B+","B-","O+","O-","AB+","AB-"],
                    index=(
                        ["","A+","A-","B+","B-","O+","O-","AB+","AB-"].index(
                            patient.get("blood_group") or ""
                        ) if patient.get("blood_group") in
                        ["","A+","A-","B+","B-","O+","O-","AB+","AB-"]
                        else 0
                    ),
                )

            new_allergies = st.text_input(
                "Allergies (comma separated)",
                value=", ".join(allergies),
            )
            new_conditions = st.text_input(
                "Chronic Conditions (comma separated)",
                value=", ".join(conditions),
            )
            new_lang = st.selectbox(
                "Language",
                ["English", "Hindi"],
                index=0 if patient.get("language","en") == "en" else 1,
            )

            save = st.form_submit_button("💾 Save Changes", use_container_width=True)

        if save:
            db.update_patient(
                pid,
                name=new_name.strip(),
                age=int(new_age),
                gender=new_gender,
                blood_group=new_bg,
                allergies=[a.strip() for a in new_allergies.split(",") if a.strip()],
                chronic_conditions=[c.strip() for c in new_conditions.split(",") if c.strip()],
                language="en" if new_lang == "English" else "hi",
            )
            st.session_state.patient_data = db.get_patient(pid)
            st.success("✅ Profile updated successfully!")
            st.rerun()