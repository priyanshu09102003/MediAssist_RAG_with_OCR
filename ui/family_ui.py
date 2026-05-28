import streamlit as st
from core.database import db


def render_family_page():
    st.markdown(
        '<div class="page-header">👨‍👩‍👧 Family Profiles</div>'
        '<div class="page-subheader">Manage health profiles for your family members</div>',
        unsafe_allow_html=True,
    )

    pid     = st.session_state.patient_id
    members = db.get_family_members(pid)

    # Existing members
    if members:
        st.markdown(f"#### {len(members)} Family Member(s)")
        for m in members:
            with st.expander(
                f"👤 {m['name']} ({m.get('relation','')}) — "
                f"{m.get('age','?')} yrs",
                expanded=False,
            ):
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown(f"**Name:** {m['name']}")
                    st.markdown(f"**Relation:** {m.get('relation','')}")
                    st.markdown(f"**Age:** {m.get('age','N/A')}")
                with c2:
                    st.markdown(f"**Gender:** {m.get('gender','N/A')}")
                    st.markdown(f"**Blood Group:** {m.get('blood_group') or 'N/A'}")

                import json
                allergies = []
                try:
                    allergies = json.loads(m.get("allergies", "[]"))
                except Exception:
                    pass
                if allergies:
                    st.warning(f"⚠️ Allergies: {', '.join(allergies)}")

                # Consult button
                if st.button(f"🩺 Consult for {m['name']}",
                             key=f"consult_{m['id']}",
                             use_container_width=True):
                    st.session_state.family_member_id = m["id"]
                    st.session_state.session_id       = None
                    st.session_state.memory           = None
                    st.session_state.chat_messages    = []
                    st.session_state.current_page     = "consult"
                    st.rerun()

        st.markdown("---")

    # Add new member form
    st.markdown("#### ➕ Add Family Member")
    with st.form("add_family_form"):
        c1, c2 = st.columns(2)
        with c1:
            name     = st.text_input("Name *")
            relation = st.selectbox(
                "Relation",
                ["spouse", "child", "parent", "sibling", "grandparent", "other"],
            )
            age      = st.number_input("Age", 0, 120, 25)
        with c2:
            gender      = st.selectbox("Gender", ["male", "female", "other"])
            blood_group = st.selectbox(
                "Blood Group",
                ["", "A+", "A-", "B+", "B-", "O+", "O-", "AB+", "AB-"],
            )
            allergies_raw = st.text_input(
                "Allergies",
                placeholder="penicillin, dust (comma separated)",
            )

        add_btn = st.form_submit_button("Add Member", use_container_width=True)

    if add_btn:
        if not name.strip():
            st.error("Please enter a name.")
        else:
            allergies = [a.strip() for a in allergies_raw.split(",") if a.strip()]
            db.add_family_member(
                patient_id=pid,
                name=name.strip(),
                relation=relation,
                age=int(age),
                gender=gender,
                blood_group=blood_group,
                allergies=allergies,
            )
            st.success(f"✅ {name} added to your family profile!")
            st.rerun()