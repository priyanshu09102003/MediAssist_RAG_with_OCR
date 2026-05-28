"""
app.py
------
MediAssist AI — Main Streamlit Application

Run with:
    streamlit run app.py
"""

import streamlit as st
from pathlib import Path

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="MediAssist AI",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

import json
from core.database import db
from core.memory import SessionMemory, PatientContextBuilder

# ── Global CSS ────────────────────────────────────────────────────────────────
def inject_css():
    st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Serif+Display&display=swap');

:root {
    --primary:      #1a6b4a;
    --primary-lt:   #2e7d52;
    --accent:       #2ecc71;
    --accent-lt:    #eaf6ef;
    --danger:       #e74c3c;
    --warning:      #f39c12;
    --info:         #3498db;
    --bg:           #f7faf8;
    --card:         #ffffff;
    --border:       #d1e8d8;
    --text:         #1a2e22;
    --text-muted:   #6b8f76;
    --sidebar-bg:   #0f3d28;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    color: var(--text);
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: var(--sidebar-bg) !important;
    border-right: none;
}
[data-testid="stSidebar"] * { color: #e8f5ec !important; }
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stRadio label { color: #a8d5b8 !important; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: #ffffff !important; }

/* Main background */
.main { background: var(--bg); }
.block-container { padding: 1.5rem 2rem; }

/* Cards */
.med-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 1rem;
    box-shadow: 0 1px 4px rgba(26,107,74,0.06);
}

/* Triage badges */
.badge-mild      { background:#e8f8f0; color:#1a6b4a; border:1px solid #a8d5b8; }
.badge-moderate  { background:#fef9e7; color:#7d6108; border:1px solid #f9e79f; }
.badge-severe    { background:#fef0e8; color:#784212; border:1px solid #f5cba7; }
.badge-emergency { background:#fde8e8; color:#922b21; border:1px solid #f1948a; }
.triage-badge {
    display:inline-flex; align-items:center; gap:6px;
    padding:6px 14px; border-radius:99px;
    font-weight:600; font-size:0.88rem;
    margin-bottom: 0.75rem;
}

/* Chat bubbles */
.chat-user {
    background: var(--primary);
    color: white;
    border-radius: 18px 18px 4px 18px;
    padding: 0.75rem 1.1rem;
    margin: 0.3rem 0;
    max-width: 80%;
    margin-left: auto;
    word-wrap: break-word;
}
.chat-assistant {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 18px 18px 18px 4px;
    padding: 0.75rem 1.1rem;
    margin: 0.3rem 0;
    max-width: 85%;
    word-wrap: break-word;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
}

/* Buttons */
.stButton > button {
    background: var(--primary) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    background: var(--primary-lt) !important;
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(26,107,74,0.25) !important;
}

/* Input fields */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea,
.stSelectbox > div > div {
    border-color: var(--border) !important;
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
}
.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px rgba(46,204,113,0.15) !important;
}

/* Expanders */
.streamlit-expanderHeader {
    background: var(--accent-lt) !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    color: var(--primary) !important;
}

/* Metric cards */
[data-testid="stMetric"] {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 0.75rem 1rem;
}
                
/* Sidebar selectbox — fix invisible text on light background */
[data-testid="stSidebar"] .stSelectbox > div > div {
    background: #1f5c38 !important;
    color: #e8f5ec !important;
    border-color: #2e7d52 !important;
}
[data-testid="stSidebar"] .stSelectbox > div > div > div {
    color: #e8f5ec !important;
}
[data-testid="stSidebar"] .stSelectbox svg {
    fill: #a8d5b8 !important;
}
/* Dropdown options list */
[data-testid="stSidebar"] [data-baseweb="select"] [role="listbox"],
[data-testid="stSidebar"] [data-baseweb="popover"] {
    background: #1a4a2e !important;
    border-color: #2e7d52 !important;
}
[data-testid="stSidebar"] [data-baseweb="select"] [role="option"] {
    background: #1a4a2e !important;
    color: #e8f5ec !important;
}
[data-testid="stSidebar"] [data-baseweb="select"] [role="option"]:hover {
    background: #2e7d52 !important;
}
                
/* User bubble — keep primary green, slightly deeper */
.chat-user {
    background: #155a3e !important;
}
/* Assistant bubble — warm off-white with a faint green tint */
.chat-assistant {
    background: #f0f7f3 !important;
    border-color: #c2deca !important;
}

/* Hide Streamlit branding */
#MainMenu, footer, header { visibility: hidden; }

/* Scrollable chat container */
.chat-container {
    height: 62vh;
    overflow-y: auto;
    padding: 1rem;
    background: var(--bg);
    border-radius: 12px;
    border: 1px solid var(--border);
    margin-bottom: 1rem;
}

/* Emergency alert */
.emergency-alert {
    background: #fde8e8;
    border: 2px solid #e74c3c;
    border-radius: 10px;
    padding: 1rem 1.25rem;
    margin-bottom: 1rem;
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0%, 100% { box-shadow: 0 0 0 0 rgba(231,76,60,0.3); }
    50%       { box-shadow: 0 0 0 8px rgba(231,76,60,0); }
}

/* Page header */
.page-header {
    font-family: 'DM Serif Display', serif;
    font-size: 1.6rem;
    color: var(--primary);
    margin-bottom: 0.25rem;
}
.page-subheader {
    color: var(--text-muted);
    font-size: 0.9rem;
    margin-bottom: 1.25rem;
}

/* Nav button active state */
.nav-active > button {
    background: rgba(46,204,113,0.15) !important;
    border-left: 3px solid var(--accent) !important;
}
</style>
""", unsafe_allow_html=True)


# ── Session state initializer ─────────────────────────────────────────────────
def init_session_state():
    defaults = {
        "patient_id":        None,
        "patient_data":      None,
        "session_id":        None,
        "memory":            None,
        "current_page":      "consult",
        "language":          "en",
        "family_member_id":  None,
        "show_ayush":        True,
        "chat_messages":     [],   # list of {"role","content","meta"} for display
        "triage_result":     None,
        "last_response":     None,
        "onboarding_done":   False,
        "input_mode":        "text",  # text | voice | image
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


# ── Sidebar ───────────────────────────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        # Logo
        st.markdown("""
        <div style='text-align:center; padding: 1rem 0 0.5rem'>
            <div style='font-size:2.5rem'>🏥</div>
            <div style='font-family:"DM Serif Display",serif; font-size:1.3rem;
                        color:white; font-weight:400'>MediAssist AI</div>
            <div style='font-size:0.75rem; color:#7dc99a; margin-top:2px'>
                Your Clinical AI Assistant</div>
        </div>
        <hr style='border-color:#1f5c38; margin:0.75rem 0'>
        """, unsafe_allow_html=True)

        # Patient info chip
        if st.session_state.patient_data:
            p = st.session_state.patient_data
            name = p.get("name", "Patient")
            age  = p.get("age", "")
            gender = p.get("gender", "")
            st.markdown(f"""
            <div style='background:rgba(46,204,113,0.12); border:1px solid #2e7d52;
                        border-radius:10px; padding:10px 14px; margin-bottom:1rem'>
                <div style='font-weight:600; font-size:0.95rem'>{name}</div>
                <div style='font-size:0.78rem; color:#7dc99a'>
                    {age} yrs • {gender.title()} • {p.get("blood_group") or "Blood group N/A"}
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Navigation
        st.markdown("<div style='font-size:0.7rem; color:#5a8a6a; "
                    "text-transform:uppercase; letter-spacing:1px; "
                    "margin-bottom:6px'>Navigation</div>", unsafe_allow_html=True)

        pages = [
            ("🩺", "Consult",       "consult"),
            ("📋", "History",       "history"),
            ("📊", "Vitals",        "vitals"),
            ("💊", "Prescriptions", "prescriptions"),
            ("👨‍👩‍👧", "Family",     "family"),
            ("👤", "My Profile",    "profile"),
        ]

        for icon, label, key in pages:
            is_active = st.session_state.current_page == key
            btn_style = "nav-active" if is_active else ""
            col = st.container()
            with col:
                if st.button(f"{icon}  {label}", key=f"nav_{key}",
                             use_container_width=True):
                    st.session_state.current_page = key
                    st.rerun()

        st.markdown("<hr style='border-color:#1f5c38; margin:1rem 0'>",
                    unsafe_allow_html=True)

        # Settings
        st.markdown("<div style='font-size:0.7rem; color:#5a8a6a; "
                    "text-transform:uppercase; letter-spacing:1px; "
                    "margin-bottom:6px'>Settings</div>", unsafe_allow_html=True)

        lang = st.selectbox(
            "Language",
            options=["English", "Hindi"],
            index=0 if st.session_state.language == "en" else 1,
            key="lang_select",
        )
        st.session_state.language = "en" if lang == "English" else "hi"

        st.session_state.show_ayush = st.toggle(
            "Show AYUSH Remedies",
            value=st.session_state.show_ayush,
        )

        # Family member selector
        if st.session_state.patient_id:
            members = db.get_family_members(st.session_state.patient_id)
            if members:
                st.markdown("<div style='margin-top:0.5rem'></div>",
                            unsafe_allow_html=True)
                options = ["Myself"] + [f"{m['name']} ({m['relation']})" for m in members]
                selected = st.selectbox("Consulting for", options=options,
                                        key="family_select")
                if selected == "Myself":
                    st.session_state.family_member_id = None
                else:
                    idx = options.index(selected) - 1
                    st.session_state.family_member_id = members[idx]["id"]

        st.markdown("<hr style='border-color:#1f5c38; margin:1rem 0'>",
                    unsafe_allow_html=True)
        st.markdown("""
        <div style='font-size:0.7rem; color:#4a7a5a; text-align:center;
                    line-height:1.6; padding:0 0.5rem'>
            ⚕️ AI-assisted. Not a licensed doctor.<br>
            Always consult a professional for serious conditions.
        </div>
        """, unsafe_allow_html=True)


# ── Onboarding ────────────────────────────────────────────────────────────────
def render_onboarding():
    st.markdown("""
    <div style='max-width:580px; margin:3rem auto 0'>
        <div style='text-align:center; margin-bottom:2rem'>
            <div style='font-size:3rem'>🏥</div>
            <div style='font-family:"DM Serif Display",serif; font-size:2rem;
                        color:#1a6b4a'>Welcome to MediAssist AI</div>
            <div style='color:#6b8f76; margin-top:0.5rem'>
                Your personal AI clinical assistant.<br>
                Let's set up your health profile to get started.
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    with st.form("onboarding_form"):
        st.markdown("#### 👤 Basic Information")
        col1, col2 = st.columns(2)
        with col1:
            name = st.text_input("Full Name *", placeholder="e.g. Priya Sharma")
            age  = st.number_input("Age *", min_value=1, max_value=120, value=25)
        with col2:
            gender = st.selectbox("Gender *", ["male", "female", "other"])
            blood_group = st.selectbox(
                "Blood Group",
                ["", "A+", "A-", "B+", "B-", "O+", "O-", "AB+", "AB-"],
            )

        st.markdown("#### ⚠️ Medical History")
        allergies_raw = st.text_input(
            "Known Allergies",
            placeholder="e.g. penicillin, aspirin, dust (comma separated)",
        )
        conditions_raw = st.text_input(
            "Chronic Conditions",
            placeholder="e.g. diabetes, hypertension (comma separated)",
        )

        language = st.selectbox("Preferred Language", ["English", "Hindi"])

        submitted = st.form_submit_button(
            "✅ Create My Profile & Start",
            use_container_width=True,
        )

    if submitted:
        if not name.strip():
            st.error("Please enter your name.")
            return

        allergies   = [a.strip() for a in allergies_raw.split(",") if a.strip()]
        conditions  = [c.strip() for c in conditions_raw.split(",") if c.strip()]
        lang_code   = "en" if language == "English" else "hi"

        pid = db.create_patient(
            name=name.strip(),
            age=int(age),
            gender=gender,
            blood_group=blood_group,
            allergies=allergies,
            chronic_conditions=conditions,
            language=lang_code,
        )

        st.session_state.patient_id    = pid
        st.session_state.patient_data  = db.get_patient(pid)
        st.session_state.language      = lang_code
        st.session_state.onboarding_done = True
        st.session_state["show_onboarding"] = False
        st.success(f"Profile created! Welcome, {name} 👋")
        st.rerun()


# ── Page router ───────────────────────────────────────────────────────────────
def route_page():
    page = st.session_state.current_page

    if page == "consult":
        from ui.chat_ui import render_consult_page
        render_consult_page()
    elif page == "history":
        from ui.history_ui import render_history_page
        render_history_page()
    elif page == "vitals":
        from ui.vitals_ui import render_vitals_page
        render_vitals_page()
    elif page == "prescriptions":
        from ui.prescriptions_ui import render_prescriptions_page
        render_prescriptions_page()
    elif page == "family":
        from ui.family_ui import render_family_page
        render_family_page()
    elif page == "profile":
        from ui.profile_ui import render_profile_page
        render_profile_page()


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    inject_css()
    init_session_state()

    # Load patient from DB if session state lost on rerun
    if st.session_state.patient_id and not st.session_state.patient_data:
        st.session_state.patient_data = db.get_patient(st.session_state.patient_id)

    # If "New Profile" was clicked, show onboarding form directly
    if st.session_state.get("show_onboarding"):
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("← Back to profile selection"):
                st.session_state["show_onboarding"] = False
                st.rerun()
        render_onboarding()
        return

    # Show onboarding if no patient profile yet
    if not st.session_state.patient_id:
        all_patients = db.get_all_patients()
        if all_patients:
            # Returning user — pick profile
            st.markdown("""
            <div style='text-align:center; margin-top:3rem'>
                <div style='font-size:3rem'>🏥</div>
                <div style='font-family:"DM Serif Display",serif;
                            font-size:1.8rem; color:#1a6b4a'>
                    Welcome back to MediAssist AI</div>
            </div>
            """, unsafe_allow_html=True)

            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                options = {p["name"]: p["id"] for p in all_patients}
                selected_name = st.selectbox(
                    "Select your profile", list(options.keys())
                )
                col_a, col_b = st.columns(2)
                with col_a:
                    if st.button("Continue →", use_container_width=True):
                        pid = options[selected_name]
                        st.session_state.patient_id   = pid
                        st.session_state.patient_data = db.get_patient(pid)
                        st.rerun()
                with col_b:
                    if st.button("New Profile", use_container_width=True):
                        st.session_state["show_onboarding"] = True
                        st.rerun()
        else:
            render_onboarding()
        return

    render_sidebar()
    route_page()


if __name__ == "__main__":
    main()