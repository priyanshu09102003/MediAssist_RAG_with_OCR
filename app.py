import streamlit as st

st.set_page_config(
    page_title="MediAssist AI",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

import json
from core.database import db
from utils.security import get_device_id, hash_pin, verify_pin, validate_pin_format

# ── SVG Logo ──────────────────────────────────────────────────────────────────
def logo_svg(size=56):
    return f"""<svg width="{size}" height="{size}" viewBox="0 0 56 56" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <linearGradient id="lg1" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" style="stop-color:#2ecc71"/>
      <stop offset="100%" style="stop-color:#1a6b4a"/>
    </linearGradient>
    <linearGradient id="lg2" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" style="stop-color:#2ecc71;stop-opacity:0"/>
      <stop offset="50%" style="stop-color:#2ecc71"/>
      <stop offset="100%" style="stop-color:#2ecc71;stop-opacity:0"/>
    </linearGradient>
  </defs>
  <rect width="56" height="56" rx="14" fill="url(#lg1)"/>
  <rect x="23" y="11" width="10" height="34" rx="5" fill="white" opacity="0.95"/>
  <rect x="11" y="23" width="34" height="10" rx="5" fill="white" opacity="0.95"/>
  <path d="M9 43 L15 43 L18 37 L22 47 L26 39 L29 45 L33 41 L37 41 L47 41"
        stroke="url(#lg2)" stroke-width="2.2" fill="none"
        stroke-linecap="round" stroke-linejoin="round"/>
</svg>"""


# ── CSS ───────────────────────────────────────────────────────────────────────
def inject_css():
    vars_css = """
        --primary:#1a6b4a; --primary-dk:#0f3d28; --accent:#2ecc71;
        --bg:#f4f9f6; --bg2:#edf7f2; --card:#ffffff;
        --border:#c8e6d4; --text:#1a2e22; --muted:#5a7a65;
        --sidebar:#0f3d28; --input:#ffffff; --exp:#edf7f2;"""

    st.markdown(f"""<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Serif+Display&display=swap');
:root {{{vars_css}}}

html,body,[class*="css"]{{font-family:'DM Sans',sans-serif!important;color:var(--text)!important;}}

/* backgrounds */
.main,[data-testid="stAppViewContainer"]{{background:var(--bg)!important;}}
.block-container{{padding:1.5rem 2rem!important;background:var(--bg)!important;}}

/* ── SIDEBAR ── */
[data-testid="stSidebar"]{{
    background:var(--sidebar)!important;
    border-right:1px solid rgba(46,204,113,0.1)!important;
    min-width:320px!important;}}

[data-testid="stSidebar"] *{{color:#d4edd9!important;}}
[data-testid="stSidebar"] h1,[data-testid="stSidebar"] h2,[data-testid="stSidebar"] h3{{color:#fff!important;}}
[data-testid="stSidebar"] .stSelectbox>div>div{{background:rgba(255,255,255,0.09)!important;border:1px solid rgba(255,255,255,0.18)!important;border-radius:8px!important;}}
[data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] span,
[data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] div{{color:#fff!important;background:transparent!important;}}
[data-testid="stSidebar"] .stSelectbox svg{{fill:#8ecfa8!important;}}
[data-baseweb="popover"] [data-baseweb="menu"]{{background:#1a5c36!important;border:1px solid #2e7d52!important;border-radius:8px!important;}}
[data-baseweb="popover"] [data-baseweb="menu"] li{{color:#e8f5ec!important;}}
[data-baseweb="popover"] [data-baseweb="menu"] li:hover{{background:rgba(46,204,113,0.2)!important;}}


/* ── CHAT BUBBLES ── */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]){{
    background:linear-gradient(135deg,#1a6b4a,#2e7d52)!important;
    border-radius:20px 20px 5px 20px!important;border:none!important;
    margin-left:6%!important;margin-bottom:.75rem!important;
    box-shadow:0 4px 16px rgba(26,107,74,.3)!important;}}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) p,
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) span,
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) div{{color:#fff!important;}}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]){{
    background:var(--card)!important;
    border-radius:20px 20px 20px 5px!important;
    border:1px solid var(--border)!important;
    margin-right:6%!important;margin-bottom:.75rem!important;
    box-shadow:0 2px 12px rgba(0,0,0,.06)!important;}}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) p,
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) li{{color:var(--text)!important;}}

/* ── BUTTONS ── */
.stButton>button{{background:linear-gradient(135deg,#1a6b4a,#2e7d52)!important;
    color:#fff!important;border:none!important;border-radius:10px!important;
    font-weight:500!important;transition:all .2s!important;
    box-shadow:0 2px 8px rgba(26,107,74,.25)!important;}}
.stButton>button:hover{{background:linear-gradient(135deg,#1f7a56,#34915f)!important;
    transform:translateY(-1px)!important;box-shadow:0 6px 16px rgba(26,107,74,.35)!important;}}

/* ── INPUTS ── */
.stTextInput>div>div>input,.stTextArea>div>div>textarea,.stNumberInput>div>div>input{{
    background:var(--input)!important;color:var(--text)!important;
    border:1.5px solid var(--border)!important;border-radius:10px!important;
    font-family:'DM Sans',sans-serif!important;}}
.stTextInput>div>div>input:focus,.stTextArea>div>div>textarea:focus{{
    border-color:var(--accent)!important;box-shadow:0 0 0 3px rgba(46,204,113,.12)!important;}}
[data-baseweb="select"]>div{{background:var(--input)!important;
    border-color:var(--border)!important;border-radius:10px!important;color:var(--text)!important;}}

/* ── TABS ── */
.stTabs [data-baseweb="tab-list"]{{background:var(--bg2)!important;border-radius:10px!important;padding:4px!important;}}
.stTabs [data-baseweb="tab"]{{border-radius:8px!important;font-weight:500!important;color:var(--muted)!important;}}
.stTabs [aria-selected="true"]{{background:var(--card)!important;color:var(--primary)!important;box-shadow:0 2px 8px rgba(0,0,0,.08)!important;}}

/* ── EXPANDERS ── */
[data-testid="stExpander"]{{background:var(--exp)!important;border:1px solid var(--border)!important;border-radius:12px!important;margin-bottom:.5rem!important;}}
[data-testid="stExpander"] summary{{color:var(--primary)!important;font-weight:600!important;}}

/* ── METRICS ── */
[data-testid="stMetric"]{{background:var(--card)!important;border:1px solid var(--border)!important;border-radius:12px!important;padding:1rem 1.25rem!important;}}
[data-testid="stMetricValue"]{{color:var(--primary)!important;font-weight:700!important;}}

/* ── FORMS ── */
[data-testid="stForm"]{{background:var(--card)!important;border:1px solid var(--border)!important;border-radius:14px!important;padding:1.5rem!important;}}

/* ── TRIAGE BADGES ── */
.badge-mild{{background:#e6f9ef;color:#1a6b4a;border:1.5px solid #a8d5b8;}}
.badge-moderate{{background:#fef9e7;color:#7d6108;border:1.5px solid #f9e79f;}}
.badge-severe{{background:#fef0e8;color:#784212;border:1.5px solid #f5cba7;}}
.badge-emergency{{background:#fde8e8;color:#922b21;border:1.5px solid #f1948a;}}
.triage-badge{{display:inline-flex;align-items:center;gap:6px;padding:5px 14px;
    border-radius:99px;font-weight:600;font-size:.85rem;margin-bottom:.6rem;}}

/* ── EMERGENCY ── */
.emergency-alert{{background:#fde8e8;border:2px solid #e74c3c;border-radius:12px;
    padding:1rem 1.25rem;margin-bottom:1rem;animation:pulsering 2s infinite;}}
@keyframes pulsering{{0%,100%{{box-shadow:0 0 0 0 rgba(231,76,60,.35);}}
    50%{{box-shadow:0 0 0 10px rgba(231,76,60,0);}}}}

/* ── PAGE HEADERS ── */
.page-header{{font-family:'DM Serif Display',serif;font-size:1.65rem;
    color:var(--primary);margin-bottom:.2rem;font-weight:400;}}
.page-subheader{{color:var(--muted);font-size:.88rem;margin-bottom:1.25rem;}}

/* ── LANDING PAGE ── */
.landing-bg{{
    min-height:100vh;
    background:linear-gradient(160deg,#0f3d28 0%,#1a6b4a 45%,#0d2e1e 100%);
    display:flex;align-items:center;justify-content:center;
    margin:-1.5rem -2rem;padding:2rem;}}

/* ── SCROLLBAR ── */
::-webkit-scrollbar{{width:5px;}}
::-webkit-scrollbar-thumb{{background:var(--border);border-radius:99px;}}
::-webkit-scrollbar-thumb:hover{{background:var(--primary);}}


#MainMenu,footer,header{{visibility:hidden!important;}}

[data-testid="stToolbar"]{{display:none!important;}}

/* ── FIX: File uploader dropzone (Streamlit defaults dark) ── */
[data-testid="stFileUploaderDropzone"],
[data-testid="stFileUploaderDropzone"] *,
[data-testid="stFileDropzoneInstructions"],
[data-testid="stFileDropzoneInstructions"] *{{
    background:var(--bg2)!important;
    color:var(--text)!important;
    border-color:var(--border)!important;}}

/* ── FIX: Radio labels ── */
[data-testid="stRadio"] label,
[data-testid="stRadio"] p{{color:var(--text)!important;}}

/* ── FIX: Tab labels ── */
.stTabs [data-baseweb="tab"] p{{color:inherit!important;}}

/* ── Keep buttons white text ── */
.stButton>button,.stButton>button *{{color:#fff!important;}}

/* ── Placeholder text visibility ── */
.stTextArea>div>div>textarea::placeholder{{color:#8aab96!important;opacity:1!important;}}
.stTextInput>div>div>input::placeholder{{color:#8aab96!important;opacity:1!important;}}

</style>""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
def init_session_state():
    defaults = dict(
        patient_id=None, patient_data=None, session_id=None, memory=None,
        current_page="consult", language="en", family_member_id=None,
        show_ayush=True, chat_messages=[], triage_result=None,
        last_response=None, show_onboarding=False,
        pin_verified=False, pin_attempts=0,
    )
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ── Sidebar ───────────────────────────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        st.markdown(f"""
        <div style="text-align:center;padding:1.25rem 0 .5rem">
          <div style="width:50px;height:50px;margin:0 auto 8px">{logo_svg(50)}</div>
          <div style="font-family:'DM Serif Display',serif;font-size:1.2rem;
                      color:#fff;letter-spacing:-.01em">MediAssist AI</div>
          <div style="font-size:.7rem;color:#6dab82;margin-top:2px;
                      text-transform:uppercase;letter-spacing:.08em">
              Clinical AI Assistant</div>
        </div>
        <div style="height:1px;background:linear-gradient(90deg,transparent,
            rgba(46,204,113,.3),transparent);margin:.5rem 0 1rem"></div>
        """, unsafe_allow_html=True)

        if st.session_state.patient_data:
            p = st.session_state.patient_data
            st.markdown(f"""
            <div style="background:rgba(46,204,113,.1);border:1px solid rgba(46,204,113,.2);
                        border-radius:12px;padding:10px 12px;margin-bottom:1rem;
                        display:flex;align-items:center;gap:10px">
              <div style="font-size:1.4rem">👤</div>
              <div>
                <div style="font-weight:600;font-size:.9rem;color:#fff">
                    {p.get('name','Patient')}</div>
                <div style="font-size:.72rem;color:#7dc99a">
                    {p.get('age','')} yrs · {(p.get('gender') or '').title()} · {p.get('blood_group') or '—'}</div>
              </div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<div style='font-size:.65rem;color:#4a7a5a;text-transform:uppercase;"
                    "letter-spacing:.1em;margin-bottom:6px'>Navigation</div>",
                    unsafe_allow_html=True)

        for icon, label, key in [
            ("🩺","Consult","consult"),("📋","History","history"),
            ("📊","Vitals","vitals"),("💊","Prescriptions","prescriptions"),
            ("👨‍👩‍👧","Family","family"),("👤","My Profile","profile"),
        ]:
            if st.button(f"{icon}  {label}", key=f"nav_{key}", use_container_width=True):
                st.session_state.current_page = key
                st.rerun()

        st.markdown("<div style='height:1px;background:rgba(46,204,113,.15);"
                    "margin:1rem 0'></div>", unsafe_allow_html=True)
        st.markdown("<div style='font-size:.65rem;color:#4a7a5a;text-transform:uppercase;"
                    "letter-spacing:.1em;margin-bottom:8px'>Settings</div>",
                    unsafe_allow_html=True)

        lang = st.selectbox("Language", ["English","Hindi"],
                            index=0 if st.session_state.language=="en" else 1,
                            key="lang_sel")
        st.session_state.language = "en" if lang=="English" else "hi"
        st.session_state.show_ayush = st.toggle("🌿 AYUSH Remedies",
                                                 value=st.session_state.show_ayush)

        if st.session_state.patient_id:
            members = db.get_family_members(st.session_state.patient_id)
            if members:
                opts = ["Myself"]+[f"{m['name']} ({m['relation']})" for m in members]
                sel  = st.selectbox("Consulting for", opts, key="fam_sel")
                st.session_state.family_member_id = (
                    None if sel=="Myself" else members[opts.index(sel)-1]["id"])

        st.markdown("<div style='height:1px;background:rgba(46,204,113,.15);"
                    "margin:1rem 0'></div>", unsafe_allow_html=True)
        if st.button("🔄 Switch Profile", use_container_width=True, key="sw_prof"):
            for k in ["patient_id","patient_data","session_id","memory","chat_messages"]:
                st.session_state[k] = None if k!="chat_messages" else []
            st.session_state.pin_verified = False   
            st.session_state.pin_attempts = 0      
            st.rerun()

        st.markdown("""<div style="font-size:.68rem;color:#3a6a4a;text-align:center;
            line-height:1.7;padding:.5rem .25rem 0">
            ⚕️ AI-assisted · Not a licensed doctor<br>
            Consult a professional for serious conditions</div>""",
            unsafe_allow_html=True)


# ── Landing page (returning user) ─────────────────────────────────────────────
import base64, os

def get_img_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

def render_landing(all_patients):
    all_patients = db.get_all_patients(device_id=get_device_id())
    if not all_patients:
        render_onboarding()
        return
    st.markdown("""<style>
    .main,[data-testid="stAppViewContainer"]{
        background:linear-gradient(160deg,#0f3d28 0%,#1a6b4a 45%,#0d2e1e 100%)!important;
        min-height:100vh;}
    .block-container{background:transparent!important;padding-top:3rem!important;}
    [data-baseweb="select"]>div{
        background:rgba(255,255,255,.1)!important;
        border:1.5px solid rgba(255,255,255,.22)!important;border-radius:10px!important;}
    [data-baseweb="select"] span,[data-baseweb="select"] div{color:#fff!important;}
    </style>""", unsafe_allow_html=True)

    # Load doctor image as base64
    doc_b64 = get_img_base64("assets/doctor-ai.png")  

    left_col, right_col = st.columns([1.4, 1], gap="medium")

    with left_col:
        st.markdown(f"""
        <div style="text-align:center;margin-bottom:1.8rem">
          <div style="width:80px;height:80px;margin:0 auto 1rem">{logo_svg(80)}</div>
          <h1 style="font-family:'DM Serif Display',serif;font-size:2.4rem;
                     color:#ffffff;margin:0;font-weight:400;letter-spacing:-.02em">
              MediAssist AI</h1>
          <p style="color:rgba(255,255,255,.55);font-size:.92rem;margin:.4rem 0 0">
              Your Personal Clinical AI Assistant</p>
        </div>""", unsafe_allow_html=True)

        # ── Glass card — dropdown now INSIDE ──
        st.markdown("""
        <div style="background:rgba(255,255,255,.06);backdrop-filter:blur(20px);
                    -webkit-backdrop-filter:blur(20px);border:1px solid rgba(255,255,255,.12);
                    border-radius:24px;padding:1.2rem 2rem 0.25rem;
                    box-shadow:0 24px 64px rgba(0,0,0,.4),inset 0 1px 0 rgba(255,255,255,.1);
                    margin-bottom:1rem">
          <p style="color:rgba(255,255,255,.65);font-size:1.25rem;
                    text-transform:uppercase;letter-spacing:.1em;margin:0 0 .6rem">
              Select Profile</p>""", unsafe_allow_html=True)

        options = {p["name"]: p["id"] for p in all_patients}
        selected = st.selectbox("profile_sel", list(options.keys()),
                                label_visibility="collapsed", key="land_sel")

        st.markdown("</div>", unsafe_allow_html=True)

        # ── Buttons ──
        st.markdown("<div style='height:.5rem'></div>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Continue →", use_container_width=True, key="land_cont"):
                pid = options[selected]
                st.session_state.patient_id   = pid
                st.session_state.patient_data = db.get_patient(pid)
                st.session_state.pin_verified = False 
                st.session_state.pin_attempts = 0 
                st.rerun()
        with c2:
            if st.button("New Profile", use_container_width=True, key="land_new"):
                st.session_state.show_onboarding = True
                st.rerun()

        # ── Pills + footer ──
        st.markdown("""
        <div style="height:1px;background:rgba(255,255,255,.08);margin:1.25rem 0 1rem"></div>
        <div style="display:flex;flex-wrap:wrap;justify-content:center;gap:6px">
            <span style="background:rgba(46,204,113,.12);border:1px solid rgba(46,204,113,.25);
                  border-radius:99px;padding:4px 11px;font-size:.75rem;color:rgba(255,255,255,.75)">
                  🩺 RAG Diagnosis</span>
            <span style="background:rgba(46,204,113,.12);border:1px solid rgba(46,204,113,.25);
                  border-radius:99px;padding:4px 11px;font-size:.75rem;color:rgba(255,255,255,.75)">
                  📷 Vision AI</span>
            <span style="background:rgba(46,204,113,.12);border:1px solid rgba(46,204,113,.25);
                  border-radius:99px;padding:4px 11px;font-size:.75rem;color:rgba(255,255,255,.75)">
                  🎤 Hindi & English</span>
            <span style="background:rgba(46,204,113,.12);border:1px solid rgba(46,204,113,.25);
                  border-radius:99px;padding:4px 11px;font-size:.75rem;color:rgba(255,255,255,.75)">
                  🌿 AYUSH</span>
            <span style="background:rgba(46,204,113,.12);border:1px solid rgba(46,204,113,.25);
                  border-radius:99px;padding:4px 11px;font-size:.75rem;color:rgba(255,255,255,.75)">
                  💊 Prescription PDF</span>
        </div>
        <p style="text-align:center;color:rgba(255,255,255,.75);font-size:.7rem;margin-top:1.25rem">
            ⚕️ AI-assisted · Not a substitute for professional medical advice</p>
        """, unsafe_allow_html=True)

# ── Right column — doctor image + stat cards below ──
    with right_col:
        cards_html = (
            "<div style='display:flex;flex-direction:column;align-items:center;"
            "justify-content:flex-start;height:100%;gap:0;margin-top:-2rem'>"

            "<img src='data:image/png;base64," + doc_b64 + "'"
            " style='width:100%;max-width:360px;height:auto;display:block;'"
            " alt='AI Doctor'/>"

            "<div style='display:grid;grid-template-columns:1fr 1fr 1fr;"
            "gap:10px;width:100%;max-width:360px;margin-top:12px'>"

            "<div style='background:rgba(255,255,255,.07);"
            "border:1px solid rgba(46,204,113,.2);"
            "border-radius:14px;padding:12px 10px;text-align:center'>"
            "<div style='font-size:1.3rem;font-weight:700;color:#2ecc71;"
            "font-family:DM Sans,sans-serif;line-height:1'>200+</div>"
            "<div style='font-size:.65rem;color:rgba(255,255,255,.5);"
            "margin-top:4px;font-family:DM Sans,sans-serif;"
            "text-transform:uppercase;letter-spacing:.06em'>AYUSH Remedies</div>"
            "</div>"

            "<div style='background:rgba(255,255,255,.07);"
            "border:1px solid rgba(46,204,113,.2);"
            "border-radius:14px;padding:12px 10px;text-align:center'>"
            "<div style='font-size:1.3rem;font-weight:700;color:#2ecc71;"
            "font-family:DM Sans,sans-serif;line-height:1'>500+</div>"
            "<div style='font-size:.65rem;color:rgba(255,255,255,.5);"
            "margin-top:4px;font-family:DM Sans,sans-serif;"
            "text-transform:uppercase;letter-spacing:.06em'>Trained Conditions</div>"
            "</div>"

            "<div style='background:rgba(255,255,255,.07);"
            "border:1px solid rgba(46,204,113,.2);"
            "border-radius:14px;padding:12px 10px;text-align:center'>"
            "<div style='font-size:1.3rem;font-weight:700;color:#2ecc71;"
            "font-family:DM Sans,sans-serif;line-height:1'>1200+</div>"
            "<div style='font-size:.65rem;color:rgba(255,255,255,.5);"
            "margin-top:4px;font-family:DM Sans,sans-serif;"
            "text-transform:uppercase;letter-spacing:.06em'>Knowledge Base Articles</div>"
            "</div>"

            "</div>"

            "<div style='width:100%;max-width:360px;margin-top:10px;"
            "background:rgba(46,204,113,.08);"
            "border:1px solid rgba(46,204,113,.18);"
            "border-radius:12px;padding:10px 14px;"
            "display:flex;align-items:center;gap:10px'>"
            "<div style='width:8px;height:8px;border-radius:50%;"
            "background:#2ecc71;flex-shrink:0'></div>"
            "<div style='font-family:DM Sans,sans-serif;font-size:.75rem;"
            "color:rgba(255,255,255,.65);line-height:1.4'>"
            "Powered by RAG &middot; Vision AI &middot; Multilingual NLP"
            "</div></div>"

            "</div>"
        )
        st.markdown(cards_html, unsafe_allow_html=True)

# ── PIN verification screen 
def render_pin_screen(patient: dict):
    _, mid, _ = st.columns([1, 1.4, 1])
    with mid:
        st.markdown(f"""
        <div style="text-align:center;margin-bottom:1.75rem">
          <div style="width:56px;height:56px;margin:0 auto .75rem">{logo_svg(56)}</div>
          <div style="font-family:'DM Serif Display',serif;font-size:1.8rem;
                      color:var(--primary);font-weight:400">Welcome back</div>
          <div style="color:var(--muted);font-size:.9rem;margin-top:.3rem">
              Enter your PIN to access
              <b>{patient.get('name','')}'s</b> profile</div>
        </div>""", unsafe_allow_html=True)

        if st.session_state.pin_attempts >= 5:
            st.error("🔒 Too many incorrect attempts. Please restart the app.")
            if st.button("← Back to profile selection"):
                st.session_state.patient_id   = None
                st.session_state.pin_attempts = 0
                st.rerun()
            return

        with st.form("pin_form", clear_on_submit=True):
            pin = st.text_input(
                "4-digit PIN", type="password",
                placeholder="● ● ● ●", max_chars=4,
                label_visibility="collapsed",
            )
            col1, col2 = st.columns(2)
            with col1:
                submit = st.form_submit_button("Unlock →", use_container_width=True)
            with col2:
                back   = st.form_submit_button("← Back",   use_container_width=True)

        if back:
            st.session_state.patient_id   = None
            st.session_state.patient_data = None
            st.session_state.pin_attempts = 0
            st.rerun()

        if submit:
            pin_hash = patient.get("pin_hash", "")
            if not pin_hash:
                # Old profile created before PIN feature — let in, no PIN set
                st.session_state.pin_verified = True
                st.session_state.pin_attempts = 0
                st.rerun()
            elif verify_pin(pin, pin_hash):
                st.session_state.pin_verified = True
                st.session_state.pin_attempts = 0
                st.rerun()
            else:
                st.session_state.pin_attempts += 1
                remaining = 5 - st.session_state.pin_attempts
                st.error(f"❌ Incorrect PIN. {remaining} attempt(s) remaining.")

        if st.session_state.pin_attempts > 0:
            st.markdown(
                f"<div style='text-align:center;color:var(--muted);"
                f"font-size:.8rem;margin-top:.5rem'>"
                f"Failed attempts: {st.session_state.pin_attempts}/5</div>",
                unsafe_allow_html=True,
            )
# ── Onboarding ────────────────────────────────────────────────────────────────
def render_onboarding():
    _, mid, _ = st.columns([1, 2, 1])
    with mid:
        if st.session_state.show_onboarding:
            if st.button("← Back", key="back_onb"):
                st.session_state.show_onboarding = False
                st.rerun()

        st.markdown(f"""
        <div style="text-align:center;margin:1rem 0 2rem">
          <div style="width:56px;height:56px;margin:0 auto .75rem">{logo_svg(56)}</div>
          <div style="font-family:'DM Serif Display',serif;font-size:1.9rem;
                      color:var(--primary);font-weight:400">Create Your Profile</div>
          <div style="color:var(--muted);font-size:.87rem;margin-top:.3rem">
              Your health data stays private on your device</div>
        </div>""", unsafe_allow_html=True)

        with st.form("onboard_form"):
            st.markdown("#### 👤 Basic Information")
            c1, c2 = st.columns(2)
            with c1:
                name = st.text_input("Full Name *", placeholder="e.g. Priya Sharma")
                age  = st.number_input("Age *", 1, 120, 25)
            with c2:
                gender = st.selectbox("Gender *", ["male","female","other"])
                bg     = st.selectbox("Blood Group",
                             ["","A+","A-","B+","B-","O+","O-","AB+","AB-"])

            st.markdown("#### ⚠️ Medical History")
            c3, c4 = st.columns(2)
            with c3:
                allg = st.text_input("Known Allergies",
                           placeholder="penicillin, aspirin (comma separated)")
            with c4:
                cond = st.text_input("Chronic Conditions",
                           placeholder="diabetes, hypertension (comma separated)")

            lang = st.selectbox("Preferred Language", ["English","Hindi"])

            st.markdown("#### 🔒 Set Your PIN")
            st.caption("A 4-digit PIN to protect your profile on this device")
            pc1, pc2 = st.columns(2)
            with pc1:
                pin1 = st.text_input("Create PIN *", type="password",
                                     max_chars=4, placeholder="e.g. 1234")
            with pc2:
                pin2 = st.text_input("Confirm PIN *", type="password",
                                     max_chars=4, placeholder="repeat PIN")

            sub  = st.form_submit_button("✅  Create Profile & Start →",
                                         use_container_width=True)

        if sub:
            if not name.strip():
                st.error("Please enter your name.")
                return
            valid, err = validate_pin_format(pin1)
            if not valid:
                st.error(f"PIN error: {err}")
                return
            if pin1 != pin2:
                st.error("PINs do not match. Please re-enter.")
                return
            lc  = "en" if lang=="English" else "hi"
            pid = db.create_patient(
                name=name.strip(), age=int(age), gender=gender,
                blood_group=bg,
                allergies=[a.strip() for a in allg.split(",") if a.strip()],
                chronic_conditions=[c.strip() for c in cond.split(",") if c.strip()],
                language=lc,
                device_id=get_device_id(),
                pin_hash=hash_pin(pin1),
            )
            st.session_state.patient_id      = pid
            st.session_state.patient_data    = db.get_patient(pid)
            st.session_state.language        = lc
            st.session_state.show_onboarding = False
            st.session_state.pin_verified    = True
            st.session_state.pin_attempts    = 0
            st.success(f"Welcome to MediAssist AI, {name.strip()}! 🎉")
            st.rerun()


# ── Page router ───────────────────────────────────────────────────────────────
def route_page():
    p = st.session_state.current_page
    if p == "consult":
        from ui.chat_ui import render_consult_page; render_consult_page()
    elif p == "history":
        from ui.history_ui import render_history_page; render_history_page()
    elif p == "vitals":
        from ui.vitals_ui import render_vitals_page; render_vitals_page()
    elif p == "prescriptions":
        from ui.prescriptions_ui import render_prescriptions_page; render_prescriptions_page()
    elif p == "family":
        from ui.family_ui import render_family_page; render_family_page()
    elif p == "profile":
        from ui.profile_ui import render_profile_page; render_profile_page()


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    init_session_state()
    inject_css()

    if st.session_state.patient_id and not st.session_state.patient_data:
        st.session_state.patient_data = db.get_patient(st.session_state.patient_id)

    if st.session_state.show_onboarding:
        render_onboarding()
        return

    if not st.session_state.patient_id:
        patients = db.get_all_patients()
        if patients:
            render_landing(patients)
        else:
            render_onboarding()
        return

    # ── PIN gate ──
    if not st.session_state.pin_verified:
        render_pin_screen(st.session_state.patient_data)
        return

    render_sidebar()

    route_page()

if __name__ == "__main__":
    main()