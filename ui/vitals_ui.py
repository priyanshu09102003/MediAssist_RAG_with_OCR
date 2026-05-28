"""ui/vitals_ui.py — Vitals tracker with logging and trend charts"""

import streamlit as st
import pandas as pd
from datetime import datetime
from core.database import db


def render_vitals_page():
    st.markdown(
        '<div class="page-header">📊 Vitals Tracker</div>'
        '<div class="page-subheader">Log and monitor your health indicators over time</div>',
        unsafe_allow_html=True,
    )

    pid = st.session_state.patient_id
    tab_log, tab_trends = st.tabs(["📝 Log Vitals", "📈 Trends"])

    # ── Log new vitals 
    with tab_log:
        st.markdown("#### Record Today's Vitals")
        with st.form("vitals_form"):
            col1, col2, col3 = st.columns(3)
            with col1:
                sys_bp  = st.number_input("Systolic BP (mmHg)",  0, 300, 0)
                dias_bp = st.number_input("Diastolic BP (mmHg)", 0, 200, 0)
            with col2:
                sugar  = st.number_input("Blood Sugar (mg/dL)", 0.0, 600.0, 0.0)
                weight = st.number_input("Weight (kg)",         0.0, 300.0, 0.0)
            with col3:
                spo2   = st.number_input("SpO₂ (%)",        0.0, 100.0, 0.0)
                hr     = st.number_input("Heart Rate (bpm)", 0,   250,   0)
                temp   = st.number_input("Temperature (°C)", 0.0, 45.0,  0.0)

            notes     = st.text_input("Notes", placeholder="e.g. after meal, morning reading")
            submitted = st.form_submit_button("💾 Save Vitals", use_container_width=True)

        if submitted:
            db.log_vitals(
                patient_id=pid,
                family_member_id=st.session_state.family_member_id,
                systolic_bp=sys_bp  or None,
                diastolic_bp=dias_bp or None,
                blood_sugar=sugar  or None,
                weight=weight or None,
                spo2=spo2  or None,
                heart_rate=hr    or None,
                temperature=temp  or None,
                notes=notes,
            )
            st.success("✅ Vitals saved successfully!")
            st.rerun()

        # Latest reading
        vitals = db.get_vitals(pid, limit=1)
        if vitals:
            v = vitals[0]
            st.markdown("#### 🔴 Latest Reading")
            c1, c2, c3, c4 = st.columns(4)
            metrics = [
                ("Blood Pressure", f"{v.get('systolic_bp','—')}/{v.get('diastolic_bp','—')}", "mmHg"),
                ("Blood Sugar",    str(v.get("blood_sugar") or "—"),                          "mg/dL"),
                ("SpO₂",          str(v.get("spo2") or "—"),                                 "%"),
                ("Heart Rate",     str(v.get("heart_rate") or "—"),                           "bpm"),
            ]
            for col, (label, val, unit) in zip([c1, c2, c3, c4], metrics):
                with col:
                    st.metric(label, f"{val} {unit}" if val != "—" else "—")

    # ── Trend charts 
    with tab_trends:
        vitals_all = db.get_vitals(pid, limit=30)
        if not vitals_all:
            st.info("No vitals recorded yet. Log your first reading above.")
            return

        df = pd.DataFrame(vitals_all)
        df["recorded_at"] = pd.to_datetime(df["recorded_at"])
        df = df.sort_values("recorded_at")

        chart_options = {
            "Blood Pressure":  ("systolic_bp", "diastolic_bp"),
            "Blood Sugar":     ("blood_sugar",),
            "Weight":          ("weight",),
            "SpO₂":            ("spo2",),
            "Heart Rate":      ("heart_rate",),
            "Temperature":     ("temperature",),
        }

        selected = st.multiselect(
            "Select metrics to chart",
            list(chart_options.keys()),
            default=["Blood Pressure", "Blood Sugar"],
        )

        for metric in selected:
            cols = chart_options[metric]
            chart_df = df[["recorded_at"] + [c for c in cols if c in df.columns]].dropna()
            if not chart_df.empty:
                st.markdown(f"**{metric}**")
                st.line_chart(chart_df.set_index("recorded_at"))
            else:
                st.caption(f"No data for {metric}")