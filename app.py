"""
TunnelSpec AI — Integrated Tender Intelligence & IoT Site Monitor
KTP Associate Demo  |  HB Tunnelling Ltd × Birmingham City University

Run:  streamlit run app.py
"""

import time
import json
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st

# ── Page config ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TunnelSpec AI | HBT",
    page_icon="⚙",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0D1B2A 0%, #1B2838 100%);
}
[data-testid="stSidebar"] * { color: #E8F4FD !important; }
[data-testid="stSidebar"] .stRadio label { font-size: 0.9rem; }

/* ── Metric cards ── */
.metric-card {
    background: linear-gradient(135deg, #1E3A5F 0%, #0D1B2A 100%);
    border: 1px solid #2E5C8A;
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    margin: 0.4rem 0;
}
.section-header {
    color: #F5A623;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    margin-bottom: 0.4rem;
}

/* ── Alerts ── */
.alert-critical {
    background: rgba(231,76,60,0.14);
    border-left: 4px solid #E74C3C;
    border-radius: 6px;
    padding: 0.7rem 1rem;
    margin: 0.3rem 0;
    color: #FF6B6B;
    font-weight: 600;
}
.alert-warning {
    background: rgba(245,166,35,0.14);
    border-left: 4px solid #F5A623;
    border-radius: 6px;
    padding: 0.7rem 1rem;
    margin: 0.3rem 0;
    color: #F5A623;
    font-weight: 600;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 6px;
    background: #0D1B2A;
    border-radius: 10px;
    padding: 4px;
}
.stTabs [data-baseweb="tab"] {
    color: #8BA7C7;
    background: transparent;
    border-radius: 8px;
    padding: 7px 18px;
    font-size: 0.85rem;
}
.stTabs [aria-selected="true"] {
    background: #1E3A5F !important;
    color: #F5A623 !important;
    font-weight: 600;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #1E3A5F, #2E5C8A);
    color: white;
    border: 1px solid #2E5C8A;
    border-radius: 8px;
    font-weight: 600;
    letter-spacing: 0.4px;
    transition: all 0.2s ease;
}
.stButton > button:hover {
    border-color: #F5A623;
    color: #F5A623;
}

/* ── Misc ── */
hr { border-color: #2E5C8A; }
div[data-testid="stMetricValue"] { color: #F5A623 !important; font-weight: 700; }
</style>
""", unsafe_allow_html=True)


# ── Session state initialisation ─────────────────────────────────────────────────
def _init_state():
    defaults = {
        "api_key":          "",
        "engine":           None,
        "tender_data":      None,
        "confidence":       None,
        "opp_score":        None,
        "boq_data":         None,
        "iot_running":      False,
        "iot_df":           None,
        "iot_tick":         0,
        "simulator":        None,
        "graph_pdf_name":   None,
        "graph_store":      None,
        "graph_vector_store": None,
        "graph_extractions": None,
        "graph_html":       None,
        "graph_qa_result":  None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_state()


# ── Sidebar ──────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙ TunnelSpec AI")
    st.markdown("**HBT Digital Innovation Platform**")
    st.markdown("*KTP Associate · BCU × HB Tunnelling*")
    st.divider()


    st.markdown('<p class="section-header">🔐 API Configuration</p>', unsafe_allow_html=True)
    api_key_input = st.text_input(
        "OpenAI API Key",
        type="password",
        value=st.session_state.api_key,
        placeholder="sk-…",
        help="Required for Modules A & B. Module C runs fully offline.",
    )
    if api_key_input:
        st.session_state.api_key = api_key_input
        st.success("API key loaded", icon="✅")

    st.divider()
    st.markdown('<p class="section-header">📌 Navigation</p>', unsafe_allow_html=True)
    module = st.radio(
        "module",
        [
            "🏗  Module A — Tender Intelligence",
            "📋  Module B — BoQ Generator",
            "📡  Module C — IoT Digital Twin",
            "🕸  Module D — Graph Reasoning",
        ],
        label_visibility="collapsed",
    )

    st.divider()
    st.markdown('<p class="section-header">⚙️ Technical Strategy (KTP Roadmap)</p>', unsafe_allow_html=True)
    st.markdown("""
<div style="font-size:0.78rem;line-height:1.6;color:#C8DCF0;">

**Current Stack**
PDF → GPT-4o · Plotly IoT · NumPy simulation

**Phase 2 — Mamba-SSM Integration**

The IoT Digital Twin module is architected for upgrade to
**Mamba (Selective State Space Models)** for long-horizon
sensor sequence modelling.

**Why Mamba beats Transformers for TBM telemetry:**

- Standard Attention scales at **O(n²)** in sequence length —
  impractical for continuous 28-month tunnel drives at
  high-frequency sampling (e.g. 10 Hz per sensor channel)
- Mamba's selective state-space mechanism scales at **O(n)**
  linearly, eliminating the *memory wall* entirely
- Enables HBT to train anomaly-detection models on the
  full project lifecycle without data truncation or
  expensive chunking strategies
- Mamba's hardware-aware recurrence is optimised for
  GPU/edge deployment — critical for on-site TBM controllers

**KTP Outcome:** A Mamba-SSM model trained on HBT's
historical drive data will predict face pressure exceedances
and settlement risk 60–90 seconds ahead of threshold breach.

</div>
""", unsafe_allow_html=True)

    st.divider()
    st.markdown('<p class="section-header">ℹ System Info</p>', unsafe_allow_html=True)
    st.caption("Model : GPT-4o")
    st.caption("RAG   : pypdf + OpenAI")
    st.caption("IoT   : NumPy + Plotly")
    st.caption(f"Time  : {datetime.now().strftime('%H:%M:%S')}")


# ── Global header ────────────────────────────────────────────────────────────────
st.markdown("""
<div style="background:linear-gradient(90deg,#0D1B2A 0%,#1E3A5F 50%,#0D1B2A 100%);
            padding:1.4rem 2rem;border-radius:12px;margin-bottom:1.2rem;
            border:1px solid #2E5C8A;">
  <h1 style="color:#F5A623;margin:0;font-size:1.75rem;font-weight:700;">
    ⚙ TunnelSpec AI
  </h1>
  <p style="color:#8BA7C7;margin:0.3rem 0 0;font-size:0.92rem;">
    Integrated Tender Intelligence &amp; IoT Site Monitor — HB Tunnelling Ltd
  </p>
</div>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════════
# MODULE A — TENDER INTELLIGENCE
# ════════════════════════════════════════════════════════════════════════════════
if "Module A" in module:
    st.markdown("## 🏗 Tender Intelligence Engine")
    st.caption(
        "Upload an ITT, Employer's Requirements, or Ground Investigation Report (PDF) "
        "to extract structured intelligence and score the opportunity against HBT's capabilities."
    )

    col_left, col_right = st.columns([3, 2], gap="large")

    with col_left:
        st.markdown("### 📄 Document Upload")
        uploaded = st.file_uploader(
            "Upload Tender / ITT / EIR document (PDF)",
            type=["pdf"],
            help="Accepts: Invitation to Tender, Employer's Information Requirements, "
                 "Geotechnical Interpretive Reports, Contract Particulars.",
        )

        if uploaded:
            st.info(
                f"📎 **{uploaded.name}** — {uploaded.size / 1024:.1f} KB  |  Ready for analysis",
                icon="📄",
            )

            if st.button("🔍  Run Tender Intelligence Analysis", use_container_width=True):
                if not st.session_state.api_key:
                    st.error("Enter your OpenAI API key in the sidebar first.")
                else:
                    with st.spinner("Indexing document and running RAG pipeline…"):
                        try:
                            from engine_ai import TenderEngine
                            engine = TenderEngine(st.session_state.api_key)
                            engine.load_pdf_document(uploaded.read(), uploaded.name)
                            st.session_state.engine = engine
                            st.session_state.tender_data = engine.extract_tender_intelligence()
                            st.session_state.opp_score = None  # reset score on new upload
                            st.session_state.confidence = None
                        except Exception as exc:
                            st.error(f"Analysis failed: {exc}")
                            st.session_state.tender_data = None

                    if st.session_state.tender_data:
                        with st.spinner("Scoring extraction confidence…"):
                            try:
                                st.session_state.confidence = st.session_state.engine.score_extraction_confidence(
                                    st.session_state.tender_data
                                )
                                st.success("Document indexed. Intelligence extracted.", icon="✅")
                            except Exception as exc:
                                st.warning(f"Confidence scoring skipped: {exc}")

        if st.session_state.tender_data:
            if st.button("📊  Calculate Opportunity Score", use_container_width=True):
                if not st.session_state.api_key:
                    st.error("API key required.")
                else:
                    with st.spinner("Scoring opportunity against HBT capability matrix…"):
                        try:
                            score = st.session_state.engine.calculate_opportunity_score(
                                st.session_state.tender_data
                            )
                            st.session_state.opp_score = score
                        except Exception as exc:
                            st.error(f"Scoring failed: {exc}")

    # ── Opportunity score panel ──
    with col_right:
        st.markdown("### 📊 Opportunity Score")
        score_data = st.session_state.opp_score

        if score_data:
            score = int(score_data.get("overall_score", 0))
            rec   = score_data.get("recommendation", "N/A")
            colour      = "#27AE60" if score >= 7 else "#F5A623" if score >= 5 else "#E74C3C"
            rec_colour  = "#27AE60" if rec == "GO" else "#F5A623" if "CONDITIONAL" in rec else "#E74C3C"

            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=score,
                domain={"x": [0, 1], "y": [0, 1]},
                title={"text": "HBT Fit Score", "font": {"color": "#8BA7C7", "size": 13}},
                gauge={
                    "axis":        {"range": [0, 10], "tickcolor": "#8BA7C7", "tickfont": {"size": 10}},
                    "bar":         {"color": colour, "thickness": 0.25},
                    "bgcolor":     "#1E3A5F",
                    "bordercolor": "#2E5C8A",
                    "steps": [
                        {"range": [0, 4],  "color": "rgba(231,76,60,0.18)"},
                        {"range": [4, 7],  "color": "rgba(245,166,35,0.18)"},
                        {"range": [7, 10], "color": "rgba(39,174,96,0.18)"},
                    ],
                    "threshold": {"line": {"color": colour, "width": 3},
                                  "thickness": 0.75, "value": score},
                },
                number={"font": {"color": colour, "size": 52}},
            ))
            fig_gauge.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=230,
                margin=dict(l=20, r=20, t=40, b=10),
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

            st.markdown(
                f'<div style="text-align:center;padding:0.7rem;background:rgba(0,0,0,0.3);'
                f'border:2px solid {rec_colour};border-radius:10px;margin:0.4rem 0;">'
                f'<span style="font-size:1.25rem;font-weight:700;color:{rec_colour};">{rec}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

            dims = score_data.get("dimensions", {})
            if dims:
                dim_map = {
                    "technical_fit":      ("Technical Fit",       4.0),
                    "ground_suitability": ("Ground Suitability",  2.5),
                    "risk_profile":       ("Risk Profile",        2.0),
                    "strategic_value":    ("Strategic Value",     1.5),
                }
                st.markdown("---")
                for key, (label, max_val) in dim_map.items():
                    if key in dims:
                        val = dims[key].get("score", 0)
                        comment = dims[key].get("comment", "")
                        pct = val / max_val
                        st.markdown(f"**{label}** — {val}/{max_val}")
                        st.progress(min(pct, 1.0))
                        if comment:
                            st.caption(comment)
        else:
            st.markdown(
                '<div class="metric-card" style="text-align:center;padding:2rem;">'
                '<p style="color:#8BA7C7;font-size:0.9rem;">'
                "Upload a tender document and run analysis to generate an opportunity score."
                "</p></div>",
                unsafe_allow_html=True,
            )

    # ── Extracted intelligence tabs ──
    if st.session_state.tender_data:
        st.divider()
        st.markdown("### 🧠 Extracted Tender Intelligence")
        t_scope, t_ground, t_risk = st.tabs(
            ["📐 Project Scope", "🪨 Ground Conditions", "⚠️ Risk Register"]
        )
        with t_scope:
            st.markdown(st.session_state.tender_data.get("project_scope", "—"))
        with t_ground:
            st.markdown(st.session_state.tender_data.get("ground_conditions", "—"))
        with t_risk:
            st.markdown(st.session_state.tender_data.get("risk_factors", "—"))

    # ── Extraction confidence panel ──
    if st.session_state.get("confidence"):
        st.divider()
        st.markdown("### 🎯 Extraction Confidence Scores")
        st.caption(
            "Each field is scored out of 10: Completeness (0–4) + Grounding (0–3) + Specificity (0–3). "
            "Fields below 5 are flagged for manual review."
        )

        FIELD_LABELS = {
            "project_scope":    "📐 Project Scope",
            "ground_conditions": "🪨 Ground Conditions",
            "risk_factors":     "⚠️ Risk Factors",
        }

        conf_cols = st.columns(3)
        for idx, (field, label) in enumerate(FIELD_LABELS.items()):
            cf = st.session_state.confidence.get(field, {})
            score  = cf.get("score", 0)
            band   = cf.get("band", "LOW")
            colour = "#27AE60" if band == "HIGH" else "#F5A623" if band == "MEDIUM" else "#E74C3C"

            with conf_cols[idx]:
                st.markdown(
                    f'<div class="metric-card">'
                    f'<p style="color:#8BA7C7;font-size:0.75rem;margin:0;">{label}</p>'
                    f'<p style="font-size:2rem;font-weight:700;color:{colour};margin:0.2rem 0;">'
                    f'{score}<span style="font-size:1rem;color:#8BA7C7;">/10</span></p>'
                    f'<p style="font-size:0.75rem;font-weight:700;color:{colour};margin:0;">{band}</p>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                st.progress(score / 10)

                # Sub-dimension breakdown
                st.caption(
                    f"Completeness {cf.get('completeness',0)}/4 · "
                    f"Grounding {cf.get('grounding',0)}/3 · "
                    f"Specificity {cf.get('specificity',0)}/3"
                )

                if cf.get("needs_review"):
                    st.warning("Manual review recommended", icon="⚠️")

                if cf.get("reviewer_note"):
                    st.caption(f"*{cf['reviewer_note']}*")

                caveats = cf.get("caveats", [])
                if caveats:
                    with st.expander("View caveats"):
                        for c in caveats:
                            st.markdown(f"- {c}")

    # ── Score narrative ──
    if st.session_state.opp_score:
        st.divider()
        score_data = st.session_state.opp_score
        col_s, col_c = st.columns(2, gap="large")
        with col_s:
            st.markdown("#### ✅ Key Strengths")
            for item in score_data.get("key_strengths", []):
                st.markdown(f"- {item}")
        with col_c:
            st.markdown("#### ⚠️ Key Concerns")
            for item in score_data.get("key_concerns", []):
                st.markdown(f"- {item}")
        st.markdown("#### 📝 Executive Summary")
        st.info(score_data.get("rationale", ""))


# ════════════════════════════════════════════════════════════════════════════════
# MODULE B — BoQ GENERATOR
# ════════════════════════════════════════════════════════════════════════════════
elif "Module B" in module:
    st.markdown("## 📋 Bill of Quantities Generator")
    st.caption(
        "Describe the tunnelling scope and generate a structured BoQ with NRM2/CESMM4 "
        "measurement conventions and current UK 2025 market rates."
    )

    # ── Example scope library ──
    EXAMPLES = {
        "— Select an example scope —": "",
        "500 mm Gravity Sewer — Pipe Jacking": (
            "Construction of 280 m of 500 mm ID pre-cast concrete jacking pipes by pipe jacking "
            "method beneath an urban road corridor. Works include: excavation of 4.5 m diameter × "
            "12 m deep drive shaft with sheet-pile retention and temporary steel strutting, "
            "excavation of 3 m diameter × 10 m deep reception shaft, installation of jacking frame "
            "and guide rails, pipe jacking drive using Herrenknecht AVN500 microtunnelling system "
            "in mixed ground (London Clay overlying River Terrace Gravels with groundwater at 4 m bgl), "
            "annulus grouting with cementitious grout, CCTV structural survey, reinstatement of highway."
        ),
        "1200 mm Water Main — EPB-TBM": (
            "Construction of 850 m of 1200 mm OD steel carrier pipe within a 1800 mm internal "
            "diameter precast concrete segmental lined tunnel by EPB-TBM. Works include: "
            "6 m ID × 18 m deep launch shaft with contiguous secant pile walls and king-post "
            "propping, 5 m ID × 16 m deep retrieval shaft, TBM mobilisation / demobilisation, "
            "ring building of 250 mm thick six-segment lining at 1.2 m ring pitch, secondary "
            "cementitious grouting, pipeline installation (butt-welded carbon steel, PE-coated), "
            "cathodic protection system, hydrostatic pressure testing to 16 bar, "
            "full highway and footway reinstatement."
        ),
        "250 mm HDPE — HDD Railway Crossing": (
            "Construction of a 180 m horizontal directional drill (HDD) crossing beneath a Network "
            "Rail operational embankment and adjacent watercourse at minimum 8 m cover below rail "
            "level. Works include: mobilisation of 150-tonne pullback HDD rig, pilot bore and "
            "prereaming to 400 mm diameter through Mercia Mudstone, installation of 250 mm PE100 "
            "SDR11 HDPE product pipe, drilling fluid management and containment, interface with "
            "Network Rail Possession management, pressure testing to 10 bar, "
            "topsoil reinstatement and SUDS drainage restoration."
        ),
    }

    col_in, col_opts = st.columns([3, 1], gap="large")

    with col_in:
        selected = st.selectbox("Load example scope", list(EXAMPLES.keys()))
        scope_text = st.text_area(
            "Scope of Works",
            value=EXAMPLES[selected],
            height=230,
            placeholder=(
                "Describe the tunnelling/trenchless scope in detail — include method, "
                "diameter, drive length, ground conditions, shaft specifications, lining type, "
                "service connections, testing, and reinstatement requirements…"
            ),
        )

    with col_opts:
        st.markdown("**BoQ Options**")
        include_contingency = st.checkbox("Include Contingency (10%)", value=True)
        include_design_risk = st.checkbox("Design Risk Allowance (5%)", value=False)
        st.divider()
        generate_btn = st.button("⚡  Generate BoQ", use_container_width=True)

    if generate_btn:
        if not st.session_state.api_key:
            st.error("Enter your OpenAI API key in the sidebar.")
        elif not scope_text.strip():
            st.warning("Enter a scope of works description before generating.")
        else:
            with st.spinner("Analysing scope and generating Bill of Quantities…"):
                try:
                    from engine_ai import TenderEngine
                    engine = TenderEngine(st.session_state.api_key)
                    st.session_state.boq_data = engine.generate_boq(scope_text)
                except Exception as exc:
                    st.error(f"BoQ generation failed: {exc}")

    # ── BoQ results ──
    if st.session_state.boq_data:
        items = st.session_state.boq_data
        if items and "error" not in items[0]:
            df_boq = pd.DataFrame(items)

            # Numeric coercion
            for col in ("quantity", "rate_gbp", "amount_gbp"):
                if col in df_boq.columns:
                    df_boq[col] = pd.to_numeric(df_boq[col], errors="coerce").fillna(0)

            subtotal   = df_boq["amount_gbp"].sum() if "amount_gbp" in df_boq.columns else 0
            contingency = subtotal * 0.10 if include_contingency else 0
            design_risk = subtotal * 0.05 if include_design_risk else 0
            grand_total = subtotal + contingency + design_risk

            # KPI strip
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Line Items",        len(df_boq))
            k2.metric("Works Subtotal",    f"£{subtotal:,.0f}")
            k3.metric("Risk Allowances",   f"£{contingency + design_risk:,.0f}")
            k4.metric("Grand Total",       f"£{grand_total:,.0f}")

            st.divider()

            # Section cost-breakdown doughnut
            if "section" in df_boq.columns:
                section_totals = df_boq.groupby("section")["amount_gbp"].sum().reset_index()
                fig_pie = px.pie(
                    section_totals, values="amount_gbp", names="section",
                    title="Cost Distribution by Section",
                    color_discrete_sequence=px.colors.qualitative.Set2,
                    hole=0.42,
                )
                fig_pie.update_traces(textposition="outside", textinfo="percent+label")
                fig_pie.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#8BA7C7"),
                    height=340,
                    showlegend=False,
                    margin=dict(l=20, r=20, t=50, b=20),
                )
                st.plotly_chart(fig_pie, use_container_width=True)

            # BoQ table
            st.markdown("### 📊 Bill of Quantities — Schedule of Rates")
            display_cols = [c for c in
                            ["item_no", "section", "description", "quantity", "unit",
                             "rate_gbp", "amount_gbp", "hbt_benchmark_gbp", "notes"]
                            if c in df_boq.columns]
            display_df = df_boq[display_cols].copy()
            if "rate_gbp"   in display_df.columns:
                display_df["rate_gbp"]   = display_df["rate_gbp"].apply(lambda x: f"£{x:,.2f}")
            if "amount_gbp" in display_df.columns:
                display_df["amount_gbp"] = display_df["amount_gbp"].apply(lambda x: f"£{x:,.2f}")

            st.dataframe(display_df, use_container_width=True, height=420, hide_index=True)

            # Summary totals
            rows = f"| Subtotal (Works) | | | | | | **£{subtotal:,.2f}** |\n"
            if include_contingency:
                rows += f"| Contingency (10%) | | | | | | £{contingency:,.2f} |\n"
            if include_design_risk:
                rows += f"| Design Risk Allowance (5%) | | | | | | £{design_risk:,.2f} |\n"
            rows += f"| **GRAND TOTAL (excl. VAT)** | | | | | | **£{grand_total:,.2f}** |"
            st.markdown("---")
            st.markdown(rows)

            # ── Exports ──
            import io
            st.markdown("#### ⬇️ Export")
            dl_col1, dl_col2 = st.columns(2)

            with dl_col1:
                csv_data = df_boq.to_csv(index=False)
                st.download_button(
                    "📄  Download as CSV",
                    csv_data,
                    f"HBT_BoQ_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    "text/csv",
                    use_container_width=True,
                )

            with dl_col2:
                # Build Excel file in memory
                excel_buf = io.BytesIO()
                export_df = df_boq[display_cols].copy()

                # Rename columns to professional headers for Excel
                col_rename = {
                    "item_no":           "Item No.",
                    "section":           "Section",
                    "description":       "Description",
                    "quantity":          "Quantity",
                    "unit":              "Unit",
                    "rate_gbp":          "Rate (£)",
                    "amount_gbp":        "Amount (£)",
                    "hbt_benchmark_gbp": "HBT Historical Benchmark",
                    "notes":             "Notes / Spec Ref",
                }
                export_df = export_df.rename(columns={k: v for k, v in col_rename.items() if k in export_df.columns})

                # Append summary rows
                summary_rows = [{"Description": "Subtotal (Works)", "Amount (£)": subtotal}]
                if include_contingency:
                    summary_rows.append({"Description": "Contingency (10%)", "Amount (£)": contingency})
                if include_design_risk:
                    summary_rows.append({"Description": "Design Risk Allowance (5%)", "Amount (£)": design_risk})
                summary_rows.append({"Description": "GRAND TOTAL (excl. VAT)", "Amount (£)": grand_total})

                with pd.ExcelWriter(excel_buf, engine="openpyxl") as writer:
                    export_df.to_excel(writer, index=False, sheet_name="Bill of Quantities")
                    pd.DataFrame(summary_rows).to_excel(
                        writer, index=False, sheet_name="Summary",
                        startrow=0,
                    )

                st.download_button(
                    "📊  Download as Excel (.xlsx)",
                    excel_buf.getvalue(),
                    f"HBT_BoQ_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True,
                )
        else:
            st.error(f"Generation error: {items[0].get('error', 'Unknown')}")
            if "raw_response" in items[0]:
                with st.expander("Raw LLM response"):
                    st.code(items[0]["raw_response"])


# ════════════════════════════════════════════════════════════════════════════════
# MODULE C — IoT DIGITAL TWIN
# ════════════════════════════════════════════════════════════════════════════════
elif "Module C" in module:
    from simulator_iot import TBMSimulator, SENSOR_CONFIGS

    st.markdown("## 📡 IoT Digital Twin — Live TBM Site Monitor")
    st.caption(
        "Real-time sensor telemetry simulation for an active TBM drive. "
        "Mimics an MQTT data pipeline ingesting field instrument data from "
        "TBM face pressure transducers, torque sensors, settlement monuments, "
        "and grout volume meters."
    )

    # ── Initialise simulator once per session ──
    if st.session_state.simulator is None:
        st.session_state.simulator = TBMSimulator(seed=42)
    if st.session_state.iot_df is None:
        st.session_state.iot_df = st.session_state.simulator.get_historical_data(120)

    sim: TBMSimulator = st.session_state.simulator

    # ── Controls ──
    c1, c2, c3, c4 = st.columns([1, 1, 1, 2])
    with c1:
        if st.button("▶  Start Live Feed", use_container_width=True):
            st.session_state.iot_running = True
    with c2:
        if st.button("⏸  Pause", use_container_width=True):
            st.session_state.iot_running = False
    with c3:
        if st.button("🔄  Reset Drive", use_container_width=True):
            st.session_state.iot_df      = sim.get_historical_data(120)
            st.session_state.iot_running = False
            st.session_state.iot_tick    = 0
            st.session_state.simulator   = TBMSimulator(seed=42)
    with c4:
        refresh_s = st.slider(
            "Refresh interval (s)", min_value=1, max_value=10, value=3,
            label_visibility="collapsed",
        )
        st.caption(f"Refresh interval: {refresh_s} s")

    # ── Append new reading if running ──
    if st.session_state.iot_running:
        new_row = pd.DataFrame([sim.generate_reading()])
        st.session_state.iot_df = pd.concat(
            [st.session_state.iot_df, new_row], ignore_index=True
        ).tail(200)
        st.session_state.iot_tick += 1

    df: pd.DataFrame = st.session_state.iot_df
    latest = df.iloc[-1].to_dict() if not df.empty else {}

    # ── Alert panel ──
    alerts = sim.get_alerts(latest) if latest else []
    if alerts:
        st.markdown("### 🚨 Active Alerts")
        for alert in alerts:
            css_class = "alert-critical" if alert["severity"] == "CRITICAL" else "alert-warning"
            st.markdown(
                f'<div class="{css_class}"><strong>{alert["severity"]}</strong> — {alert["message"]}</div>',
                unsafe_allow_html=True,
            )
    else:
        st.success("✅  All sensors nominal — no active alerts", icon="✅")

    st.divider()

    # ── KPI strip ──
    kpis = sim.get_kpis(df)
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Drive Chainage",       f"{kpis.get('total_drive_m', 0):.1f} m")
    k2.metric("Rings Built",          f"{kpis.get('rings_built', 0)}")
    k3.metric("Avg Advance Rate",     f"{kpis.get('avg_advance_rate', 0):.1f} mm/min")
    k4.metric("Avg Face Pressure",    f"{kpis.get('avg_face_pressure', 0):.2f} bar")
    k5.metric("Max Settlement",       f"{kpis.get('max_settlement', 0):.1f} mm")
    k6.metric("TBM Uptime",           f"{kpis.get('uptime_pct', 0):.1f} %")

    st.divider()

    # ── Chart layout helpers ──
    CHART_LAYOUT = dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(13,27,42,0.85)",
        font=dict(color="#8BA7C7", size=11),
        margin=dict(l=60, r=20, t=45, b=25),
        xaxis=dict(gridcolor="#1E3A5F", zerolinecolor="#1E3A5F"),
        yaxis=dict(gridcolor="#1E3A5F", zerolinecolor="#1E3A5F"),
        showlegend=False,
    )

    def _add_threshold(fig, y, row, col=1):
        fig.add_hline(
            y=y, line_dash="dash", line_color="#E74C3C",
            line_width=1, row=row, col=col,
            annotation_text=f"Alert {y}", annotation_font_color="#E74C3C",
            annotation_font_size=9,
        )

    # ── Tabs ──
    t1, t2, t3, t4 = st.tabs([
        "⚙  TBM Performance", "🏗  Structural", "💉  Grouting & Water", "📊  All Sensors"
    ])

    # ── Tab 1: TBM Performance ──
    with t1:
        fig1 = make_subplots(
            rows=3, cols=1,
            subplot_titles=[
                "TBM Face Pressure (bar)",
                "Cutterhead Torque (kN·m)",
                "Advance Rate (mm/min)",
            ],
            shared_xaxes=True,
            vertical_spacing=0.08,
        )
        sensor_rows = [
            ("tbm_face_pressure", 3.8, 1),
            ("cutterhead_torque", 620, 2),
            ("advance_rate",       75, 3),
        ]
        for key, thr, row in sensor_rows:
            cfg = SENSOR_CONFIGS[key]
            fig1.add_trace(go.Scatter(
                x=df["timestamp"], y=df[key],
                name=cfg.name,
                line=dict(color=cfg.line_color, width=1.8),
                fill="tozeroy",
                fillcolor=cfg.fill_color,
            ), row=row, col=1)
            _add_threshold(fig1, thr, row)

        fig1.update_layout(**CHART_LAYOUT, height=500)
        for i in range(1, 4):
            fig1.update_xaxes(gridcolor="#1E3A5F", zerolinecolor="#1E3A5F", row=i, col=1)
            fig1.update_yaxes(gridcolor="#1E3A5F", zerolinecolor="#1E3A5F", row=i, col=1)
        st.plotly_chart(fig1, use_container_width=True)

    # ── Tab 2: Structural ──
    with t2:
        fig2 = make_subplots(
            rows=2, cols=1,
            subplot_titles=["Vibration RMS (mm/s)", "Surface Settlement (mm)"],
            shared_xaxes=True,
            vertical_spacing=0.12,
        )
        for key, row in [("vibration_rms", 1), ("surface_settlement", 2)]:
            cfg = SENSOR_CONFIGS[key]
            fig2.add_trace(go.Scatter(
                x=df["timestamp"], y=df[key],
                name=cfg.name,
                line=dict(color=cfg.line_color, width=1.8),
                fill="tozeroy",
                fillcolor=cfg.fill_color,
            ), row=row, col=1)
            _add_threshold(fig2, cfg.alert_high, row)

        fig2.update_layout(**CHART_LAYOUT, height=420)
        for i in range(1, 3):
            fig2.update_xaxes(gridcolor="#1E3A5F", zerolinecolor="#1E3A5F", row=i, col=1)
            fig2.update_yaxes(gridcolor="#1E3A5F", zerolinecolor="#1E3A5F", row=i, col=1)
        st.plotly_chart(fig2, use_container_width=True)

    # ── Tab 3: Grouting & Water ──
    with t3:
        fig3 = make_subplots(
            rows=2, cols=1,
            subplot_titles=["Tail Grout Pressure (bar)", "Water Ingress (L/min)"],
            shared_xaxes=True,
            vertical_spacing=0.12,
        )
        cfg_g = SENSOR_CONFIGS["grout_pressure"]
        fig3.add_trace(go.Scatter(
            x=df["timestamp"], y=df["grout_pressure"],
            name=cfg_g.name,
            line=dict(color=cfg_g.line_color, width=1.8),
            fill="tozeroy", fillcolor=cfg_g.fill_color,
        ), row=1, col=1)
        _add_threshold(fig3, cfg_g.alert_high, 1)

        cfg_w = SENSOR_CONFIGS["water_ingress"]
        fig3.add_trace(go.Bar(
            x=df["timestamp"], y=df["water_ingress"],
            name=cfg_w.name,
            marker_color="rgba(0,188,212,0.65)",
        ), row=2, col=1)
        _add_threshold(fig3, cfg_w.alert_high, 2)

        fig3.update_layout(**CHART_LAYOUT, height=420)
        for i in range(1, 3):
            fig3.update_xaxes(gridcolor="#1E3A5F", zerolinecolor="#1E3A5F", row=i, col=1)
            fig3.update_yaxes(gridcolor="#1E3A5F", zerolinecolor="#1E3A5F", row=i, col=1)
        st.plotly_chart(fig3, use_container_width=True)

    # ── Tab 4: All sensors ──
    with t4:
        sensor_keys = list(SENSOR_CONFIGS.keys())
        avail = [k for k in sensor_keys if k in df.columns]

        # Sensor correlation heatmap
        corr = df[avail].corr()
        sensor_labels = [SENSOR_CONFIGS[k].name for k in avail]

        fig_corr = go.Figure(go.Heatmap(
            z=corr.values,
            x=sensor_labels,
            y=sensor_labels,
            colorscale="RdBu_r",
            zmid=0,
            text=np.round(corr.values, 2),
            texttemplate="%{text}",
            textfont={"size": 8},
        ))
        fig_corr.update_layout(
            title="Sensor Correlation Matrix",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(13,27,42,0.85)",
            height=460,
            font=dict(color="#8BA7C7", size=9),
            margin=dict(l=160, r=20, t=60, b=160),
        )
        st.plotly_chart(fig_corr, use_container_width=True)

        # Live readings table
        st.markdown("#### Current Sensor Readings")
        if latest:
            rows_live = []
            for key, cfg in SENSOR_CONFIGS.items():
                val = float(latest.get(key, 0))
                if val >= cfg.alert_high:
                    status = "🔴  ALERT"
                elif val >= cfg.alert_high * 0.85:
                    status = "🟡  CAUTION"
                else:
                    status = "🟢  NORMAL"
                rows_live.append({
                    "Sensor":           cfg.name,
                    "Reading":          f"{val:.3f}",
                    "Unit":             cfg.unit,
                    "Alert Threshold":  cfg.alert_high,
                    "Status":           status,
                })
            st.dataframe(pd.DataFrame(rows_live), use_container_width=True, hide_index=True)

    # ── Auto-refresh ──
    if st.session_state.iot_running:
        time.sleep(refresh_s)
        st.rerun()

    # ── Persistent status badge ──
    dot_col  = "#27AE60" if st.session_state.iot_running else "#8BA7C7"
    dot_text = "● LIVE" if st.session_state.iot_running else "○ PAUSED"
    st.markdown(
        f'<div style="position:fixed;bottom:22px;right:22px;background:#0D1B2A;'
        f'border:1px solid #2E5C8A;border-radius:20px;padding:5px 15px;font-size:0.78rem;">'
        f'<span style="color:{dot_col};font-weight:700;">{dot_text}</span>'
        f'<span style="color:#8BA7C7;margin-left:8px;">Tick #{st.session_state.iot_tick}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )


# ════════════════════════════════════════════════════════════════════════════════
# MODULE D — GRAPH REASONING (GraphRAG)
# ════════════════════════════════════════════════════════════════════════════════
elif "Module D" in module:
    import os
    import tempfile

    import pypdf
    from openai import OpenAI as _OpenAI

    from graph_module.graph_builder import build_graph_from_chunks
    from graph_module.graphrag_pipeline import graphrag_query
    from graph_module.semantic_query import (
        find_cascading_risks,
        find_critical_dependencies,
        find_unmitigated_risks,
    )
    from graph_module.vector_store import VectorStore, chunk_text

    ENTITY_TYPE_COLOURS = {
        "Project":       "#F5A623",
        "Risk":          "#E74C3C",
        "Mitigation":    "#27AE60",
        "Phase":         "#3498DB",
        "Stakeholder":   "#9B59B6",
        "Material":      "#1ABC9C",
        "Specification": "#E67E22",
        "Cost_Item":     "#F1C40F",
        "Constraint":    "#8E44AD",
        "Location":      "#95A5A6",
    }

    st.markdown("## 🕸 Knowledge Graph + Semantic Reasoning")
    st.caption(
        "An additional reasoning layer alongside the vector RAG pipeline. Entities and "
        "relations are extracted from the document, deduplicated, and stored in a "
        "NetworkX graph. Queries that need multi-hop reasoning (e.g. cascading risks "
        "across phases) traverse the graph; the answer is rendered side-by-side with "
        "the vector-only baseline so the comparison is honest."
    )

    col_up, col_info = st.columns([3, 2], gap="large")
    with col_up:
        st.markdown("### 📄 Document Upload")
        st.caption(
            "Try the bundled sample at `samples/riverside_trunk_sewer_ITT.pdf` — "
            "it has a deliberate cascading risk chain spanning multiple sections."
        )
        graph_pdf = st.file_uploader(
            "Upload tender / ITT / GIR document (PDF)",
            type=["pdf"],
            key="graph_pdf_uploader",
        )
        build_btn = st.button(
            "🧠  Build Knowledge Graph + Vector Index",
            use_container_width=True,
            disabled=graph_pdf is None,
        )

    with col_info:
        st.markdown("### 🎨 Entity Type Legend")
        legend_html = "".join(
            f'<span style="display:inline-block;margin:3px 6px;padding:3px 9px;'
            f'border-radius:10px;font-size:0.72rem;background:{c};color:#0D1B2A;'
            f'font-weight:700;">{t}</span>'
            for t, c in ENTITY_TYPE_COLOURS.items()
        )
        st.markdown(f'<div style="line-height:2.2;">{legend_html}</div>', unsafe_allow_html=True)

    if build_btn:
        if not st.session_state.api_key:
            st.error("Enter your OpenAI API key in the sidebar first.")
        else:
            with st.spinner("Extracting text, chunking, embedding, and building graph…"):
                try:
                    pdf_bytes = graph_pdf.read()
                    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                        tmp.write(pdf_bytes)
                        tmp_path = tmp.name
                    try:
                        reader = pypdf.PdfReader(tmp_path)
                        full_text = "\n\n".join((p.extract_text() or "") for p in reader.pages)
                    finally:
                        os.unlink(tmp_path)

                    client = _OpenAI(api_key=st.session_state.api_key)
                    chunks = chunk_text(full_text)
                    st.info(f"Chunked into {len(chunks)} segments. Extracting entities in parallel…")
                    store, extractions = build_graph_from_chunks(
                        chunks, client, source_doc=graph_pdf.name,
                    )
                    vector_store = VectorStore.from_text(full_text, client)

                    st.session_state.graph_pdf_name = graph_pdf.name
                    st.session_state.graph_store = store
                    st.session_state.graph_vector_store = vector_store
                    st.session_state.graph_extractions = extractions
                    st.session_state.graph_html = None
                    st.session_state.graph_qa_result = None
                    st.success(
                        f"Graph built: {len(store.all_entities())} entities, "
                        f"{len(store.all_relations())} relations.",
                        icon="✅",
                    )
                except Exception as exc:
                    st.error(f"Graph build failed: {exc}")

    store = st.session_state.graph_store
    if store is not None:
        st.divider()
        st.markdown(f"### 🧬 Graph for `{st.session_state.graph_pdf_name}`")

        ents = store.all_entities()
        rels = store.all_relations()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Entities",   len(ents))
        c2.metric("Relations",  len(rels))
        c3.metric("Risks",      len(store.all_nodes_of_type("Risk")))
        c4.metric("Phases",     len(store.all_nodes_of_type("Phase")))

        # ── pyvis viz (built once per graph) ──
        if st.session_state.graph_html is None and ents:
            try:
                from pyvis.network import Network
                net = Network(
                    height="520px", width="100%",
                    bgcolor="#0D1B2A", font_color="#E8F4FD",
                    directed=True, notebook=False,
                )
                for e in ents:
                    net.add_node(
                        e.id,
                        label=e.name,
                        title=f"{e.type}: {e.name}",
                        color=ENTITY_TYPE_COLOURS.get(e.type, "#8BA7C7"),
                        shape="dot", size=18,
                    )
                for r in rels:
                    net.add_edge(
                        r.source_id, r.target_id,
                        label=r.relation_type,
                        title=f"{r.relation_type} (conf={r.confidence:.2f})\n{r.evidence_span}",
                        arrows="to",
                    )
                net.toggle_physics(True)
                st.session_state.graph_html = net.generate_html(notebook=False)
            except Exception as exc:
                st.warning(f"Graph visualisation skipped: {exc}")

        if st.session_state.graph_html:
            st.components.v1.html(st.session_state.graph_html, height=540, scrolling=False)

        # ── Demo query buttons ──
        st.markdown("### 🔎 Built-in Semantic Queries")
        risk_ids = [r.id for r in store.all_nodes_of_type("Risk")]
        phase_ids = [p.id for p in store.all_nodes_of_type("Phase")]

        q_col1, q_col2 = st.columns(2)
        with q_col1:
            if risk_ids:
                root_risk = st.selectbox("Root risk for cascade trace", risk_ids, key="cascade_root")
                if st.button("Trace cascading risks", use_container_width=True):
                    cascades = find_cascading_risks(store, root_risk, max_hops=3)
                    if not cascades:
                        st.info("No downstream risks reachable along TRIGGERS / RELATED_TO.")
                    for hit in cascades:
                        st.markdown(
                            f"**{hit['entity'].name}** "
                            f"`{hit['entity'].id}` — {hit['hops']} hop(s)"
                        )
                        for step in hit["path"]:
                            st.caption(
                                f"  {step['source']} —[{step['relation']}]→ {step['target']}  "
                                f"_(conf {step['confidence']:.2f})_  «{step['evidence_span']}»"
                            )

            unmitigated = find_unmitigated_risks(store)
            with st.expander(f"Unmitigated risks ({len(unmitigated)})"):
                for r in unmitigated:
                    st.markdown(f"- **{r.name}** `{r.id}`")

        with q_col2:
            if phase_ids:
                phase = st.selectbox("Phase to inspect", phase_ids, key="phase_inspect")
                if st.button("Show critical dependencies", use_container_width=True):
                    deps = find_critical_dependencies(store, phase)
                    st.markdown("**Depends on phases:**")
                    for p in deps["depends_on_phases"]:
                        st.markdown(f"- {p.name} `{p.id}`")
                    st.markdown("**Required materials:**")
                    for m in deps["requires_materials"]:
                        st.markdown(f"- {m.name} `{m.id}`")
                    st.markdown("**Required specifications:**")
                    for s in deps["requires_specifications"]:
                        st.markdown(f"- {s.name} `{s.id}`")

        # ── GraphRAG side-by-side ──
        st.divider()
        st.markdown("### ⚖ Vector RAG vs GraphRAG — side-by-side")
        st.caption(
            "Both paths see the same chunks embedded the same way and call GPT-4o "
            "with the same instruction shape. The only difference is retrieval. "
            "Try a question that requires connecting facts across sections — e.g. "
            "*\"What downstream commercial impact could a groundwater ingress event "
            "ultimately cause?\"*"
        )
        question = st.text_input(
            "Question",
            value="What downstream commercial impact could a groundwater ingress event ultimately cause?",
            key="graphrag_question",
        )
        if st.button("Run side-by-side query", use_container_width=True):
            if not st.session_state.api_key:
                st.error("API key required.")
            elif st.session_state.graph_vector_store is None:
                st.error("Vector store missing — rebuild the index.")
            else:
                client = _OpenAI(api_key=st.session_state.api_key)
                with st.spinner("Running both retrieval paths…"):
                    try:
                        st.session_state.graph_qa_result = graphrag_query(
                            question,
                            st.session_state.graph_vector_store,
                            store,
                            client,
                        )
                    except Exception as exc:
                        st.error(f"Query failed: {exc}")

        result = st.session_state.graph_qa_result
        if result:
            ac1, ac2 = st.columns(2, gap="large")
            with ac1:
                st.markdown("#### 📚 Vector RAG answer")
                st.markdown(
                    f'<div class="metric-card">{result["vector_answer"]}</div>',
                    unsafe_allow_html=True,
                )
                with st.expander(f"Retrieved chunks ({len(result['vector_hits'])})"):
                    for idx, score, preview in result["vector_hits"]:
                        st.caption(f"chunk {idx} · sim={score:.3f}")
                        st.code(preview, language="text")
            with ac2:
                st.markdown("#### 🕸 GraphRAG answer")
                st.markdown(
                    f'<div class="metric-card" style="border-color:#F5A623;">'
                    f'{result["graphrag_answer"]}</div>',
                    unsafe_allow_html=True,
                )
                with st.expander(
                    f"Seed entities ({len(result['graph_seed_entities'])}) "
                    f"+ subgraph ({len(result['graph_subgraph_entities'])} entities, "
                    f"{len(result['graph_subgraph_relations'])} relations)"
                ):
                    st.markdown("**Seeded from question:**")
                    for e in result["graph_seed_entities"]:
                        st.markdown(f"- `{e.id}` — {e.name}")
                    st.markdown("**Traversed edges:**")
                    for r in result["graph_subgraph_relations"]:
                        st.caption(
                            f"{r.source_id} —[{r.relation_type}]→ {r.target_id}  "
                            f"_(conf {r.confidence:.2f})_"
                        )

            st.markdown("#### 🔬 Why the answers differ")
            st.info(result["diagnostic"])
    else:
        st.markdown(
            '<div class="metric-card" style="text-align:center;padding:2rem;">'
            '<p style="color:#8BA7C7;font-size:0.9rem;">'
            "Upload a PDF and build the index to populate the knowledge graph."
            "</p></div>",
            unsafe_allow_html=True,
        )
