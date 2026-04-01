<div align="center">

# ⚙ TunnelSpec AI
### Integrated Tender Intelligence & IoT Site Monitor

**A KTP Associate Prototype — HB Tunnelling Ltd × Birmingham City University**

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![OpenAI](https://img.shields.io/badge/GPT--4o-OpenAI-412991?style=for-the-badge&logo=openai&logoColor=white)](https://openai.com)
[![Plotly](https://img.shields.io/badge/Plotly-Interactive-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)](https://plotly.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-F5A623?style=for-the-badge)](LICENSE)

---

> *Transforming HBT's estimating and site-monitoring workflows through applied AI — from tender PDF to live TBM dashboard.*

</div>

---

## 📸 Screenshots

<!-- ─────────────────────────────────────────────────────────────────────────
     SCREENSHOT PLACEMENT GUIDE
     Upload your screenshots to a folder called /assets/ in this repo,
     then replace the placeholder paths below with the actual filenames.
     ───────────────────────────────────────────────────────────────────── -->

### Module A — Tender Intelligence Engine
> **📌 Place screenshot here:** Upload a screenshot of the Module A page showing the extracted Project Scope, Ground Conditions, and Risk Factors tabs, plus the Opportunity Score gauge. Save it as `assets/module_a_tender.png`.

![Module A – Tender Intelligence](assets/module_a_tender.png)

---

### Module A — Extraction Confidence Scores
> **📌 Place screenshot here:** Upload a screenshot of the three confidence score cards (Completeness / Grounding / Specificity) below the intelligence tabs. Save it as `assets/module_a_confidence.png`.

![Module A – Confidence Scores](assets/module_a_confidence.png)

---

### Module B — Bill of Quantities Generator
> **📌 Place screenshot here:** Upload a screenshot of the generated BoQ table including the HBT Historical Benchmark column, the cost distribution pie chart, and the Export buttons. Save it as `assets/module_b_boq.png`.

![Module B – BoQ Generator](assets/module_b_boq.png)

---

### Module C — IoT Digital Twin (Live TBM Monitor)
> **📌 Place screenshot here:** Upload a screenshot of the live sensor dashboard with the TBM Performance charts running. Save it as `assets/module_c_iot.png`.

![Module C – IoT Digital Twin](assets/module_c_iot.png)

---

## 🧭 Project Overview

**TunnelSpec AI** is a Python-based digital innovation platform built to address three core operational challenges faced by specialist trenchless contractors:

| Challenge | TunnelSpec AI Solution |
|-----------|----------------------|
| Estimators spending days manually reading tender PDFs | **Module A** — GPT-4o extracts structured intelligence in under 60 seconds |
| Inconsistent BoQ formats across subcontractors | **Module B** — AI-generated NRM2/CESMM4-compliant schedules with historical benchmarks |
| No real-time visibility of TBM drive telemetry | **Module C** — Live IoT digital twin dashboard simulating MQTT sensor pipelines |

This prototype was developed as part of a **Knowledge Transfer Partnership (KTP)** application between **HB Tunnelling Ltd (HBT)** and **Birmingham City University (BCU)**, demonstrating the technical vision outlined in the KTP workplan.

---

## 🏗 Architecture

```
tunnelspec-ai/
│
├── app.py               # Streamlit UI — all three modules
├── engine_ai.py         # AI engine — PDF parsing, GPT-4o calls, BoQ generation
├── simulator_iot.py     # IoT simulator — TBM sensor time-series engine
├── requirements.txt     # Python dependencies
└── assets/              # Screenshots for this README
```

### Data Flow

```
                    ┌─────────────────────────────────────────┐
                    │           TunnelSpec AI Platform          │
                    └─────────────────────────────────────────┘
                                        │
          ┌─────────────────────────────┼─────────────────────────────┐
          │                             │                             │
   ┌──────▼──────┐              ┌───────▼──────┐              ┌───────▼──────┐
   │  Module A   │              │   Module B   │              │   Module C   │
   │  Tender     │              │   BoQ        │              │   IoT        │
   │  Intelligence│             │   Generator  │              │   Digital    │
   └──────┬──────┘              └───────┬──────┘              │   Twin       │
          │                             │                      └───────┬──────┘
   PDF → pypdf               Free-text scope →              NumPy simulation →
   → GPT-4o (3 RAG calls)    GPT-4o → structured           8 TBM sensor streams
   → Scope / Ground /        JSON BoQ with NRM2             → Plotly live charts
     Risk extraction         rates + HBT benchmarks         → Alert detection
   → Confidence scoring                                      → KPI dashboard
   → Opportunity Score       → CSV / Excel export
     (GO / COND / NO GO)
```

---

## 🔬 Module Deep-Dives

### Module A — Tender Intelligence Engine

Processes any construction tender PDF (ITT, Employer's Requirements, Geotechnical Interpretive Report) and returns:

- **Project Scope** — tunnelling method, diameter, drive length, shaft dimensions, programme milestones
- **Ground Conditions** — soil classification, SPT N-values, groundwater depth, mixed-face risk
- **Risk Register** — categorised under Geotechnical, Structural, Programme, Environmental, and Commercial

**Confidence Scoring** evaluates each extracted field across three dimensions:

| Dimension | Max Score | What it measures |
|-----------|-----------|-----------------|
| Completeness | 4 | Were all expected data elements present in the document? |
| Grounding | 3 | Are claims traceable to source text vs. hallucinated? |
| Specificity | 3 | Quantified figures and precise engineering terminology? |

**Opportunity Score** rates HBT's fit using a weighted rubric:

| Dimension | Weight | Description |
|-----------|--------|-------------|
| Technical Fit | 4.0 | Alignment with HBT trenchless methods and plant capability |
| Ground Suitability | 2.5 | Suitability of ground conditions for HBT's equipment |
| Risk Profile | 2.0 | Overall risk level — lower risk scores higher |
| Strategic Value | 1.5 | Geography, client relationship, market sector |

Output: **GO / CONDITIONAL GO / NO GO** recommendation with executive rationale.

---

### Module B — Bill of Quantities Generator

Converts an unstructured scope description into a fully structured BoQ using GPT-4o, applying:

- **NRM2 / CESMM4** measurement conventions
- **2025 UK market rates** for specialist trenchless works
- **HBT Historical Benchmark** column — typical past project cost ranges per item type (simulating integration with HBT's internal cost database)

Sections generated: Preliminaries · Shaft Construction · Main Tunnel Drive · Pipeline & Lining · Grouting & Ground Treatment · Utilities Diversions · Reinstatement · Risk Allowances

Built-in example scopes:
- 500 mm Gravity Sewer — Pipe Jacking (Herrenknecht AVN500)
- 1200 mm Water Main — EPB-TBM (segmental lining)
- 250 mm HDPE — HDD Railway Crossing (Network Rail interface)

**Export:** CSV and professionally formatted `.xlsx` with two sheets (BoQ + Summary).

---

### Module C — IoT Digital Twin

Simulates a live TBM sensor telemetry pipeline, mimicking what an MQTT broker would receive from field instruments on an active tunnel drive:

| Sensor | Unit | Alert Threshold |
|--------|------|----------------|
| TBM Face Pressure | bar | 3.8 bar |
| Cutterhead Torque | kN·m | 620 kN·m |
| Advance Rate | mm/min | 75 mm/min |
| Vibration (RMS) | mm/s | 3.0 mm/s |
| Water Ingress | L/min | 2.5 L/min |
| Tail Grout Pressure | bar | 3.0 bar |
| Surface Settlement | mm | 8.0 mm |
| Annulus Grout Volume | L/ring | 350 L/ring |

Each sensor stream includes: Gaussian noise · linear geological trend drift · sinusoidal stratigraphy variation · 2% probability operational spike events.

**KPI Dashboard:** Drive Chainage · Rings Built · Average Advance Rate · Average Face Pressure · Max Settlement · TBM Uptime %

**Sensor Correlation Matrix** (All Sensors tab) identifies cross-sensor dependencies in real time.

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10 or higher
- An OpenAI API key ([platform.openai.com](https://platform.openai.com)) — required for Modules A & B only
- Module C runs fully offline with no API key

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/Soheil-jafari/tunnelspec-ai-.git
cd tunnelspec-ai-

# 2. Create and activate a virtual environment (recommended)
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch the app
streamlit run app.py
```

The app will open automatically at `http://localhost:8501`.

### Configuration

No `.env` file or config needed. Enter your OpenAI API key directly in the sidebar when the app loads.

---

## 📦 Dependencies

| Package | Purpose |
|---------|---------|
| `streamlit` | Web UI framework |
| `openai` | GPT-4o API client |
| `pypdf` | PDF text extraction |
| `pandas` | Data manipulation and BoQ structuring |
| `numpy` | Sensor time-series simulation |
| `plotly` | Interactive charts and live dashboards |
| `openpyxl` | Excel (.xlsx) export |

---

## 🗺 KTP Technical Roadmap

This prototype establishes the foundation for a 28-month KTP workplan. Planned Phase 2 upgrades:

### Mamba-SSM for Long-Horizon IoT Monitoring

The current IoT module uses NumPy simulation. The production architecture will integrate **Mamba (Selective State Space Models)** for real sensor sequence modelling:

| Aspect | Transformer (Attention) | Mamba-SSM |
|--------|------------------------|-----------|
| Time complexity | O(n²) — quadratic | O(n) — linear |
| Memory scaling | Memory wall at long sequences | Constant memory footprint |
| 28-month telemetry | Requires aggressive chunking | Full sequence, no truncation |
| Edge deployment | High compute cost | Hardware-aware recurrence, GPU/edge optimised |

**Target outcome:** Predictive anomaly detection 60–90 seconds ahead of threshold breach — giving TBM operators intervention time before a ground loss or structural event.

### Phase 2 Feature Pipeline

- [ ] Live MQTT broker integration (Eclipse Mosquitto / AWS IoT Core)
- [ ] Mamba-SSM anomaly prediction model trained on HBT historical drive data
- [ ] Automated tender shortlisting from procurement portal APIs
- [ ] Multi-project portfolio dashboard (concurrent drive monitoring)
- [ ] RAG over HBT's internal cost database for live benchmark pricing
- [ ] BIM/GIS integration for alignment and geology visualisation

---

## 👤 Author

**Soheil Jafari**
AI Engineer | MSc [Your Degree] | BEng Electrical Engineering

KTP Associate Candidate — HB Tunnelling Ltd × Birmingham City University

[![GitHub](https://img.shields.io/badge/GitHub-Soheil--jafari-181717?style=flat-square&logo=github)](https://github.com/Soheil-jafari)

---

## 📄 License

This project is licensed under the MIT License.

---

<div align="center">

*Built to demonstrate the technical vision for the HBT-BCU Knowledge Transfer Partnership.*
*Trenchless · Pipe Jacking · Micro-tunnelling · EPB-TBM · HDD*

</div>
