"""
One-shot generator for samples/riverside_trunk_sewer_ITT.pdf.

The synthetic document is laid out so that the cascading risk chain is
deliberately split across separate sections — each link in the chain
appears in a different chapter. A naive vector retriever asking
"what could a groundwater ingress event ultimately do to project cost?"
will retrieve the groundwater section but is unlikely to retrieve all
intermediate risks in a single shot. The graph traversal can follow
the chain regardless of chunk boundaries.

Run from the project root:
    python samples/generate_sample_pdf.py
"""

from __future__ import annotations

from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
)


OUTPUT_PATH = Path(__file__).parent / "riverside_trunk_sewer_ITT.pdf"


SECTIONS = [
    (
        "1. Project Overview",
        [
            "This Invitation to Tender (ITT) covers the Riverside Trunk Sewer Upgrade, "
            "a 1,200 m segmental concrete tunnel of 1,800 mm internal diameter to be "
            "constructed by Earth Pressure Balance Tunnel Boring Machine (EPB-TBM) "
            "beneath the Riverside Quarter mixed-use development in central Birmingham. "
            "The Project is procured by Severn Trent Water under an NEC4 Option C target-cost contract. "
            "HB Tunnelling Ltd (HBT) is invited to bid as principal contractor, "
            "with City Council acting as highway authority and Network Rail as adjacent "
            "third-party asset owner along Chainage 600 m to 720 m.",
            "The Works are split into six phases: Phase 1 Site Setup, Phase 2 Launch Shaft "
            "Construction, Phase 3 Tunnel Drive, Phase 4 Reception Shaft Construction, "
            "Phase 5 Pipeline and Lining Works, and Phase 6 Highway and Public Realm "
            "Reinstatement. The intended sequencing is strictly linear: Phase 3 depends on "
            "completion of Phase 2, and Phase 5 depends on completion of Phase 3.",
        ],
    ),
    (
        "2. Ground Conditions",
        [
            "The Geotechnical Interpretive Report (GIR) characterises the drive horizon as "
            "Mercia Mudstone Group overlain by 4 m to 6 m of River Terrace Gravels and "
            "made ground. Standard Penetration Test results across the route give "
            "N = 15 to 25 in the gravels and N > 50 in the weathered mudstone. "
            "The groundwater table is recorded at 4.0 m below ground level in piezometers "
            "BH-04 and BH-07, sitting within the gravels above the drive crown.",
            "The combination of permeable gravels at the drive crown and a high "
            "groundwater table presents a credible risk of Groundwater Ingress at the "
            "TBM face during Phase 3 Tunnel Drive. This risk is the dominant geotechnical "
            "hazard for the Project. Tenderers must demonstrate adequate mitigation; "
            "the Employer's Requirements specify pre-drive ground treatment (cementitious "
            "permeation grouting) along Chainage 240 m to 360 m as the principal mitigation "
            "for groundwater ingress.",
        ],
    ),
    (
        "3. Risk Register — Geotechnical",
        [
            "Risk G-01: Groundwater Ingress at TBM Face. Likelihood: High. Severity: High. "
            "Triggered by encountering high-permeability gravels with hydraulic head above "
            "drive crown. Mitigated by pre-drive cementitious permeation grouting and by "
            "maintaining EPB face pressure at 1.4 to 1.8 bar.",
            "Risk G-02: Surface Settlement. Likelihood: Medium. Severity: High. "
            "Settlement risk is closely related to groundwater ingress: dewatering at the "
            "face draws fines from the overlying gravels and mobilises consolidation "
            "settlement of the made ground. Mitigated by precise face pressure control "
            "and by the Settlement Monitoring Programme described in Section 6.",
        ],
    ),
    (
        "4. Risk Register — Programme & Operations",
        [
            "Risk P-01: TBM Stoppage. Likelihood: Medium. Severity: High. A sustained "
            "groundwater ingress event triggers TBM stoppage because the EPB cannot "
            "maintain face pressure once the spoil becomes excessively fluidised. "
            "Recovery requires intervention through hyperbaric chamber access, "
            "typically adding 10 to 20 working days per event.",
            "Risk P-02: Programme Delay. Likelihood: Medium. Severity: High. "
            "TBM stoppage on the critical path triggers Programme Delay across "
            "Phase 3 Tunnel Drive and, by the linear dependency described in "
            "Section 1, into Phase 5 Pipeline and Lining Works. The Employer "
            "imposes liquidated damages of £25,000 per calendar day beyond "
            "Sectional Completion Date for Phase 3.",
        ],
    ),
    (
        "5. Risk Register — Third Party & Commercial",
        [
            "Risk T-01: Third-Party Building Damage. Likelihood: Low. Severity: High. "
            "Surface Settlement above the drive corridor can cause cosmetic and "
            "structural damage to the listed Edwardian terrace at 14-22 Riverside Walk. "
            "This risk is presently UNMITIGATED in the Employer's Requirements and "
            "tenderers are invited to propose a Party Wall Agreement strategy and "
            "compensation grouting allowance.",
            "Risk C-01: Cost Overrun. Likelihood: Medium. Severity: High. "
            "Programme Delay triggers Cost Overrun through prolongation of "
            "site-establishment costs (Preliminaries running at £42,000 per week) "
            "and through liquidated damages exposure. Cost Overrun is also related "
            "to Third-Party Building Damage exposure should compensation grouting "
            "or remediation be required.",
        ],
    ),
    (
        "6. Mitigation Measures",
        [
            "M-01 Pre-Drive Ground Treatment. Cementitious permeation grouting along "
            "Chainage 240 m to 360 m. This mitigates Groundwater Ingress.",
            "M-02 Settlement Monitoring Programme. A grid of automated total stations "
            "and precise levelling along the drive corridor, with trigger and action "
            "levels set at 5 mm and 10 mm respectively. This mitigates Surface Settlement.",
            "Note that no mitigation is currently proposed for Third-Party Building Damage; "
            "tenderers are required to price and design this themselves.",
        ],
    ),
    (
        "7. Phase Dependencies & Required Materials",
        [
            "Phase 2 Launch Shaft Construction requires Sheet Piles (Larssen 605 grade), "
            "King-post propping, and a 6 m internal diameter excavation to 18 m depth. "
            "Phase 2 is the predecessor to Phase 3.",
            "Phase 3 Tunnel Drive requires the EPB-TBM (Herrenknecht S-1010 class), "
            "Precast Concrete Segmental Lining (250 mm thick six-segment ring at 1.2 m pitch), "
            "Cementitious Tail Grout, and a Drive Length specification of 1,200 m. "
            "Phase 3 depends on Phase 2 and is the predecessor to Phase 5.",
            "Phase 5 Pipeline and Lining Works requires the carrier pipe, internal "
            "secondary lining, and pressure-testing infrastructure. Phase 5 depends "
            "on completion of Phase 3.",
        ],
    ),
    (
        "8. Stakeholder Responsibilities",
        [
            "HB Tunnelling Ltd is responsible for Phase 2, Phase 3, Phase 4, and Phase 5 delivery. "
            "Severn Trent Water is responsible for Project sponsorship and acceptance testing. "
            "City Council is responsible for highway closure approvals during Phase 1 and Phase 6. "
            "Network Rail is responsible for possession management between Chainage 600 m and 720 m.",
        ],
    ),
    (
        "9. Programme Milestones",
        [
            "Sectional Completion Date for Phase 2: T0 + 16 weeks. "
            "Sectional Completion Date for Phase 3: T0 + 52 weeks. "
            "Sectional Completion Date for Phase 5: T0 + 70 weeks. "
            "Practical Completion: T0 + 80 weeks.",
        ],
    ),
    (
        "10. Bill of Quantities — Indicative Summary",
        [
            "Section A Preliminaries: site establishment £180,000, project management £210,000, "
            "temporary works design £75,000, insurance and bonds £95,000, environmental "
            "monitoring £60,000. Section B Shafts: launch shaft excavation and sheet piling "
            "£640,000, reception shaft excavation and sheet piling £420,000. Section C "
            "Tunnelling: EPB-TBM mobilisation and demobilisation £540,000, segmental lining "
            "supply £1,260,000, primary tunnel drive at £4,800 per linear metre over 1,200 m, "
            "tail grouting £180,000.",
            "Section D Ground Treatment: cementitious permeation grouting along Chainage "
            "240 m to 360 m £310,000, contingency for additional treatment if encountered "
            "£120,000. Section E Pipeline: carrier pipe supply and installation £490,000, "
            "internal secondary lining £180,000, hydrostatic pressure testing £60,000. "
            "Section F Reinstatement: highway reinstatement £215,000, public realm reinstatement "
            "£140,000. Section G Risk Allowances: contingency 8 percent of works value, "
            "design risk allowance 4 percent.",
        ],
    ),
    (
        "11. HSE & Environmental Constraints",
        [
            "Working hours are restricted to 07:00 to 19:00 Monday to Friday, with "
            "no weekend working permitted along Chainage 0 m to 240 m due to residential "
            "amenity. Noise emission at the site boundary is capped at 70 dB(A) during "
            "permitted hours. Dust suppression is required at the launch and reception "
            "shafts during excavation. The drive corridor passes within 30 m of the "
            "Riverside Walk Conservation Area and within 15 m of two protected London Plane "
            "trees, both of which are subject to Tree Preservation Orders.",
            "Spoil management constraint: contaminated arisings from the made ground at "
            "the launch shaft must be characterised under WM3 guidance and disposed of "
            "to a permitted facility. The tendered rate must include contingent provision "
            "for hazardous classification. Discharge of any TBM process water to surface "
            "water is prohibited; tenderers must propose a closed-loop water recycling and "
            "treatment arrangement at the launch shaft.",
        ],
    ),
    (
        "12. Quality, Testing & Acceptance",
        [
            "Acceptance of the segmental tunnel lining is conditional on internal CCTV "
            "structural survey demonstrating no longitudinal cracking exceeding 0.3 mm and "
            "no offsets between rings exceeding 5 mm. Carrier pipe acceptance requires "
            "hydrostatic pressure testing at 1.5 times working pressure for two hours with "
            "permissible pressure loss not exceeding 2 percent. Tail grouting is to be "
            "verified by post-grout core sampling at 5 percent of rings, randomly selected.",
            "Settlement monitoring data is to be reported daily to the Employer's Engineer "
            "during Phase 3 Tunnel Drive. Trigger Level breach (5 mm) requires written "
            "notification within 4 hours; Action Level breach (10 mm) requires immediate "
            "notification and triggers a joint review of TBM operating parameters. The "
            "Employer's Engineer reserves the right to instruct a temporary halt to the "
            "drive at Action Level until corrective measures are agreed.",
        ],
    ),
]


def build_pdf(path: Path = OUTPUT_PATH) -> Path:
    doc = SimpleDocTemplate(
        str(path),
        pagesize=A4,
        leftMargin=2.2 * cm, rightMargin=2.2 * cm,
        topMargin=2.0 * cm,  bottomMargin=2.0 * cm,
        title="Riverside Trunk Sewer Upgrade — ITT (Synthetic)",
        author="TunnelSpec AI demo fixture",
    )
    styles = getSampleStyleSheet()
    h1 = ParagraphStyle("H1", parent=styles["Heading1"], spaceAfter=14, fontSize=16)
    h2 = ParagraphStyle("H2", parent=styles["Heading2"], spaceBefore=14, spaceAfter=8, fontSize=12)
    body = ParagraphStyle(
        "Body", parent=styles["BodyText"], fontSize=10.5, leading=15,
        spaceAfter=8, alignment=4,
    )
    notice = ParagraphStyle(
        "Notice", parent=styles["BodyText"], fontSize=9, leading=12,
        textColor="#888888", spaceAfter=14,
    )

    story = [
        Paragraph("Riverside Trunk Sewer Upgrade", h1),
        Paragraph("Invitation to Tender — Synthetic Document for TunnelSpec AI Demo", h2),
        Paragraph(
            "This document is fabricated for demonstration purposes only. Names, "
            "chainages, parties, and rates are illustrative and do not refer to any real project.",
            notice,
        ),
    ]
    for title, paragraphs in SECTIONS:
        story.append(Paragraph(title, h2))
        for p in paragraphs:
            story.append(Paragraph(p, body))
        story.append(Spacer(1, 0.4 * cm))
    story.append(PageBreak())

    doc.build(story)
    return path


if __name__ == "__main__":
    out = build_pdf()
    print(f"Wrote {out} ({out.stat().st_size / 1024:.1f} KB)")
