"""
TunnelSpec AI — Intelligence Engine
PDF parsing with pypdf + direct OpenAI GPT-4o calls.
No LlamaIndex dependency required.
"""

import json
import tempfile
import os
from typing import Any, Dict, List

import pypdf
from openai import OpenAI as _OpenAI


def _extract_pdf_text(pdf_bytes: bytes) -> str:
    """Extract all text from a PDF given its raw bytes."""
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(pdf_bytes)
        tmp_path = tmp.name
    try:
        reader = pypdf.PdfReader(tmp_path)
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n\n".join(pages)
    finally:
        os.unlink(tmp_path)


def _clean_json(raw: str) -> str:
    """Strip markdown code fences from an LLM response."""
    if "```" in raw:
        raw = raw.split("```")[1]
        if raw.startswith("json\n"):
            raw = raw[5:]
    return raw.strip()


class TenderEngine:
    """
    Main AI engine.
    - Reads PDF tender documents with pypdf.
    - Extracts structured tender intelligence via GPT-4o.
    - Scores opportunities against HBT's capability profile.
    - Scores extraction confidence per field.
    - Generates Bill of Quantities from free-text scope descriptions.
    """

    def __init__(self, api_key: str, model: str = "gpt-4o"):
        self.api_key = api_key
        self.model = model
        self._client = _OpenAI(api_key=api_key)
        self._doc_text: str = ""

    # ------------------------------------------------------------------
    # Document loading
    # ------------------------------------------------------------------

    def load_pdf_document(self, pdf_bytes: bytes, filename: str) -> int:
        """Extract text from a PDF. Returns page count."""
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(pdf_bytes)
            tmp_path = tmp.name
        try:
            reader = pypdf.PdfReader(tmp_path)
            pages = [page.extract_text() or "" for page in reader.pages]
            self._doc_text = "\n\n".join(pages)
            return len(reader.pages)
        finally:
            os.unlink(tmp_path)

    def _chat(self, system: str, user: str, temperature: float = 0.1) -> str:
        """Single GPT-4o call. Returns the assistant message text."""
        response = self._client.chat.completions.create(
            model=self.model,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
        )
        return response.choices[0].message.content.strip()

    # ------------------------------------------------------------------
    # Tender intelligence extraction
    # ------------------------------------------------------------------

    def extract_tender_intelligence(self) -> Dict[str, str]:
        """
        Extract Project Scope, Ground Conditions, and Risk Factors
        from the loaded document text.
        """
        if not self._doc_text:
            return {k: "No document loaded." for k in
                    ("project_scope", "ground_conditions", "risk_factors")}

        system = (
            "You are a senior civil engineer specialising in trenchless construction. "
            "Answer only from the provided tender document text. "
            "Use precise civil/geotechnical engineering terminology."
        )

        queries = {
            "project_scope": (
                "Summarise the Project Scope from the document below. Include: tunnelling method "
                "(TBM, pipe jacking, micro-tunnelling, HDD, auger boring), pipe/tunnel diameter, "
                "total drive length, cover depth, launch/reception shaft dimensions, key "
                "deliverables and programme milestones.\n\nDOCUMENT:\n" + self._doc_text
            ),
            "ground_conditions": (
                "Describe the Ground and Geotechnical Conditions from the document below. Include: "
                "soil/rock classification, SPT N-values or CPT qc if stated, groundwater table "
                "depth, mixed-face risk, settlement sensitivity of overlying structures, "
                "obstructions, and any ground treatment measures.\n\nDOCUMENT:\n" + self._doc_text
            ),
            "risk_factors": (
                "List the key Risk Factors from the document below as bullet points, categorised "
                "under: 1. Geotechnical & Ground Risks, 2. Structural & Third-Party Risks, "
                "3. Programme & Sequencing Risks, 4. Environmental & HSE Risks, "
                "5. Commercial & Contractual Risks. Include likelihood/severity where stated."
                "\n\nDOCUMENT:\n" + self._doc_text
            ),
        }

        results: Dict[str, str] = {}
        for key, user_prompt in queries.items():
            try:
                results[key] = self._chat(system, user_prompt)
            except Exception as exc:
                results[key] = f"Extraction failed: {exc}"
        return results

    # ------------------------------------------------------------------
    # Extraction confidence scoring
    # ------------------------------------------------------------------

    def score_extraction_confidence(self, tender_data: Dict[str, str]) -> Dict[str, Any]:
        """
        Assess the confidence of each extracted field on three dimensions:
          Completeness (0–4) + Grounding (0–3) + Specificity (0–3) = total 0–10.
        Bands: HIGH >= 8, MEDIUM >= 5, LOW < 5.
        Fields scored LOW have needs_review = True.
        """
        FIELD_CRITERIA = {
            "project_scope": (
                "Expected: tunnelling method, diameter, drive length, cover depth, "
                "shaft dimensions, key deliverables, programme milestones."
            ),
            "ground_conditions": (
                "Expected: soil/rock classification, SPT/CPT values, groundwater depth, "
                "mixed-face risk, settlement sensitivity, ground treatment measures."
            ),
            "risk_factors": (
                "Expected: geotechnical risks, third-party/structural risks, programme risks, "
                "environmental/HSE risks, commercial risks, likelihood/severity ratings."
            ),
        }

        system = "You are a QA reviewer for an AI tender analysis system. Return only valid JSON."

        results: Dict[str, Any] = {}

        for field, criteria in FIELD_CRITERIA.items():
            text = tender_data.get(field, "")

            if not text or text.startswith("No document"):
                results[field] = {
                    "score": 0, "band": "LOW",
                    "completeness": 0, "grounding": 0, "specificity": 0,
                    "caveats": ["No content extracted."],
                    "reviewer_note": "Field is empty.",
                    "needs_review": True,
                }
                continue

            user_prompt = f"""Assess this extracted tender intelligence against the criteria below.

EXTRACTED TEXT:
{text}

EXPECTED CONTENT:
{criteria}

Score each dimension (integers only):
  completeness 0-4  (4 = all expected elements present)
  grounding    0-3  (3 = every claim traceable to source, 0 = vague/hallucinated)
  specificity  0-3  (3 = quantified figures and precise terms throughout)

Return ONLY this JSON:
{{
  "completeness": <int>,
  "grounding": <int>,
  "specificity": <int>,
  "caveats": ["<gap or concern>"],
  "reviewer_note": "<one sentence summary of main quality issue if any>"
}}"""

            try:
                raw = self._chat(system, user_prompt, temperature=0.0)
                dims = json.loads(_clean_json(raw))

                completeness = int(dims.get("completeness", 0))
                grounding    = int(dims.get("grounding",    0))
                specificity  = int(dims.get("specificity",  0))
                score        = completeness + grounding + specificity

                band = "HIGH" if score >= 8 else "MEDIUM" if score >= 5 else "LOW"

                results[field] = {
                    "score":         score,
                    "band":          band,
                    "completeness":  completeness,
                    "grounding":     grounding,
                    "specificity":   specificity,
                    "caveats":       dims.get("caveats", []),
                    "reviewer_note": dims.get("reviewer_note", ""),
                    "needs_review":  band == "LOW",
                }
            except Exception as exc:
                results[field] = {
                    "score": 0, "band": "LOW",
                    "completeness": 0, "grounding": 0, "specificity": 0,
                    "caveats": [f"Confidence scoring failed: {exc}"],
                    "reviewer_note": "",
                    "needs_review": True,
                }

        return results

    # ------------------------------------------------------------------
    # Opportunity scoring
    # ------------------------------------------------------------------

    def calculate_opportunity_score(self, tender_data: Dict[str, str]) -> Dict[str, Any]:
        """
        Score HBT's fit for this tender (1–10) using a weighted rubric.
        Returns structured dict with recommendation: GO / CONDITIONAL GO / NO GO.
        """
        system = (
            "You are the Technical Director at HB Tunnelling (HBT), a leading UK specialist "
            "contractor in trenchless construction: EPB-TBM tunnelling, pipe jacking, "
            "micro-tunnelling, auger boring, and HDD. "
            "HBT's sweet spot: 300–3000 mm diameter drives in mixed ground urban environments. "
            "Return only valid JSON — no prose."
        )

        user_prompt = f"""Evaluate this tender and return a JSON scoring object.

PROJECT SCOPE:
{tender_data.get("project_scope", "Not extracted")}

GROUND CONDITIONS:
{tender_data.get("ground_conditions", "Not extracted")}

RISK FACTORS:
{tender_data.get("risk_factors", "Not extracted")}

Scoring rubric:
  technical_fit        max 4.0  (alignment with HBT trenchless methods & plant)
  ground_suitability   max 2.5  (suitability of ground conditions for HBT plant)
  risk_profile         max 2.0  (lower/manageable risk = higher score)
  strategic_value      max 1.5  (location, client relationship, market sector)

Return ONLY this JSON:
{{
  "overall_score": <int 1-10>,
  "dimensions": {{
    "technical_fit":      {{"score": <float>, "comment": "<note>"}},
    "ground_suitability": {{"score": <float>, "comment": "<note>"}},
    "risk_profile":       {{"score": <float>, "comment": "<note>"}},
    "strategic_value":    {{"score": <float>, "comment": "<note>"}}
  }},
  "recommendation": "<GO | CONDITIONAL GO | NO GO>",
  "key_strengths": ["<strength>", "<strength>", "<strength>"],
  "key_concerns":  ["<concern>", "<concern>"],
  "rationale": "<3-4 sentence executive summary>"
}}"""

        try:
            raw = self._chat(system, user_prompt)
            return json.loads(_clean_json(raw))
        except Exception as exc:
            return {
                "overall_score": 5,
                "recommendation": "CONDITIONAL GO",
                "rationale": f"Auto-scoring unavailable: {exc}",
                "key_strengths": [], "key_concerns": [], "dimensions": {},
            }

    # ------------------------------------------------------------------
    # BoQ generation
    # ------------------------------------------------------------------

    def generate_boq(self, scope_text: str) -> List[Dict[str, Any]]:
        """
        Generate a structured Bill of Quantities from an unstructured scope description.
        NRM2 / CESMM4 conventions, UK 2025 market rates.
        """
        system = (
            "You are a Senior Quantity Surveyor specialising in underground construction "
            "and trenchless technology in the UK. Return only valid JSON."
        )

        user_prompt = f"""Generate a comprehensive Bill of Quantities for the scope below.
Use current UK market rates (2025). Apply NRM2 / CESMM4 measurement conventions.

SCOPE OF WORKS:
{scope_text}

Return ONLY a JSON array. Each item must have exactly:
  "item_no"           — string (e.g. "A.1", "B.2")
  "section"           — string (e.g. "Preliminaries", "Shaft Construction", "Tunnelling Works",
                        "Pipeline & Lining", "Grouting & Ground Treatment", "Reinstatement", "Risk Allowances")
  "description"       — string (detailed, civil engineering terminology)
  "quantity"          — number
  "unit"              — string (m, m², m³, nr, sum, t, wk)
  "rate_gbp"          — number (realistic 2025 UK rate)
  "amount_gbp"        — number (quantity × rate, 2 d.p.)
  "hbt_benchmark_gbp" — string: typical HBT historical cost range for this item type drawn from
                        past UK trenchless projects, formatted as e.g. "£140 - £165" per unit.
                        Base this on realistic UK specialist contractor historical data for the
                        item category (e.g. pipe jacking drives, shaft excavation, grout works).
  "notes"             — string (spec ref, assumption, or NRM2 clause)

Include: Preliminaries, Shaft Construction (launch + reception), Main Tunnel Drive,
Pipeline/Lining, Grouting, Utilities, Reinstatement, Contingency & Risk Allowances.
Generate at least 18 line items. Return ONLY the JSON array."""

        try:
            raw = self._chat(system, user_prompt, temperature=0.2)
            return json.loads(_clean_json(raw))
        except Exception as exc:
            return [{"error": str(exc), "raw_response": raw[:600]}]
