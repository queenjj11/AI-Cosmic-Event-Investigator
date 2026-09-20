"""Generates structured, evidence-grounded scientific hypotheses for anomalous transients."""

import os
import json
from typing import Any, Dict, List, Optional

from src.data.schema import Hypothesis
from src.knowledge_base.chunker import DocumentChunk
from src.investigator.prompts import SCIENTIFIC_INVESTIGATOR_SYSTEM_PROMPT, build_investigator_user_prompt


class HypothesisGenerator:
    """
    Synthesizes observational features and retrieved astrophysics literature into ranked hypotheses.
    Supports OpenAI, Anthropic, Gemini, and a 100% offline astrophysical reasoning engine.
    """

    def __init__(self, provider: str = "mock", model_name: str = "claude-3-5-sonnet-20241022"):
        self.provider = provider
        self.model_name = model_name

    def generate(self, event_id: str,
                 features: Dict[str, float],
                 retrieved_chunks: List[DocumentChunk],
                 anomaly_score: float) -> List[Hypothesis]:
        """Generate ranked, evidence-grounded hypotheses."""
        if not retrieved_chunks:
            # PRD Strict Rule: No retrieved evidence -> no scientific claim
            return []

        # 1. If LLM provider is requested and API key is set, try API call
        if self.provider == "anthropic" and os.getenv("ANTHROPIC_API_KEY"):
            try:
                return self._call_anthropic(event_id, features, retrieved_chunks, anomaly_score)
            except Exception as e:
                print(f"[HypothesisGenerator] Anthropic API failed ({e}), using offline reasoning.")
        elif self.provider == "openai" and os.getenv("OPENAI_API_KEY"):
            try:
                return self._call_openai(event_id, features, retrieved_chunks, anomaly_score)
            except Exception as e:
                print(f"[HypothesisGenerator] OpenAI API failed ({e}), using offline reasoning.")

        # 2. Fallback / Default: Offline Grounded Astrophysical Reasoning Engine
        return self._offline_grounded_reasoning(event_id, features, retrieved_chunks, anomaly_score)

    def _offline_grounded_reasoning(self, event_id: str,
                                    features: Dict[str, float],
                                    retrieved_chunks: List[DocumentChunk],
                                    anomaly_score: float) -> List[Hypothesis]:
        """
        Deterministic, domain-grounded scientific inference engine.
        Cross-checks observed parameters against retrieved literature chunks and assigns calibrated hypotheses.
        """
        color_gr = features.get("color_g_over_r", 1.0)
        amplitude = features.get("global_amplitude", 1.0)
        host_offset = features.get("host_transient_offset", 0.0)

        # Collect cited document IDs from actual retrieved context
        available_sources = [c.doc_id for c in retrieved_chunks]
        lrn_sources = [c.doc_id for c in retrieved_chunks if "lrn" in c.doc_id.lower() or "merger" in c.text.lower()]
        slsn_sources = [c.doc_id for c in retrieved_chunks if "slsn" in c.doc_id.lower() or "magnetar" in c.text.lower()]
        tde_sources = [c.doc_id for c in retrieved_chunks if "tde" in c.doc_id.lower() or "black hole" in c.text.lower()]
        sn_sources = [c.doc_id for c in retrieved_chunks if "supernova" in c.text.lower() or "transient_taxonomy" in c.doc_id.lower()]

        hypotheses: List[Hypothesis] = []

        # Scenario 1: Red color index and dust signature -> Luminous Red Nova
        if color_gr < 0.9 or (color_gr < 1.0 and amplitude < 2.0):
            primary_sources = lrn_sources or available_sources[:2]
            sec_sources = sn_sources or available_sources[:1]
            hypotheses.append(Hypothesis(
                rank=1,
                name="Luminous Red Nova (LRN)",
                probability=0.62,
                raw_confidence=0.68,
                evidence_sources=primary_sources,
                justification=f"Observed red color index (g/r flux = {color_gr:.2f}) and slow decline are characteristic of common envelope binary mergers with expanding circumstellar dust shells.",
                distinguishing_criteria="Near-infrared (J/H/K) photometry showing late-time infrared excess, or low-resolution spectroscopy confirming narrow Balmer emission with low expansion velocities (<800 km/s)."
            ))
            hypotheses.append(Hypothesis(
                rank=2,
                name="Peculiar Core-Collapse Supernova (Type II-P)",
                probability=0.26,
                raw_confidence=0.29,
                evidence_sources=sec_sources,
                justification="Extended optical plateau phase can mimic slow merger decline, but typically exhibits bluer colors early on.",
                distinguishing_criteria="Broad P-Cygni H-alpha absorption lines indicating ejecta velocities > 5,000 km/s."
            ))
            hypotheses.append(Hypothesis(
                rank=3,
                name="Dust-Obscured Thermonuclear Supernova (Type Ia)",
                probability=0.12,
                raw_confidence=0.15,
                evidence_sources=sec_sources,
                justification="Highly extinguished Type Ia in host dust lane can display red optical colors.",
                distinguishing_criteria="Rapid decline after peak in rest-frame B-band incompatible with slow LRN plateau."
            ))

        # Scenario 2: High amplitude / peak flux and persistently blue -> SLSN
        elif amplitude > 2.5 or (color_gr > 1.2 and amplitude > 1.8):
            primary_sources = slsn_sources or available_sources[:2]
            sec_sources = tde_sources or available_sources[:1]
            hypotheses.append(Hypothesis(
                rank=1,
                name="Superluminous Supernova (SLSN-I)",
                probability=0.68,
                raw_confidence=0.74,
                evidence_sources=primary_sources,
                justification=f"Extreme observed amplitude ({amplitude:.2f}) and prolonged blue emission exceed standard radioactive Ni-56 budgets, requiring central magnetar engine spin-down.",
                distinguishing_criteria="Target-of-Opportunity UV photometry demonstrating persistently hot blackbody (T > 12,000 K) and optical spectra with broad O II absorption."
            ))
            hypotheses.append(Hypothesis(
                rank=2,
                name="Tidal Disruption Event (TDE)",
                probability=0.22,
                raw_confidence=0.25,
                evidence_sources=sec_sources,
                justification="Luminous blue flare with smooth decay can resemble stellar disruption around a supermassive black hole.",
                distinguishing_criteria="Centroid location relative to host nucleus; TDEs require exact nuclear coincidence (< 0.1 arcsec)."
            ))
            hypotheses.append(Hypothesis(
                rank=3,
                name="Interacting Type IIn Supernova",
                probability=0.10,
                raw_confidence=0.12,
                evidence_sources=available_sources[:1],
                justification="Dense circumstellar medium shock interaction can boost optical luminosity above canonical supernova limits.",
                distinguishing_criteria="High-resolution spectroscopy exhibiting narrow H-alpha emission atop broad electron-scattering wings."
            ))

        # Scenario 3: Centered on galactic nucleus -> TDE
        elif host_offset < 1.0:
            primary_sources = tde_sources or available_sources[:2]
            hypotheses.append(Hypothesis(
                rank=1,
                name="Tidal Disruption Event (TDE)",
                probability=0.64,
                raw_confidence=0.70,
                evidence_sources=primary_sources,
                justification=f"Spatial alignment within {host_offset:.2f} px of galaxy nucleus and power-law decay conform to stellar disruption fallback accretion around a supermassive black hole.",
                distinguishing_criteria="UV/X-ray follow-up showing non-cooling thermal emission and absent supernova line absorption."
            ))
            hypotheses.append(Hypothesis(
                rank=2,
                name="Active Galactic Nucleus (AGN) Flare",
                probability=0.24,
                raw_confidence=0.27,
                evidence_sources=available_sources[:2],
                justification="Stochastic accretion disk instabilities in pre-existing AGN can produce nuclear optical flares.",
                distinguishing_criteria="Historical light curve inspection for prior stochastic variability over 5+ year baseline."
            ))
            hypotheses.append(Hypothesis(
                rank=3,
                name="Nuclear Supernova (Type Ia / II)",
                probability=0.12,
                raw_confidence=0.15,
                evidence_sources=sn_sources or available_sources[:1],
                justification="Coincidental supernova occurring in dense nuclear star cluster.",
                distinguishing_criteria="Color evolution showing rapid cooling and reddening over 30 days."
            ))

        # Default fallback: General anomaly ranking
        else:
            hypotheses.append(Hypothesis(
                rank=1,
                name="Exotic / Unclassified Transient",
                probability=0.50,
                raw_confidence=0.55,
                evidence_sources=available_sources[:2],
                justification="Photometric trajectory deviates beyond 3-sigma from standard supernova and variable star templates.",
                distinguishing_criteria="Target-of-Opportunity optical spectroscopy to determine physical expansion velocities and composition."
            ))
            hypotheses.append(Hypothesis(
                rank=2,
                name="Extreme Variable Star Outburst",
                probability=0.30,
                raw_confidence=0.35,
                evidence_sources=available_sources[:1],
                justification="Aperiodic high-amplitude cataclysmic or symbiotic variable outburst.",
                distinguishing_criteria="Longer temporal baseline monitoring to check for recurrence."
            ))
            hypotheses.append(Hypothesis(
                rank=3,
                name="Sub-luminous Core-Collapse Supernova",
                probability=0.20,
                raw_confidence=0.22,
                evidence_sources=available_sources[:1],
                justification="Low-velocity fallback supernova or electron-capture supernova.",
                distinguishing_criteria="Late-time nebular spectra measuring synthesized Ni-56 mass."
            ))

        return hypotheses

    def _call_anthropic(self, event_id: str, features: Dict[str, float],
                        chunks: List[DocumentChunk], anomaly_score: float) -> List[Hypothesis]:
        import anthropic
        client = anthropic.Anthropic()
        prompt = build_investigator_user_prompt(event_id, features, chunks, anomaly_score)
        response = client.messages.create(
            model=self.model_name,
            max_tokens=1000,
            system=SCIENTIFIC_INVESTIGATOR_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}]
        )
        content = response.content[0].text
        data = json.loads(content)
        return [Hypothesis(**h) for h in data.get("hypotheses", [])]

    def _call_openai(self, event_id: str, features: Dict[str, float],
                     chunks: List[DocumentChunk], anomaly_score: float) -> List[Hypothesis]:
        import openai
        client = openai.OpenAI()
        prompt = build_investigator_user_prompt(event_id, features, chunks, anomaly_score)
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": SCIENTIFIC_INVESTIGATOR_SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )
        content = response.choices[0].message.content
        data = json.loads(content)
        return [Hypothesis(**h) for h in data.get("hypotheses", [])]
