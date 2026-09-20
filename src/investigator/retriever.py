"""Astrophysical literature retriever mapping observational features to search queries."""

from typing import Dict, List, Optional, Tuple
from src.knowledge_base.vector_store import VectorStore
from src.knowledge_base.chunker import DocumentChunk


class LiteratureRetriever:
    """Constructs astrophysical queries from event light curve features and retrieves evidence."""

    def __init__(self, vector_store: VectorStore, top_k: int = 4):
        self.vector_store = vector_store
        self.top_k = top_k

    def build_query_from_features(self, features: Dict[str, float]) -> str:
        """Formulate a targeted astronomical query reflecting dominant photometric anomalies."""
        terms = ["astronomical transient", "optical light curve"]

        color_gr = features.get("color_g_over_r", 1.0)
        amplitude = features.get("global_amplitude", 1.0)
        rise_rate = features.get("g_rise_rate", 0.0)
        decay_rate = features.get("r_decay_rate", 0.0)
        host_offset = features.get("host_transient_offset", 0.0)

        # 1. Color indicators
        if color_gr < 0.8:
            terms.append("red color infrared excess dust formation stellar merger")
        elif color_gr > 1.2:
            terms.append("blue UV excess high temperature black hole magnetar")

        # 2. Amplitude / Luminosity
        if amplitude > 2.5:
            terms.append("superluminous extreme peak luminosity magnetar spin-down")

        # 3. Time evolution / Rise & Decay
        if rise_rate > 0 and rise_rate < 0.05:
            terms.append("prolonged slow rise time")
        elif rise_rate > 0.5:
            terms.append("rapid flare minutes hours")

        # 4. Spatial centroid
        if host_offset < 1.0:
            terms.append("nuclear host galaxy center tidal disruption event")

        return " ".join(terms)

    def retrieve(self, features: Dict[str, float], top_k: Optional[int] = None) -> List[DocumentChunk]:
        """Query vector store and return relevant research chunks."""
        query_text = self.build_query_from_features(features)
        k = top_k or self.top_k
        results = self.vector_store.query(query_text, top_k=k)
        return [chunk for chunk, _ in results]
