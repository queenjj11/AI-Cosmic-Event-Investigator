"""Unit tests for the RAG knowledge base and hypothesis generation."""

import pytest
from src.knowledge_base.document_loader import AstroDocument
from src.knowledge_base.chunker import DocumentChunker
from src.knowledge_base.vector_store import VectorStore
from src.investigator.retriever import LiteratureRetriever
from src.investigator.hypothesis_generator import HypothesisGenerator
from src.investigator.calibration import ConfidenceCalibrator
import numpy as np


def test_chunker_and_vector_store():
    doc = AstroDocument(
        doc_id="paper_1",
        title="Superluminous Supernovae Magnetar Models",
        content="### Section 1\nSLSNe are powered by millisecond magnetar spin-down.\n### Section 2\nDiagnostic signatures include broad O II absorption lines.",
        source_type="paper",
        filepath="/fake/path"
    )
    chunker = DocumentChunker(chunk_size=100)
    chunks = chunker.chunk_document(doc)
    assert len(chunks) >= 2

    vs = VectorStore()
    vs.add_chunks(chunks)

    results = vs.query("magnetar spin-down broad lines", top_k=2)
    assert len(results) > 0
    top_chunk, sim = results[0]
    assert "magnetar" in top_chunk.text.lower() or "broad" in top_chunk.text.lower()


def test_strict_grounding_rule():
    """PRD rule: No retrieved evidence -> no scientific claim."""
    generator = HypothesisGenerator(provider="mock")
    # Empty retrieved chunks
    hypotheses = generator.generate(
        event_id="TEST_01",
        features={"global_amplitude": 2.5},
        retrieved_chunks=[],
        anomaly_score=0.9
    )
    assert len(hypotheses) == 0  # Must refuse to output ungrounded claims


def test_confidence_calibrator():
    calibrator = ConfidenceCalibrator(method="isotonic")
    raw_confs = np.array([0.95, 0.90, 0.85, 0.80, 0.70, 0.60, 0.50, 0.40])
    correctness = np.array([1, 1, 1, 0, 1, 0, 0, 0])

    calibrator.fit(raw_confs, correctness)
    cal_p = calibrator.calibrate(np.array([0.95, 0.40]))

    assert cal_p[0] >= cal_p[1]
    assert np.all(cal_p <= 1.0)
    assert np.all(cal_p >= 0.0)
