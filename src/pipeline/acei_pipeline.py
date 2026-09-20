"""Master end-to-end orchestration pipeline for the AI Cosmic Event Investigator (ACEI)."""

import os
import time
import argparse
from typing import Any, Dict, List, Optional
import numpy as np
import torch

from src.data.schema import AstronomicalEvent, InvestigationResult, Hypothesis, RecommendedObservation
from src.data.dataset_builder import generate_benchmark_events, AstronomicalDataset
from src.data.splitter import ObjectLevelSplitter
from src.features.lightcurve_features import LightCurveFeatureExtractor
from src.features.image_features import ImageFeatureExtractor
from src.models.multimodal_model import MultimodalTransientModel
from src.anomaly_detection.autoencoder import MultimodalAutoencoder
from src.anomaly_detection.mahalanobis import MahalanobisDetector
from src.anomaly_detection.energy_score import EnergyOODDetector
from src.anomaly_detection.anomaly_ensemble import AnomalyEnsemble
from src.knowledge_base.document_loader import DocumentLoader
from src.knowledge_base.chunker import DocumentChunker
from src.knowledge_base.embedder import TextEmbedder
from src.knowledge_base.vector_store import VectorStore
from src.investigator.retriever import LiteratureRetriever
from src.investigator.hypothesis_generator import HypothesisGenerator
from src.investigator.hypothesis_ranker import HypothesisRanker
from src.investigator.calibration import ConfidenceCalibrator
from src.recommender.recommender import ActiveObservationRecommender


class ACEIPipeline:
    """
    End-to-end scientific reasoning system:
    Observe -> Detect Anomaly -> Retrieve Evidence -> Generate Hypotheses -> Calibrate -> Recommend Next Observation.
    """

    def __init__(self,
                 known_classes: Optional[List[str]] = None,
                 anomaly_threshold: float = 0.65,
                 device: str = "cpu"):
        self.device = device
        self.known_classes = known_classes or ["SN_Ia", "SN_II", "Stellar_Flare", "Variable_Star"]
        self.class_to_idx = {cls: i for i, cls in enumerate(self.known_classes)}
        self.idx_to_class = {i: cls for i, cls in enumerate(self.known_classes)}
        self.anomaly_threshold = anomaly_threshold

        # 1. Feature extractors
        self.lc_extractor = LightCurveFeatureExtractor()
        self.img_extractor = ImageFeatureExtractor()

        # 2. Deep Multimodal Model
        self.model = MultimodalTransientModel(num_classes=len(self.known_classes))

        # 3. Anomaly Detectors
        self.autoencoder = MultimodalAutoencoder(input_dim=256, latent_dim=64)
        self.mahalanobis = MahalanobisDetector()
        self.energy_detector = EnergyOODDetector(temperature=1.0)
        self.anomaly_ensemble = AnomalyEnsemble(
            autoencoder=self.autoencoder,
            mahalanobis=self.mahalanobis,
            energy_detector=self.energy_detector,
            threshold=anomaly_threshold
        )

        # 4. Knowledge Base & RAG
        self.vector_store = VectorStore(embedder=TextEmbedder(use_neural=False))
        self.retriever = LiteratureRetriever(self.vector_store, top_k=4)

        # 5. Investigator & Calibrator
        self.hypothesis_generator = HypothesisGenerator(provider="mock")
        self.calibrator = ConfidenceCalibrator(method="isotonic")

        # 6. Active Observation Recommender
        self.recommender = ActiveObservationRecommender()

        self.is_initialized = False

    def initialize_system(self, papers_dir: str = "knowledge_base/papers",
                          catalogs_dir: str = "knowledge_base/catalogs") -> "ACEIPipeline":
        """Bootstrap the knowledge base and train baseline models on synthetic benchmark data."""
        print("[ACEI] Initializing Knowledge Base...")
        loader = DocumentLoader(papers_dir=papers_dir, catalogs_dir=catalogs_dir)
        docs = loader.load_all()
        chunker = DocumentChunker()
        chunks = chunker.chunk_all(docs)
        self.vector_store.add_chunks(chunks)
        print(f"[ACEI] Knowledge base loaded with {len(chunks)} search chunks across {len(docs)} documents.")

        print("[ACEI] Bootstrapping Multimodal Transient Model & Anomaly Detectors...")
        benchmark_events = generate_benchmark_events(num_known=120, num_anomalies=30)
        splitter = ObjectLevelSplitter(self.known_classes, ["LRN", "SLSN", "TDE"])
        splits = splitter.split_events(benchmark_events)

        train_events = [e for e in benchmark_events if e.object_id in splits.train_ids]
        val_events = [e for e in benchmark_events if e.object_id in splits.val_ids]

        # Train Multimodal Model
        self.model.fit(train_events, self.class_to_idx, epochs=5, batch_size=32, device=self.device)

        # Extract embeddings and logits for training anomaly detectors (strictly on training set)
        self.model.eval()
        train_dataset = AstronomicalDataset(train_events, self.class_to_idx)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=len(train_dataset))
        train_batch = next(iter(train_loader))

        with torch.no_grad():
            train_logits, train_embeds = self.model(
                train_batch["image"].to(self.device),
                train_batch["lightcurve"].to(self.device),
                train_batch["mask"].to(self.device)
            )

        np_train_embeds = train_embeds.cpu().numpy()
        np_train_labels = train_batch["label"].cpu().numpy()

        # Fit Autoencoder strictly on normal training embeddings
        self.autoencoder.fit(np_train_embeds, epochs=25, device=self.device)

        # Fit Mahalanobis on normal training centroids
        self.mahalanobis.fit(np_train_embeds, np_train_labels)

        # Extract embeddings and logits strictly on held-out validation set for anomaly calibration
        val_dataset = AstronomicalDataset(val_events, self.class_to_idx)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=len(val_dataset))
        val_batch = next(iter(val_loader))

        with torch.no_grad():
            val_logits, val_embeds = self.model(
                val_batch["image"].to(self.device),
                val_batch["lightcurve"].to(self.device),
                val_batch["mask"].to(self.device)
            )

        np_val_embeds = val_embeds.cpu().numpy()
        np_val_logits = val_logits.cpu().numpy()

        # Calibrate Anomaly Ensemble strictly on held-out validation distributions (prevents data leakage)
        self.anomaly_ensemble.calibrate(np_val_embeds, np_val_logits)

        # Fit Calibrator
        dummy_conf = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3])
        dummy_corr = np.array([1, 1, 1, 0, 1, 0, 0])
        self.calibrator.fit(dummy_conf, dummy_corr)

        self.is_initialized = True
        print("[ACEI] System initialization complete. Ready for closed-loop triage.")
        return self

    def investigate_event(self, event: AstronomicalEvent) -> InvestigationResult:
        """Execute the full closed-loop triage pipeline on an event."""
        start_time = time.time()
        if not self.is_initialized:
            self.initialize_system()

        # 1. Forward pass through Multimodal Neural Model
        dummy_dataset = AstronomicalDataset([event], self.class_to_idx)
        batch = dummy_dataset[0]

        img_t = batch["image"].unsqueeze(0).to(self.device)
        lc_t = batch["lightcurve"].unsqueeze(0).to(self.device)
        mask_t = batch["mask"].unsqueeze(0).to(self.device)

        self.model.eval()
        with torch.no_grad():
            logits, embedding = self.model(img_t, lc_t, mask_t)

        np_logits = logits.cpu().numpy()[0]
        np_embed = embedding.cpu().numpy()[0]

        # Softmax probabilities over known classes
        exp_l = np.exp(np_logits - np.max(np_logits))
        probs = exp_l / np.sum(exp_l)
        class_probs = {self.idx_to_class[i]: float(probs[i]) for i in range(len(self.known_classes))}
        best_class_idx = int(np.argmax(probs))
        predicted_class = self.idx_to_class[best_class_idx]
        classifier_conf = float(probs[best_class_idx])

        # 2. Compute 3-Signal Anomaly Ensemble Score
        anomaly_score, is_flagged, detector_details = self.anomaly_ensemble.score_event(np_embed, np_logits)

        hypotheses: List[Hypothesis] = []
        recommended_action: Optional[RecommendedObservation] = None

        # 3. Closed-Loop Scientific Investigation (if flagged anomalous)
        if is_flagged:
            # Extract combined photometric & image features
            lc_features = self.lc_extractor.extract_features(event.lightcurve)
            img_features = self.img_extractor.extract_features(event.image.data) if event.image else {}
            all_features = {**lc_features, **img_features}

            # Retrieve evidence from astrophysical literature
            retrieved_chunks = self.retriever.retrieve(all_features, top_k=4)

            # Generate evidence-grounded hypotheses
            raw_hypotheses = self.hypothesis_generator.generate(
                event_id=event.object_id,
                features=all_features,
                retrieved_chunks=retrieved_chunks,
                anomaly_score=anomaly_score
            )

            # Calibrate confidence scores
            calibrated_hyps = []
            for h in raw_hypotheses:
                cal_p = float(self.calibrator.calibrate(np.array([h.raw_confidence]))[0])
                calibrated_hyps.append(Hypothesis(
                    rank=h.rank,
                    name=h.name,
                    probability=cal_p,
                    raw_confidence=h.raw_confidence,
                    evidence_sources=h.evidence_sources,
                    justification=h.justification,
                    distinguishing_criteria=h.distinguishing_criteria
                ))

            # Re-rank and normalize to probability simplex
            hypotheses = HypothesisRanker.rank_and_normalize(calibrated_hyps)

            # Active observation recommendation
            recommended_action = self.recommender.recommend_best_action(event, hypotheses)

        elapsed = time.time() - start_time

        return InvestigationResult(
            event_id=event.object_id,
            is_anomaly=is_flagged,
            anomaly_score=anomaly_score,
            detector_scores=detector_details,
            classifier_prediction=predicted_class,
            classifier_confidence=classifier_conf,
            class_probabilities=class_probs,
            hypotheses=hypotheses,
            recommended_action=recommended_action,
            execution_time_seconds=elapsed
        )


def run_pipeline_demo():
    """Run interactive CLI demo illustrating normal classification vs anomalous scientific triage."""
    torch.manual_seed(42)
    np.random.seed(42)

    print("=" * 80)
    print("      AI COSMIC EVENT INVESTIGATOR (ACEI) — CLOSED-LOOP SCIENTIFIC DEMO")
    print("=" * 80)

    pipeline = ACEIPipeline(anomaly_threshold=0.65)
    pipeline.initialize_system()

    def format_conclusion(res: InvestigationResult, true_label: Optional[str]) -> str:
        if res.is_anomaly:
            return "Conclusion: Astronomical anomaly detected. Proceeding to evidence-grounded investigation and follow-up recommendation."
        elif true_label is not None and res.classifier_prediction == true_label:
            return "Conclusion: Consistent with the known transient class. No urgent follow-up triage required."
        else:
            return "Conclusion: No anomaly detected, but transient classification is uncertain/mismatched. No urgent anomaly triage triggered."

    # Case 1: Known normal event (SN Ia)
    print("\n" + "#" * 80)
    print(">>> CASE 1: Standard In-Distribution Transient (Type Ia Supernova)")
    print("#" * 80)
    from src.data.lightcurve_loader import LightCurveLoader
    from src.data.image_loader import ImageLoader
    from src.data.schema import CutoutImage

    normal_event = AstronomicalEvent(
        object_id="ZTF23ab_normal_sn_ia",
        ra=184.2345, dec=29.8765,
        lightcurve=LightCurveLoader.generate_synthetic_lightcurve("SN_Ia", is_anomaly=False),
        image=CutoutImage(data=ImageLoader.generate_synthetic_cutout(is_anomaly=False, event_type="SN_Ia")),
        true_label="SN_Ia",
        is_anomaly=False
    )
    res_normal = pipeline.investigate_event(normal_event)
    print(f"Object ID:              {res_normal.event_id}")
    print(f"Anomaly Score:          {res_normal.anomaly_score:.4f}  [Flagged Anomaly: {res_normal.is_anomaly}]")
    print(f"Classifier Prediction:  {res_normal.classifier_prediction} (Confidence: {res_normal.classifier_confidence*100:.1f}%)")
    print(format_conclusion(res_normal, normal_event.true_label))

    # Case 2: Held-out Rare Anomaly (Luminous Red Nova)
    print("\n" + "#" * 80)
    print(">>> CASE 2: Held-Out Rare Anomaly (Luminous Red Nova — Binary Stellar Merger)")
    print("#" * 80)
    anom_event = AstronomicalEvent(
        object_id="ZTF23xyz_exotic_merger",
        ra=210.1234, dec=-12.3456,
        lightcurve=LightCurveLoader.generate_synthetic_lightcurve("LRN", is_anomaly=True),
        image=CutoutImage(data=ImageLoader.generate_synthetic_cutout(is_anomaly=True, event_type="LRN")),
        true_label="LRN",
        is_anomaly=True
    )
    res_anom = pipeline.investigate_event(anom_event)
    print(f"Object ID:              {res_anom.event_id}")
    print(
    f"Ensemble Anomaly Score: {res_anom.anomaly_score:.4f}  "
    f"[Flagged Anomaly: {res_anom.is_anomaly}]")
    print(f"Autoencoder Recon Err:  {res_anom.detector_scores.get('autoencoder_norm', 0.0):.4f}")
    print(f"Mahalanobis Distance:   {res_anom.detector_scores.get('mahalanobis_norm', 0.0):.4f}")
    print(f"Energy OOD Score:       {res_anom.detector_scores.get('energy_norm', 0.0):.4f}")
    print(format_conclusion(res_anom, anom_event.true_label))

    print("\n--- AI INVESTIGATOR: EVIDENCE-GROUNDED HYPOTHESES (RAG) ---")
    for h in res_anom.hypotheses:
        print(f"Rank {h.rank}: {h.name} (Calibrated P = {h.probability*100:.1f}%)")
        print(f"   Supporting Evidence: {', '.join(h.evidence_sources)}")
        print(f"   Physical Rationale:  {h.justification}")
        print(f"   Distinguishing Test: {h.distinguishing_criteria}\n")

    if res_anom.recommended_action:
        rec = res_anom.recommended_action
        print("--- ACTIVE OBSERVATION RECOMMENDER (BAYESIAN EXPERIMENTAL DESIGN) ---")
        print(f"Recommended Next Action:  ⭐ {rec.name}")
        print(f"Filter Band:              {rec.band}")
        print(f"Telescope Cost:           {rec.cost} hours")
        print(f"Expected Information Gain:{rec.expected_information_gain:.4f} nats (Shannon entropy reduction)")
        print(f"Scientific Justification: {rec.rationale}")

    print("\n" + "=" * 80)
    print(f"Execution completed in {res_anom.execution_time_seconds:.2f} seconds.")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true", help="Run end-to-end pipeline demonstration")
    args = parser.parse_args()
    if args.demo:
        run_pipeline_demo()
    else:
        run_pipeline_demo()
