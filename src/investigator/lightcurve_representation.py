"""Light-curve representation service utilizing the frozen production encoder."""

import os
import csv
from typing import Any, Dict, List, Optional
import numpy as np
import torch

from src.models.checkpoint_manager import load_production_lightcurve_encoder, DEFAULT_CHECKPOINT_PATH
from src.data.real_ztf_preprocessing import RealZTFPreprocessor, PreprocessedZTFEvent
from src.investigator.schemas import RepresentationSummary


class LightCurveRepresentationService:
    """
    Extracts 128-D light-curve representations from raw or preprocessed ZTF photometry.
    Reuses the frozen production checkpoint encoder in strictly inference mode (torch.no_grad).
    """

    def __init__(self,
                 checkpoint_path: str = DEFAULT_CHECKPOINT_PATH,
                 device: str = "cpu",
                 preprocessor: Optional[RealZTFPreprocessor] = None):
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.encoder, self.payload = load_production_lightcurve_encoder(
            checkpoint_path=checkpoint_path,
            device=device
        )
        self.encoder.eval()
        # Freeze parameters explicitly
        for p in self.encoder.parameters():
            p.requires_grad = False

        self.preprocessor = preprocessor or RealZTFPreprocessor(max_sequence_length=50)

    def represent_preprocessed_event(self, pre: PreprocessedZTFEvent) -> RepresentationSummary:
        """Extract embedding from an already preprocessed ZTF event."""
        with torch.no_grad():
            x = pre.feature_tensor.unsqueeze(0).to(self.device)  # (1, 50, 4)
            m = pre.mask_tensor.unsqueeze(0).to(self.device)     # (1, 50)
            emb = self.encoder(x, mask=m)                       # (1, 128)
            emb_vec = emb.squeeze(0).cpu().numpy().astype(float)

        emb_norm = float(np.linalg.norm(emb_vec))
        if not np.all(np.isfinite(emb_vec)):
            raise ValueError(f"Non-finite values detected in embedding for {pre.object_id}")

        valid_tokens = pre.valid_token_count
        padding_fraction = 1.0 - (valid_tokens / 50.0)

        # Filters with at least 1 observation
        available_filters = [b for b, count in pre.band_counts.items() if count > 0]

        # Time span calculation
        time_span = 0.0
        if pre.window_report.window_end_mjd > pre.window_report.window_start_mjd:
            time_span = float(pre.window_report.window_end_mjd - pre.window_report.window_start_mjd)
        elif valid_tokens > 1:
            times = x[0, :valid_tokens, 0].cpu().numpy()
            time_span = float(np.max(times) - np.min(times))

        status = "SUCCESS" if valid_tokens >= 5 else "INSUFFICIENT_OBSERVATIONS"

        return RepresentationSummary(
            object_id=pre.object_id,
            embedding_dimension=int(emb_vec.shape[0]),
            embedding_norm=emb_norm,
            embedding=emb_vec.tolist(),
            valid_token_count=valid_tokens,
            padding_fraction=padding_fraction,
            available_filters=available_filters,
            time_span_days=time_span,
            preprocessing_status=status,
            warnings=list(pre.warnings),
            encoder_model="Production LightCurveEncoder (128-D Transformer)",
            checkpoint_source=self.checkpoint_path
        )

    def represent_records(self, records: List[Dict[str, Any]], object_id: str) -> RepresentationSummary:
        """Preprocess raw observation records and extract representation."""
        pre = self.preprocessor.process_records(records=records, object_id=object_id)
        return self.represent_preprocessed_event(pre)

    def represent_ztf_object(self, object_id: str, raw_csv_path: Optional[str] = None) -> RepresentationSummary:
        """Load raw IRSA CSV from candidate directory and extract representation."""
        path = raw_csv_path
        if not path:
            # Check default candidate directory
            default_candidate_dir = f"data/real_ztf_benchmark/raw/{object_id}/raw_irsa.csv"
            if os.path.exists(default_candidate_dir):
                path = default_candidate_dir
            else:
                pilot_candidate_dir = f"data/real_ztf_pilot/raw/{object_id}/raw_irsa.csv"
                if os.path.exists(pilot_candidate_dir):
                    path = pilot_candidate_dir
                else:
                    raise FileNotFoundError(f"Could not locate raw photometry CSV for object {object_id}")

        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            records = list(reader)

        return self.represent_records(records=records, object_id=object_id)
