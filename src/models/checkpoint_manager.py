"""Checkpoint manager for ACEI production models and encoders.

Handles deterministic serialization and reconstruction of the production
MultimodalTransientModel and LightCurveEncoder checkpoints with complete
astronomical and training provenance.

STRICT CONSTRAINTS:
- No architecture changes
- No hyperparameter changes
- Zero real ZTF data used
- 100% bitwise parity upon reload
"""

import os
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch

from src.models.multimodal_model import MultimodalTransientModel
from src.models.lightcurve_encoder import LightCurveEncoder
from src.data.dataset_builder import generate_benchmark_events
from src.data.splitter import ObjectLevelSplitter


DEFAULT_CHECKPOINT_PATH = "models/checkpoints/acei_multimodal_production.pt"

CANONICAL_KNOWN_CLASSES = ["SN_Ia", "SN_II", "Stellar_Flare", "Variable_Star"]
CANONICAL_HELD_OUT_CLASSES = ["LRN", "SLSN", "TDE"]
CANONICAL_CLASS_TO_IDX = {cls: i for i, cls in enumerate(CANONICAL_KNOWN_CLASSES)}
CANONICAL_IDX_TO_CLASS = {i: cls for i, cls in enumerate(CANONICAL_KNOWN_CLASSES)}


def save_production_checkpoint(output_path: str = DEFAULT_CHECKPOINT_PATH,
                               seed: int = 42,
                               device: str = "cpu") -> str:
    """
    Execute the exact deterministic production baseline training run (5 epochs, seed=42)
    and serialize the trained model, isolated LightCurveEncoder weights, and full provenance.

    Parameters:
        output_path: Target filepath for the checkpoint (.pt).
        seed: Random seed for synthetic generation, splitting, and training.
        device: Execution device (default 'cpu').

    Returns:
        Absolute filepath of the saved checkpoint.
    """
    # 1. Enforce strict deterministic seeds
    torch.manual_seed(seed)
    np.random.seed(seed)

    # 2. Generate identical synthetic benchmark events (120 known, 30 anomalies)
    benchmark_events = generate_benchmark_events(
        num_known=120,
        num_anomalies=30,
        known_classes=CANONICAL_KNOWN_CLASSES,
        anomaly_classes=CANONICAL_HELD_OUT_CLASSES
    )

    # 3. Partition events strictly by object ID using ObjectLevelSplitter
    splitter = ObjectLevelSplitter(
        known_classes=CANONICAL_KNOWN_CLASSES,
        held_out_classes=CANONICAL_HELD_OUT_CLASSES,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=seed
    )
    splits = splitter.split_events(benchmark_events)

    train_events = [e for e in benchmark_events if e.object_id in splits.train_ids]
    val_events = [e for e in benchmark_events if e.object_id in splits.val_ids]
    test_known_events = [e for e in benchmark_events if e.object_id in splits.test_known_ids]
    test_anomaly_events = [e for e in benchmark_events if e.object_id in splits.test_anomaly_ids]

    # 4. Instantiate production MultimodalTransientModel
    model = MultimodalTransientModel(
        num_classes=len(CANONICAL_CLASS_TO_IDX),
        img_dim=128,
        lc_dim=128,
        fused_dim=256,
        fusion_method="cross_attention",
        dropout=0.2
    )

    # 5. Execute production training (5 epochs, batch_size=32, lr=0.0005, CrossEntropy)
    model.fit(
        train_events=train_events,
        class_to_idx=CANONICAL_CLASS_TO_IDX,
        val_events=val_events,
        epochs=5,
        batch_size=32,
        lr=0.0005,
        device=device
    )
    model.eval()

    # 6. Extract full state_dict and standalone LightCurveEncoder state_dict
    model_state_dict = model.state_dict()
    lc_state_dict = model.lc_encoder.state_dict()

    # 7. Construct comprehensive checkpoint payload
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    payload: Dict[str, Any] = {
        "model_state_dict": model_state_dict,
        "lc_encoder_state_dict": lc_state_dict,
        "architecture": {
            "model_class": "MultimodalTransientModel",
            "version": "1.0.0-production-baseline",
            "num_classes": 4,
            "img_dim": 128,
            "lc_dim": 128,
            "fused_dim": 256,
            "fusion_method": "cross_attention",
            "dropout": 0.2,
            "lc_encoder_config": {
                "embedding_dim": 128,
                "d_model": 128,
                "nhead": 4,
                "num_layers": 3,
                "dim_feedforward": 256,
                "dropout": 0.2,
                "time_embed_dim": 32,
                "num_bands": 5
            },
            "image_encoder_config": {
                "embedding_dim": 128,
                "in_channels": 3,
                "dropout": 0.2
            }
        },
        "class_to_idx": dict(CANONICAL_CLASS_TO_IDX),
        "idx_to_class": dict(CANONICAL_IDX_TO_CLASS),
        "training_provenance": {
            "seed": seed,
            "epochs": 5,
            "batch_size": 32,
            "learning_rate": 0.0005,
            "optimizer": "AdamW",
            "weight_decay": 1e-4,
            "loss_function": "CrossEntropyLoss",
            "device": device
        },
        "dataset_provenance": {
            "dataset_type": "synthetic_benchmark",
            "num_known_events": 120,
            "num_anomaly_events": 30,
            "known_classes": list(CANONICAL_KNOWN_CLASSES),
            "held_out_classes": list(CANONICAL_HELD_OUT_CLASSES),
            "split_ratios": {"train": 0.70, "val": 0.15, "test": 0.15},
            "split_seed": seed,
            "train_event_count": len(train_events),
            "val_event_count": len(val_events),
            "test_known_count": len(test_known_events),
            "test_anomaly_count": len(test_anomaly_events),
            "train_object_ids": list(splits.train_ids),
            "val_object_ids": list(splits.val_ids),
            "test_known_object_ids": list(splits.test_known_ids),
            "test_anomaly_object_ids": list(splits.test_anomaly_ids),
            "real_ztf_data_used": False
        },
        "metadata": {
            "created_at": timestamp,
            "framework": f"PyTorch {torch.__version__}",
            "description": "Deterministic production baseline weights for ACEI MultimodalTransientModel and LightCurveEncoder",
            "scientific_validity_note": "Checkpoint proves reproducibility and persistence of existing synthetic baseline, not astrophysical validity."
        },
        "real_ztf_data_used": False
    }

    # 8. Save checkpoint file atomically
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(payload, output_path)

    return os.path.abspath(output_path)


def load_production_model(checkpoint_path: str = DEFAULT_CHECKPOINT_PATH,
                          device: str = "cpu") -> Tuple[MultimodalTransientModel, Dict[str, Any]]:
    """
    Instantiate and reload the full production MultimodalTransientModel from checkpoint.

    Parameters:
        checkpoint_path: Path to the .pt checkpoint.
        device: Device to map tensors to ('cpu' or 'cuda').

    Returns:
        Tuple of (model, checkpoint_payload).
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    payload = torch.load(checkpoint_path, map_location=device)
    arch = payload["architecture"]

    model = MultimodalTransientModel(
        num_classes=arch["num_classes"],
        img_dim=arch["img_dim"],
        lc_dim=arch["lc_dim"],
        fused_dim=arch["fused_dim"],
        fusion_method=arch["fusion_method"],
        dropout=arch["dropout"]
    )

    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(device)
    model.eval()

    return model, payload


def load_production_lightcurve_encoder(checkpoint_path: str = DEFAULT_CHECKPOINT_PATH,
                                      device: str = "cpu") -> Tuple[LightCurveEncoder, Dict[str, Any]]:
    """
    Instantiate and reload a standalone LightCurveEncoder using the checkpoint's trained weights.

    Parameters:
        checkpoint_path: Path to the .pt checkpoint.
        device: Device to map tensors to ('cpu' or 'cuda').

    Returns:
        Tuple of (standalone_encoder, checkpoint_payload).
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    payload = torch.load(checkpoint_path, map_location=device)
    lc_conf = payload["architecture"]["lc_encoder_config"]

    encoder = LightCurveEncoder(
        embedding_dim=lc_conf["embedding_dim"],
        num_bands=lc_conf["num_bands"],
        d_model=lc_conf["d_model"],
        nhead=lc_conf["nhead"],
        num_layers=lc_conf["num_layers"],
        dim_feedforward=lc_conf["dim_feedforward"],
        dropout=lc_conf["dropout"],
        time_embed_dim=lc_conf["time_embed_dim"]
    )

    encoder.load_state_dict(payload["lc_encoder_state_dict"], strict=True)
    encoder.to(device)
    encoder.eval()

    return encoder, payload
