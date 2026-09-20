"""Dataset builder, PyTorch Dataset wrapper, and benchmark data generator."""

import os
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from src.data.schema import AstronomicalEvent, CutoutImage, LightCurve
from src.data.image_loader import ImageLoader
from src.data.lightcurve_loader import LightCurveLoader
from src.data.preprocessing import Preprocessor


class AstronomicalDataset(Dataset):
    """PyTorch Dataset yielding paired image cutouts, light curves, masks, and labels."""

    def __init__(self, events: List[AstronomicalEvent],
                 class_to_idx: Dict[str, int],
                 max_seq_len: int = 50,
                 image_size: Tuple[int, int] = (64, 64)):
        self.events = events
        self.class_to_idx = class_to_idx
        self.preprocessor = Preprocessor(max_length=max_seq_len)
        self.image_loader = ImageLoader(target_size=image_size)

    def __len__(self) -> int:
        return len(self.events)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        event = self.events[idx]

        # 1. Process Light Curve
        lc_feats, lc_mask = self.preprocessor.lightcurve_to_padded_tensor(event.lightcurve)
        lc_tensor = torch.from_numpy(lc_feats).float()
        mask_tensor = torch.from_numpy(lc_mask).bool()

        # 2. Process Image Cutout
        if event.image is not None and isinstance(event.image.data, np.ndarray):
            img_data = self.image_loader.preprocess_image(event.image.data)
        else:
            # Generate or default based on event type
            img_data = ImageLoader.generate_synthetic_cutout(
                event_type=event.true_label or "SN_Ia"
            )
        img_tensor = torch.from_numpy(img_data).float()

        # 3. Label
        label_str = event.true_label or "Unknown"
        label_idx = self.class_to_idx.get(label_str, -1)
        label_tensor = torch.tensor(label_idx, dtype=torch.long)
        is_anomaly_tensor = torch.tensor(1 if event.is_anomaly else 0, dtype=torch.float)

        return {
            "image": img_tensor,             # Shape: (3, H, W)
            "lightcurve": lc_tensor,         # Shape: (seq_len, 4)
            "mask": mask_tensor,             # Shape: (seq_len,)
            "label": label_tensor,           # Scalar int
            "is_anomaly": is_anomaly_tensor, # Scalar float
            "object_id": event.object_id
        }


def create_dataloader(events: List[AstronomicalEvent],
                      class_to_idx: Dict[str, int],
                      batch_size: int = 32,
                      shuffle: bool = True,
                      max_seq_len: int = 50) -> DataLoader:
    """Create a standard DataLoader for astronomical events."""
    dataset = AstronomicalDataset(events, class_to_idx, max_seq_len=max_seq_len)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def generate_benchmark_events(num_known: int = 160,
                              num_anomalies: int = 40,
                              known_classes: Optional[List[str]] = None,
                              anomaly_classes: Optional[List[str]] = None) -> List[AstronomicalEvent]:
    """
    Generate a diverse, benchmark collection of astronomical events with realistic
    multi-band light curves and 3-channel cutouts for development and testing.
    """
    known_classes = known_classes or ["SN_Ia", "SN_II", "Stellar_Flare", "Variable_Star"]
    anomaly_classes = anomaly_classes or ["LRN", "SLSN", "TDE"]

    events: List[AstronomicalEvent] = []

    # Known class events
    for i in range(num_known):
        cls_name = known_classes[i % len(known_classes)]
        obj_id = f"ZTF_{cls_name}_{i:04d}"
        lc = LightCurveLoader.generate_synthetic_lightcurve(event_type=cls_name, is_anomaly=False)
        img_arr = ImageLoader.generate_synthetic_cutout(is_anomaly=False, event_type=cls_name)

        event = AstronomicalEvent(
            object_id=obj_id,
            ra=float(np.random.uniform(0.0, 360.0)),
            dec=float(np.random.uniform(-30.0, 80.0)),
            lightcurve=lc,
            image=CutoutImage(data=img_arr),
            true_label=cls_name,
            is_anomaly=False,
            metadata={"peak_mag": float(np.random.uniform(14.5, 19.5))}
        )
        events.append(event)

    # Held-out anomaly events
    for j in range(num_anomalies):
        anomaly_cls = anomaly_classes[j % len(anomaly_classes)]
        obj_id = f"ZTF_ANOMALY_{anomaly_cls}_{j:04d}"
        lc = LightCurveLoader.generate_synthetic_lightcurve(event_type=anomaly_cls, is_anomaly=True)
        img_arr = ImageLoader.generate_synthetic_cutout(is_anomaly=True, event_type=anomaly_cls)

        event = AstronomicalEvent(
            object_id=obj_id,
            ra=float(np.random.uniform(0.0, 360.0)),
            dec=float(np.random.uniform(-30.0, 80.0)),
            lightcurve=lc,
            image=CutoutImage(data=img_arr),
            true_label=anomaly_cls,
            is_anomaly=True,
            metadata={"peak_mag": float(np.random.uniform(13.0, 18.0))}
        )
        events.append(event)

    return events
