"""Object-level dataset splitting to prevent observation-level data leakage."""

import os
import json
import random
from typing import Any, Dict, List, Tuple
from dataclasses import dataclass

from src.data.schema import AstronomicalEvent


@dataclass
class DatasetSplits:
    """Container holding object IDs partitioned by split role."""
    train_ids: List[str]
    val_ids: List[str]
    test_known_ids: List[str]
    test_anomaly_ids: List[str]

    def to_dict(self) -> Dict[str, List[str]]:
        return {
            "train": self.train_ids,
            "val": self.val_ids,
            "test_known": self.test_known_ids,
            "test_anomalies": self.test_anomaly_ids
        }


class ObjectLevelSplitter:
    """Partitions events by object ID into known-class splits and held-out anomaly classes."""

    def __init__(self,
                 known_classes: List[str],
                 held_out_classes: List[str],
                 train_ratio: float = 0.70,
                 val_ratio: float = 0.15,
                 test_ratio: float = 0.15,
                 seed: int = 42):
        self.known_classes = sorted(list(set(known_classes)))
        self.held_out_classes = sorted(list(set(held_out_classes)))
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed

    def split_events(self, events: List[AstronomicalEvent]) -> DatasetSplits:
        """Split a collection of events cleanly without observation or object leakage."""
        rng = random.Random(self.seed)

        known_events_by_class: Dict[str, List[str]] = {cls: [] for cls in self.known_classes}
        anomaly_ids: List[str] = []

        for e in events:
            label = e.true_label or "Unknown"
            if label in self.held_out_classes or e.is_anomaly:
                anomaly_ids.append(e.object_id)
            elif label in self.known_classes:
                known_events_by_class[label].append(e.object_id)
            else:
                # Default non-classified or novel objects go to anomaly evaluation set
                anomaly_ids.append(e.object_id)

        train_ids: List[str] = []
        val_ids: List[str] = []
        test_known_ids: List[str] = []

        # Stratified object split across known classes
        for cls, obj_ids in known_events_by_class.items():
            shuffled = list(obj_ids)
            rng.shuffle(shuffled)
            n = len(shuffled)
            n_train = int(n * self.train_ratio)
            n_val = int(n * self.val_ratio)

            train_ids.extend(shuffled[:n_train])
            val_ids.extend(shuffled[n_train:n_train + n_val])
            test_known_ids.extend(shuffled[n_train + n_val:])

        rng.shuffle(train_ids)
        rng.shuffle(val_ids)
        rng.shuffle(test_known_ids)
        rng.shuffle(anomaly_ids)

        return DatasetSplits(
            train_ids=train_ids,
            val_ids=val_ids,
            test_known_ids=test_known_ids,
            test_anomaly_ids=anomaly_ids
        )

    def save_splits(self, splits: DatasetSplits, output_filepath: str) -> None:
        """Save split IDs to disk for reproducible experimentation."""
        os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
        with open(output_filepath, "w") as f:
            json.dump(splits.to_dict(), f, indent=2)

    @staticmethod
    def load_splits(filepath: str) -> DatasetSplits:
        """Load saved split object IDs from JSON."""
        with open(filepath, "r") as f:
            data = json.load(f)
        return DatasetSplits(
            train_ids=data.get("train", []),
            val_ids=data.get("val", []),
            test_known_ids=data.get("test_known", []),
            test_anomaly_ids=data.get("test_anomalies", [])
        )
