"""CNN baseline classifier operating exclusively on astronomical image cutouts."""

import os
from typing import Dict, List, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.data.schema import AstronomicalEvent
from src.data.dataset_builder import AstronomicalDataset


class SimpleConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class CNNImageBaseline(nn.Module):
    """Convolutional neural network for classification from image cutouts alone."""

    def __init__(self, num_classes: int = 4, in_channels: int = 3, hidden_dim: int = 128):
        super().__init__()
        self.num_classes = num_classes
        self.features = nn.Sequential(
            SimpleConvBlock(in_channels, 32),    # 64 -> 32
            SimpleConvBlock(32, 64),             # 32 -> 16
            SimpleConvBlock(64, 128),            # 16 -> 8
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass taking (B, C, H, W) images and returning logits (B, num_classes)."""
        feats = self.features(x)
        logits = self.classifier(feats)
        return logits

    def fit(self, train_events: List[AstronomicalEvent],
            class_to_idx: Dict[str, int],
            epochs: int = 15,
            batch_size: int = 32,
            lr: float = 0.001,
            device: str = "cpu") -> "CNNImageBaseline":
        """Train CNN baseline on image cutouts."""
        self.to(device)
        self.train()

        dataset = AstronomicalDataset(train_events, class_to_idx)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(epochs):
            total_loss = 0.0
            for batch in loader:
                images = batch["image"].to(device)
                labels = batch["label"].to(device)
                # Filter out unmapped labels if any
                valid = labels >= 0
                if not torch.any(valid):
                    continue

                images = images[valid]
                labels = labels[valid]

                optimizer.zero_grad()
                logits = self(images)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

        return self

    @torch.no_grad()
    def predict_proba(self, events: List[AstronomicalEvent],
                      class_to_idx: Dict[str, int],
                      device: str = "cpu") -> np.ndarray:
        """Compute softmax class probabilities for events."""
        self.eval()
        self.to(device)
        dataset = AstronomicalDataset(events, class_to_idx)
        loader = DataLoader(dataset, batch_size=32, shuffle=False)

        all_probs = []
        for batch in loader:
            images = batch["image"].to(device)
            logits = self(images)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            all_probs.append(probs)

        return np.concatenate(all_probs, axis=0) if all_probs else np.zeros((0, self.num_classes))
