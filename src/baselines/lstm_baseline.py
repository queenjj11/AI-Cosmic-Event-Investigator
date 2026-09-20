"""Bi-LSTM baseline classifier operating exclusively on light curve time series."""

import os
from typing import Dict, List, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.data.schema import AstronomicalEvent
from src.data.dataset_builder import AstronomicalDataset


class LSTMLightCurveBaseline(nn.Module):
    """Bidirectional LSTM network for classification from light curve sequences alone."""

    def __init__(self, num_classes: int = 4, input_dim: int = 4,
                 hidden_dim: int = 64, num_layers: int = 2):
        super().__init__()
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim

        # Input features: [time, flux, flux_err, band_idx]
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.2 if num_layers > 1 else 0.0
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        x: (B, seq_len, 4)
        mask: (B, seq_len) boolean mask where True = valid point
        """
        # Run LSTM
        lstm_out, _ = self.lstm(x)  # (B, seq_len, 2 * hidden_dim)

        # Masked average pooling across valid time steps
        mask_expanded = mask.unsqueeze(-1).float()  # (B, seq_len, 1)
        sum_masked = torch.sum(lstm_out * mask_expanded, dim=1)
        count_valid = torch.clamp(torch.sum(mask_expanded, dim=1), min=1.0)
        pooled = sum_masked / count_valid  # (B, 2 * hidden_dim)

        logits = self.classifier(pooled)
        return logits

    def fit(self, train_events: List[AstronomicalEvent],
            class_to_idx: Dict[str, int],
            epochs: int = 15,
            batch_size: int = 32,
            lr: float = 0.001,
            device: str = "cpu") -> "LSTMLightCurveBaseline":
        """Train LSTM baseline on light curves."""
        self.to(device)
        self.train()

        dataset = AstronomicalDataset(train_events, class_to_idx)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(epochs):
            for batch in loader:
                lcs = batch["lightcurve"].to(device)
                masks = batch["mask"].to(device)
                labels = batch["label"].to(device)

                valid = labels >= 0
                if not torch.any(valid):
                    continue

                lcs = lcs[valid]
                masks = masks[valid]
                labels = labels[valid]

                optimizer.zero_grad()
                logits = self(lcs, masks)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

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
            lcs = batch["lightcurve"].to(device)
            masks = batch["mask"].to(device)
            logits = self(lcs, masks)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            all_probs.append(probs)

        return np.concatenate(all_probs, axis=0) if all_probs else np.zeros((0, self.num_classes))
