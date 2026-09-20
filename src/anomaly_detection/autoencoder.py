"""Multimodal Latent Autoencoder for reconstruction-based anomaly detection."""

import os
from typing import List, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class MultimodalAutoencoder(nn.Module):
    """
    Autoencoder trained exclusively on embeddings of known/normal astronomical transients.
    High reconstruction error indicates out-of-distribution / novel physical behavior.
    """

    def __init__(self, input_dim: int = 256, latent_dim: int = 64):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # Encoder: compresses multimodal manifold to low-dimensional bottleneck
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Linear(128, latent_dim)
        )

        # Decoder: reconstructs the normal multimodal embedding
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Linear(128, input_dim)
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Forward pass reconstructing embedding z."""
        latent = self.encoder(z)
        reconstruction = self.decoder(latent)
        return reconstruction

    def compute_reconstruction_error(self, z: torch.Tensor) -> np.ndarray:
        """
        Compute per-sample Mean Squared Error ||z - z_hat||^2.
        Returns: 1D numpy array of reconstruction errors.
        """
        self.eval()
        with torch.no_grad():
            reconstruction = self(z)
            errors = torch.mean((z - reconstruction)**2, dim=-1).cpu().numpy()
        return errors

    def fit(self, normal_embeddings: np.ndarray,
            epochs: int = 25,
            batch_size: int = 32,
            lr: float = 0.001,
            device: str = "cpu") -> "MultimodalAutoencoder":
        """Train autoencoder strictly on in-distribution known classes."""
        self.to(device)
        self.train()

        tensor_data = torch.from_numpy(normal_embeddings).float()
        dataset = torch.utils.data.TensorDataset(tensor_data)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=1e-5)
        criterion = nn.MSELoss()

        for epoch in range(epochs):
            for (batch_z,) in loader:
                batch_z = batch_z.to(device)
                optimizer.zero_grad()
                recon = self(batch_z)
                loss = criterion(recon, batch_z)
                loss.backward()
                optimizer.step()

        return self
