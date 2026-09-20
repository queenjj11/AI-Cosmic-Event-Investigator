"""Transformer-based light curve encoder with continuous Time2Vec positional encodings."""

import torch
import torch.nn as nn
import numpy as np


class Time2Vec(nn.Module):
    """
    Time2Vec: Learning a representation of continuous, irregular time.
    tau[0] = omega_0 * t + phi_0 (linear component)
    tau[i] = sin(omega_i * t + phi_i) for i in 1..k (periodic components)
    """
    def __init__(self, output_dim: int = 32):
        super().__init__()
        self.output_dim = output_dim
        self.w0 = nn.Parameter(torch.randn(1, 1))
        self.p0 = nn.Parameter(torch.randn(1, 1))
        self.w = nn.Parameter(torch.randn(1, output_dim - 1))
        self.p = nn.Parameter(torch.randn(1, output_dim - 1))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        t: (B, seq_len, 1) continuous time steps
        Returns: (B, seq_len, output_dim)
        """
        linear = torch.matmul(t, self.w0) + self.p0
        periodic = torch.sin(torch.matmul(t, self.w) + self.p)
        return torch.cat([linear, periodic], dim=-1)


class LightCurveEncoder(nn.Module):
    """Transformer encoder for irregularly-sampled multi-band astronomical light curves."""

    def __init__(self,
                 embedding_dim: int = 128,
                 num_bands: int = 5,
                 d_model: int = 128,
                 nhead: int = 4,
                 num_layers: int = 3,
                 dim_feedforward: int = 256,
                 dropout: float = 0.1,
                 time_embed_dim: int = 32):
        super().__init__()
        self.d_model = d_model
        self.embedding_dim = embedding_dim

        # Encoders for input modalities
        self.time2vec = Time2Vec(output_dim=time_embed_dim)
        self.band_embedding = nn.Embedding(num_bands, 16)
        # Numerical input: [flux, flux_err] (dim = 2)
        self.photometry_proj = nn.Linear(2, 32)

        # Combined input projection to d_model
        total_in_dim = time_embed_dim + 16 + 32
        self.input_projection = nn.Sequential(
            nn.Linear(total_in_dim, d_model),
            nn.LayerNorm(d_model)
        )

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output projection head
        self.output_head = nn.Sequential(
            nn.Linear(d_model, embedding_dim),
            nn.LayerNorm(embedding_dim)
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        x: (B, seq_len, 4) where cols are [time, flux, flux_err, band_idx]
        mask: (B, seq_len) boolean mask where True = valid observation, False = padded
        Returns: (B, embedding_dim)
        """
        time = x[:, :, 0:1]         # (B, seq_len, 1)
        photometry = x[:, :, 1:3]   # (B, seq_len, 2)
        band_idx = torch.clamp(x[:, :, 3].long(), 0, self.band_embedding.num_embeddings - 1)

        # Feature representations
        t_embed = self.time2vec(time)
        b_embed = self.band_embedding(band_idx)
        p_embed = self.photometry_proj(photometry)

        combined = torch.cat([t_embed, b_embed, p_embed], dim=-1)
        seq_features = self.input_projection(combined)  # (B, seq_len, d_model)

        # PyTorch Transformer expects src_key_padding_mask where True = IGNORE (padded)
        padding_mask = ~mask

        encoded = self.transformer_encoder(seq_features, src_key_padding_mask=padding_mask)

        # Masked average pooling across valid time points
        mask_expanded = mask.unsqueeze(-1).float()
        sum_masked = torch.sum(encoded * mask_expanded, dim=1)
        count_valid = torch.clamp(torch.sum(mask_expanded, dim=1), min=1.0)
        pooled = sum_masked / count_valid  # (B, d_model)

        embedding = self.output_head(pooled)
        return embedding
