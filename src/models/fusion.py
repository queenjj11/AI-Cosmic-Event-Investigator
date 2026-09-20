"""Multimodal fusion layers: Cross-Attention and Concat-MLP."""

import torch
import torch.nn as nn


class ConcatMLPFusion(nn.Module):
    """Simple baseline fusion: Concatenate embeddings and project through MLP."""

    def __init__(self, img_dim: int = 128, lc_dim: int = 128, fused_dim: int = 256, dropout: float = 0.2):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(img_dim + lc_dim, fused_dim),
            nn.BatchNorm1d(fused_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fused_dim, fused_dim),
            nn.LayerNorm(fused_dim)
        )

    def forward(self, z_img: torch.Tensor, z_lc: torch.Tensor) -> torch.Tensor:
        """
        z_img: (B, img_dim)
        z_lc:  (B, lc_dim)
        Returns: (B, fused_dim)
        """
        concatenated = torch.cat([z_img, z_lc], dim=-1)
        return self.mlp(concatenated)


class CrossAttentionFusion(nn.Module):
    """
    Bidirectional cross-attention fusion:
    Allows visual morphology features to query photometric variability features and vice versa.
    """

    def __init__(self, img_dim: int = 128, lc_dim: int = 128, fused_dim: int = 256,
                 num_heads: int = 4, dropout: float = 0.2):
        super().__init__()
        self.fused_dim = fused_dim

        # Project modalities to a shared latent attention dimension
        self.img_proj = nn.Linear(img_dim, fused_dim)
        self.lc_proj = nn.Linear(lc_dim, fused_dim)

        # Multihead Attention layers
        self.cross_attn_img_to_lc = nn.MultiheadAttention(
            embed_dim=fused_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.cross_attn_lc_to_img = nn.MultiheadAttention(
            embed_dim=fused_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )

        self.norm_img = nn.LayerNorm(fused_dim)
        self.norm_lc = nn.LayerNorm(fused_dim)

        # Fusion aggregation gate
        self.gate = nn.Sequential(
            nn.Linear(fused_dim * 2, fused_dim),
            nn.Sigmoid()
        )
        self.final_proj = nn.Sequential(
            nn.Linear(fused_dim, fused_dim),
            nn.LayerNorm(fused_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

    def forward(self, z_img: torch.Tensor, z_lc: torch.Tensor) -> torch.Tensor:
        """
        z_img: (B, img_dim)
        z_lc:  (B, lc_dim)
        Returns: (B, fused_dim)
        """
        # Expand dimensions to (B, 1, fused_dim) for attention
        h_img = self.img_proj(z_img).unsqueeze(1)
        h_lc = self.lc_proj(z_lc).unsqueeze(1)

        # Image attends to Light Curve (Q=img, K=lc, V=lc)
        attn_img, _ = self.cross_attn_img_to_lc(query=h_img, key=h_lc, value=h_lc)
        h_img = self.norm_img(h_img + attn_img).squeeze(1)  # (B, fused_dim)

        # Light Curve attends to Image (Q=lc, K=img, V=img)
        attn_lc, _ = self.cross_attn_lc_to_img(query=h_lc, key=h_img.unsqueeze(1), value=h_img.unsqueeze(1))
        h_lc = self.norm_lc(h_lc + attn_lc).squeeze(1)      # (B, fused_dim)

        # Gated combination
        concat = torch.cat([h_img, h_lc], dim=-1)
        g = self.gate(concat)
        fused = g * h_img + (1.0 - g) * h_lc

        return self.final_proj(fused)
