"""Astronomical image encoder using a ResNet architecture with projection head."""

import torch
import torch.nn as nn
import torchvision.models as models


class ImageEncoder(nn.Module):
    """Encodes astronomical image cutouts (science, template, difference) into vector embeddings."""

    def __init__(self, embedding_dim: int = 128, in_channels: int = 3,
                 pretrained: bool = False, dropout: float = 0.2):
        super().__init__()
        self.embedding_dim = embedding_dim

        # Use ResNet-18 backbone
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        resnet = models.resnet18(weights=weights)

        # Modify first layer if in_channels is different from 3
        if in_channels != 3:
            resnet.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Remove default classification head (fc)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])  # Output shape: (B, 512, 1, 1)

        # Projection head to latent embedding space
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, embedding_dim),
            nn.LayerNorm(embedding_dim)
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        images: (B, C, H, W) e.g., (B, 3, 64, 64)
        Returns: (B, embedding_dim)
        """
        features = self.backbone(images)
        embeddings = self.projection(features)
        return embeddings
