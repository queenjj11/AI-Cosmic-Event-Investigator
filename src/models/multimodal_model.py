"""End-to-end Multimodal Transient Model combining vision, light curve encoders, and fusion."""

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.models.image_encoder import ImageEncoder
from src.models.lightcurve_encoder import LightCurveEncoder
from src.models.fusion import CrossAttentionFusion, ConcatMLPFusion
from src.data.schema import AstronomicalEvent
from src.data.dataset_builder import AstronomicalDataset


class MultimodalTransientModel(nn.Module):
    """
    Multimodal neural architecture that encodes both image cutouts and light curves,
    fuses them, and outputs both joint latent representations and class logits.
    """

    def __init__(self,
                 num_classes: int = 4,
                 img_dim: int = 128,
                 lc_dim: int = 128,
                 fused_dim: int = 256,
                 fusion_method: str = "cross_attention",
                 dropout: float = 0.2):
        super().__init__()
        self.num_classes = num_classes
        self.fusion_method = fusion_method

        # Modality encoders
        self.image_encoder = ImageEncoder(embedding_dim=img_dim, dropout=dropout)
        self.lc_encoder = LightCurveEncoder(embedding_dim=lc_dim, dropout=dropout)

        # Fusion module
        if fusion_method == "cross_attention":
            self.fusion = CrossAttentionFusion(img_dim=img_dim, lc_dim=lc_dim, fused_dim=fused_dim, dropout=dropout)
        else:
            self.fusion = ConcatMLPFusion(img_dim=img_dim, lc_dim=lc_dim, fused_dim=fused_dim, dropout=dropout)

        # Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, images: torch.Tensor,
                lightcurves: torch.Tensor,
                masks: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        images: (B, C, H, W)
        lightcurves: (B, seq_len, 4)
        masks: (B, seq_len)
        Returns:
        - logits: (B, num_classes)
        - fused_embedding: (B, fused_dim)
        """
        z_img = self.image_encoder(images)
        z_lc = self.lc_encoder(lightcurves, masks)
        z_fused = self.fusion(z_img, z_lc)
        logits = self.classifier(z_fused)
        return logits, z_fused

    @torch.no_grad()
    def get_embeddings(self, images: torch.Tensor,
                       lightcurves: torch.Tensor,
                       masks: torch.Tensor) -> torch.Tensor:
        """Extract joint multimodal latent representations."""
        self.eval()
        _, z_fused = self.forward(images, lightcurves, masks)
        return z_fused

    def fit(self, train_events: List[AstronomicalEvent],
            class_to_idx: Dict[str, int],
            val_events: Optional[List[AstronomicalEvent]] = None,
            epochs: int = 20,
            batch_size: int = 32,
            lr: float = 0.0005,
            device: str = "cpu") -> "MultimodalTransientModel":
        """Train the multimodal classifier end-to-end on known-class events."""
        self.to(device)
        train_dataset = AstronomicalDataset(train_events, class_to_idx)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        optimizer = optim.AdamW(self.parameters(), lr=lr, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(epochs):
            self.train()
            total_loss = 0.0
            for batch in train_loader:
                images = batch["image"].to(device)
                lcs = batch["lightcurve"].to(device)
                masks = batch["mask"].to(device)
                labels = batch["label"].to(device)

                valid = labels >= 0
                if not torch.any(valid):
                    continue

                optimizer.zero_grad()
                logits, _ = self(images[valid], lcs[valid], masks[valid])
                loss = criterion(logits, labels[valid])
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

        return self
