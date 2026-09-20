"""Unit tests for neural architectures and multimodal encoders."""

import pytest
import torch
import numpy as np

from src.models.image_encoder import ImageEncoder
from src.models.lightcurve_encoder import LightCurveEncoder, Time2Vec
from src.models.fusion import CrossAttentionFusion, ConcatMLPFusion
from src.models.multimodal_model import MultimodalTransientModel


def test_time2vec_shape():
    t2v = Time2Vec(output_dim=32)
    t = torch.linspace(0, 10, 15).unsqueeze(0).unsqueeze(-1)  # (1, 15, 1)
    out = t2v(t)
    assert out.shape == (1, 15, 32)


def test_image_encoder_shape():
    encoder = ImageEncoder(embedding_dim=128, in_channels=3, pretrained=False)
    images = torch.randn(4, 3, 64, 64)
    embeddings = encoder(images)
    assert embeddings.shape == (4, 128)


def test_lightcurve_encoder_shape():
    encoder = LightCurveEncoder(embedding_dim=128, d_model=64, nhead=2, num_layers=2)
    x = torch.randn(4, 20, 4)
    mask = torch.ones(4, 20, dtype=torch.bool)
    embeddings = encoder(x, mask)
    assert embeddings.shape == (4, 128)


def test_multimodal_model_forward():
    model = MultimodalTransientModel(
        num_classes=4,
        img_dim=64,
        lc_dim=64,
        fused_dim=128,
        fusion_method="cross_attention"
    )
    images = torch.randn(2, 3, 64, 64)
    lcs = torch.randn(2, 25, 4)
    masks = torch.ones(2, 25, dtype=torch.bool)

    logits, z_fused = model(images, lcs, masks)
    assert logits.shape == (2, 4)
    assert z_fused.shape == (2, 128)
