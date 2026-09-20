"""Baseline models for unimodal and classical benchmark comparisons."""
from src.baselines.random_forest import RandomForestBaseline
from src.baselines.cnn_baseline import CNNImageBaseline
from src.baselines.lstm_baseline import LSTMLightCurveBaseline

__all__ = ["RandomForestBaseline", "CNNImageBaseline", "LSTMLightCurveBaseline"]
