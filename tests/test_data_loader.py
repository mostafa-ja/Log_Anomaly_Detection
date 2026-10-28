"""
Unit tests for data loaders, windowing, and synthetic generation.
"""

import pytest
import numpy as np
from src.data_loader import (
    generate_synthetic_log_dataset,
    build_sliding_windows,
    build_semantic_windows,
    LogDataset,
)


def test_synthetic_generation():
    sessions, labels = generate_synthetic_log_dataset(
        num_sessions=100, anomaly_ratio=0.2, num_classes=28, random_state=42
    )
    assert len(sessions) == 100
    assert len(labels) == 100
    assert sum(labels) == 20  # 20% anomalies
    assert all(isinstance(s, list) for s in sessions)


def test_sliding_window_builder():
    sessions = [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]]
    seqs, targets, quants = build_sliding_windows(
        sessions, window_size=5, num_classes=20, include_quant=True
    )
    # len(sessions[0]) - window_size = 12 - 5 = 7
    assert len(seqs) == 7
    assert len(targets) == 7
    assert seqs.shape == (7, 5)
    assert quants.shape == (7, 20)
    assert targets[0] == 6  # Next key after [1, 2, 3, 4, 5]


def test_semantic_window_builder():
    sessions = [[0, 1, 2, 3]]
    embeddings = np.random.randn(10, 8).astype(np.float32)
    windows = build_semantic_windows(sessions, embeddings, window_size=3)
    assert windows.shape == (2, 3, 8)
