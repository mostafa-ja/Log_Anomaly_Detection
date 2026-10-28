"""
Unit tests for all deep learning model architectures.
"""

import pytest
import torch
from src.models import (
    DeepLog,
    LogAnomaly,
    TemporalConvNet,
    MLPAutoencoder,
    LSTMAutoencoder,
)


def test_deeplog_forward_and_session_prediction():
    model = DeepLog(num_classes=20, hidden_size=32, num_layers=2)
    # Batch of 4 sequences of length 10
    x = torch.randint(0, 20, (4, 10))
    logits = model(x)
    assert logits.shape == (4, 20)

    # Test session prediction
    session = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    is_ano, score = model.predict_session(session, window_size=5, top_k=5)
    assert isinstance(is_ano, bool)
    assert 0.0 <= score <= 1.0


def test_loganomaly_forward_and_session_prediction():
    model = LogAnomaly(num_classes=20, hidden_size=32, num_layers=2)
    seqs = torch.randint(0, 20, (4, 10))
    quants = torch.randn(4, 20)
    logits = model(seqs, quants)
    assert logits.shape == (4, 20)

    session = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    is_ano, score = model.predict_session(session, window_size=5, top_k=5)
    assert isinstance(is_ano, bool)
    assert 0.0 <= score <= 1.0


def test_tcn_forward_and_session_prediction():
    model = TemporalConvNet(num_classes=20, num_channels=[32, 32])
    x = torch.randint(0, 20, (4, 10))
    logits = model(x)
    assert logits.shape == (4, 20)

    session = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    is_ano, score = model.predict_session(session, window_size=5, top_k=5)
    assert isinstance(is_ano, bool)
    assert 0.0 <= score <= 1.0


def test_autoencoders_forward():
    mlp_ae = MLPAutoencoder(input_dim=16, hidden_dim=32)
    x = torch.randn(8, 16)
    recon = mlp_ae(x)
    assert recon.shape == (8, 16)

    lstm_ae = LSTMAutoencoder(input_dim=16, hidden_dim=32)
    seq = torch.randn(8, 10, 16)
    recon_seq = lstm_ae(seq)
    assert recon_seq.shape == (8, 10, 16)
