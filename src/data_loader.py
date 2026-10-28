"""
Data Loader & Preprocessing Pipeline
====================================
Handles dataset ingestion, sliding window generation, sessionization,
quantitative count vector extraction, and synthetic fallback generation.
"""

import os
import re
from typing import List, Dict, Tuple, Optional
from collections import Counter
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


class LogDataset(Dataset):
    """PyTorch Dataset for sequential log events and quantitative count vectors."""

    def __init__(self, sequences: np.ndarray, targets: np.ndarray, quant_vectors: Optional[np.ndarray] = None):
        self.sequences = torch.tensor(sequences, dtype=torch.long)
        self.targets = torch.tensor(targets, dtype=torch.long)
        self.quant_vectors = (
            torch.tensor(quant_vectors, dtype=torch.float32)
            if quant_vectors is not None
            else None
        )

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int):
        if self.quant_vectors is not None:
            return (self.sequences[idx], self.quant_vectors[idx]), self.targets[idx]
        return self.sequences[idx], self.targets[idx]


class SemanticSequenceDataset(Dataset):
    """PyTorch Dataset for dense semantic embedding sequences (Autoencoders)."""

    def __init__(self, sequences: np.ndarray, labels: Optional[np.ndarray] = None):
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long) if labels is not None else None

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int):
        if self.labels is not None:
            return self.sequences[idx], self.labels[idx]
        return self.sequences[idx]


def build_sliding_windows(
    sessions: List[List[int]],
    window_size: int = 10,
    num_classes: int = 28,
    include_quant: bool = True,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Extract sliding window sequences, next-event targets, and optional quantitative vectors.
    """
    sequences = []
    targets = []
    quant_vectors = [] if include_quant else None

    for sess in sessions:
        if len(sess) <= window_size:
            # Pad session to window_size + 1
            padded = sess + [0] * (window_size + 1 - len(sess))
            seq = padded[:window_size]
            tgt = padded[window_size]
            sequences.append(seq)
            targets.append(tgt)
            if include_quant:
                counts = Counter(seq)
                q_vec = [counts.get(k, 0) for k in range(num_classes)]
                quant_vectors.append(q_vec)
        else:
            for i in range(len(sess) - window_size):
                seq = sess[i : i + window_size]
                tgt = sess[i + window_size]
                sequences.append(seq)
                targets.append(tgt)
                if include_quant:
                    counts = Counter(seq)
                    q_vec = [counts.get(k, 0) for k in range(num_classes)]
                    quant_vectors.append(q_vec)

    seq_arr = np.array(sequences, dtype=np.int64)
    tgt_arr = np.array(targets, dtype=np.int64)
    q_arr = np.array(quant_vectors, dtype=np.float32) if include_quant else None
    return seq_arr, tgt_arr, q_arr


def build_semantic_windows(
    sessions: List[List[int]],
    embeddings: np.ndarray,
    window_size: int = 10,
) -> np.ndarray:
    """
    Transform integer event sequences into dense semantic window sequences of shape (N, window_size, embed_dim).
    """
    embed_dim = embeddings.shape[1]
    windows = []

    for sess in sessions:
        if len(sess) <= window_size:
            padded_indices = sess + [0] * (window_size - len(sess))
            win = [embeddings[min(idx, len(embeddings) - 1)] for idx in padded_indices[:window_size]]
            windows.append(win)
        else:
            for i in range(len(sess) - window_size + 1):
                win = [embeddings[min(idx, len(embeddings) - 1)] for idx in sess[i : i + window_size]]
                windows.append(win)

    return np.array(windows, dtype=np.float32)


def generate_synthetic_log_dataset(
    num_sessions: int = 2000,
    anomaly_ratio: float = 0.1,
    num_classes: int = 28,
    min_len: int = 12,
    max_len: int = 30,
    random_state: int = 42,
) -> Tuple[List[List[int]], List[int]]:
    """
    Generate synthetic log event sessions with controlled normal and anomalous patterns.
    Guarantees reproducible execution without needing large external file downloads.
    """
    rng = np.random.RandomState(random_state)
    sessions = []
    labels = []

    split_idx = max(2, int(num_classes * 0.7))
    normal_events = list(range(split_idx))
    normal_motifs = [
        normal_events[i : i + 4]
        for i in range(0, len(normal_events), 4)
        if len(normal_events[i : i + 4]) > 0
    ]
    if not normal_motifs:
        normal_motifs = [[0]]

    # Rare / unexpected transitions indicating anomalies
    anomaly_events = list(range(split_idx, num_classes))
    if not anomaly_events:
        anomaly_events = list(range(num_classes))

    num_anomalies = int(num_sessions * anomaly_ratio)
    num_normals = num_sessions - num_anomalies

    # Generate normal sessions
    for _ in range(num_normals):
        sess_len = rng.randint(min_len, max_len)
        sess = []
        while len(sess) < sess_len:
            motif = normal_motifs[rng.choice(len(normal_motifs))]
            sess.extend(motif)
        sessions.append(sess[:sess_len])
        labels.append(0)  # 0 = Normal

    # Generate anomalous sessions (unexpected order, high retry spikes, or anomalous events)
    for _ in range(num_anomalies):
        sess_len = rng.randint(min_len, max_len)
        sess = []
        while len(sess) < sess_len - 3:
            motif = normal_motifs[rng.choice(len(normal_motifs))]
            sess.extend(motif)
        # Inject anomalous disruption
        sess.extend([rng.choice(anomaly_events), rng.choice(anomaly_events), rng.choice(anomaly_events)])
        rng.shuffle(sess[-6:])  # Disordered sequence
        sessions.append(sess[:sess_len])
        labels.append(1)  # 1 = Anomaly

    # Shuffle sessions
    indices = rng.permutation(len(sessions))
    sessions = [sessions[i] for i in indices]
    labels = [labels[i] for i in indices]

    return sessions, labels


def get_data_loaders(
    train_sessions: List[List[int]],
    test_sessions: List[List[int]],
    test_labels: List[int],
    window_size: int = 10,
    num_classes: int = 28,
    batch_size: int = 64,
    include_quant: bool = False,
) -> Tuple[DataLoader, Dict[str, any]]:
    """
    Construct training DataLoader and test session dictionary for evaluation.
    """
    X_train, y_train, q_train = build_sliding_windows(
        train_sessions, window_size=window_size, num_classes=num_classes, include_quant=include_quant
    )
    train_ds = LogDataset(X_train, y_train, quant_vectors=q_train)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)

    test_data = {
        "sessions": test_sessions,
        "labels": test_labels,
        "window_size": window_size,
        "num_classes": num_classes,
    }

    return train_loader, test_data
