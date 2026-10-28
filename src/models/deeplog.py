"""
DeepLog Architecture
====================
Sequential log key anomaly detection model using stacked LSTM layers
with top-g candidate probability verification.

Reference:
- Du, M., Li, F., Zheng, G., & Srikumar, V. (2017).
  DeepLog: Anomaly Detection and Diagnosis from System Logs through Deep Learning.
  Proceedings of the 2017 ACM SIGSAC Conference on Computer and Communications Security (CCS).
"""

from typing import List, Tuple
import torch
import torch.nn as nn


class DeepLog(nn.Module):
    """
    DeepLog: Multi-layer LSTM model for sequential log anomaly detection.
    Predicts probability distribution over next possible log event keys.
    """

    def __init__(
        self,
        num_classes: int = 28,
        hidden_size: int = 64,
        num_layers: int = 2,
        embedding_dim: int = 32,
        dropout: float = 0.1,
    ):
        super(DeepLog, self).__init__()
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Token embedding for discrete event indices (+1 for padding / unknown token 0)
        self.embedding = nn.Embedding(num_classes + 1, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x: Tensor of shape (batch_size, sequence_length) with integer event IDs.
        Returns:
            logits: Tensor of shape (batch_size, num_classes)
        """
        # Embed token sequences
        embeds = self.embedding(x)  # (batch_size, seq_len, embedding_dim)

        # Initialize hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)

        out, _ = self.lstm(embeds, (h0, c0))  # (batch_size, seq_len, hidden_size)
        logits = self.fc(out[:, -1, :])      # Project last time step
        return logits

    @torch.no_grad()
    def predict_session(
        self,
        session: List[int],
        window_size: int = 10,
        top_k: int = 9,
        device: torch.device = torch.device("cpu"),
    ) -> Tuple[bool, float]:
        """
        Evaluate an entire session sequence using sliding windows.
        If any actual transition falls outside the top-k candidate set,
        the session is flagged as anomalous.

        Returns:
            is_anomaly: bool (True if anomalous, False if normal)
            anomaly_score: float (ratio of anomalous windows)
        """
        self.eval()
        if len(session) <= window_size:
            session = session + [0] * (window_size + 1 - len(session))

        anomalous_windows = 0
        total_windows = len(session) - window_size

        if total_windows <= 0:
            return False, 0.0

        for i in range(total_windows):
            seq = torch.tensor([session[i : i + window_size]], dtype=torch.long, device=device)
            target = session[i + window_size]

            logits = self.forward(seq)
            # Retrieve top_k candidate indices
            top_candidates = torch.topk(logits, k=min(top_k, self.num_classes), dim=1).indices[0].tolist()

            if target not in top_candidates:
                anomalous_windows += 1

        is_anomaly = anomalous_windows > 0
        score = anomalous_windows / total_windows
        return is_anomaly, score
