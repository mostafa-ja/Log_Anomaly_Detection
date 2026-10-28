"""
LogAnomaly Architecture
=======================
Dual-stream LSTM model combining sequential log order and quantitative
event frequency vectors with self-attention mechanism.

Reference:
- Meng, W., Liu, Y., Zhu, Y., Zhang, S., Pei, D., Liu, Y., ... & Tao, M. (2019).
  LogAnomaly: Unsupervised Detection of Sequential and Quantitative Anomalies in Unstructured Logs.
  Proceedings of the 28th International Joint Conference on Artificial Intelligence (IJCAI).
"""

from typing import List, Tuple
from collections import Counter
import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    """Self-attention mechanism over temporal recurrent states."""

    def __init__(self, hidden_size: int):
        super(SelfAttention, self).__init__()
        self.w_omega = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.u_omega = nn.Parameter(torch.empty(hidden_size, 1))
        nn.init.xavier_uniform_(self.w_omega)
        nn.init.xavier_uniform_(self.u_omega)

    def forward(self, lstm_output: torch.Tensor) -> torch.Tensor:
        """
        Args:
            lstm_output: Tensor of shape (batch_size, seq_len, hidden_size)
        Returns:
            attn_output: Tensor of shape (batch_size, hidden_size)
        """
        # u = tanh(H * W) -> (batch_size, seq_len, hidden_size)
        u = torch.tanh(torch.matmul(lstm_output, self.w_omega))
        # att = u * u_omega -> (batch_size, seq_len, 1)
        att = torch.matmul(u, self.u_omega)
        # weights = softmax(att) -> (batch_size, seq_len, 1)
        weights = F.softmax(att, dim=1)
        # output = sum(weights * H) -> (batch_size, hidden_size)
        attn_output = torch.sum(lstm_output * weights, dim=1)
        return attn_output


class LogAnomaly(nn.Module):
    """
    LogAnomaly: Dual-stream neural network for log anomaly detection.
    Stream 0: Sequential LSTM over event token sequence with self-attention.
    Stream 1: Quantitative LSTM / Dense projection over event frequency counts.
    """

    def __init__(
        self,
        num_classes: int = 28,
        hidden_size: int = 64,
        num_layers: int = 2,
        embedding_dim: int = 32,
        dropout: float = 0.1,
    ):
        super(LogAnomaly, self).__init__()
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Sequential stream
        self.embedding = nn.Embedding(num_classes + 1, embedding_dim, padding_idx=0)
        self.seq_lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.attention = SelfAttention(hidden_size)

        # Quantitative stream
        self.quant_fc = nn.Sequential(
            nn.Linear(num_classes, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )

        # Multi-modal fusion classification head
        self.fc = nn.Linear(2 * hidden_size, num_classes)

    def forward(self, seq_tokens: torch.Tensor, quant_vector: torch.Tensor) -> torch.Tensor:
        """
        Args:
            seq_tokens: (batch_size, seq_len)
            quant_vector: (batch_size, num_classes)
        Returns:
            logits: (batch_size, num_classes)
        """
        # Sequential stream
        embeds = self.embedding(seq_tokens)
        seq_out, _ = self.seq_lstm(embeds)
        seq_rep = self.attention(seq_out)  # (batch_size, hidden_size)

        # Quantitative stream
        quant_rep = self.quant_fc(quant_vector)  # (batch_size, hidden_size)

        # Concat and project
        fused = torch.cat([seq_rep, quant_rep], dim=-1)
        logits = self.fc(fused)
        return logits

    @torch.no_grad()
    def predict_session(
        self,
        session: List[int],
        window_size: int = 10,
        top_k: int = 9,
        device: torch.device = torch.device("cpu"),
    ) -> Tuple[bool, float]:
        """Evaluate session under both sequential and quantitative expectations."""
        self.eval()
        if len(session) <= window_size:
            session = session + [0] * (window_size + 1 - len(session))

        anomalous_windows = 0
        total_windows = len(session) - window_size

        if total_windows <= 0:
            return False, 0.0

        for i in range(total_windows):
            seq = session[i : i + window_size]
            target = session[i + window_size]

            # Build quantitative vector
            counts = Counter(seq)
            q_vec = [counts.get(k, 0) for k in range(self.num_classes)]

            seq_tensor = torch.tensor([seq], dtype=torch.long, device=device)
            quant_tensor = torch.tensor([q_vec], dtype=torch.float32, device=device)

            logits = self.forward(seq_tensor, quant_tensor)
            top_candidates = torch.topk(logits, k=min(top_k, self.num_classes), dim=1).indices[0].tolist()

            if target not in top_candidates:
                anomalous_windows += 1

        is_anomaly = anomalous_windows > 0
        score = anomalous_windows / total_windows
        return is_anomaly, score
