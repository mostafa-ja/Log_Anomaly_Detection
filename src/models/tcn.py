"""
Temporal Convolutional Network (TCN) Architecture
=================================================
Dilated Causal 1D Convolutions with residual blocks for high-throughput,
long-range log sequence anomaly detection.

Reference:
- Bai, S., Kolter, J. Z., & Koltun, V. (2018).
  An empirical evaluation of generic convolutional and recurrent networks for sequence modeling.
  arXiv preprint arXiv:1803.01271.
"""

from typing import List, Tuple
import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import weight_norm


class Chomp1d(nn.Module):
    """Causal truncation: trims extra padded elements from the temporal dimension."""

    def __init__(self, chomp_size: int):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.chomp_size == 0:
            return x
        return x[:, :, : -self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """A single residual block with two dilated causal convolution layers."""

    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        kernel_size: int,
        stride: int,
        dilation: int,
        padding: int,
        dropout: float = 0.2,
    ):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(
            nn.Conv1d(
                n_inputs,
                n_outputs,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
        )
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(
            nn.Conv1d(
                n_outputs,
                n_outputs,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1,
            self.chomp1,
            self.relu1,
            self.dropout1,
            self.conv2,
            self.chomp2,
            self.relu2,
            self.dropout2,
        )
        self.downsample = (
            nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        )
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    """TCN model for sequence modeling and log key prediction."""

    def __init__(
        self,
        num_classes: int = 28,
        embedding_dim: int = 32,
        num_channels: List[int] = [64, 64, 64],
        kernel_size: int = 2,
        dropout: float = 0.2,
    ):
        super(TemporalConvNet, self).__init__()
        self.num_classes = num_classes
        self.embedding = nn.Embedding(num_classes + 1, embedding_dim, padding_idx=0)

        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2**i
            in_channels = embedding_dim if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            padding = (kernel_size - 1) * dilation_size
            layers.append(
                TemporalBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=1,
                    dilation=dilation_size,
                    padding=padding,
                    dropout=dropout,
                )
            )

        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1], num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Integer event tokens of shape (batch_size, seq_len)
        Returns:
            logits: Next-event prediction logits of shape (batch_size, num_classes)
        """
        # Embed and permute to (batch_size, channels, seq_len)
        embeds = self.embedding(x).transpose(1, 2)
        tcn_out = self.network(embeds)  # (batch_size, num_channels[-1], seq_len)
        last_step = tcn_out[:, :, -1]   # Extract last temporal step
        logits = self.fc(last_step)
        return logits

    @torch.no_grad()
    def predict_session(
        self,
        session: List[int],
        window_size: int = 10,
        top_k: int = 9,
        device: torch.device = torch.device("cpu"),
    ) -> Tuple[bool, float]:
        """Evaluate session under TCN causal convolutions."""
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
            top_candidates = torch.topk(logits, k=min(top_k, self.num_classes), dim=1).indices[0].tolist()

            if target not in top_candidates:
                anomalous_windows += 1

        is_anomaly = anomalous_windows > 0
        score = anomalous_windows / total_windows
        return is_anomaly, score
