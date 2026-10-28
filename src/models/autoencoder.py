"""
Autoencoder Architectures for Unsupervised Log Anomaly Detection
================================================================
Includes Deep Multi-Layer Perceptron (MLP) and LSTM Autoencoders trained on
normal log representations to detect deviations via reconstruction error.
"""

from typing import List, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn


class MLPAutoencoder(nn.Module):
    """Deep Multi-Layer Perceptron Autoencoder for dense vector reconstruction."""

    def __init__(self, input_dim: int = 20, hidden_dim: int = 32, dropout: float = 0.1):
        super(MLPAutoencoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass reconstructing input vector."""
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)
        return reconstruction


class LSTMAutoencoder(nn.Module):
    """Recurrent LSTM Autoencoder for temporal sequence reconstruction."""

    def __init__(self, input_dim: int = 20, hidden_dim: int = 32, num_layers: int = 1):
        super(LSTMAutoencoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.decoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.output_fc = nn.Linear(hidden_dim, input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, input_dim)
        Returns:
            reconstructed: Tensor of shape (batch_size, seq_len, input_dim)
        """
        batch_size, seq_len, _ = x.shape
        _, (hn, _) = self.encoder(x)  # hn: (num_layers, batch_size, hidden_dim)

        # Bottleneck latent vector from last layer
        bottleneck = hn[-1].unsqueeze(1).repeat(1, seq_len, 1)  # (batch_size, seq_len, hidden_dim)

        dec_out, _ = self.decoder(bottleneck)
        reconstructed = self.output_fc(dec_out)
        return reconstructed


class ReconstructionAnomalyDetector:
    """Wrapper that handles reconstruction scoring, threshold calibration, and inference."""

    def __init__(
        self,
        model: nn.Module,
        threshold: float = 0.005,
        criterion: Optional[nn.Module] = None,
        device: torch.device = torch.device("cpu"),
    ):
        self.model = model.to(device)
        self.threshold = threshold
        self.criterion = criterion if criterion is not None else nn.MSELoss(reduction="none")
        self.device = device

    def compute_errors(self, dataloader) -> np.ndarray:
        """Compute sample-wise reconstruction errors across a DataLoader."""
        self.model.eval()
        errors = []
        with torch.no_grad():
            for batch in dataloader:
                x = batch[0] if isinstance(batch, (list, tuple)) else batch
                x = x.to(self.device)
                recon = self.model(x)
                # Compute mean squared error per sample
                loss = torch.mean((x - recon) ** 2, dim=-1)
                if loss.ndim > 1:
                    loss = torch.mean(loss, dim=-1)
                errors.extend(loss.cpu().numpy().tolist())
        return np.array(errors)

    def calibrate_threshold(self, train_errors: np.ndarray, percentile: float = 95.0) -> float:
        """Calibrate anomaly threshold based on percentile of normal reconstruction errors."""
        self.threshold = float(np.percentile(train_errors, percentile))
        return self.threshold

    def calibrate_optimal_f1(
        self, val_errors: np.ndarray, val_labels: np.ndarray, num_steps: int = 100
    ) -> Tuple[float, float]:
        """Sweep thresholds to locate optimal F1 score threshold on validation set."""
        from sklearn.metrics import f1_score

        min_err = np.min(val_errors)
        max_err = np.max(val_errors)
        thresholds = np.linspace(min_err, max_err, num_steps)

        best_f1 = 0.0
        best_thresh = thresholds[0]

        for thresh in thresholds:
            preds = (val_errors > thresh).astype(int)
            f1 = f1_score(val_labels, preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = thresh

        self.threshold = float(best_thresh)
        return float(best_thresh), float(best_f1)

    def predict(self, errors: np.ndarray) -> np.ndarray:
        """Predict binary anomaly flags (0=Normal, 1=Anomaly)."""
        return (errors > self.threshold).astype(int)
