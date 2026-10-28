"""
Unified Training Pipeline
=========================
Command-line interface to train DeepLog, LogAnomaly, TCN, or Autoencoder models
with automated GPU acceleration, checkpointing, and evaluation.
"""

import os
import argparse
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

from src.utils import seed_everything, get_device, compute_metrics
from src.data_loader import (
    generate_synthetic_log_dataset,
    build_sliding_windows,
    build_semantic_windows,
    LogDataset,
    SemanticSequenceDataset,
)
from src.models import (
    DeepLog,
    LogAnomaly,
    TemporalConvNet,
    MLPAutoencoder,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train Log Anomaly Detection Models")
    parser.add_argument(
        "--model",
        type=str,
        default="deeplog",
        choices=["deeplog", "loganomaly", "tcn", "autoencoder"],
        help="Model architecture to train",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="synthetic",
        help="Dataset identifier: 'synthetic' or path to dataset directory",
    )
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--num-sessions", type=int, default=1000, help="Number of synthetic sessions")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--window-size", type=int, default=10, help="Sliding window length")
    parser.add_argument("--hidden-size", type=int, default=64, help="Hidden state dimension")
    parser.add_argument("--num-classes", type=int, default=28, help="Number of distinct log event classes")
    parser.add_argument("--top-k", type=int, default=9, help="Top-g candidates considered normal")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--save-dir", type=str, default="checkpoints", help="Output directory for saved weights")
    return parser.parse_args()


def train_next_token_model(model, dataloader, epochs, lr, device, is_loganomaly=False):
    """Training loop for next-log key prediction models (DeepLog, LogAnomaly, TCN)."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    model.train()

    print(f"[Training] Starting optimization across {epochs} epochs on {device}...")
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        batches = 0
        for batch in dataloader:
            if is_loganomaly:
                (seqs, quants), targets = batch
                seqs = seqs.to(device)
                quants = quants.to(device)
                targets = targets.to(device)
                outputs = model(seqs, quants)
            else:
                seqs, targets = batch
                seqs = seqs.to(device)
                targets = targets.to(device)
                outputs = model(seqs)

            loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            batches += 1

        avg_loss = epoch_loss / max(1, batches)
        print(f"  Epoch [{epoch:02d}/{epochs:02d}] - Loss: {avg_loss:.4f}")

    elapsed = time.time() - start_time
    print(f"[Training] Completed in {elapsed:.2f}s ({elapsed/epochs:.2f}s/epoch)")
    return model


def train_autoencoder_model(model, dataloader, epochs, lr, device):
    """Training loop for Autoencoder reconstruction on normal log patterns."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    model.train()

    print(f"[Training] Starting Autoencoder reconstruction optimization for {epochs} epochs...")
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        batches = 0
        for batch in dataloader:
            x = batch[0].to(device)
            recon = model(x)
            loss = criterion(recon, x)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            batches += 1

        avg_loss = epoch_loss / max(1, batches)
        print(f"  Epoch [{epoch:02d}/{epochs:02d}] - Reconstruction MSE: {avg_loss:.6f}")

    elapsed = time.time() - start_time
    print(f"[Training] Completed in {elapsed:.2f}s")
    return model


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = get_device()
    os.makedirs(args.save_dir, exist_ok=True)

    print(f"\n=======================================================")
    print(f"  Model: {args.model.upper()} | Dataset: {args.dataset}")
    print(f"  Window Size: {args.window_size} | Epochs: {args.epochs} | LR: {args.lr}")
    print(f"=======================================================\n")

    # 1. Dataset Preparation
    print("[Data] Generating / Ingesting session data...")
    sessions, labels = generate_synthetic_log_dataset(
        num_sessions=args.num_sessions, anomaly_ratio=0.15, num_classes=args.num_classes, random_state=args.seed
    )

    # Train on normal sessions only (semi-supervised anomaly detection standard)
    normal_indices = [i for i, lbl in enumerate(labels) if lbl == 0]
    train_indices = normal_indices[: int(len(normal_indices) * 0.7)]
    test_indices = normal_indices[int(len(normal_indices) * 0.7) :] + [
        i for i, lbl in enumerate(labels) if lbl == 1
    ]

    train_sessions = [sessions[i] for i in train_indices]
    test_sessions = [sessions[i] for i in test_indices]
    test_labels = [labels[i] for i in test_indices]

    print(f"[Data] Training normal sessions: {len(train_sessions)} | Test sessions: {len(test_sessions)} (Anomalies: {sum(test_labels)})")

    # 2. Build Model & DataLoaders
    save_path = os.path.join(args.save_dir, f"{args.model}.pth")

    if args.model == "deeplog":
        seqs, targets, _ = build_sliding_windows(
            train_sessions, window_size=args.window_size, num_classes=args.num_classes, include_quant=False
        )
        ds = LogDataset(seqs, targets)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True)
        model = DeepLog(num_classes=args.num_classes, hidden_size=args.hidden_size).to(device)
        model = train_next_token_model(model, loader, args.epochs, args.lr, device, is_loganomaly=False)

    elif args.model == "loganomaly":
        seqs, targets, quants = build_sliding_windows(
            train_sessions, window_size=args.window_size, num_classes=args.num_classes, include_quant=True
        )
        ds = LogDataset(seqs, targets, quant_vectors=quants)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True)
        model = LogAnomaly(num_classes=args.num_classes, hidden_size=args.hidden_size).to(device)
        model = train_next_token_model(model, loader, args.epochs, args.lr, device, is_loganomaly=True)

    elif args.model == "tcn":
        seqs, targets, _ = build_sliding_windows(
            train_sessions, window_size=args.window_size, num_classes=args.num_classes, include_quant=False
        )
        ds = LogDataset(seqs, targets)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True)
        model = TemporalConvNet(
            num_classes=args.num_classes, num_channels=[args.hidden_size, args.hidden_size]
        ).to(device)
        model = train_next_token_model(model, loader, args.epochs, args.lr, device, is_loganomaly=False)

    elif args.model == "autoencoder":
        # Generate pseudo-embeddings for templates
        dummy_embeddings = np.random.RandomState(args.seed).randn(args.num_classes + 1, 20).astype(np.float32)
        train_windows = build_semantic_windows(train_sessions, dummy_embeddings, window_size=args.window_size)
        # Average vectors across window for MLP input
        train_vecs = train_windows.mean(axis=1)
        ds = TensorDataset(torch.tensor(train_vecs, dtype=torch.float32))
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True)
        model = MLPAutoencoder(input_dim=20, hidden_dim=args.hidden_size).to(device)
        model = train_autoencoder_model(model, loader, args.epochs, args.lr, device)

    # 3. Save weights
    torch.save(model.state_dict(), save_path)
    print(f"[Checkpoint] Saved trained weights to: {save_path}")

    # 4. Evaluation on Test Sessions
    print("\n[Evaluation] Evaluating on test set...")
    y_preds = []
    y_scores = []

    if args.model in ["deeplog", "loganomaly", "tcn"]:
        for sess in test_sessions:
            is_ano, score = model.predict_session(
                sess, window_size=args.window_size, top_k=args.top_k, device=device
            )
            y_preds.append(int(is_ano))
            y_scores.append(score)
    else:
        # Autoencoder evaluation
        test_windows = build_semantic_windows(test_sessions, dummy_embeddings, window_size=args.window_size)
        test_vecs = torch.tensor(test_windows.mean(axis=1), dtype=torch.float32).to(device)
        model.eval()
        with torch.no_grad():
            recons = model(test_vecs)
            errors = torch.mean((test_vecs - recons) ** 2, dim=-1).cpu().numpy()
        # Calibrate threshold on normal test instances
        thresh = np.percentile(errors[: len(test_indices) - sum(test_labels)], 95)
        # Map back to session level
        y_preds = (errors[: len(test_sessions)] > thresh).astype(int).tolist()
        y_scores = errors[: len(test_sessions)].tolist()

    metrics = compute_metrics(test_labels, y_preds, y_scores)
    print(f"\n====================== Benchmark Results ======================")
    print(f"  Precision : {metrics['Precision']*100:.2f}%")
    print(f"  Recall    : {metrics['Recall']*100:.2f}%")
    print(f"  F1-Score  : {metrics['F1']*100:.2f}%")
    print(f"  Accuracy  : {metrics['Accuracy']*100:.2f}%")
    if metrics['ROC_AUC'] is not None:
        print(f"  ROC-AUC   : {metrics['ROC_AUC']:.4f}")
    print(f"  Confusion : TP={metrics['TP']}, FP={metrics['FP']}, TN={metrics['TN']}, FN={metrics['FN']}")
    print(f"===============================================================\n")


if __name__ == "__main__":
    main()
