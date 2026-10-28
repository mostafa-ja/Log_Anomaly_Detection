"""
Standalone Evaluation & Reporting Module
========================================
Loads trained models, executes multi-metric evaluation, and generates
publication-quality confusion matrices and ROC curves.
"""

import os
import argparse
import numpy as np
import torch
from sklearn.metrics import classification_report

from src.utils import (
    seed_everything,
    get_device,
    compute_metrics,
    plot_confusion_matrix,
    plot_roc_curve,
)
from src.data_loader import generate_synthetic_log_dataset, build_semantic_windows
from src.models import DeepLog, LogAnomaly, TemporalConvNet, MLPAutoencoder


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Trained Anomaly Detection Model")
    parser.add_argument(
        "--model",
        type=str,
        default="deeplog",
        choices=["deeplog", "loganomaly", "tcn", "autoencoder"],
        help="Model architecture to evaluate",
    )
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint (.pth)")
    parser.add_argument("--window-size", type=int, default=10, help="Sliding window length")
    parser.add_argument("--hidden-size", type=int, default=64, help="Hidden state dimension")
    parser.add_argument("--num-classes", type=int, default=28, help="Number of distinct log classes")
    parser.add_argument("--top-k", type=int, default=9, help="Top-g candidates considered normal")
    parser.add_argument("--output-dir", type=str, default="reports", help="Directory to save metric charts")
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    return parser.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = get_device()
    os.makedirs(args.output_dir, exist_ok=True)

    checkpoint_path = (
        args.checkpoint
        if args.checkpoint is not None
        else os.path.join("checkpoints", f"{args.model}.pth")
    )

    # Check if checkpoint exists and infer num_classes
    num_classes = args.num_classes
    state_dict = None
    if os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
        if "fc.bias" in state_dict:
            num_classes = state_dict["fc.bias"].shape[0]
            print(f"[Evaluation] Inferred num_classes={num_classes} from checkpoint.")
        print(f"[Evaluation] Successfully loaded checkpoint weights.")
    else:
        print(f"[Warning] Checkpoint not found at {checkpoint_path}. Evaluating uninitialized model.")

    # Build model architecture
    if args.model == "deeplog":
        model = DeepLog(num_classes=num_classes, hidden_size=args.hidden_size).to(device)
    elif args.model == "loganomaly":
        model = LogAnomaly(num_classes=num_classes, hidden_size=args.hidden_size).to(device)
    elif args.model == "tcn":
        model = TemporalConvNet(
            num_classes=num_classes, num_channels=[args.hidden_size, args.hidden_size]
        ).to(device)
    elif args.model == "autoencoder":
        model = MLPAutoencoder(input_dim=20, hidden_dim=args.hidden_size).to(device)

    if state_dict is not None:
        model.load_state_dict(state_dict)

    model.eval()

    # Generate evaluation sessions
    sessions, labels = generate_synthetic_log_dataset(
        num_sessions=600, anomaly_ratio=0.2, num_classes=num_classes, random_state=args.seed + 100
    )

    print(f"[Evaluation] Benchmarking across {len(sessions)} test sessions ({sum(labels)} anomalies)...")

    y_preds = []
    y_scores = []

    if args.model in ["deeplog", "loganomaly", "tcn"]:
        for sess in sessions:
            is_ano, score = model.predict_session(
                sess, window_size=args.window_size, top_k=args.top_k, device=device
            )
            y_preds.append(int(is_ano))
            y_scores.append(score)
    else:
        dummy_embeddings = np.random.RandomState(args.seed).randn(num_classes + 1, 20).astype(np.float32)
        test_windows = build_semantic_windows(sessions, dummy_embeddings, window_size=args.window_size)
        test_vecs = torch.tensor(test_windows.mean(axis=1), dtype=torch.float32).to(device)
        with torch.no_grad():
            recons = model(test_vecs)
            errors = torch.mean((test_vecs - recons) ** 2, dim=-1).cpu().numpy()
        thresh = np.percentile(errors[: len(sessions) - sum(labels)], 90)
        y_preds = (errors[: len(sessions)] > thresh).astype(int).tolist()
        y_scores = errors[: len(sessions)].tolist()

    metrics = compute_metrics(labels, y_preds, y_scores)
    print("\n" + "=" * 50)
    print(classification_report(labels, y_preds, target_names=["Normal", "Anomaly"], digits=4))
    print("=" * 50)
    print(f"Summary Metrics:")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:12s}: {v:.4f}")
        else:
            print(f"  {k:12s}: {v}")
    print("=" * 50)

    # Save visual reports
    cm_path = os.path.join(args.output_dir, f"confusion_matrix_{args.model}.png")
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(labels, y_preds, labels=[0, 1])
    plot_confusion_matrix(cm, title=f"Confusion Matrix ({args.model.upper()})", save_path=cm_path)
    print(f"[Report] Saved confusion matrix heatmap to: {cm_path}")

    if len(set(y_scores)) > 1:
        roc_path = os.path.join(args.output_dir, f"roc_curve_{args.model}.png")
        plot_roc_curve(labels, y_scores, title=f"ROC Curve ({args.model.upper()})", save_path=roc_path)
        print(f"[Report] Saved ROC curve to: {roc_path}")


if __name__ == "__main__":
    main()
