from __future__ import annotations

import argparse
from collections import defaultdict

import torch
from torch.utils.data import DataLoader

from data_utils import AthleticMMDataset, load_athletic_mm_samples
from model import MambaSportsNet


def rmse(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    return torch.sqrt(torch.mean((y_pred - y_true) ** 2)).item()


def mae(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    return torch.mean(torch.abs(y_pred - y_true)).item()


def peak_error_percent(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    pred_peak = y_pred.max().item()
    true_peak = y_true.max().item()
    return abs(pred_peak - true_peak) / (abs(true_peak) + 1e-8) * 100.0


def macro_f1_from_labels(pred: torch.Tensor, true: torch.Tensor, num_classes: int) -> float:
    scores = []
    for c in range(num_classes):
        tp = ((pred == c) & (true == c)).sum().float()
        fp = ((pred == c) & (true != c)).sum().float()
        fn = ((pred != c) & (true == c)).sum().float()
        denom = 2 * tp + fp + fn
        f1 = (2 * tp / denom) if denom > 0 else torch.tensor(0.0)
        scores.append(f1)
    return torch.stack(scores).mean().item()


def evaluate(model, loader, device, fatigue_classes):
    model.eval()

    all_grf_pred = []
    all_grf_true = []
    all_fatigue_pred = []
    all_fatigue_true = []

    subject_grf_pred = defaultdict(list)
    subject_grf_true = defaultdict(list)
    subject_fatigue_pred = defaultdict(list)
    subject_fatigue_true = defaultdict(list)

    with torch.no_grad():
        for batch in loader:
            kin = batch["kinematic"].to(device)
            phy = batch["physiological"].to(device)
            grf = batch["grf"].to(device)
            fatigue = batch["fatigue"].to(device)
            subject_ids = batch["subject_id"]

            outputs = model(kin, phy)
            grf_pred = outputs["grf_pred"]
            fatigue_pred = outputs["fatigue_logits"].argmax(dim=-1)

            all_grf_pred.append(grf_pred.cpu())
            all_grf_true.append(grf.cpu())
            all_fatigue_pred.append(fatigue_pred.cpu())
            all_fatigue_true.append(fatigue.cpu())

            for i, sid in enumerate(subject_ids):
                subject_grf_pred[sid].append(grf_pred[i].cpu())
                subject_grf_true[sid].append(grf[i].cpu())
                subject_fatigue_pred[sid].append(fatigue_pred[i].cpu().view(1))
                subject_fatigue_true[sid].append(fatigue[i].cpu().view(1))

    y_grf_pred = torch.cat(all_grf_pred, dim=0)
    y_grf_true = torch.cat(all_grf_true, dim=0)
    y_fatigue_pred = torch.cat(all_fatigue_pred, dim=0)
    y_fatigue_true = torch.cat(all_fatigue_true, dim=0)

    window_metrics = {
        "grf_rmse": rmse(y_grf_pred, y_grf_true),
        "grf_mae": mae(y_grf_pred, y_grf_true),
        "grf_peak_error_pct": peak_error_percent(y_grf_pred, y_grf_true),
        "fatigue_acc": (y_fatigue_pred == y_fatigue_true).float().mean().item(),
        "fatigue_macro_f1": macro_f1_from_labels(y_fatigue_pred, y_fatigue_true, fatigue_classes),
    }

    subject_rmse = []
    subject_f1 = []
    for sid in sorted(subject_grf_pred.keys()):
        pred_grf = torch.cat(subject_grf_pred[sid], dim=0)
        true_grf = torch.cat(subject_grf_true[sid], dim=0)
        pred_f = torch.cat(subject_fatigue_pred[sid], dim=0)
        true_f = torch.cat(subject_fatigue_true[sid], dim=0)

        subject_rmse.append(rmse(pred_grf, true_grf))
        subject_f1.append(macro_f1_from_labels(pred_f, true_f, fatigue_classes))

    subject_metrics = {
        "grf_rmse_mean": float(torch.tensor(subject_rmse).mean().item()),
        "grf_rmse_std": float(torch.tensor(subject_rmse).std(unbiased=False).item()),
        "fatigue_f1_mean": float(torch.tensor(subject_f1).mean().item()),
        "fatigue_f1_std": float(torch.tensor(subject_f1).std(unbiased=False).item()),
        "num_subjects": len(subject_rmse),
    }

    return window_metrics, subject_metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate Mamba-SportsNet checkpoint")
    parser.add_argument("--data", type=str, default="athletic_mm_v1.pt")
    parser.add_argument("--ckpt", type=str, default="checkpoints/best.pt")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    checkpoint = torch.load(args.ckpt, map_location="cpu")
    samples = load_athletic_mm_samples(args.data)

    normalizer = checkpoint["normalizer"]
    split = checkpoint["split"]
    fatigue_classes = int(checkpoint["fatigue_classes"])
    technique_classes = int(checkpoint["technique_classes"])

    test_ds = AthleticMMDataset(samples, split["test_subjects"], normalizer)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = MambaSportsNet(
        kinematic_dim=18,
        physiological_dim=4,
        d_model=128,
        depth=4,
        fatigue_classes=fatigue_classes,
        technique_classes=technique_classes,
    ).to(args.device)
    model.load_state_dict(checkpoint["model_state"])

    window_metrics, subject_metrics = evaluate(model, test_loader, args.device, fatigue_classes)

    print("=== Window-level metrics ===")
    for k, v in window_metrics.items():
        print(f"{k}: {v:.6f}")

    print("\n=== Subject-level metrics (paper-style aggregation) ===")
    for k, v in subject_metrics.items():
        if isinstance(v, float):
            print(f"{k}: {v:.6f}")
        else:
            print(f"{k}: {v}")


if __name__ == "__main__":
    main()
