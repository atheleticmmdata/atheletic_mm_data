from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data_utils import (
    AthleticMMDataset,
    compute_train_normalizer,
    create_subject_split,
    infer_num_classes,
    load_athletic_mm_samples,
    set_seed,
)
from model import MambaSportsNet


def macro_f1_score(logits: torch.Tensor, labels: torch.Tensor, num_classes: int) -> float:
    preds = logits.argmax(dim=-1)
    f1_values = []
    for c in range(num_classes):
        tp = ((preds == c) & (labels == c)).sum().float()
        fp = ((preds == c) & (labels != c)).sum().float()
        fn = ((preds != c) & (labels == c)).sum().float()
        denom = 2 * tp + fp + fn
        f1 = (2 * tp / denom) if denom > 0 else torch.tensor(0.0, device=logits.device)
        f1_values.append(f1)
    return torch.stack(f1_values).mean().item()


def run_epoch(model, loader, optimizer, device, lambda_reg, lambda_cls, fatigue_classes, technique_classes, train):
    mse = nn.MSELoss()
    ce = nn.CrossEntropyLoss()

    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_rmse = 0.0
    total_f1 = 0.0
    total_tech_acc = 0.0
    steps = 0

    for batch in loader:
        kin = batch["kinematic"].to(device)
        phy = batch["physiological"].to(device)
        grf = batch["grf"].to(device)
        fatigue = batch["fatigue"].to(device)
        technique = batch["technique"].to(device)

        if train:
            optimizer.zero_grad(set_to_none=True)

        outputs = model(kin, phy)
        loss_reg = mse(outputs["grf_pred"], grf)
        loss_fatigue = ce(outputs["fatigue_logits"], fatigue)
        loss_technique = ce(outputs["technique_logits"], technique)
        loss_cls = 0.5 * (loss_fatigue + loss_technique)
        loss = lambda_reg * loss_reg + lambda_cls * loss_cls

        if train:
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            rmse = torch.sqrt(torch.mean((outputs["grf_pred"] - grf) ** 2)).item()
            f1 = macro_f1_score(outputs["fatigue_logits"], fatigue, fatigue_classes)
            tech_acc = (outputs["technique_logits"].argmax(dim=-1) == technique).float().mean().item()

        total_loss += loss.item()
        total_rmse += rmse
        total_f1 += f1
        total_tech_acc += tech_acc
        steps += 1

    return {
        "loss": total_loss / max(steps, 1),
        "grf_rmse": total_rmse / max(steps, 1),
        "fatigue_f1": total_f1 / max(steps, 1),
        "technique_acc": total_tech_acc / max(steps, 1),
    }


def main():
    parser = argparse.ArgumentParser(description="Train Mamba-SportsNet on Athletic-MM")
    parser.add_argument("--data", type=str, default="athletic_mm_v1.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--lambda-reg", type=float, default=1.0)
    parser.add_argument("--lambda-cls", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out-dir", type=str, default="checkpoints")
    args = parser.parse_args()

    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    samples = load_athletic_mm_samples(args.data)
    fatigue_classes, technique_classes = infer_num_classes(samples)

    split = create_subject_split([s["subject_id"] for s in samples], seed=args.seed)
    normalizer = compute_train_normalizer(samples, split.train_subjects)

    train_ds = AthleticMMDataset(samples, split.train_subjects, normalizer)
    val_ds = AthleticMMDataset(samples, split.val_subjects, normalizer)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = MambaSportsNet(
        kinematic_dim=18,
        physiological_dim=4,
        d_model=128,
        depth=4,
        fatigue_classes=fatigue_classes,
        technique_classes=technique_classes,
    ).to(args.device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_rmse = float("inf")
    history = []

    for epoch in range(1, args.epochs + 1):
        train_stats = run_epoch(
            model,
            train_loader,
            optimizer,
            args.device,
            args.lambda_reg,
            args.lambda_cls,
            fatigue_classes,
            technique_classes,
            train=True,
        )
        with torch.no_grad():
            val_stats = run_epoch(
                model,
                val_loader,
                optimizer,
                args.device,
                args.lambda_reg,
                args.lambda_cls,
                fatigue_classes,
                technique_classes,
                train=False,
            )

        scheduler.step()

        row = {"epoch": epoch, "train": train_stats, "val": val_stats}
        history.append(row)
        print(
            f"Epoch {epoch:03d} | "
            f"Train Loss {train_stats['loss']:.4f} RMSE {train_stats['grf_rmse']:.4f} F1 {train_stats['fatigue_f1']:.4f} | "
            f"Val Loss {val_stats['loss']:.4f} RMSE {val_stats['grf_rmse']:.4f} F1 {val_stats['fatigue_f1']:.4f}"
        )

        if val_stats["grf_rmse"] < best_val_rmse:
            best_val_rmse = val_stats["grf_rmse"]
            ckpt = {
                "model_state": model.state_dict(),
                "normalizer": normalizer,
                "split": {
                    "train_subjects": split.train_subjects,
                    "val_subjects": split.val_subjects,
                    "test_subjects": split.test_subjects,
                },
                "config": vars(args),
                "fatigue_classes": fatigue_classes,
                "technique_classes": technique_classes,
                "epoch": epoch,
            }
            torch.save(ckpt, out_dir / "best.pt")
            print(f"  Saved new best checkpoint at epoch {epoch} (Val RMSE={best_val_rmse:.4f})")

    with open(out_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
