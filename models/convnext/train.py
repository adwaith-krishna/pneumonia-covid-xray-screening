import os
import sys
import json
import time
import argparse
import random
from typing import Optional, Dict, List
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.dataset import get_dataloaders, CLASS_TO_IDX, IDX_TO_CLASS
from src.metrics import compute_classification_metrics, plot_training_curves
from models.convnext.model import build_convnext_tiny


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_one_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[torch.amp.GradScaler],
    device: torch.device,
    use_amp: bool = True,
):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    pbar = tqdm(loader, desc="  Train", leave=False, dynamic_ncols=True)
    for images, targets, _ in pbar:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if use_amp and device.type == "cuda":
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                outputs = model(images)
                loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = torch.argmax(outputs, dim=1).detach().cpu().numpy()
        all_preds.extend(preds)
        all_targets.extend(targets.cpu().numpy())

        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    epoch_loss = running_loss / len(loader.dataset)
    metrics = compute_classification_metrics(all_targets, all_preds)
    return epoch_loss, metrics["accuracy"], metrics["macro_f1"]


@torch.no_grad()
def evaluate_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool = True,
    class_names=None,
):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []
    all_probs = []

    pbar = tqdm(loader, desc="  Val  ", leave=False, dynamic_ncols=True)
    for images, targets, _ in pbar:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if use_amp and device.type == "cuda":
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                outputs = model(images)
                loss = criterion(outputs, targets)
        else:
            outputs = model(images)
            loss = criterion(outputs, targets)

        running_loss += loss.item() * images.size(0)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()
        preds = np.argmax(probs, axis=1)

        all_probs.append(probs)
        all_preds.extend(preds)
        all_targets.extend(targets.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    all_probs = np.vstack(all_probs)
    metrics = compute_classification_metrics(
        y_true=all_targets,
        y_pred=all_preds,
        y_probs=all_probs,
        class_names=class_names,
    )
    return epoch_loss, metrics


def main():
    parser = argparse.ArgumentParser(description="Train ConvNeXt-Tiny on Chest X-Rays")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Peak learning rate for AdamW")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay for AdamW")
    parser.add_argument("--drop-rate", type=float, default=0.2, help="Dropout probability in classifier head")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience (epochs)")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader background worker count")
    parser.add_argument("--seed", type=int, default=42, help="Random reproducibility seed")
    parser.add_argument("--no-class-weights", action="store_true", help="Disable class-weighted CrossEntropyLoss")
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision training")
    parser.add_argument(
        "--metadata-csv",
        type=str,
        default="dataset_processed/dataset_metadata.csv",
        help="Path to preprocessed dataset metadata CSV",
    )
    args = parser.parse_args()

    set_seed(args.seed)

    # Directories
    model_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoint_dir = os.path.join(model_dir, "checkpoints")
    results_dir = os.path.join(model_dir, "results")
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = (not args.no_amp) and (device.type == "cuda")

    print("\n" + "=" * 65)
    print("      CONVNEXT-TINY CHEST X-RAY SCREENING TRAINING PIPELINE")
    print("=" * 65)
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  GPU Name:         {torch.cuda.get_device_name(0)}")
        print(f"  VRAM Available:   {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
        print(f"  Mixed Precision:  {use_amp} (AMP float16)")
    else:
        print("  Warning: Running on CPU. Training will be significantly slower.")
    print(f"Hyperparameters:")
    print(f"  Epochs:           {args.epochs}")
    print(f"  Batch Size:       {args.batch_size}")
    print(f"  Learning Rate:    {args.lr}")
    print(f"  Weight Decay:     {args.weight_decay}")
    print(f"  Early Stopping:   patience = {args.patience}")
    print("=" * 65)

    # 1. DataLoaders
    print("\n[1/4] Loading Datasets & Computing Split Weights...")
    train_loader, val_loader, test_loader, class_weights = get_dataloaders(
        metadata_csv=args.metadata_csv,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        use_weighted_sampler=False,
        return_mask=True,
    )

    class_names = [IDX_TO_CLASS[i] for i in range(len(CLASS_TO_IDX))]
    print(f"  Classes: {class_names}")
    print(f"  Train samples: {len(train_loader.dataset)} | Batches: {len(train_loader)}")
    print(f"  Val samples:   {len(val_loader.dataset)}   | Batches: {len(val_loader)}")
    print(f"  Test samples:  {len(test_loader.dataset)}  | Batches: {len(test_loader)}")

    # 2. Model & Loss Setup
    print("\n[2/4] Initializing Pretrained ConvNeXt-Tiny Architecture...")
    model = build_convnext_tiny(
        num_classes=len(CLASS_TO_IDX),
        pretrained=True,
        drop_rate=args.drop_rate,
    ).to(device)

    if not args.no_class_weights:
        class_weights = class_weights.to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        print(f"  Using Weighted CrossEntropy: {class_weights.cpu().numpy().round(3)}")
    else:
        criterion = nn.CrossEntropyLoss()
        print("  Using Standard Unweighted CrossEntropy")

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    # 3. Training Loop
    print("\n[3/4] Starting Model Optimization...")
    best_val_f1 = 0.0
    epochs_no_improve = 0
    history = {
        "train_loss": [],
        "train_acc": [],
        "train_f1": [],
        "val_loss": [],
        "val_acc": [],
        "val_f1": [],
        "val_precision": [],
        "val_recall": [],
        "lr": [],
    }

    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        train_loss, train_acc, train_f1 = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            use_amp=use_amp,
        )

        val_loss, val_metrics = evaluate_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            use_amp=use_amp,
            class_names=class_names,
        )

        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        # Update History
        history["train_loss"].append(float(train_loss))
        history["train_acc"].append(float(train_acc))
        history["train_f1"].append(float(train_f1))
        history["val_loss"].append(float(val_loss))
        history["val_acc"].append(float(val_metrics["accuracy"]))
        history["val_f1"].append(float(val_metrics["macro_f1"]))
        history["val_precision"].append(float(val_metrics["macro_precision"]))
        history["val_recall"].append(float(val_metrics["macro_recall"]))
        history["lr"].append(float(current_lr))

        epoch_duration = time.time() - epoch_start

        # Display Epoch Results
        print(
            f"Epoch [{epoch:02d}/{args.epochs:02d}] ({epoch_duration:.1f}s) | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc*100:.1f}% F1: {train_f1*100:.1f}% | "
            f"Val Loss: {val_loss:.4f} Acc: {val_metrics['accuracy']*100:.1f}% "
            f"Macro-F1: {val_metrics['macro_f1']*100:.2f}% | "
            f"LR: {current_lr:.1e}"
        )

        # Print per-class F1 for clinical clarity
        per_class = val_metrics["per_class"]
        f1_summary = "  -> Per-Class F1: " + " | ".join(
            [f"{cls}: {per_class[cls]['f1']*100:.1f}%" for cls in class_names]
        )
        print(f1_summary)

        # Checkpointing
        is_best = val_metrics["macro_f1"] > best_val_f1
        checkpoint_data = {
            "epoch": epoch,
            "model_name": "ConvNeXt-Tiny",
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_f1": max(best_val_f1, val_metrics["macro_f1"]),
            "metrics": val_metrics,
            "class_to_idx": CLASS_TO_IDX,
            "args": vars(args),
        }

        # Save last checkpoint
        last_ckpt_path = os.path.join(checkpoint_dir, "last_model.pth")
        torch.save(checkpoint_data, last_ckpt_path)

        if is_best:
            best_val_f1 = val_metrics["macro_f1"]
            epochs_no_improve = 0
            best_ckpt_path = os.path.join(checkpoint_dir, "best_model.pth")
            torch.save(checkpoint_data, best_ckpt_path)
            print(f"  ★ New best model saved to: {best_ckpt_path} (Macro-F1: {best_val_f1*100:.2f}%)")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"\n[Early Stopping Triggered] No improvement in Macro F1 for {args.patience} epochs.")
                break

    total_training_time = (time.time() - start_time) / 60.0
    print("\n" + "=" * 65)
    print(f"Training completed in {total_training_time:.2f} minutes.")
    print(f"Best Validation Macro F1: {best_val_f1*100:.2f}%")
    print(f"Best model weights saved: {os.path.join(checkpoint_dir, 'best_model.pth')}")

    # 4. Save History & Training Curves
    print("\n[4/4] Saving Training History and Generating Curves...")
    history_path = os.path.join(results_dir, "training_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    curves_path = os.path.join(results_dir, "training_curves.png")
    plot_training_curves(history, curves_path, model_name="ConvNeXt-Tiny")
    print(f"  Training curves plot saved: {curves_path}")
    print(f"  History metrics saved:     {history_path}")
    print("=" * 65)
    print("\nTo evaluate this model on the unseen test set, run:")
    print("  uv run python models/convnext/evaluate.py")


if __name__ == "__main__":
    main()
