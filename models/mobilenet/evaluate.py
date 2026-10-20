import os
import sys
import json
import argparse
import numpy as np
import torch
from tqdm import tqdm

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.dataset import get_dataloaders, CLASS_TO_IDX, IDX_TO_CLASS
from src.metrics import compute_classification_metrics, plot_confusion_matrix
from models.mobilenet.model import build_mobilenet_v3


def evaluate_test_set(
    checkpoint_path: str = "models/mobilenet/checkpoints/best_model.pth",
    batch_size: int = 32,
    metadata_csv: str = "dataset_processed/dataset_metadata.csv",
):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found at: {checkpoint_path}. Please train MobileNet first!"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n" + "=" * 65)
    print("      MOBILENETV3-LARGE TEST SET EVALUATION & BENCHMARKING")
    print("=" * 65)
    print(f"Loading checkpoint: {checkpoint_path}")
    print(f"Evaluation Device:  {device}")

    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device)
    print(f"  Model saved from Epoch: {ckpt.get('epoch', 'N/A')}")
    print(f"  Validation Macro-F1:    {ckpt.get('best_val_f1', 0.0)*100:.2f}%")

    # Load DataLoaders
    _, _, test_loader, _ = get_dataloaders(
        metadata_csv=metadata_csv,
        batch_size=batch_size,
        num_workers=2,
        pin_memory=(device.type == "cuda"),
        return_mask=True,
    )

    class_names = [IDX_TO_CLASS[i] for i in range(len(CLASS_TO_IDX))]
    print(f"  Test samples: {len(test_loader.dataset)} across classes {class_names}")

    # Build model & load state dict
    model = build_mobilenet_v3(num_classes=len(CLASS_TO_IDX), pretrained=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    all_preds = []
    all_targets = []
    all_probs = []

    print("\nRunning inference on test dataset...")
    with torch.no_grad():
        for images, targets, _ in tqdm(test_loader, desc="Testing", dynamic_ncols=True):
            images = images.to(device, non_blocking=True)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)

            all_probs.append(probs)
            all_preds.extend(preds)
            all_targets.extend(targets.numpy())

    all_probs = np.vstack(all_probs)
    test_metrics = compute_classification_metrics(
        y_true=all_targets,
        y_pred=all_preds,
        y_probs=all_probs,
        class_names=class_names,
    )

    # Print Report
    print("\n" + "=" * 65)
    print("                 TEST SET PERFORMANCE REPORT")
    print("=" * 65)
    print(f"Overall Accuracy:       {test_metrics['accuracy']*100:.2f}%")
    print(f"Macro F1-Score:         {test_metrics['macro_f1']*100:.2f}%")
    print(f"Macro Precision:        {test_metrics['macro_precision']*100:.2f}%")
    print(f"Macro Recall:           {test_metrics['macro_recall']*100:.2f}%")
    if test_metrics.get("macro_roc_auc") is not None:
        print(f"Macro ROC-AUC:          {test_metrics['macro_roc_auc']:.4f}")
    print("-" * 65)
    print(f"{'Class':<18} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<8}")
    print("-" * 65)
    for cls in class_names:
        stats = test_metrics["per_class"][cls]
        print(
            f"{cls:<18} "
            f"{stats['precision']*100:>8.2f}%    "
            f"{stats['recall']*100:>8.2f}%    "
            f"{stats['f1']*100:>8.2f}%    "
            f"{stats['support']:>6d}"
        )
    print("=" * 65)

    # Save results
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(results_dir, exist_ok=True)

    cm_path = os.path.join(results_dir, "test_confusion_matrix.png")
    plot_confusion_matrix(
        y_true=all_targets,
        y_pred=all_preds,
        class_names=class_names,
        save_path=cm_path,
        title="MobileNetV3-Large Test Confusion Matrix",
        normalize=True,
    )
    print(f"Normalized Confusion Matrix saved to: {cm_path}")

    metrics_path = os.path.join(results_dir, "test_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(test_metrics, f, indent=2)
    print(f"Test metrics JSON saved to:           {metrics_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate MobileNetV3-Large on Test Set")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="models/mobilenet/checkpoints/best_model.pth",
        help="Path to trained model checkpoint",
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Mini-batch size")
    parser.add_argument(
        "--metadata-csv",
        type=str,
        default="dataset_processed/dataset_metadata.csv",
        help="Path to preprocessed dataset metadata CSV",
    )
    args = parser.parse_args()

    evaluate_test_set(
        checkpoint_path=args.checkpoint,
        batch_size=args.batch_size,
        metadata_csv=args.metadata_csv,
    )


if __name__ == "__main__":
    main()
