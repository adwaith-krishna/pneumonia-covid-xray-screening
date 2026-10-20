# DenseNet-121 Screening Pipeline

This folder contains the complete, dedicated training and evaluation suite for **DenseNet-121** on the 3-class Chest X-Ray dataset (`COVID`, `Normal`, `Viral_Pneumonia`).

---

## File Structure

```
models/densenet/
├── __init__.py           # Package exports
├── model.py              # DenseNet-121 architecture + classifier head + CAM hook
├── train.py              # GPU-accelerated training pipeline with AMP & class weighting
├── evaluate.py           # Test set evaluation, metrics calculation & confusion matrix
├── checkpoints/          # Directory where best_model.pth and last_model.pth are saved
└── results/              # Directory where training curves and metrics are saved
```

---

## How to Train

To start training DenseNet-121 with default settings (15 epochs, batch size 32, AdamW `lr=1e-4`, AMP mixed precision, and class-weighted Cross-Entropy):

```bash
uv run python models/densenet/train.py
```

### Configurable Hyperparameters

You can customize training with command-line arguments:

```bash
uv run python models/densenet/train.py \
  --epochs 20 \
  --batch-size 32 \
  --lr 0.0001 \
  --drop-rate 0.2 \
  --patience 5 \
  --num-workers 2
```

| Argument | Default | Description |
| :--- | :---: | :--- |
| `--epochs` | `15` | Total training epochs |
| `--batch-size` | `32` | Mini-batch size (tuned for 6GB RTX 4050 VRAM) |
| `--lr` | `1e-4` | Peak learning rate for AdamW |
| `--weight-decay`| `1e-4` | L2 regularization penalty |
| `--drop-rate` | `0.2` | Dropout rate in classifier head |
| `--patience` | `5` | Early stopping patience monitoring Validation Macro F1 |
| `--num-workers` | `2` | DataLoader worker processes |
| `--no-amp` | `False` | Flag to disable Automatic Mixed Precision (`torch.amp`) |
| `--no-class-weights` | `False` | Flag to disable class-frequency reweighting in loss |

---

## Checkpoints & Outputs

During and after training, the following artifacts are automatically generated:
1. **Best Model**: `models/densenet/checkpoints/best_model.pth` (saved whenever validation Macro F1 reaches a new high).
2. **Last Checkpoint**: `models/densenet/checkpoints/last_model.pth` (saved after every epoch).
3. **Training Curves**: `models/densenet/results/training_curves.png` (Training & Validation loss, Accuracy, and Macro F1).
4. **Metric History**: `models/densenet/results/training_history.json`.

---

## How to Evaluate on the Test Set

Once training completes, evaluate the saved `best_model.pth` on the independent **test set** (2,273 unseen images):

```bash
uv run python models/densenet/evaluate.py
```

This will output:
- Per-class Precision, Recall, and F1-score for `COVID`, `Normal`, and `Viral_Pneumonia`.
- Macro F1 and ROC-AUC scores.
- High-resolution normalized Confusion Matrix saved to `models/densenet/results/test_confusion_matrix.png`.
