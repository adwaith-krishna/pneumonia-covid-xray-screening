# ConvNeXt-Tiny Screening Pipeline

This folder contains the complete, dedicated training and evaluation suite for **ConvNeXt-Tiny** on the 3-class Chest X-Ray dataset (`COVID`, `Normal`, `Viral_Pneumonia`).

---

## File Structure

```
models/convnext/
├── __init__.py           # Package exports
├── model.py              # ConvNeXt-Tiny architecture + classifier head + CAM hook
├── train.py              # GPU-accelerated training pipeline with AMP & class weighting
├── evaluate.py           # Test set evaluation, metrics calculation & confusion matrix
├── checkpoints/          # Directory where best_model.pth and last_model.pth are saved
└── results/              # Directory where training curves and metrics are saved
```

---

## How to Train

To start training ConvNeXt-Tiny with default settings (15 epochs, batch size 32, AdamW `lr=1e-4`, AMP mixed precision, and class-weighted Cross-Entropy):

```bash
uv run python models/convnext/train.py
```

### Configurable Hyperparameters

```bash
uv run python models/convnext/train.py \
  --epochs 15 \
  --batch-size 32 \
  --lr 0.0001 \
  --drop-rate 0.2 \
  --patience 5 \
  --num-workers 2
```

---

## How to Evaluate on the Test Set

Once training completes, evaluate the saved `best_model.pth` on the independent **test set** (2,273 unseen images):

```bash
uv run python models/convnext/evaluate.py
```
