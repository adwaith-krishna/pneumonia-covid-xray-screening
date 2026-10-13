import sys
import os
import torch
import numpy as np

# Ensure root directory is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.dataset import (
    ChestXRayDataset,
    CLASS_TO_IDX,
    IDX_TO_CLASS,
    compute_class_weights,
    get_dataloaders,
    DEFAULT_MEAN,
    DEFAULT_STD,
)

def test_dataset_pipeline():
    print("=" * 60)
    print("Verifying PyTorch Data Loading & Preprocessing Pipeline")
    print("=" * 60)

    # 1. Test class weight computation
    print("\n1. Testing Class Weights Computation...")
    weights = compute_class_weights()
    print("Class mappings:", CLASS_TO_IDX)
    for cls_name, idx in CLASS_TO_IDX.items():
        print(f"  Class '{cls_name}' (idx {idx}): Weight = {weights[idx]:.4f}")
    assert len(weights) == 3, "Expected 3 class weights"
    assert torch.all(weights > 0), "All weights must be positive"
    # Viral Pneumonia is minority class, so its weight should be highest
    assert weights[CLASS_TO_IDX["Viral_Pneumonia"]] > weights[CLASS_TO_IDX["Normal"]]
    print("  Class weighting passed!")

    # 2. Test Single Dataset Access
    print("\n2. Testing ChestXRayDataset items (Train, Val, Test)...")
    for split in ["train", "val", "test"]:
        ds = ChestXRayDataset(split=split, return_mask=True)
        print(f"  [{split.upper()}] Total samples: {len(ds)}")
        img, label, mask = ds[0]

        assert img.shape == (3, 224, 224), f"Unexpected img shape: {img.shape}"
        assert mask.shape == (1, 224, 224), f"Unexpected mask shape: {mask.shape}"
        assert label.ndim == 0, f"Label must be a scalar tensor, got {label.shape}"
        assert 0 <= label.item() <= 2, f"Label index out of bounds: {label.item()}"

        # Mask binary validation
        unique_mask_vals = torch.unique(mask).tolist()
        assert all(v in [0.0, 1.0] for v in unique_mask_vals), f"Mask must be binary: {unique_mask_vals}"

    print("  Dataset item shapes, dtypes, and mask values passed!")

    # 3. Test DataLoader mini-batch execution
    print("\n3. Testing DataLoader Batch Generation...")
    batch_size = 16
    train_loader, val_loader, test_loader, cls_weights = get_dataloaders(
        batch_size=batch_size,
        num_workers=0,  # Single-process for quick verification
        use_weighted_sampler=True,
        return_mask=True,
    )

    # Fetch one batch from train loader
    images, labels, masks = next(iter(train_loader))
    print(f"  Train Batch - Images: {images.shape}, Labels: {labels.shape}, Masks: {masks.shape}")
    assert images.shape == (batch_size, 3, 224, 224)
    assert labels.shape == (batch_size,)
    assert masks.shape == (batch_size, 1, 224, 224)

    # Check normalization range
    print(f"  Image Batch Stats - Min: {images.min().item():.3f}, Max: {images.max().item():.3f}, Mean: {images.mean().item():.3f}")

    # Fetch one batch from val loader
    val_images, val_labels, val_masks = next(iter(val_loader))
    print(f"  Val Batch   - Images: {val_images.shape}, Labels: {val_labels.shape}, Masks: {val_masks.shape}")
    assert val_images.shape == (batch_size, 3, 224, 224)

    print("\n" + "=" * 60)
    print("All Data Loading & Preprocessing Checks Passed Successfully!")
    print("=" * 60)

if __name__ == "__main__":
    test_dataset_pipeline()
