import os
import json
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROCESSED_DIR = "dataset_processed"

def main():
    print("=" * 60)
    print("Verifying Processed Dataset Integrity")
    print("=" * 60)

    # 1. Check metadata and summary
    meta_path = os.path.join(PROCESSED_DIR, "dataset_metadata.csv")
    summary_path = os.path.join(PROCESSED_DIR, "dataset_summary.json")

    assert os.path.exists(meta_path), f"Missing {meta_path}"
    assert os.path.exists(summary_path), f"Missing {summary_path}"

    df = pd.read_csv(meta_path)
    with open(summary_path, "r") as f:
        summary = json.load(f)

    print(f"Total entries in metadata CSV: {len(df)}")
    print("Classes found:", df["class_name"].unique().tolist())
    print("Splits found:", df["split"].unique().tolist())

    crosstab = pd.crosstab(df["split"], df["class_name"])
    print("\nSamples per Split & Class:")
    print(crosstab)

    # 2. Check image & mask files integrity
    print("\nValidating files on disk...")
    corrupt_images = 0
    corrupt_masks = 0
    non_binary_masks = 0

    # Sample check 500 images across the dataset for fast thorough verification
    sample_indices = np.random.choice(len(df), size=min(500, len(df)), replace=False)
    
    for idx in sample_indices:
        row = df.iloc[idx]
        img_path = row["proc_img_path"]
        mask_path = row["proc_mask_path"]

        if not os.path.exists(img_path):
            corrupt_images += 1
            continue
        if not os.path.exists(mask_path):
            corrupt_masks += 1
            continue

        img = cv2.imread(img_path)
        if img is None or img.shape != (224, 224, 3):
            corrupt_images += 1

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None or mask.shape != (224, 224):
            corrupt_masks += 1
        else:
            vals = np.unique(mask)
            if not np.all(np.isin(vals, [0, 255])):
                non_binary_masks += 1

    print(f"Sample Check Results (Checked {len(sample_indices)} random items):")
    print(f"  Corrupt/Missing Images: {corrupt_images}")
    print(f"  Corrupt/Missing Masks: {corrupt_masks}")
    print(f"  Non-binary Masks: {non_binary_masks}")

    assert corrupt_images == 0, "Found corrupt or missing images!"
    assert corrupt_masks == 0, "Found corrupt or missing masks!"
    assert non_binary_masks == 0, "Found non-binary masks!"

    print("\nIntegrity check passed with zero errors!")

    # 3. Generate Visual Comparison Figure
    print("\nGenerating sample visual inspection collage...")
    classes = ["COVID", "Normal", "Viral_Pneumonia"]
    fig, axes = plt.subplots(len(classes), 4, figsize=(14, 10))

    for row_idx, cls_name in enumerate(classes):
        cls_sample = df[(df["class_name"] == cls_name) & (df["split"] == "test")].iloc[0]

        # Raw image
        raw_img = cv2.imread(cls_sample["orig_img_path"], cv2.IMREAD_GRAYSCALE)
        # Processed image (RGB)
        proc_bgr = cv2.imread(cls_sample["proc_img_path"])
        proc_rgb = cv2.cvtColor(proc_bgr, cv2.COLOR_BGR2RGB)
        # Mask
        mask = cv2.imread(cls_sample["proc_mask_path"], cv2.IMREAD_GRAYSCALE)

        # Create overlay (translucent red on lung fields)
        overlay = proc_rgb.copy()
        lung_pixels = mask > 127
        overlay[lung_pixels, 0] = np.clip(overlay[lung_pixels, 0] * 0.6 + 255 * 0.4, 0, 255)
        overlay[lung_pixels, 1] = np.clip(overlay[lung_pixels, 1] * 0.6, 0, 255)
        overlay[lung_pixels, 2] = np.clip(overlay[lung_pixels, 2] * 0.6, 0, 255)

        # Plot Row
        axes[row_idx, 0].imshow(raw_img, cmap="gray")
        axes[row_idx, 0].set_title(f"Raw {cls_name} (299x299)", fontsize=11, fontweight="bold")
        axes[row_idx, 0].axis("off")

        axes[row_idx, 1].imshow(proc_rgb)
        axes[row_idx, 1].set_title(f"CLAHE Enhanced (224x224)", fontsize=11, fontweight="bold")
        axes[row_idx, 1].axis("off")

        axes[row_idx, 2].imshow(mask, cmap="gray")
        axes[row_idx, 2].set_title("Aligned Lung Mask", fontsize=11, fontweight="bold")
        axes[row_idx, 2].axis("off")

        axes[row_idx, 3].imshow(overlay)
        axes[row_idx, 3].set_title("Mask Overlay (ROI)", fontsize=11, fontweight="bold")
        axes[row_idx, 3].axis("off")

    plt.suptitle("Chest X-Ray Preprocessing & Mask Alignment Verification", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()

    out_fig_path = os.path.join(PROCESSED_DIR, "preprocessing_samples.png")
    plt.savefig(out_fig_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved visual verification figure to: {out_fig_path}")

    print("\nSummary Information:")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
