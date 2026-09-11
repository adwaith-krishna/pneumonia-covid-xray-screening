import os
import sys
import json
import numpy as np
import pandas as pd
import cv2
from glob import glob
from sklearn.model_selection import train_test_split
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

SOURCE_DIR = "COVID-19_Radiography_Dataset"
TARGET_DIR = "dataset_processed"
TARGET_SIZE = (224, 224)
RANDOM_SEED = 42

CLASSES_MAP = {
    "COVID": "COVID",
    "Normal": "Normal",
    "Viral Pneumonia": "Viral_Pneumonia"
}

def process_single_item(item):
    """
    Process one image and its corresponding mask.
    """
    orig_img_path = item["orig_img_path"]
    orig_mask_path = item["orig_mask_path"]
    proc_img_path = item["proc_img_path"]
    proc_mask_path = item["proc_mask_path"]

    # 1. Read CXR as grayscale
    img_gray = cv2.imread(orig_img_path, cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        raise ValueError(f"Could not read image at {orig_img_path}")

    # 2. Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_clahe = clahe.apply(img_gray)

    # 3. Resize image to target resolution (224x224) using bicubic interpolation
    img_resized = cv2.resize(img_clahe, TARGET_SIZE, interpolation=cv2.INTER_CUBIC)

    # 4. Convert grayscale to 3-channel RGB representation
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB)

    # 5. Read and resize matching lung mask (use nearest neighbor for crisp binary boundaries)
    mask_gray = cv2.imread(orig_mask_path, cv2.IMREAD_GRAYSCALE)
    if mask_gray is None:
        raise ValueError(f"Could not read mask at {orig_mask_path}")
    mask_resized = cv2.resize(mask_gray, TARGET_SIZE, interpolation=cv2.INTER_NEAREST)
    _, mask_binary = cv2.threshold(mask_resized, 127, 255, cv2.THRESH_BINARY)

    # 6. Save processed outputs
    cv2.imwrite(proc_img_path, cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(proc_mask_path, mask_binary)

    # Pixel statistics for normalization reference
    norm_vals = img_rgb.astype(np.float32) / 255.0
    mean_val = float(np.mean(norm_vals))
    std_val = float(np.std(norm_vals))

    return {
        "sample_id": item["sample_id"],
        "class_name": item["class_name"],
        "split": item["split"],
        "orig_img_path": orig_img_path,
        "orig_mask_path": orig_mask_path,
        "proc_img_path": proc_img_path,
        "proc_mask_path": proc_mask_path,
        "mean": mean_val,
        "std": std_val
    }

def main():
    print("=" * 60, flush=True)
    print("Chest X-Ray Preprocessing Pipeline (Offline)", flush=True)
    print("=" * 60, flush=True)

    # Collect dataset samples
    records = []
    for raw_cls, clean_cls in CLASSES_MAP.items():
        img_dir = os.path.join(SOURCE_DIR, raw_cls, "images")
        mask_dir = os.path.join(SOURCE_DIR, raw_cls, "masks")

        img_files = sorted(glob(os.path.join(img_dir, "*.png")))
        print(f"Found {len(img_files)} images for class: {raw_cls} (mapped to '{clean_cls}')", flush=True)

        for f in img_files:
            fname = os.path.basename(f)
            mask_file = os.path.join(mask_dir, fname)
            if not os.path.exists(mask_file):
                raise FileNotFoundError(f"Missing corresponding mask for {f}")
            sample_id = os.path.splitext(fname)[0]
            records.append({
                "sample_id": sample_id,
                "raw_class": raw_cls,
                "class_name": clean_cls,
                "orig_img_path": f,
                "orig_mask_path": mask_file
            })

    df = pd.DataFrame(records)
    print(f"\nTotal dataset samples across 3 classes: {len(df)}", flush=True)

    # Stratified Train/Val/Test Split (70% Train, 15% Val, 15% Test)
    print("\nPerforming Stratified 70/15/15 Train-Val-Test Split...", flush=True)
    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        random_state=RANDOM_SEED,
        stratify=df["class_name"]
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=RANDOM_SEED,
        stratify=temp_df["class_name"]
    )

    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()

    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"

    full_split_df = pd.concat([train_df, val_df, test_df], ignore_index=True)

    print("\nDataset Split Summary:", flush=True)
    split_summary = pd.crosstab(full_split_df["split"], full_split_df["class_name"])
    print(split_summary, flush=True)

    # Pre-create all destination directories
    for split in ["train", "val", "test"]:
        for cls_name in CLASSES_MAP.values():
            os.makedirs(os.path.join(TARGET_DIR, split, cls_name), exist_ok=True)
            os.makedirs(os.path.join(TARGET_DIR, "masks", split, cls_name), exist_ok=True)

    items_to_process = []
    for _, row in full_split_df.iterrows():
        split = row["split"]
        cls_name = row["class_name"]
        sample_id = row["sample_id"]
        fname = f"{sample_id}.png"

        proc_img_path = os.path.join(TARGET_DIR, split, cls_name, fname)
        proc_mask_path = os.path.join(TARGET_DIR, "masks", split, cls_name, fname)

        items_to_process.append({
            "sample_id": sample_id,
            "class_name": cls_name,
            "split": split,
            "orig_img_path": row["orig_img_path"],
            "orig_mask_path": row["orig_mask_path"],
            "proc_img_path": proc_img_path,
            "proc_mask_path": proc_mask_path
        })

    # Execute image processing with ThreadPoolExecutor
    workers = min(16, (os.cpu_count() or 4) * 2)
    print(f"\nProcessing {len(items_to_process)} images and masks using {workers} threads...", flush=True)

    results = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(process_single_item, item) for item in items_to_process]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Preprocessing"):
            results.append(future.result())

    # Build and save metadata CSV
    meta_df = pd.DataFrame(results)
    csv_path = os.path.join(TARGET_DIR, "dataset_metadata.csv")
    meta_df.to_csv(csv_path, index=False)
    print(f"\nSaved metadata index to {csv_path}", flush=True)

    # Compute training set normalization statistics
    train_meta = meta_df[meta_df["split"] == "train"]
    dataset_mean = float(train_meta["mean"].mean())
    dataset_std = float(train_meta["std"].mean())

    summary_info = {
        "target_resolution": TARGET_SIZE,
        "channels": 3,
        "clahe_parameters": {
            "clipLimit": 2.0,
            "tileGridSize": [8, 8]
        },
        "classes": list(CLASSES_MAP.values()),
        "split_counts": {
            split: {
                cls: int((meta_df[(meta_df["split"] == split) & (meta_df["class_name"] == cls)]).shape[0])
                for cls in CLASSES_MAP.values()
            }
            for split in ["train", "val", "test"]
        },
        "total_samples": len(meta_df),
        "train_normalization": {
            "mean": [round(dataset_mean, 4)] * 3,
            "std": [round(dataset_std, 4)] * 3
        }
    }

    summary_json_path = os.path.join(TARGET_DIR, "dataset_summary.json")
    with open(summary_json_path, "w") as f:
        json.dump(summary_info, f, indent=2)
    print(f"Saved dataset summary configuration to {summary_json_path}", flush=True)
    print("\nPreprocessing Completed Successfully!", flush=True)

if __name__ == "__main__":
    main()
