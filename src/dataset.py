import os
import random
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms.functional as TF
from torchvision import transforms

# Target class mappings
CLASS_TO_IDX = {
    "COVID": 0,
    "Normal": 1,
    "Viral_Pneumonia": 2
}

IDX_TO_CLASS = {v: k for k, v in CLASS_TO_IDX.items()}

# Computed dataset normalization parameters from offline preprocessing
DEFAULT_MEAN = [0.5363, 0.5363, 0.5363]
DEFAULT_STD = [0.2455, 0.2455, 0.2455]


class JointCXRTransform:
    """
    Joint spatial & photometric transforms for Chest X-Ray images and paired lung masks.
    Ensures spatial augmentations (rotation, scaling, translation) are applied identically
    to both the image and the anatomical mask, while photometric augmentations
    (brightness, contrast) are only applied to the image.
    """

    def __init__(
        self,
        is_training: bool = True,
        mean: List[float] = DEFAULT_MEAN,
        std: List[float] = DEFAULT_STD,
        rotation_degrees: float = 10.0,
        scale_range: Tuple[float, float] = (0.95, 1.05),
        translate_ratio: float = 0.04,
        brightness_jitter: float = 0.1,
        contrast_jitter: float = 0.1,
    ):
        self.is_training = is_training
        self.mean = mean
        self.std = std
        self.rotation_degrees = rotation_degrees
        self.scale_range = scale_range
        self.translate_ratio = translate_ratio
        self.brightness_jitter = brightness_jitter
        self.contrast_jitter = contrast_jitter

    def __call__(
        self, image: Image.Image, mask: Optional[Image.Image] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self.is_training:
            # 1. Joint geometric transform parameters
            angle = random.uniform(-self.rotation_degrees, self.rotation_degrees)
            scale = random.uniform(self.scale_range[0], self.scale_range[1])
            max_dx = self.translate_ratio * image.size[0]
            max_dy = self.translate_ratio * image.size[1]
            translate = (random.uniform(-max_dx, max_dx), random.uniform(-max_dy, max_dy))

            # Apply identical affine transformation to image (bilinear) and mask (nearest)
            image = TF.affine(
                image,
                angle=angle,
                translate=list(translate),
                scale=scale,
                shear=[0.0, 0.0],
                interpolation=TF.InterpolationMode.BILINEAR,
            )

            if mask is not None:
                mask = TF.affine(
                    mask,
                    angle=angle,
                    translate=list(translate),
                    scale=scale,
                    shear=[0.0, 0.0],
                    interpolation=TF.InterpolationMode.NEAREST,
                )

            # 2. Photometric jitter on image only
            brightness_factor = random.uniform(1.0 - self.brightness_jitter, 1.0 + self.brightness_jitter)
            contrast_factor = random.uniform(1.0 - self.contrast_jitter, 1.0 + self.contrast_jitter)
            image = TF.adjust_brightness(image, brightness_factor)
            image = TF.adjust_contrast(image, contrast_factor)

        # 3. Convert image to Tensor & Normalize
        img_tensor = TF.to_tensor(image)  # [3, H, W] in [0.0, 1.0]
        img_tensor = TF.normalize(img_tensor, mean=self.mean, std=self.std)

        # 4. Convert mask to binary Tensor [1, H, W] in {0.0, 1.0}
        mask_tensor = None
        if mask is not None:
            mask_arr = np.array(mask, dtype=np.float32)
            mask_binary = (mask_arr > 127).astype(np.float32)
            mask_tensor = torch.from_numpy(mask_binary).unsqueeze(0)  # [1, H, W]

        return img_tensor, mask_tensor


class ChestXRayDataset(Dataset):
    """
    PyTorch Dataset for Chest X-Ray images, class labels, and aligned anatomical lung masks.
    """

    def __init__(
        self,
        metadata_csv: Union[str, pd.DataFrame] = "dataset_processed/dataset_metadata.csv",
        split: str = "train",
        transform: Optional[JointCXRTransform] = None,
        return_mask: bool = True,
    ):
        super().__init__()
        self.split = split
        self.return_mask = return_mask

        if isinstance(metadata_csv, str):
            if not os.path.exists(metadata_csv):
                raise FileNotFoundError(f"Metadata file not found at: {metadata_csv}")
            df = pd.read_csv(metadata_csv)
        else:
            df = metadata_csv.copy()

        if split is not None:
            df = df[df["split"] == split].reset_index(drop=True)

        if len(df) == 0:
            raise ValueError(f"No records found for split '{split}' in metadata.")

        self.df = df
        self.samples = self.df.to_dict("records")

        # Fallback default transform if none provided
        if transform is None:
            self.transform = JointCXRTransform(is_training=(split == "train"))
        else:
            self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def get_labels(self) -> List[int]:
        """Returns integer class labels for all samples in the dataset."""
        return [CLASS_TO_IDX[s["class_name"]] for s in self.samples]

    def __getitem__(self, idx: int):
        sample = self.samples[idx]

        # Load processed image
        img_path = sample["proc_img_path"]
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found at {img_path}")
        image = Image.open(img_path).convert("RGB")

        # Load processed mask if requested
        mask = None
        if self.return_mask:
            mask_path = sample["proc_mask_path"]
            if not os.path.exists(mask_path):
                raise FileNotFoundError(f"Mask not found at {mask_path}")
            mask = Image.open(mask_path).convert("L")

        # Apply transforms
        img_tensor, mask_tensor = self.transform(image, mask)
        label_idx = CLASS_TO_IDX[sample["class_name"]]
        label_tensor = torch.tensor(label_idx, dtype=torch.long)

        if self.return_mask:
            return img_tensor, label_tensor, mask_tensor
        return img_tensor, label_tensor


def compute_class_weights(
    metadata_csv: Union[str, pd.DataFrame] = "dataset_processed/dataset_metadata.csv",
    split: str = "train",
) -> torch.FloatTensor:
    """
    Computes balanced class weights for CrossEntropyLoss:
    weight[c] = total_samples / (num_classes * count[c])
    """
    if isinstance(metadata_csv, str):
        df = pd.read_csv(metadata_csv)
    else:
        df = metadata_csv

    if split is not None:
        df = df[df["split"] == split]

    counts = df["class_name"].value_counts().to_dict()
    total = len(df)
    n_classes = len(CLASS_TO_IDX)

    weights = []
    for cls_name in sorted(CLASS_TO_IDX, key=CLASS_TO_IDX.get):
        count = counts.get(cls_name, 1)
        w = total / (n_classes * count)
        weights.append(w)

    return torch.tensor(weights, dtype=torch.float32)


def get_dataloaders(
    metadata_csv: str = "dataset_processed/dataset_metadata.csv",
    batch_size: int = 32,
    num_workers: int = 2,
    pin_memory: bool = True,
    use_weighted_sampler: bool = False,
    return_mask: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader, torch.FloatTensor]:
    """
    Creates and returns train, val, and test DataLoaders along with balanced class weights.

    Args:
        metadata_csv: Path to metadata CSV file.
        batch_size: Mini-batch size.
        num_workers: Number of background data loader processes.
        pin_memory: Pin memory for GPU transfer speedup.
        use_weighted_sampler: If True, uses WeightedRandomSampler for train_loader
                              instead of random shuffling.
        return_mask: Whether each batch returns (images, labels, masks) or (images, labels).

    Returns:
        (train_loader, val_loader, test_loader, class_weights_tensor)
    """
    class_weights = compute_class_weights(metadata_csv, split="train")

    train_transform = JointCXRTransform(is_training=True)
    val_test_transform = JointCXRTransform(is_training=False)

    train_dataset = ChestXRayDataset(
        metadata_csv=metadata_csv,
        split="train",
        transform=train_transform,
        return_mask=return_mask,
    )
    val_dataset = ChestXRayDataset(
        metadata_csv=metadata_csv,
        split="val",
        transform=val_test_transform,
        return_mask=return_mask,
    )
    test_dataset = ChestXRayDataset(
        metadata_csv=metadata_csv,
        split="test",
        transform=val_test_transform,
        return_mask=return_mask,
    )

    sampler = None
    shuffle = True
    if use_weighted_sampler:
        train_labels = train_dataset.get_labels()
        sample_weights = [class_weights[lbl].item() for lbl in train_labels]
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )
        shuffle = False

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader, class_weights
