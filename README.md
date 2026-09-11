# pneumonia-covid-xray-screening

Automated Pneumonia & COVID-19 screening and interpretability analysis from Chest X-Rays.

## Overview
This project performs multi-class classification across three clinical categories:
- **Normal**
- **COVID-19**
- **Viral Pneumonia**

The core objective is to evaluate diagnostic accuracy and model faithfulness by comparing modern deep learning architectures (**DenseNet-121**, **ConvNeXt**, and **MobileNetV3**) and validating interpretability methods (**Grad-CAM** vs. **Integrated Gradients**) against ground-truth anatomical lung masks to verify whether models learn true pulmonary pathology or spurious hospital artifacts (text markers, monitoring leads).

## Repository Structure
```
├── scripts/
│   ├── preprocess_dataset.py    # Multi-threaded offline preprocessing pipeline
│   └── verify_preprocessing.py  # Integrity checks & visual verification collage
├── .gitignore                   # Excludes raw/processed data & heavy artifacts
└── README.md
```

## Dataset & Preprocessing Pipeline
- **Dataset Source**: COVID-19 Radiography Database
- **Standardized Dimensions**: 224x224 RGB PNGs
- **Contrast Enhancement**: Contrast Limited Adaptive Histogram Equalization (CLAHE, `clipLimit=2.0`, `tileGridSize=(8,8)`)
- **Mask Alignment**: 224x224 nearest-neighbor binarized lung masks
- **Data Splits**: 70% Train, 15% Validation, 15% Test (Stratified)
  - Train: 10,607 images (`COVID`: 2,531 | `Normal`: 7,134 | `Viral Pneumonia`: 942)
  - Val: 2,273 images (`COVID`: 542 | `Normal`: 1,529 | `Viral Pneumonia`: 202)
  - Test: 2,273 images (`COVID`: 543 | `Normal`: 1,529 | `Viral Pneumonia`: 201)

## Usage
### 1. Run Offline Preprocessing
```bash
python scripts/preprocess_dataset.py
```
Outputs the preprocessed images and aligned masks into `dataset_processed/`.

### 2. Verify Preprocessed Data
```bash
python scripts/verify_preprocessing.py
```
Validates file integrity and generates visual comparison grids.
