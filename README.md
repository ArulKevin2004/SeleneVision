# Lunar Crater Classification — Classical CV + SVM

This project is adapted specifically for the uploaded `archive.zip` dataset.

## Important dataset facts

The archive contains a mixed Moon/Mars YOLO object-detection dataset.

- Images are 640 × 640.
- YOLO labels use `class x_center y_center width height` in normalized coordinates.
- Mars filenames contain `mars_crater`.
- Lunar filenames look like `01_png...`, `010_png...`, etc.
- `best.pt` is a deep-learning model and is **not used** in this project.

The preprocessing script automatically keeps only lunar images and converts the crater bounding boxes into a classical binary image-classification dataset.

## Folder setup

Extract the downloaded archive so you have:

```text
project/
├── craters/
│   ├── train/
│   │   ├── images/
│   │   └── labels/
│   ├── valid/
│   │   ├── images/
│   │   └── labels/
│   └── test/
│       ├── images/
│       └── labels/
├── prepare_dataset.py
├── features.py
├── train.py
└── experiment.py
```

## 1. Install

```bash
pip install -r requirements.txt
```

## 2. Convert YOLO annotations into crater/non-crater patches

```bash
python prepare_dataset.py --source craters
```

This creates:

```text
processed_dataset/
├── train/
│   ├── crater/
│   └── non_crater/
├── valid/
│   ├── crater/
│   └── non_crater/
└── test/
    ├── crater/
    └── non_crater/
```

The original train/valid/test split is preserved. This is important because patches from one 640×640 source image must not appear in both training and testing.

## 3. Train

```bash
python train.py
```

Pipeline:

```text
Lunar optical image
       ↓
YOLO crater annotations
       ↓
Crater / non-crater patch generation
       ↓
Resize to 128×128
       ↓
CLAHE
       ↓
Gaussian blur
       ↓
HOG features
       ↓
Linear SVM
       ↓
Crater / Non-crater
```

## 4. Run your CV experiment

```bash
python experiment.py
```

It compares:

1. HOG only
2. CLAHE + HOG
3. CLAHE + Gaussian blur + HOG
4. CLAHE + Gaussian blur + HOG + LBP

Use accuracy, precision, recall and F1 in your report.

## Why this is good for a CV lab

You are not simply loading a ready-made classification dataset. You perform:

- YOLO annotation parsing
- crater-region extraction
- negative-region sampling
- image resizing
- contrast enhancement (CLAHE)
- smoothing
- handcrafted feature extraction (HOG/LBP)
- classical machine learning (SVM)

No deep learning is used.
