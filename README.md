

# Puffin Counting Demo with YOLOv8

This repository contains a simple demonstration project for counting Puffins using **YOLOv8**. It was developed as the final exam project for the **"Machine Learning for Computer Vision"** course, part of the **2026 Spring International Top-Notch Innovative Talent Training Program**.

## Table of Contents

- [Requirements](#0-requirements)
- [Dataset](#1-dataset)
- [Training](#2-trainpy)
- [Application](#3-applypy)
- [Contributor](#contributor)

---

## 0. Requirements

To run this project, please install the necessary dependencies listed in `requirements.txt`. The project relies on [PyTorch](https://pytorch.org/) and [Ultralytics YOLO](https://github.com/ultralytics).

```bash
pip install -r requirements.txt
```

**Key Dependencies:**
- `ultralytics==8.4.53`
- `torch==2.8.0`
- `opencv-python`
- `numpy`

---

## 1. Dataset

> **Note:** This dataset is a **very simplified demo** intended for proof-of-concept purposes only. It is extremely small and not suitable for training a robust, production-level model.

The dataset consists of just **30 images** of Puffins collected via Google Search, along with their corresponding YOLO-format annotations.

> **Copyright:** All images were downloaded from Google Search. If there are any copyright infringement issues, please contact the contributor to have them removed.

### Configuration (`dataset.yaml`)
The dataset configuration is defined in `dataset.yaml`, pointing to the training, validation, and test directories.

### Annotation Guidelines
The dataset follows specific rules to ensure data quality:
- A head is annotated **only if at least 50% of its area is visible** and the annotator is confident it is a puffin.
- Heads with less than 50% visibility or those that cannot be reliably identified (e.g., occluded by vegetation or facing away) are treated as **ignore regions** and left unlabelled.

---

## 2. train.py

This script handles the training process of the YOLOv8 model.

**Features:**
- Loads the `yolo26n.pt` pre-trained weights.
- Trains for 50 epochs with a batch size of 16.
- Includes data augmentation (degrees, flip left-right, mosaic).
- Automatically evaluates the model on the validation set and prints test metrics (mAP50, Precision, Recall) upon completion.

**Usage:**
```bash
python train.py
```

---

## 3. apply.py

This script is used for inference on the test set and calculates specific counting metrics.

**Functionality:**
- Loads the best model weights (`best.pt`) generated from training.
- Iterates through the test images to predict puffin counts.
- Compares predictions against Ground Truth labels.
- Calculates and prints the following error metrics:
    - Mean Relative Counting Error
    - MAE (Mean Absolute Error)
    - MSE (Mean Squared Error)
    - RMSE (Root Mean Squared Error)

**Usage:**
```bash
python apply.py
```

---

## Contributor

**Zijie Guo**
*Spring 2026*
