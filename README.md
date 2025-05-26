# 🩺 Pneumonia Detection from Chest X-Ray Images using DenseNet-121

A deep learning project for automatic detection of pneumonia using chest X-ray images. This model utilizes **DenseNet-121**, **Focal Loss**, and **Test-Time Augmentation (TTA)** to enhance classification accuracy, especially on imbalanced datasets.

## 🚀 Features

- ✅ DenseNet-121 with transfer learning
- 🎯 Class imbalance handled using Focal Loss
- 🔁 Robust predictions using Test-Time Augmentation (TTA)
- 🔍 Image augmentation using Albumentations
- ⚡ Real-time prediction support

---

## 🧠 Model Overview

- **Architecture**: DenseNet-121
- **Input Size**: 224×224×3
- **Loss Function**: Focal Loss
- **Optimizer**: AdamW
- **Epochs**: 30
- **Batch Size**: 32

---

## 📁 Dataset

- **Source**: [Kaggle Chest X-ray Pneumonia Dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- **Classes**: `Pneumonia` and `Normal`
- **Total Images**: 5,863

---

## 📊 Results

| Metric     | Pneumonia | Normal |
|------------|-----------|--------|
| Precision  | 88%       | 93%    |
| Recall     | 97%       | 79%    |
| F1-Score   | 92%       | 86%    |
| Accuracy   | **91%**   |        |
