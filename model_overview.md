# Face Mask Detection — YOLOv8 Classification Model Overview

## Project Goal

The goal of this project is to build a **YOLOv8 Classification Model** that can look at a photo of a person's face and classify it into one of three categories:

- **Masked** — The person is wearing a face mask properly.
- **Unmasked** — The person is not wearing any face mask.
- **Improperly Worn** — The person has a mask but it is not covering the nose and mouth correctly (e.g., pulled under the chin).

---

## Dataset

The dataset consists of **535 images** split across the three classes. The images are divided into:

- **Training set (80%):** ~428 images — used to teach the model.
- **Validation set (20%):** ~107 images — used to test how well the model performs on images it has never seen before.

The dataset is organized into folders matching YOLOv8's expected structure:

```
Dataset_YOLOv8/
├── train/
│   ├── Improperly Wearing Facemask/
│   ├── Not Wearing Facemask/
│   └── Wearing Facemask/
└── val/
    ├── Improperly Wearing Facemask/
    ├── Not Wearing Facemask/
    └── Wearing Facemask/
```

This is a relatively **small dataset**, which makes transfer learning from a pretrained model essential for achieving good performance.

---

## Model Architecture

The model uses **YOLOv8 Nano Classification** (`yolov8n-cls`), a lightweight classification model from the Ultralytics YOLO family. Unlike the previous custom CNN approach, this model comes pretrained on **ImageNet** (1000 classes, ~1.2M images) and is fine-tuned on our face mask dataset.

### Why YOLOv8 Classification?

| Aspect | Custom CNN (Previous) | YOLOv8 Classification (Current) |
|--------|----------------------|--------------------------------|
| **Architecture** | 2-layer Conv2D + Dense | Modified CSPNet backbone with classification head |
| **Parameters** | ~1.7M (untrained) | ~2.7M (pretrained on ImageNet) |
| **Transfer Learning** | None | ImageNet pretrained weights |
| **Feature Extraction** | Learns from scratch on 428 images | Leverages features learned from 1.2M images |
| **Augmentation** | Manual ImageDataGenerator | Built-in (mosaic, mixup, HSV jitter, flip, etc.) |
| **Training Optimizations** | Manual setup | Auto LR scheduling, early stopping, best checkpoint |

### Architecture Details

YOLOv8n-cls uses a **CSPNet (Cross Stage Partial Network)** backbone, which is significantly more powerful than a simple 2-layer CNN:

1. **Backbone**: CSPDarknet with cross-stage partial connections — efficiently extracts multi-scale features while keeping the model small.
2. **Classification Head**: Global Average Pooling → Linear layer with Softmax — maps extracted features to 3 class probabilities.

The key advantage is **transfer learning**: the backbone starts with weights trained on ImageNet, meaning it already knows how to detect edges, textures, shapes, and objects. We only need to fine-tune these features for our specific mask classification task.

---

## Techniques Used to Improve Generalization

### 1. Transfer Learning (Pretrained Weights)

**What it does:** Instead of training from random weights, the model starts with weights learned from classifying 1000 categories on 1.2 million ImageNet images.

**Why it helps:** With only 428 training images, learning visual features from scratch is extremely difficult. Transfer learning provides a strong starting point — the model already understands basic visual concepts (edges, textures, shapes) and only needs to learn how to apply them to face masks.

### 2. Built-in Data Augmentation

**What it does:** YOLOv8 automatically applies a suite of augmentations during training, including:

| Technique | What It Simulates |
|-----------|-------------------|
| **HSV Jitter** | Different lighting, white balance, and camera settings |
| **Horizontal Flip** | Person facing left or right |
| **Rotation** | Natural head tilts |
| **Scale Variation** | Different distances from the camera |
| **Translation** | Different positions in the frame |
| **Mosaic** | Combining 4 images for context diversity |
| **Mixup** | Blending two images for smoother decision boundaries |

**Why it helps:** These augmentations artificially increase the variety of training data, forcing the model to learn general patterns rather than memorizing specific images.

### 3. Dropout

**What it does:** During training, 20% of neurons are randomly disabled in each forward pass.

**Why it helps:** Prevents the model from over-relying on specific neurons, encouraging distributed and redundant feature representations that generalize better to unseen data.

### 4. Learning Rate Scheduling

**What it does:** YOLOv8 uses a **warmup + cosine annealing** schedule:
- **Warmup** (first 3 epochs): Learning rate gradually increases from near-zero to the target rate, preventing unstable early training.
- **Cosine Annealing** (remaining epochs): Learning rate smoothly decreases following a cosine curve, allowing fine-grained optimization in later epochs.

**Why it helps:** Prevents overshooting in early training and allows precise convergence in later epochs.

### 5. Early Stopping

**What it does:** Training automatically stops if validation performance doesn't improve for 10 consecutive epochs (`patience=10`).

**Why it helps:** Prevents overfitting by stopping training at the optimal point rather than continuing until the model memorizes the training data.

---

## Training Configuration

| Setting | Value | Purpose |
|---------|-------|---------|
| **Base Model** | yolov8n-cls.pt | Nano classification model pretrained on ImageNet |
| **Optimizer** | Adam | Adaptive optimizer with automatic per-parameter learning rate |
| **Initial Learning Rate** | 0.001 | Starting learning rate before cosine annealing |
| **Loss Function** | Cross Entropy | Standard loss for multi-class classification |
| **Epochs** | 50 (max) | Maximum training epochs (early stopping may trigger sooner) |
| **Patience** | 10 | Early stopping patience — stops if no improvement for 10 epochs |
| **Batch Size** | 32 | Number of images processed per weight update |
| **Image Size** | 128 × 128 pixels | All images resized to this standard size |
| **Dropout** | 0.2 | 20% dropout for regularization |

---

## Results

*Run the notebook to populate results.*

---

## Summary

This model demonstrates how **transfer learning with YOLOv8** dramatically improves classification performance on small datasets compared to training a custom CNN from scratch. The key advantages are:

- **Transfer learning** provides a strong foundation of visual features learned from millions of images.
- **Built-in augmentation** automatically increases training data variety without manual configuration.
- **Automatic training optimizations** (LR scheduling, early stopping, best checkpoint) eliminate manual tuning.
- **Simpler code** — the entire training pipeline is ~10 lines of code compared to ~50+ for the custom CNN approach.

The shift from a custom 2-layer CNN to YOLOv8 represents a move from "learning features from scratch" to "fine-tuning existing knowledge," which is the industry-standard approach for small-dataset image classification tasks.
