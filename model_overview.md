# Face Mask Detection — Improved Model Overview

## Project Goal

The goal of this project is to build a **Convolutional Neural Network (CNN)** that can look at a photo of a person's face and classify it into one of three categories:

- **Masked** — The person is wearing a face mask properly.
- **Unmasked** — The person is not wearing any face mask.
- **Improperly Worn** — The person has a mask but it is not covering the nose and mouth correctly (e.g., pulled under the chin).

---

## Dataset

The dataset consists of **535 images** split across the three classes. The images are divided into:

- **Training set (80%):** 428 images — used to teach the model.
- **Validation set (20%):** 107 images — used to test how well the model performs on images it has never seen before.

This is a relatively **small dataset**, which makes overfitting (the model memorizing training images instead of learning general patterns) a major challenge throughout the project.

---

## Model Architecture

The model is a **Sequential CNN** built using TensorFlow/Keras. Below is a breakdown of each layer and its purpose.

### Convolutional Layers

```
Conv2D(32 filters, 3×3 kernel, ReLU activation, L2 regularization)
MaxPooling2D(2×2)
Dropout(0.2)

Conv2D(64 filters, 3×3 kernel, ReLU activation, L2 regularization)
MaxPooling2D(2×2)
Dropout(0.2)
```

- **Conv2D** is the core of a CNN. It slides small filters (3×3 windows) across the image to detect visual features like edges, textures, and shapes. The first layer learns simple features (e.g., lines and corners), while the second layer combines those into more complex patterns (e.g., the outline of a mask).
- **ReLU activation** introduces non-linearity, allowing the network to learn complex patterns instead of just simple linear relationships.
- **MaxPooling2D** reduces the size of the feature maps by keeping only the maximum value in each 2×2 region. This makes the model faster, reduces memory usage, and helps the model focus on the most important features rather than exact pixel positions.

### Classification Layers

```
Flatten()
Dense(64 neurons, ReLU activation, L2 regularization)
Dropout(0.4)
Dense(3 neurons, Softmax activation)
```

- **Flatten** converts the 2D feature maps from the convolutional layers into a 1D list of numbers so that the Dense (fully connected) layers can process them.
- **Dense(64)** is a fully connected layer that combines all the extracted features to learn the relationship between them and the three output classes.
- **Dense(3, softmax)** is the output layer. It produces three probability scores (one for each class) that add up to 1.0. The class with the highest probability is the model's prediction.

---

## Techniques Used to Improve Generalization

### 1. Data Augmentation

**What it does:** Creates modified versions of training images on-the-fly during each training cycle so the model sees slightly different images every time.

**Why it helps:** With only 428 training images, the model can easily memorize the exact images. Augmentation artificially increases the variety of training data without collecting new images, which forces the model to learn general features instead of specific ones.

**Augmentations applied (training only):**

| Technique | Setting | What It Simulates |
|---|---|---|
| **Brightness Jitter** | ±20% | Different lighting conditions (indoor vs. outdoor, day vs. night) |
| **Color/Saturation Shift** | Channel shift of 30 | Different camera sensors and white balance settings |
| **Horizontal Flip** | 50% chance | The model should recognize a mask whether the person faces left or right |
| **Mild Rotation** | Max ±10° | Natural head tilts when a person is not perfectly upright |
| **Gaussian Blur** | 30% chance, mild intensity | Motion blur or lower-quality cameras |

**What was intentionally avoided:**
- **No vertical flipping** — faces should always remain upright.
- **No random erasing/cutout** — could block the mouth, nose, or chin which are the defining features for all three classes.
- **No aggressive cropping/zooming** — could cut off the chin or nose area, making it impossible to tell "Masked" from "Improperly Worn."

**Important:** These augmentations are applied **only to the training set**. The validation set receives only standard rescaling so that evaluation reflects real-world performance.

### 2. L2 Regularization (Weight Decay)

**What it does:** Adds a small penalty to the loss function based on how large the model's weights become. The penalty is `0.001 × sum of squared weights`.

**Why it helps:** Without regularization, the model is free to assign very large weight values to memorize specific training images. L2 regularization forces the weights to stay small, which encourages the model to find simpler, more general patterns that work across many images — not just the training set.

**Where it is applied:** On both Conv2D layers and the Dense(64) layer using `kernel_regularizer=regularizers.l2(0.001)`.

### 3. Dropout

**What it does:** During each training step, a random percentage of neurons are temporarily turned off (set to zero). Different neurons are disabled each time.

**Why it helps:** Without Dropout, the model can rely on a few specific neurons to memorize training data. By randomly disabling neurons, the model is forced to spread knowledge across many neurons, creating redundant pathways. This means the model doesn't break when it encounters new, unseen images.

**Dropout rates used:**
- **0.2 (20%)** after each convolutional block — mildly prevents over-reliance on specific feature maps.
- **0.4 (40%)** after the Dense(64) layer — more aggressive since the dense layer is the most prone to memorization.

**Important:** Dropout is automatically turned off during validation and prediction, so the full model capacity is always used when making predictions.

### 4. Separate Data Generators (Train vs. Validation)

**What it does:** Uses two different `ImageDataGenerator` objects — one with augmentations for training, and one with only rescaling for validation.

**Why it helps:** If augmentations were also applied to the validation set, the validation accuracy would not reflect real-world performance. The validation set should always be a clean, unmodified representation of what the model will encounter in production.

---

## Training Configuration

| Setting | Value | Purpose |
|---|---|---|
| **Optimizer** | Adam | An adaptive optimizer that automatically adjusts the learning rate for each parameter. Good default choice for most tasks |
| **Loss Function** | Categorical Crossentropy | Standard loss function for multi-class classification. Measures how far the predicted probabilities are from the true labels |
| **Epochs** | 12 | Number of complete passes through the training data |
| **Batch Size** | 32 | Number of images processed before updating the model's weights |
| **Image Size** | 128 × 128 pixels | All images are resized to this standard size before being fed to the model |

---

## Results

| Metric | Value |
|---|---|
| **Final Training Accuracy** | ~78.7% |
| **Final Validation Accuracy** | ~72.9% |
| **Best Validation Accuracy** | ~72.9% (Epoch 12) |

The ~6% gap between training and validation accuracy shows the model is generalizing reasonably well given the small dataset size. The regularization techniques (L2 + Dropout) successfully reduced overfitting compared to the baseline model, which had a much larger gap (80%+ training vs. ~53% validation).

---

## Summary

This model demonstrates how a simple CNN can be improved for a small-dataset scenario using a combination of **data augmentation**, **L2 regularization**, and **Dropout**. These techniques work together to reduce overfitting:

- **Data augmentation** increases the effective size and variety of the training set.
- **L2 regularization** keeps the model's weights small and prevents memorization.
- **Dropout** forces the model to learn redundant, generalizable features.

While the validation accuracy (~72.9%) shows there is room for improvement, the narrowed gap between training and validation performance confirms that the model is learning patterns that generalize to unseen data rather than simply memorizing the training set.
