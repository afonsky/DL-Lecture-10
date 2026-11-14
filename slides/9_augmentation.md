---
layout: section
---

# Image Augmentation
More data without more data

---
zoom: 0.87
---

# Why Augment?

<div class="grid grid-cols-2 gap-6">
<div>

### The Motivation
- Deep networks are **data hungry**
- Collecting labeled data is **expensive**
- Models can **overfit** to specific positions, colors, orientations

### Image Augmentation
Generate **new training examples** by applying random transformations:
- The label stays the same
- The image looks different
- Model learns to be **invariant** to these transformations

> *"The best regularizer is more data. Augmentation is the cheapest way to get it."*

</div>
<div>

### Common Augmentation Techniques

| Technique | Effect |
|-----------|--------|
| **Random Horizontal Flip** | Left-right invariance |
| **Random Crop + Resize** | Position/scale invariance |
| **Color Jitter** | Lighting condition robustness |
| **Random Rotation** | Orientation invariance |
| **Gaussian Blur** | Focus/quality robustness |
| **Random Erasing** | Occlusion robustness |

<div class="mt-2 text-sm text-gray-500">

⚠️ Augmentations must be **domain-appropriate**: don't flip medical scans vertically, don't rotate digit "6" into "9"!

</div>

</div>
</div>

---
zoom: 0.92
---

# Modern Augmentation Strategies

<div class="grid grid-cols-2 gap-6">
<div>

### [CutOut (2017)](https://arxiv.org/abs/1708.04552)
- Randomly **mask** a square region with zeros
- Forces model to use **global** context, not just key features

### [MixUp (2017)](https://arxiv.org/abs/1710.09412)
- **Blend** two images: $\tilde{x} = \lambda x_i + (1-\lambda) x_j$
- **Blend** labels too: $\tilde{y} = \lambda y_i + (1-\lambda) y_j$
- Smooth decision boundaries, better calibration

### [CutMix (2019)](https://arxiv.org/abs/1905.04899)
- **Cut** a patch from one image and **paste** onto another
- Labels mixed proportionally to area
- Combines benefits of CutOut + MixUp

</div>
<div>

### Visual Comparison

```
Original:     CutOut:       MixUp:        CutMix:
┌────────┐    ┌────────┐    ┌────────┐    ┌────────┐
│  🐱    │    │  🐱    │    │ 🐱+🐕  │    │  🐱    │
│        │    │ ██████ │    │ blended│    │  ┌───┐ │
│        │    │ ██████ │    │        │    │  │🐕 │ │
└────────┘    └────────┘    └────────┘    └────────┘
Label: cat   Label: cat     Label: 0.7    Label: 0.75
                            cat+0.3 dog   cat+0.25 dog
```
<br>

### AutoAugment & RandAugment
- [**Automatically search** (2019)](https://arxiv.org/abs/1805.09501) for best augmentation policies
- [**RandAugment** (2020)](https://arxiv.org/abs/1909.13719): Simplified — just 2 hyperparameters!
  - $N$ = number of transforms
  - $M$ = magnitude

</div>
</div>

---
zoom: 0.99
---

# Augmentation in Practice: PyTorch

<div class="grid grid-cols-2 gap-6">
<div>

### Training Transforms

```python
train_transform = transforms.Compose([
    # Spatial augmentations
    transforms.RandomResizedCrop(224,
        scale=(0.08, 1.0)),
    transforms.RandomHorizontalFlip(),

    # Color augmentations
    transforms.ColorJitter(
        brightness=0.4,
        contrast=0.4,
        saturation=0.4,
        hue=0.1),

    # Convert & normalize
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
])
```

</div>
<div>

### Test Transforms (no randomness!)

```python
test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
])
```

<div class="mt-4 bg-yellow-50 dark:bg-yellow-900 rounded p-3 text-sm">

### ⚠️ **Common mistake**: Applying augmentation to test data! Augmentations are **training-only**.

<br>

#### Test-time augmentation (TTA) is a separate technique where you average predictions over augmented copies at inference.

</div>

</div>
</div>
