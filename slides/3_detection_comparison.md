---
layout: section
---

# One-Stage vs Two-Stage Detectors
Choosing the right approach

---

# Detection Paradigms Comparison

<div class="grid grid-cols-2 gap-6">
<div>

### Two-Stage Detectors
<div class="bg-blue-50 dark:bg-blue-900 rounded p-3 text-sm mb-3">

**Step 1**: Generate region proposals<br>
**Step 2**: Classify each proposal

</div>

- R-CNN → Fast R-CNN → Faster R-CNN → Mask R-CNN
- ✅ Higher accuracy (especially small objects)
- ✅ Better localization precision
- ❌ Slower inference
- **Use when**: Accuracy matters most (medical imaging, satellite)

</div>
<div>

### One-Stage Detectors
<div class="bg-green-50 dark:bg-green-900 rounded p-3 text-sm mb-3">

**Single step**: Predict all boxes + classes at once

</div>

- SSD, YOLO, RetinaNet, EfficientDet
- ✅ Much faster (real-time capable)
- ✅ Simpler pipeline
- ❌ Historically lower accuracy
- **Use when**: Speed matters (video, autonomous driving)

</div>
</div>

<div class="mt-4 text-center text-sm bg-gray-50 dark:bg-gray-800 rounded p-3">

**Closing the gap**: Modern one-stage detectors (YOLOv8, EfficientDet) now approach two-stage accuracy thanks to innovations like **Feature Pyramid Networks** and **Focal Loss**.

</div>

---
zoom: 0.9
---

# The Class Imbalance Problem & Focal Loss

<div class="grid grid-cols-[4fr_2fr] gap-6">
<div>

### Why one-stage detectors struggled

In one-stage detectors, most anchors are **background**:
- ~100,000 anchor locations
- Only ~10 contain objects
- **99.99% are easy negatives** → dominate the loss

This makes training difficult — the detector learns to just say "background" for everything.

### Solution: Focal Loss ([Lin et al., 2017](https://arxiv.org/abs/1708.02002))

Modify cross-entropy to **down-weight easy examples**:

$$FL(p_t) = -\alpha_t (1 - p_t)^{\gamma} \log(p_t)$$

- When $p_t$ is high (easy example): $(1-p_t)^\gamma \approx 0$ → low loss
- When $p_t$ is low (hard example): full loss

</div>
<div>

### Effect of Focal Loss

```
Standard CE Loss:
  Easy backgrounds ████████████████ dominate
  Hard examples    ██ drowned out

Focal Loss (γ=2):
  Easy backgrounds ██ down-weighted
  Hard examples    ████████ focused on
```

<br>

### RetinaNet

Applied Focal Loss to a simple one-stage detector → **matched two-stage accuracy** for the first time!

<div class="mt-3 bg-yellow-50 dark:bg-yellow-900 rounded p-2 text-sm">

**Key insight** (from Kaiming He's team): The accuracy gap wasn't about architecture — it was about the **loss function**.

</div>

</div>
</div>
