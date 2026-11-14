---
layout: section
---

# Single Shot Multibox Detection (SSD)
One-stage object detection

---

# Motivation: Why Single-Shot?

<div class="grid grid-cols-2 gap-6">
<div>

### Two-Stage Detectors (R-CNN family)
- ✅ High accuracy
- ❌ **Slow**: separate region proposal + classification
- ❌ Not suitable for real-time applications

### One-Stage Detectors
- ✅ **Fast**: detect objects in a single pass
- ✅ End-to-end training
- ⚠️ Slightly lower accuracy (but improving!)

</div>
<div>

### The Speed Problem

| Model | FPS | mAP |
|-------|-----|-----|
| R-CNN | ~0.02 | 58.5 |
| Fast R-CNN | ~0.5 | 70.0 |
| Faster R-CNN | ~7 | 73.2 |
| **SSD** | **~46** | 74.3 |

<div class="text-xs text-gray-400 mt-2">

*Approximate values on [Pascal VOC 2007](https://www.robots.ox.ac.uk/~vgg/projects/pascal/VOC/voc2007/). Speed measured on similar hardware.*

</div>

</div>
</div>

<br>

> 💡 **Key idea**: Skip the region proposal step — predict object classes and bounding boxes **directly** from feature maps at multiple scales.

---

# SSD Architecture

<div class="grid grid-cols-[5fr_3fr] gap-6">
<div>

### Design
1. **Base network** (e.g., VGG-16) extracts features
2. **Multiscale feature map blocks** progressively reduce spatial size
3. At each scale, predict:
   - **Class scores** for each anchor box
   - **Bounding box offsets** (4 values)
</div>
<div>

<figure>
  <img src="/img/ssd.svg" class="h-70 mx-auto">
  <figcaption class="text-xs text-gray-400 text-center mt-1">SSD architecture (d2l.ai)</figcaption>
</figure>

</div>
</div>

### Why multiscale?
- **Large feature maps** (early layers) → detect **small** objects
- **Small feature maps** (later layers) → detect **large** objects

#### Each location generates multiple anchor boxes with different aspect ratios.

---
zoom: 0.99
---

# SSD: How Predictions Work

<div class="grid grid-cols-2 gap-6">
<div>

### Class Prediction Layer
- Uses **convolution** (not fully connected!)
- For each spatial position with $a$ anchors and $q$ classes:
  - Output channels: $a \times (q + 1)$
  - +1 for background class

### Bounding Box Prediction Layer
- Same idea, but predict **4 offsets** per anchor
- Output channels: $a \times 4$

</div>
<div>

### Multiscale Feature Maps

```
Feature Map 1: 38×38  → many small anchors
Feature Map 2: 19×19  → medium anchors
Feature Map 3: 10×10  → ...
Feature Map 4:  5×5   → ...
Feature Map 5:  3×3   → ...
Feature Map 6:  1×1   → few large anchors
```

**Total**: ~8732 anchor boxes across all scales

</div>
</div>

<div class="mt-4 text-center bg-blue-50 dark:bg-blue-900 rounded p-3">

**SSD = Base CNN + Multiscale Feature Maps + Conv Predictions + NMS**

</div>

---
zoom: 0.97
---

# SSD: Training & Loss

<div class="grid grid-cols-2 gap-6">
<div>

### Training Procedure

1. **Match** anchor boxes to ground truth via IoU
2. **Hard negative mining**: most anchors are background — sample hardest negatives<br> (3:1 ratio)
3. **Joint loss** = Classification loss + Localization loss

### Loss Function
- **Classification**: Cross-entropy loss
- **Localization**: Smooth L1 loss on box offsets
  - Only for **positive** (matched) anchors

</div>
<div>

### SSD in Practice

**Strengths**:
- Real-time speed (~46 FPS)
- Good accuracy across object sizes
- Simple, end-to-end training

**Weaknesses**:
- Struggles with very small objects
- Many anchors needed → memory heavy
- Manual anchor box design

</div>
</div>

<div class="text-sm text-gray-500 mt-4">

📑 [Liu et al., "SSD: Single Shot MultiBox Detector", ECCV 2016](https://arxiv.org/abs/1512.02325)

</div>
