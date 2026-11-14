---
layout: section
---

# Modern CV Landscape<br> &<br> Practical Tips
What works in practice

---
zoom: 0.87
---

# The Modern CV Toolbox (Pre-Transformer)

<div class="grid grid-cols-2 gap-6">
<div>

### Detection
- **Faster R-CNN** — accuracy benchmark
- **YOLO** family — real-time king
- **SSD / RetinaNet** — balanced approaches
- **EfficientDet** — efficiency-focused (2020)

<br>

### Segmentation
- **U-Net** — medical & small data
- **DeepLab v3+** — atrous convolutions
- **Mask R-CNN** — instance segmentation
- **PointRend** — rendering-based refinement (2020)

</div>
<div>

### Key Building Blocks

**Feature Pyramid Networks (FPN)**
- Multi-scale feature extraction
- Top-down pathway + lateral connections
- Used in Mask R-CNN, RetinaNet, EfficientDet

**Dilated (Atrous) Convolutions**
- Expand receptive field **without** losing resolution
- Key component of DeepLab series

**Depthwise Separable Convolutions**
- Factorize standard conv → cheaper
- MobileNet, EfficientNet for mobile/edge

</div>
</div>

<div class="mt-2 bg-blue-50 dark:bg-blue-900 rounded p-2 text-sm text-center">

#### 🔮 **Coming soon**: Vision Transformers (ViT), DETR, and Segment Anything (SAM)

</div>

---
zoom: 0.89
---

# EfficientNet & Compound Scaling

<div class="grid grid-cols-2 gap-6">
<div>

### The Scaling Question
How to scale up a CNN for better accuracy?

- **Wider** (more channels)?
- **Deeper** (more layers)?
- **Higher resolution** input?

### EfficientNet ([Tan & Le, 2019](https://proceedings.mlr.press/v97/tan19a.html))

**Compound scaling** — scale all three dimensions together!

$$d = \alpha^\phi, \quad w = \beta^\phi, \quad r = \gamma^\phi$$

$$\text{s.t. } \alpha \cdot \beta^2 \cdot \gamma^2 \approx 2$$

- $\phi$ = compound coefficient (user-specified)
- $\alpha, \beta, \gamma$ found by grid search on small model

</div>
<div>

### Why It Matters

| Model | Top-1 Acc | Params | FLOPs |
|-------|-----------|--------|-------|
| ResNet-50 | 76.0% | 26M | 4.1B |
| EfficientNet-B0 | 77.3% | 5.3M | 0.4B |
| EfficientNet-B4 | 83.0% | 19M | 4.2B |
| EfficientNet-B7 | 84.4% | 66M | 37B |

**8.4× smaller** and **6.1× faster** than best previous CNN at same accuracy!

<div class="mt-3 bg-green-50 dark:bg-green-900 rounded p-3 text-sm">

#### 💡 **Lesson**: It's not just about making networks bigger — **how** you scale matters as much as **how much** you scale.

</div>

</div>
</div>

---
zoom: 0.78
---

# Karpathy's Recipe for Training Neural Nets

<div class="grid grid-cols-[3fr_4fr] gap-16">
<div>

### The Most Common Mistakes

> *"The most common neural net mistakes: 1) you didn't try to overfit a single batch first; 2) you forgot to toggle train/eval mode; 3) you turned off gradient computation somewhere; 4) you passed softmaxed outputs to a loss that expects raw logits."*<br> — Andrej Karpathy

<br>

### The Recipe 🧑‍🍳
1. **Understand the data** — visualize, check distributions
2. **Set up a baseline** — simplest model, known results
3. **Overfit one batch** — verify the pipeline works
4. **Regularize** — augmentation, dropout, weight decay
5. **Tune** — learning rate is the most important hyperparameter

</div>
<div>

### Practical Tips for CV Projects

**Data**:
- Look at your data! Plot examples, check labels
- Start simple: resize to **224×224**, use standard augmentation
- Use **pretrained models** — almost always

**Training**:
- Start with **Adam** (`lr=3e-4`), switch to **SGD** for fine-tuning
- Use **learning rate warmup** + cosine annealing
- **Mixed precision** training for speed

**Debugging**:
- If loss doesn't decrease: LR too high, bug in data loading
- If training acc is high but val is low: **overfitting**
- If both are low: model too small or bug

<div class="mt-2 text-sm text-gray-500">

📝 Based on Karpathy's ["A Recipe for Training Neural Networks" blog post](http://karpathy.github.io/2019/04/25/recipe/)

</div>

</div>
</div>

---
zoom: 0.9
---

# Summary: Computer Vision II

<div class="grid grid-cols-2 gap-6">
<div>

### Object Detection
- **SSD**: Single-shot multiscale detection
- **YOLO**: Grid-based, real-time detection
- **Focal Loss**: Solved class imbalance for one-stage detectors
- One-stage detectors now match two-stage accuracy

### Segmentation
- **Semantic**: Per-pixel classification
- **Instance**: Individual object masks (Mask R-CNN)
- **FCN**: First end-to-end segmentation
- **U-Net**: Encoder-decoder with skip connections

</div>
<div>

### Practical Techniques
- **Transfer Learning**: Pretrained → fine-tune
- **Augmentation**: More data without more data
- **Neural Style Transfer**: Content + style = art

### Key Takeaways
1. **Use pretrained models** — always your starting point
2. **Data augmentation** — cheap and effective regularization
3. **Choice of detector** depends on speed vs accuracy needs
4. **Multi-scale features** are crucial for CV
5. **Next lectures**: Transformers will change everything 🚀

</div>
</div>
