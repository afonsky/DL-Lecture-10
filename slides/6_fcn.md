---
layout: section
---

# Fully Convolutional Networks (FCN)
The pioneer of semantic segmentation

---

# FCN: Key Idea

<div class="grid grid-cols-[3fr_1fr] gap-6">
<div>

### The Problem with Classification CNNs

Standard CNNs (AlexNet, VGG, ResNet) end with:
- Global average pooling
- Fully connected layers
- → Fixed-size output (e.g., 1000 classes)

**We need**: Per-pixel classification → same resolution as input!

### FCN Solution ([Long et al., 2015](https://arxiv.org/abs/1411.4038))
1. **Remove** FC layers from pretrained CNN
2. **Add** 1×1 conv to predict classes per pixel
3. **Upsample** with transposed convolution back to input size

> *"Any classification network can be turned into a segmentation network."*

</div>
<div>

<figure>
  <img src="/img/fcn.svg" class="h-100 mx-auto">
  <figcaption class="text-xs text-gray-400 text-center mt-1">FCN architecture (d2l.ai)</figcaption>
</figure>

</div>
</div>

---
zoom: 0.95
---

# FCN: Architecture Details

<div class="grid grid-cols-2 gap-10">
<div>

### Using Pretrained ResNet-18 as Backbone

```python
# Take all layers except final FC
pretrained_net = torchvision.models.resnet18(
    pretrained=True)
net = nn.Sequential(
    *list(pretrained_net.children())[:-2])

# Input: 3×320×480
# After backbone: 512×10×15
# (downsampled by 32×)
```
<br>

### Add prediction layers:
```python
# 1×1 conv: 512 channels → 21 classes
net.add_module('final_conv',
    nn.Conv2d(512, 21, kernel_size=1))

# Transposed conv: upsample 32×
net.add_module('transpose_conv',
    nn.ConvTranspose2d(21, 21,
        kernel_size=64, padding=16, stride=32))
```

</div>
<div>

### Pipeline Visualization

```
Input image: 3 × 320 × 480
        ↓
  [Pretrained ResNet backbone]
        ↓
Feature maps: 512 × 10 × 15
        ↓
  [1×1 Convolution]
        ↓
Class scores: 21 × 10 × 15   ← per-class score
        ↓                        at low resolution
  [Transposed Conv, stride=32]
        ↓
Output:       21 × 320 × 480  ← per-pixel class
                                  prediction!
```

<div class="mt-3 text-sm">

**Final step**: `argmax` along channel dim → pixel labels

</div>

</div>
</div>

---
zoom: 0.9
---

# FCN: Skip Connections for Finer Output

<div class="grid grid-cols-2 gap-6">
<div>

### The Problem with FCN-32s
- Upsampling 32× in one step → **coarse** boundaries
- Fine spatial details are lost

### Solution: Skip Connections
Combine predictions from **multiple scales**:

- **FCN-32s**: Upsample pool5 by 32× (coarse)
- **FCN-16s**: Combine pool5 + pool4, upsample 16×
- **FCN-8s**: Combine pool5 + pool4 + pool3, upsample 8×

Each skip connection adds **finer spatial information**.

</div>
<div>

### Visual Comparison

```
FCN-32s:  ░░▓▓▓▓░░░░  (blocky)
           coarse boundaries

FCN-16s:  ░░▓▓▓▓░░░░  (better)
           medium detail

FCN-8s:   ░▓▓▓▓▓░░░░  (best)
           fine boundaries
```
<br>

### Why it works
- **Deep layers**: WHAT is in the image (semantics)
- **Shallow layers**: WHERE exactly (spatial detail)
- **Skip connections**: Get the best of both!

<div class="mt-3 text-sm text-gray-500">

📑 [Long, Shelhamer, Darrell — "Fully Convolutional Networks for Semantic Segmentation", CVPR 2015](https://arxiv.org/abs/1411.4038)

</div>

</div>
</div>
