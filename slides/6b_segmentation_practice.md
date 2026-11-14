---
zoom: 0.95
---

# Segmentation Output = Class Logits at Every Pixel

<div class="grid grid-cols-[2fr_2fr] gap-7">
<div>

```python
import torch
from torch import nn, functional as F
from torchvision import models

weights = models.ResNet18_Weights.DEFAULT
resnet = models.resnet18(weights=weights)
backbone = nn.Sequential(*list(resnet.children())[:-2])
head = nn.Conv2d(512, 21, kernel_size=1)
up = nn.ConvTranspose2d(21, 21, kernel_size=64,
                        stride=32, padding=16)

X = torch.randn(8, 3, 320, 480)
Y = torch.randint(0, 21, (8, 320, 480))
logits = up(head(backbone(X)))
loss = F.cross_entropy(logits, Y)
prediction = logits.argmax(dim=1)
```

<br>
<br>

<span class="refs">Source: [d2l.ai §14.11, Fully Convolutional Networks](https://d2l.ai/chapter_computer-vision/fcn.html) · [torchvision model weights](https://pytorch.org/vision/stable/models.html) · [S. Raschka, CNNs in PyTorch](https://sebastianraschka.com/blog/2021/dl-course.html#l13-introduction-to-convolutional-neural-networks)</span>

</div>
<div class="text-sm">

### Tensor shapes

```text
X             [8,  3, 320, 480]
backbone(X)   [8, 512, 10, 15]
logits        [8,  21, 320, 480]
Y             [8,      320, 480]
prediction    [8,      320, 480]
```

<div class="mt-4 rounded p-3 bg-blue-50 dark:bg-blue-900">

### **One classification problem per pixel**  
#### `21` = 20 VOC classes + background
</div>

<div class="mt-3 rounded p-3 bg-orange-50 dark:bg-orange-900">

### **Do not softmax first**  
#### `F.cross_entropy` expects raw logits and integer class IDs. **See Lecture 2**
</div>

<div class="mt-3 text-sm text-gray-500">

#### Modern `weights=` API calling also supplies the matching evaluation preprocessing.
</div>
</div>
</div>

<!--
Two minutes. Follow one tensor through the network. The classifier's global average pool and fully connected layer are removed; a 1x1 convolution creates 21 low-resolution class maps; the transposed convolution restores H and W. Cross-entropy then compares 21 logits with one integer class ID at each of 320 x 480 positions.

The expected logits and target shapes are shown explicitly. `argmax(dim=1)` is for inference, not before the loss. The API fragment is documentation-checked but not executed in the authoring environment because PyTorch is unavailable; do not claim a measured loss value. Full d2l training remains the seminar exercise.
-->

---
zoom: 0.9
---

# Segmentation Metrics: Pixel Accuracy Is Not Enough

<figure>
  <img src="/img/lecture10/segmentation_metrics.svg" class="h-72 mx-auto">
  <figcaption style="color:#b3b3b3ff; font-size: 11px; position: relative; top: 10px">Image source: original 10×10 mask example computed for this lecture; metric context: <a href="https://arxiv.org/abs/1411.4038">Long et al., Fully Convolutional Networks, pixel accuracy and mean IU</a></figcaption>
</figure>

<div class="grid grid-cols-4 gap-3 text-center text-sm mt-4">
<div class="rounded p-3 bg-gray-50 dark:bg-gray-800">

### **Pixel accuracy**  
#### all pixels, background included
</div>
<div class="rounded p-3 bg-orange-50 dark:bg-orange-900">

### **IoU per class**  
#### TP / (TP + FP + FN)
</div>
<div class="rounded p-3 bg-blue-50 dark:bg-blue-900">

### **mIoU**  
#### mean of class IoUs
</div>
<div class="rounded p-3 bg-green-50 dark:bg-green-900">

### **Dice**  
#### 2TP / (2TP + FP + FN)
</div>
</div>

<div class="mt-3 text-sm text-gray-500">

#### Small objects and boundary errors need class-wise or boundary-aware evaluation — not a single impressive accuracy.

</div>

<!--
Two minutes. The ground truth contains 10 object pixels. Prediction has TP=8, FN=2, FP=5, TN=85. Pixel accuracy is 93%, but foreground IoU is only 8/15 = 0.533 and Dice is 16/23 = 0.696. Background IoU is 85/92 = 0.924, so the two-class mIoU is (0.533 + 0.924)/2 = 0.729.

The extra material after the end slide introduces pixel-wise cross-entropy, weighted CE, Dice and IoU losses. Keep training loss and evaluation metric conceptually separate: Dice/IoU losses optimize differentiable surrogates; this slide evaluates a hard predicted mask. For boundary-critical applications, inspect a boundary metric as well.

Reproduction: <code>scripts/lecture10/make_figures.py</code>
-->
