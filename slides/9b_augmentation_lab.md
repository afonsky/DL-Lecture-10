---
zoom: 0.98
---

# Augmentation Lab: Image and Target Are One Example

<figure>
  <AugmentationLab />
  <figcaption style="color:#b3b3b3ff; font-size: 11px; position: relative; top: 10px">Image source: interactive original illustration after <a href="https://d2l.ai/chapter_computer-vision/image-augmentation.html">d2l.ai §14.1, image augmentation</a> and <a href="https://d2l.ai/chapter_computer-vision/semantic-segmentation-and-dataset.html#data-preprocessing">d2l.ai §14.9.3, shared image/label cropping</a></figcaption>
</figure>

<!--
Three minutes, guaranteed quiz demonstration. Work through three states:
1. Detection + horizontal flip: xyxy changes to [W - x2, y1, W - x1, y2]; the class labels do not change.
2. Detection + shared crop: both boxes are clipped to the crop and the dog disappears, so its target is removed. An empty target is valid; a stale dog box is a bug.
3. Segmentation + shared crop, then color jitter: crop changes image and mask together; color jitter changes only image appearance, never class IDs.

For classification, the single cat label survives all displayed transforms because the cat remains visible. Do not generalize this to arbitrary crops or arbitrary domains: augmentations must preserve the intended task and labels. Static fallback is the next slide's torchvision v2 pipeline plus the flip/crop readouts described above.

Component: <code>components/AugmentationLab.vue</code>
-->

---
zoom: 0.9
---

# Torchvision v2: Geometry Together, Appearance Separately

<br>
<br>

<div class="grid grid-cols-[2fr_2fr] gap-7">
<div>

```python
from PIL import Image
import torch
from torchvision import tv_tensors
from torchvision.transforms import v2

image = v2.ToImage()(Image.open("image.jpg"))
mask = tv_tensors.Mask(v2.ToImage()(Image.open("mask.png")))

geometric = v2.Compose([
    v2.RandomResizedCrop(224, scale=(0.5, 1.0)),
    v2.RandomHorizontalFlip(),
])
image, mask = geometric(image, mask)

image = v2.ColorJitter(brightness=0.2, contrast=0.2)(image)
image = v2.ToDtype(torch.float32, scale=True)(image)
image = v2.Normalize(mean=[0.485, 0.456, 0.406],
                     std=[0.229, 0.224, 0.225])(image)
```

</div>
<div class="text-sm">

<div class="rounded p-3 bg-blue-50 dark:bg-blue-900">

### **Geometry**  
#### Use the same sampled crop/flip for image and `Mask` / `BoundingBox` targets.
</div>
<div class="mt-3 rounded p-3 bg-purple-50 dark:bg-purple-900">

### **Appearance**  
#### Color jitter and normalization apply to the image — not class IDs.
</div>
<div class="mt-3 rounded p-3 bg-green-50 dark:bg-green-900">

### **Validation / test**  
#### Deterministic resize or crop; no random training augmentation. **See Lecture 2**
</div>
</div>
</div>

<br>

<span class="refs">Source: [d2l.ai §14.9.3, shared crop](https://d2l.ai/chapter_computer-vision/semantic-segmentation-and-dataset.html#data-preprocessing) · [torchvision transforms v2](https://pytorch.org/vision/stable/transforms.html) · [S. Raschka, data-augmentation notebook](https://github.com/rasbt/stat453-deep-learning-ss21/blob/main/L10/code/data-augmentation.ipynb)</span>

<!--
Ninety seconds. The important line is `image, mask = geometric(image, mask)`: torchvision v2 understands the target type and applies compatible geometry. Mask values remain integer class IDs; only the image is converted to scaled float32 and normalized.

For detection, construct a tv_tensors.BoundingBox with its format and canvas_size and pass it through the geometric transform, then remove targets invalidated by the task's crop policy. The interactive previous slide demonstrates the semantics rather than asking students to infer them from API calls.

The fragment follows the current v2 documentation but was not executed in this environment. It is a static fallback for the demo, not a measured training result.
-->
