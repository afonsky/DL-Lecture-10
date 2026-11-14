---
layout: section
---

# U-Net & Modern Segmentation
Encoder-decoder architectures

---
zoom: 0.93
---

# U-Net: The Symmetric Architecture

<div class="grid grid-cols-2 gap-6">
<div>

### Design (Ronneberger et al., 2015)

**Encoder** (contracting path):
- Repeated conv + pooling → downsample
- Capture context ("what")

**Decoder** (expanding path):
- Transposed conv → upsample
- Recover spatial detail ("where")

**Skip connections**: Concatenate encoder features to decoder — preserve fine-grained details!

### Why "U-Net"?
The architecture looks like a **U** shape.

</div>
<div>

### Architecture Diagram

```
Input ──[Conv]──[Conv]──────────────── Concat ──[Conv]──[Conv]── Output
         │                                ↑
         ↓ Pool                     UpConv │
        [Conv]──[Conv]──────────── Concat ──[Conv]──[Conv]
           │                          ↑
           ↓ Pool                UpConv│
          [Conv]──[Conv]──── Concat ──[Conv]──[Conv]
             │                    ↑
             ↓ Pool          UpConv│
            [Conv]──[Conv]──[Conv]──[Conv]
                   Bottleneck
```

<div class="text-sm mt-2">

**Key difference from FCN**: Skip connections **concatenate** features (not add), giving the decoder much richer information.

</div>

</div>
</div>

---
zoom: 0.78
---

# U-Net: Why So Successful?

<div class="grid grid-cols-[4fr_4fr] gap-6">
<div>

### Designed for Small Datasets
- Originally for **biomedical image segmentation**
- Works well with **very few** training images
- Heavy use of data augmentation

<br>

### Key Innovations
1. **Symmetric encoder-decoder** — balanced capacity
2. **Skip connections via concatenation** — preserves spatial info at every resolution
3. **Overlap-tile strategy** — segment large images in patches

<br>

### Medical Imaging Impact
- Cell segmentation
- Organ delineation
- Tumor boundary detection
- Retinal vessel segmentation

</div>
<div>

### U-Net vs FCN

| Feature | FCN | U-Net |
|---------|-----|-------|
| Skip connections | Addition | **Concatenation** |
| Decoder | Single transposed conv | **Multi-step decoder** |
| Small data | Needs large dataset | **Works with few images** |
| Resolution | Coarse (FCN-32s) | **Fine details** |
| Origin | General CV | **Medical imaging** |

<div class="mt-4 bg-green-50 dark:bg-green-900 rounded p-3 text-sm">

#### 💡 U-Net has become the **go-to architecture** for segmentation tasks, especially in medical imaging. Its variations (U-Net++, Attention U-Net) dominate almost every medical imaging challenge.

</div>

</div>
</div>

---
zoom: 0.85
---

# Mask R-CNN: Instance Segmentation

<div class="grid grid-cols-[3fr_2fr] gap-6">
<div>

### Extending Faster R-CNN ([He et al., 2017](https://arxiv.org/abs/1703.06870))

Mask R-CNN = Faster R-CNN + **pixel-level mask branch**

### How it works
1. **Backbone** (ResNet + FPN) extracts features
2. **RPN** proposes regions of interest
3. **RoI Align** (not RoI Pool!) extracts fixed-size features
   - Uses bilinear interpolation — no quantization!
4. **Three parallel heads**:
   - Classification (what?)
   - Bounding box regression (where?)
   - **Mask prediction** (pixel-level shape!)

### RoI Align vs RoI Pooling
- RoI Pooling rounds coordinates → misalignment
- **RoI Align** uses bilinear interpolation → precise alignment

</div>
<div>

<figure>
  <img src="/img/mask-rcnn.svg" class="h-95 mx-auto">
  <figcaption class="text-xs text-gray-400 text-center mt-1">Mask R-CNN (d2l.ai)</figcaption>
</figure>

<div class="mt-3 text-sm">

### Results
- **COCO** instance segmentation champion
- Extends to: keypoint detection, pose estimation
- Modular: can add new heads for new tasks

</div>

</div>
</div>

---
zoom: 0.78
---

# Segmentation Architecture Evolution

<script setup>
const segmentationTimelineSource = `
> ORDERBY start
> DEFAULTVIEW 2014|2022
> HEIGHT 300
> NOTODAY
- [2015] #2a78d6 FCN · fully convolutional | Replace FC with conv + transposed conv · Semantic segmentation
- [2015] #4c9a2a U-Net · skip concat | Symmetric encoder–decoder + skip concat · Semantic segmentation
- [2015] #8e6bbf SegNet · pooling indices | Encoder–decoder with pooling indices · Semantic segmentation
- [2017] #ed7d31 Mask R-CNN · mask head | Faster R-CNN + mask head · Instance segmentation
- [2017] #2a78d6 PSPNet · pyramid pooling | Pyramid pooling module · Semantic segmentation
- [2017] #2a78d6 DeepLab v3 · atrous + ASPP | Atrous convolutions + ASPP · Semantic segmentation
- [2018] #4c9a2a DeepLab v3+ · encoder–decoder | Encoder–decoder + atrous conv · Semantic segmentation
- [2019] #8e6bbf Panoptic FPN · things + stuff | Unified stuff + things · Panoptic segmentation
`
</script>

<div class="text-xs text-gray-400 text-right mb-1">Hover for task details · drag to pan · scroll to zoom</div>

<ChronosTimeline :source="segmentationTimelineSource" />

<br>

<div class="mt-3 bg-yellow-50 dark:bg-yellow-900 rounded p-3 text-sm">

#### **Common pattern**: All modern segmentation architectures use some form of:
#### 1. **Pretrained encoder** (backbone) for feature extraction
#### 2. **Decoder** with upsampling for resolution recovery
#### 3. **Multi-scale features** (FPN, ASPP, pyramid pooling)
#### 4. **Skip connections** to preserve spatial detail

</div>
