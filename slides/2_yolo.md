---
layout: section
---

# YOLO: You Only Look Once
The fastest family of object detectors

---

# YOLO: Core Philosophy

<div class="grid grid-cols-[4fr_3fr] gap-6">
<div>

### The Big Idea ([Redmon et al., 2016](https://arxiv.org/abs/1506.02640))

<br>

> "We frame object detection as a **single regression problem**, straight from image pixels to bounding box coordinates and class probabilities."

<br>

### How it works
1. Divide image into an **S × S grid**
2. Each grid cell predicts:
   - **B** bounding boxes (x, y, w, h, confidence)
   - **C** class probabilities
3. **Single forward pass** → all predictions at once!

</div>
<div>

### YOLO vs R-CNN Approach

```
R-CNN family:
  Image → Regions → CNN → Classify each
         (thousands of forward passes)

YOLO:
  Image → Single CNN → All boxes + classes
         (ONE forward pass!)
```

<div class="mt-4 bg-green-50 dark:bg-green-900 rounded p-3 text-sm">

#### **Result**: YOLO processes images<br> at **45 FPS** (base) or **155 FPS** (Fast YOLO) — enabling real-time detection on video streams.

</div>

</div>
</div>

---
zoom: 1.1
---

# YOLO: Grid-Based Detection

<div class="grid grid-cols-[3fr_2fr] gap-6">
<div>

### Step by step
1. Input image → **S × S grid** (e.g., 7×7)
2. Cell responsible for object if object **center** falls in it
3. Each cell predicts:
   - **B** boxes: $(x, y, w, h, \text{conf})$
   - Confidence = $P(\text{object}) \times \text{IoU}$
   - **C** class conditional probs

### Final prediction
$$P(\text{class}_i | \text{object}) \times P(\text{object}) \times \text{IoU}$$

</div>
<div>

### Visual Example

```
┌────┬────┬────┬────┬────┬────┬────┐
│    │    │    │    │    │    │    │
├────┼────┼────┼────┼────┼────┼────┤
│    │    │ 🚗 │    │    │    │    │
├────┼────┼────┼────┼────┼────┼────┤
│    │    │    │    │    │    │    │
├────┼────┼────┼────┼────┼────┼────┤
│    │    │    │    │ 🐕 │    │    │
├────┼────┼────┼────┼────┼────┼────┤
│    │    │    │    │    │    │    │
├────┼────┼────┼────┼────┼────┼────┤
│    │    │    │    │    │    │    │
├────┼────┼────┼────┼────┼────┼────┤
│    │    │    │    │    │    │    │
└────┴────┴────┴────┴────┴────┴────┘
```

Each cell with an object center predicts boxes + classes.

</div>
</div>

---
zoom: 0.87
---

# YOLO Evolution: A Success Story

<script setup>
const yoloTimelineSource = `
> ORDERBY start
> DEFAULTVIEW 2015|2028
> HEIGHT 300
> NOTODAY
- [2016] #2a78d6 YOLOv1 · grid single-shot | Grid-based single-shot · 45 FPS · first real-time detector
- [2017] #2a78d6 YOLOv2 · BN + anchors | Batch norm, anchor boxes, multiscale · 67 FPS · Better, Faster, Stronger
- [2018] #2a78d6 YOLOv3 · FPN + 3 scales | Feature Pyramid Network, 3 scales · 30 FPS · multiscale predictions
- [2020] #4c9a2a YOLOv4 · CSPNet + freebies | Bag of freebies/specials, CSPNet · 65 FPS · practical optimizations
- [2020] #4c9a2a YOLOv5 · PyTorch native | PyTorch native, easy to use · 140 FPS · industry adoption
- [2023] #8e6bbf YOLOv8 · anchor-free | Anchor-free, decoupled head · 80+ FPS · unified framework
- [2024] #8e6bbf YOLO11 · C3k2 + attention | C3k2 blocks, improved attention · 90+ FPS · efficient across tasks
- [2025] #ed7d31 YOLO26 · NMS-free | NMS-free inference, MuSGD, ProgLoss + STAL · edge AI, IoT, robotics
`
</script>

<ChronosTimeline :source="yoloTimelineSource" />

<div class="flex justify-between text-xs text-gray-400 mt-1">
  <span>Hover for details · drag to pan · scroll to zoom</span>
  <a href="https://arxiv.org/abs/2509.25164">YOLO26 paper</a>
</div>

<div class="mt-3 bg-yellow-50 dark:bg-yellow-900 rounded p-3 text-sm">

#### 📌 **Trends**:
#### 1. From hand-designed anchors → **anchor-free** methods.
#### 2. From separate tasks → **unified** detection + segmentation + pose estimation.

</div>

---

# YOLO Evolution: A Success Story

<figure>
  <img src="/YOLO_performance-comparison.png" class="h-95 mx-auto">
  <figcaption class="text-xs text-gray-400 text-center mt-1"> Performance comparison of YOLO26 under TensorRT FP16 on an NVIDIA T4 GPU.<br> Image source: <a href="https://docs.ultralytics.com/models/yolo26/">https://docs.ultralytics.com/models/yolo26/</a></figcaption>
</figure>

---

# YOLO26 Architecture

<figure>
  <img src="/YOLO_architecture.png" class="h-100 mx-auto">
  <figcaption class="text-xs text-gray-400 text-center mt-1"> Core architecture diagram of Ultralytics YOLO26 object detection and segmentation algorithm.<br> Image source: <a href="https://arxiv.org/abs/2509.25164">arXiv:2509.25164</a></figcaption>
</figure>

---

# YOLO: Practical Impact

<div class="grid grid-cols-2 gap-6">
<div>

### Where YOLO is used today
- 🚗 **Self-driving cars** — real-time obstacle detection
- 📹 **Video surveillance** — person & object tracking
- 🏭 **Manufacturing** — defect detection on assembly lines
- 🏥 **Medical imaging** — lesion detection
- 📱 **Mobile apps** — on-device object detection
- 🤖 **Robotics** — environment understanding

</div>
<div>

### Why so popular?
1. **Speed**: Processes video in real-time
2. **Simplicity**: Single network, end-to-end
3. **Global reasoning**: Sees entire image → fewer background errors
4. **Generalizable**: Learns general representations
5. **Community**: Well-maintained, easy to fine-tune

<div class="mt-3 text-sm text-gray-500">

> *"YOLO is extremely fast. Since we frame detection as a regression problem we don't need a complex pipeline. We simply run our neural network on a new image at test time to predict detections."*<br> — Joseph Redmon

</div>

</div>
</div>
