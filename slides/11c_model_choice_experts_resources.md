---
zoom: 0.89
---

# Before Downloading the Newest Model: Four Decisions

<div class="grid grid-cols-4 gap-4 text-sm">
<div class="rounded-lg p-4 bg-blue-50 dark:bg-blue-900">
<div class="text-2xl font-bold">1</div>

### Output + labels
<br>

#### Classes, boxes, semantic masks, instances, tracks?
<br>

#### What annotations actually exist?
</div>
<div class="rounded-lg p-4 bg-green-50 dark:bg-green-900">
<div class="text-2xl font-bold">2</div>

### Metric + split
<br>

#### AP₅₀ or AP₅₀:₉₅? mIoU or Dice?
<br>

#### Validation for choices; test once. See Lectures 2 and 8
</div>
<div class="rounded-lg p-4 bg-orange-50 dark:bg-orange-900">
<div class="text-2xl font-bold">3</div>

### Baseline + budget
<br>

#### Start with U-Net, Faster R-CNN, YOLO or a linear probe.
<br>

#### Measure memory, latency and preprocessing on **your device**.
</div>
<div class="rounded-lg p-4 bg-purple-50 dark:bg-purple-900">
<div class="text-2xl font-bold">4</div>

### Robustness + license
<br>

#### Try shifted data and several seeds. Inspect failures.
<br>

#### Check checkpoint license, export and deployment cost.
</div>
</div>

<div class="mt-6 rounded p-4 text-center bg-gray-50 dark:bg-gray-800 text-lg">

### **Output → metric → baseline → constraints → controlled comparison**  
#### The best model is the one that survives this workflow
</div>

<span class="refs">Validation: [Y. Abu-Mostafa, Learning From Data, Lecture 13](https://www.youtube.com/watch?v=o7zzaKd0Lkk) · Practical debugging: [A. Karpathy, A Recipe for Training Neural Networks](https://karpathy.github.io/2019/04/25/recipe/) · Controlled optimizer comparisons: **See Lecture 8**</span>

<!--
Ninety seconds. This is the lecture's closing decision procedure. A model that wins COCO may lose on a small medical dataset, exceed an edge-memory budget, have an incompatible license or require unavailable preprocessing operators. A simpler baseline measured under the same protocol is the meaningful comparison.

Insist on validation-only model and threshold selection, with a final untouched test set. Where images from one patient, camera, location or video are correlated, split by that group rather than treating neighboring frames as independent examples. Report the complete inference path and the exact model version, not only the network's forward pass.
-->

---
zoom: 0.77
---

# Learn More from the Experts

<div class="experts">

| Expert | Watch / read | Today's connection |
|---|---|---|
| Andrew Ng | [Convolutional Neural Networks, Weeks 3–4](https://www.coursera.org/learn/convolutional-neural-networks) | YOLO, detection and neural style transfer |
| Yaser Abu-Mostafa | [Learning From Data, Lecture 13: Validation](https://www.youtube.com/watch?v=o7zzaKd0Lkk) | choosing models and thresholds without test-set snooping |
| Yann LeCun | [NYU Deep Learning: CNNs](https://atcold.github.io/NYU-DLSP21/) · [LeCun, Bengio & Hinton, Deep Learning](https://www.nature.com/articles/nature14539) | locality, learned hierarchies and reusable visual features |
| Yoshua Bengio | [Goodfellow, Bengio & Courville, Deep Learning, Ch. 9](https://www.deeplearningbook.org/contents/convnets.html) | convolution and representation learning behind today's backbones |
| Andrej Karpathy | [CS231n Detection and Segmentation](https://cs231n.stanford.edu/slides/2017/cs231n_2017_lecture11.pdf) · [Transfer Learning](https://cs231n.github.io/transfer-learning/) · [Training Recipe](https://karpathy.github.io/2019/04/25/recipe/) | architectures, fine-tuning and disciplined debugging |
| Sebastian Raschka | [STAT 453: CNNs](https://sebastianraschka.com/blog/2021/dl-course.html#l13-introduction-to-convolutional-neural-networks) · [Data-augmentation notebook](https://github.com/rasbt/stat453-deep-learning-ss21/blob/main/L10/code/data-augmentation.ipynb) | PyTorch tensors and augmentation practice |
| Josh Starmer | [StatQuest: Convolutional Neural Networks](https://www.youtube.com/watch?v=HGwBXDKFk9I) | visual intuition for features reused by detection and segmentation |

</div>

<style>
.experts td, .experts th { padding-top: 0.32em; padding-bottom: 0.32em; }
</style>

<span class="refs">Main text: [d2l.ai, Chapter 14 — Computer Vision](https://d2l.ai/chapter_computer-vision/index.html), especially §14.1–14.2 and §14.7–14.12</span>

<!--
Self-study slide; do not spend lecture time reading the table. Ng supplies the closest detection and neural-style walkthroughs; Karpathy and Raschka the practical workflow; LeCun and Bengio the representation-learning foundations; Abu-Mostafa the validation discipline; Starmer a visual CNN refresher. No unverified quotation is attributed to these instructors. Some Coursera videos require enrollment.
-->

---
zoom: 1.1
hideInToc: true
---

# Further Reading

<div class="grid grid-cols-2 gap-8 text-sm">
<div>

### Main text: d2l.ai Chapter 14
- [14.1 Image Augmentation](https://d2l.ai/chapter_computer-vision/image-augmentation.html) · [14.2 Fine-Tuning](https://d2l.ai/chapter_computer-vision/fine-tuning.html)
- [14.7 SSD](https://d2l.ai/chapter_computer-vision/ssd.html) · [14.8 R-CNNs](https://d2l.ai/chapter_computer-vision/rcnn.html)
- [14.9 Semantic Segmentation](https://d2l.ai/chapter_computer-vision/semantic-segmentation-and-dataset.html)
- [14.10 Transposed Convolution](https://d2l.ai/chapter_computer-vision/transposed-conv.html) · [14.11 FCN](https://d2l.ai/chapter_computer-vision/fcn.html)
- [14.12 Neural Style Transfer](https://d2l.ai/chapter_computer-vision/neural-style.html)

<br>

### Current model sources
- [YOLO26](https://arxiv.org/abs/2606.03748) · [RF-DETR benchmarks](https://rfdetr.roboflow.com/latest/learn/benchmarks/)
- [DINOv3](https://github.com/facebookresearch/dinov3) · [SigLIP 2](https://arxiv.org/abs/2502.14786)
- [SAM 3.1](https://github.com/facebookresearch/sam3/blob/main/RELEASE_SAM3p1.md)
</div>
<div>

### APIs
- [torchvision transforms v2](https://pytorch.org/vision/stable/transforms.html)
- [Model weights](https://pytorch.org/vision/stable/models.html)
- [ConvTranspose2d](https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html)
</div>
</div>


<!--
Leave this slide available for self-study. The generation script needs Python, NumPy, Matplotlib and Pillow. It writes six SVGs and asserts every displayed numerical example. The interactive augmentation component is separate and runs entirely in the browser.

The inherited extra section on segmentation losses follows the end slide; use it for self-study or questions about weighted cross-entropy, focal, Dice, IoU and boundary losses. The external quiz is not part of the Slidev deck.
-->
