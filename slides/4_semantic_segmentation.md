---
layout: section
---

# Semantic Segmentation
Pixel-level scene understanding

---

# What is Semantic Segmentation?

<div class="grid grid-cols-[3fr_2fr] gap-6">
<div>

### Definition
Assign a **class label to every pixel** in the image.

### Compared to Object Detection
- **Detection**: Where are objects? → bounding boxes
- **Segmentation**: What is each pixel? → pixel-level masks

### Applications
- 🚗 Self-driving: road, sidewalk, pedestrian, car
- 🏥 Medical: tumor boundaries, organ segmentation
- 🛰️ Satellite: land use classification
- 🤖 Robotics: scene understanding

</div>
<div>

<figure>
  <img src="/img/segmentation.svg" class="h-55 mx-auto">
  <figcaption class="text-xs text-gray-400 text-center mt-1">Semantic segmentation labels (d2l.ai)</figcaption>
</figure>

<div class="mt-3 text-sm">

**Output**: Same spatial resolution as input, with each pixel labeled to a class.

</div>

</div>
</div>

---
zoom: 1.05
---

# Segmentation Flavors

<div class="grid grid-cols-3 gap-4 text-sm">
<div class="bg-blue-50 dark:bg-blue-900 rounded p-3">

### Semantic Segmentation
- Every pixel gets a class
- **Same class** → same label
- Two dogs = same color
- *"What is here?"*

</div>
<div class="bg-green-50 dark:bg-green-900 rounded p-3">

### Instance Segmentation
- Distinguishes **individual objects**
- Two dogs = **different** labels
- Combines detection + segmentation
- *"Which object is here?"*

</div>
<div class="bg-purple-50 dark:bg-purple-900 rounded p-3">

### Panoptic Segmentation
- **Combines** semantic + instance
- Labels every pixel
- Distinguishes instances of "things" (countable: cars, people)
- Unifies "stuff" (uncountable: sky, road)

</div>
</div>

<br>

<div class="grid grid-cols-3 gap-4 text-sm">
<div>

```
⬜⬜⬜⬜⬜
⬜🟩🟩⬜⬜    🟩 = dog (all)
⬜🟩🟩⬜⬜    ⬜ = background
⬜⬜🟩🟩⬜
⬜⬜🟩🟩⬜
```

</div>
<div>

```
⬜⬜⬜⬜⬜
⬜🟩🟩⬜⬜    🟩 = dog #1
⬜🟩🟩⬜⬜    🟥 = dog #2
⬜⬜🟥🟥⬜    ⬜ = background
⬜⬜🟥🟥⬜
```

</div>
<div>

```
🟦🟦🟦🟦🟦
⬜🟩🟩⬜⬜    🟩 = dog #1
⬜🟩🟩⬜⬜    🟥 = dog #2
⬜⬜🟥🟥⬜    ⬜ = grass
⬜⬜🟥🟥⬜    🟦 = sky
```

</div>
</div>

---
zoom: 1.05
---

# Common Segmentation Datasets

<div class="grid grid-cols-3 gap-4 text-sm">
<div>

### [Pascal VOC 2012](https://www.robots.ox.ac.uk/~vgg/projects/pascal/VOC/voc2012/)
- **20** object classes + background
- ~10K images
- Pioneering benchmark
- Classes: person, car, cat, dog, …

</div>
<div>

### [Microsoft COCO](https://cocodataset.org/#home)
- **80** object classes
- 200K+ labeled images
- Instance segmentation masks
- Richer annotations
- Current standard benchmark

</div>
<div>

### [Cityscapes](https://www.cityscapes-dataset.com/)
- **30** classes for urban scenes
- 5K fine + 20K coarse annotations
- Street-level imagery
- Focus: autonomous driving

</div>
</div>

<br>

<div class="text-sm">

### Other notable datasets
- [**ADE20K**](https://ade20k.csail.mit.edu/): 150 classes, 20K+ images — diverse scenes
- [**LVIS**](https://www.lvisdataset.org/): long-tail distribution, 1000+ categories
- [**Mapillary Vistas**](https://www.mapillary.com/dataset/vistas): street-level, 124 semantic object categories, global coverage

</div>

<div class="mt-2 bg-blue-50 dark:bg-blue-900 rounded p-2 text-sm">

💡 **Andrew Ng**: *"In practice, having more data usually beats having a better algorithm. But high-quality labels matter enormously for segmentation."*

</div>
