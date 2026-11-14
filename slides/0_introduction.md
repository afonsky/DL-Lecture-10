---
hideInToc: true
---

# Last Lecture Recap: Computer Vision I

<div class="grid grid-cols-2 gap-6">
<div>

### Object Detection Pipeline
- **Bounding boxes** & IoU metric
- **Anchor boxes** & region proposals
- **Non-Maximum Suppression** (NMS)

<br>

### R-CNN Family (Two-Stage Detectors)
- **R-CNN** — selective search + CNN per region
- **Fast R-CNN** — shared CNN + RoI pooling
- **Faster R-CNN** — Region Proposal Network (RPN)

</div>
<div>

<br>


<figure>
  <img src="/img/faster-rcnn.svg" class="h-70 mx-auto">
  <figcaption class="text-xs text-gray-400 text-center mt-1">Faster R-CNN (d2l.ai)</figcaption>
</figure>

</div>
</div>

<div class="text-sm text-gray-500 mt-2">

**Key insight**: Two-stage detectors are accurate but slow — first propose regions, then classify each one.

</div>