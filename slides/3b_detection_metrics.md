---
zoom: 0.9
---

# Focal Loss: Make Easy Background Cheap

<figure>
  <img src="/img/lecture10/focal_loss_weights.svg" class="h-80 mx-auto">
  <figcaption style="color:#b3b3b3ff; font-size: 11px; position: relative; top: 10px">Image source: computed from <a href="https://arxiv.org/abs/1708.02002">Lin et al., Focal Loss for Dense Object Detection</a> and <a href="https://d2l.ai/chapter_computer-vision/ssd.html#exercises">d2l.ai §14.7, exercise 1</a></figcaption>
</figure>

<div class="grid grid-cols-3 gap-5 mt-4 text-center text-sm">
<div class="rounded p-3 bg-gray-50 dark:bg-gray-800">

### $\gamma = 0$ 
#### ordinary cross-entropy
</div>
<div class="rounded p-3 bg-orange-50 dark:bg-orange-900">

### $p_t = 0.9$, $\gamma = 2$
#### loss multiplier = **0.01**
</div>
<div class="rounded p-3 bg-blue-50 dark:bg-blue-900">

### **Hard / wrong examples**  
#### keep relatively more weight
</div>
</div>

<span class="refs">Cross-entropy: **See Lecture 2** · Visual companion: [A. Karpathy, CS231n Detection and Segmentation](https://cs231n.stanford.edu/slides/2017/cs231n_2017_lecture11.pdf)</span>

<!--
Ninety seconds. Read the curve, not a derivation. At p_t = 0.9 and gamma = 2, the focal multiplier is (1 - 0.9)^2 = 0.01: that example contributes one hundredth of the cross-entropy weight. At p_t = 0.5 the multiplier is 0.25. The underlying -log(p_t) term is still present.

The point is not to memorize gamma. It is to prevent a huge collection of easy background predictions from drowning out difficult objects during training. Focal loss is a training objective; it is not NMS and not an evaluation metric. Keep this slide even if the inherited formula slide is skipped.

reproduction: <code>scripts/lecture10/make_figures.py</code>
-->

---
zoom: 0.8
---

# Detection Evaluation: Ranked Boxes → AP → COCO AP

<figure>
  <img src="/img/lecture10/detection_ap.svg" class="h-78 mx-auto">
  <figcaption style="color:#b3b3b3ff; font-size: 11px; position: relative; top: 10px">Image source: original toy example computed for this lecture after the <a href="https://cocodataset.org/#detection-eval">COCO detection evaluation definitions</a></figcaption>
</figure>

<div class="grid grid-cols-3 gap-4 text-sm mt-2">
<div>

### **1 · Match predictions**  
Same class + sufficient IoU; a ground-truth object accepts one TP. Later duplicates are FP.
</div>
<div>

### **2 · Sweep the score threshold**  
Ranked TP/FP outcomes trace precision–recall. **AP** summarizes the area.
</div>
<div>

### **3 · Average explicitly**  
COCO AP averages IoU thresholds **0.50:0.05:0.95** and classes. AP₅₀ uses only IoU 0.50.
</div>
</div>

<div class="mt-3 rounded p-2 text-center bg-yellow-50 dark:bg-yellow-900 text-sm">

#### **Score cutoff ≠ NMS IoU threshold ≠ evaluation IoU threshold.**<br> Report the AP variant, dataset and max-detection setting.
</div>

<span class="refs">IoU and NMS: **See Lecture 9** · Precision/recall: **See Lecture 2** · [COCO evaluation protocol](https://cocodataset.org/#detection-eval)</span>

<!--
Three minutes. The toy example uses one class, three ground-truth objects and evaluation IoU 0.5. The ranked outcomes are TP, TP, FP, TP, FP, FP. With interpolation at the three distinct recall levels, AP = (1 + 1 + 0.75) / 3 = 0.9167. This is a deliberately small continuous-interpolation example, not a claim about COCO's exact 101-point implementation.

Stress the pipeline: confidence ranking determines the precision-recall curve; NMS is earlier post-processing that can change the predicted set; evaluation IoU decides whether a predicted box can match ground truth. These are three different roles.

COCO's headline AP, often written AP50:95 or informally "mAP", averages over 10 IoU thresholds and categories. It is not VOC 2007 mAP at IoU 0.5. Never compare two numbers printed only as "mAP" without checking the protocol. This closes Lecture 9's AP/mAP gap and is guaranteed quiz material.

reproduction: <code>scripts/lecture10/make_figures.py</code>
-->
