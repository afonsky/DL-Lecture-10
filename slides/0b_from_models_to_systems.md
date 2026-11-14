---
hideInToc: true
zoom: 0.92
---

# Today: From Architectures to Reliable Vision Systems

<div class="grid grid-cols-4 gap-4 text-center mt-8">
<div class="rounded-lg p-4 bg-blue-50 dark:bg-blue-900">
<div class="text-3xl font-bold text-blue-700 dark:text-blue-200">01</div>

### Detect
boxes → ranked predictions

**Evaluate with AP**, not anecdotes
</div>
<div class="rounded-lg p-4 bg-green-50 dark:bg-green-900">
<div class="text-3xl font-bold text-green-700 dark:text-green-200">02</div>

### Segment
image → pixel classes

**Measure overlap**, not just accuracy
</div>
<div class="rounded-lg p-4 bg-orange-50 dark:bg-orange-900">
<div class="text-3xl font-bold text-orange-700 dark:text-orange-200">03</div>

### Adapt
pretrained → your data

**Transform image + target together**
</div>
<div class="rounded-lg p-4 bg-purple-50 dark:bg-purple-900">
<div class="text-3xl font-bold text-purple-700 dark:text-purple-200">04</div>

### Select
2026y model → your constraints

**Benchmark the whole pipeline**
</div>
</div>

<div class="mt-8 text-center text-xl">

**Guiding question:** what must be true before a good demo becomes a dependable system?
</div>

<span class="refs">Main source: [d2l.ai, Chapter 14: Computer Vision](https://d2l.ai/chapter_computer-vision/index.html) · Earlier foundations: **See Lectures 2, 4, 5, 8 and 9**</span>

<!--
Timing plan: 80 minutes total, including the separate 10-minute quiz.
- Opening, recap and roadmap: 4 min
- SSD, YOLO, focal loss and detection evaluation: 17 min
- Segmentation, upsampling, FCN/U-Net and segmentation metrics: 19 min
- Transfer learning and augmentation lab: 10 min
- Neural style transfer: 4 min
- 2026 landscape and model-selection workflow: 11 min
- Wrap-up/resources: 1 min
  Teaching total: 66 min + 10 min quiz = 76 min, leaving 4 min for questions.

Default cuts if running late (the additions are separate; no inherited slide is edited):
1. Skip the inherited YOLO26 performance/architecture image slides; the new real-time comparison carries the current facts.
2. Skip the inherited SSD speed table and focal-loss formula slide; keep the new focal-loss visual and AP slide.
3. Skip the inherited segmentation-dataset and architecture-evolution slides.
4. Skip the inherited neural-style content/style formula slide; keep the idea and results.
5. Skip the inherited EfficientNet slide and the new foundation-backbone slide if necessary.

The quiz is external: shared/DL-L10_quiz.md. Main quiz questions do not depend on these cuts.
-->
