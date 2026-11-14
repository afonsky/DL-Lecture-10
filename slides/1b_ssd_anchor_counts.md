---
zoom: 0.84
---

# SSD Anchor Counts: Two Models, Two Correct Totals

<figure>
  <img src="/img/lecture10/ssd_anchor_counts.svg" class="h-82 mx-auto">
  <figcaption style="color:#b3b3b3ff; font-size: 11px; position: relative; top: 10px">Image source: computed from the <a href="https://arxiv.org/abs/1512.02325">original SSD300 feature maps and boxes per location</a> and <a href="https://d2l.ai/chapter_computer-vision/ssd.html#the-complete-model">d2l.ai §14.7.1, TinySSD complete model</a></figcaption>
</figure>

<div class="grid grid-cols-2 gap-6 mt-4 text-sm">
<div class="rounded p-3 bg-blue-50 dark:bg-blue-900">

### **Original SSD300**  
#### Six scales; **4, 6, 6, 6, 4, 4** anchors per cell  
$(38^2\!\cdot4)+(19^2\!\cdot6)+(10^2\!\cdot6)+(5^2\!\cdot6)+(3^2\!\cdot4)+(1^2\!\cdot4)=\mathbf{8732}$
</div>
<div class="rounded p-3 bg-orange-50 dark:bg-orange-900">

### **d2l TinySSD**  
#### Five scales; **4** anchors per cell  
$(32^2+16^2+8^2+4^2+1^2)\times4=\mathbf{5444}$
</div>
</div>

<div class="text-center mt-3 font-semibold">Always count from the implementation in front of you — not from the model name.</div>

<!--
Two minutes. The inherited architecture slide lists the original SSD300 feature maps and the original 8,732 total. The main d2l text instead builds a small five-block teaching model and prints 5,444. These are not contradictory: the map sizes and the number of boxes per location differ.

Recompute both totals before class with `python3 scripts/lecture10/make_figures.py`. Ask which term dominates: the finest map, because its number of spatial locations is quadratic. This is also why multiscale detection can produce many easy background candidates.

Optional if the lecture is running late, but do not mix the two totals when discussing the inherited slide.

reproduction: <code>scripts/lecture10/make_figures.py</code>
-->
