---
zoom: 0.89
---

# One Image, Three Different Targets

<figure>
  <img src="/img/lecture10/segmentation_flavors.svg" class="h-72 mx-auto">
  <figcaption style="color:#b3b3b3ff; font-size: 11px; position: relative; top: 10px">Image source: original computed illustration after <a href="https://d2l.ai/chapter_computer-vision/semantic-segmentation-and-dataset.html">d2l.ai §14.9.1–14.9.2, semantic and instance segmentation definitions</a> and <a href="https://arxiv.org/abs/1801.00868">Kirillov et al., Panoptic Segmentation</a></figcaption>
</figure>

<br>

<div class="grid grid-cols-3 gap-5 mt-4 text-center text-sm">
<div class="rounded p-3 bg-blue-50 dark:bg-blue-900">

### **Semantic**  
#### pixel → **class ID**  
#### both dogs share one label
</div>
<div class="rounded p-3 bg-green-50 dark:bg-green-900">

### **Instance**  
#### pixel → **instance ID + class**  
#### each dog is separate
</div>
<div class="rounded p-3 bg-purple-50 dark:bg-purple-900">

### **Panoptic**  
#### every pixel: countable **things** + uncountable **stuff**
</div>
</div>

<div class="mt-4 text-center">Different targets require different annotations, architectures and metrics.</div>

<!--
Ninety seconds. Point at the same two dogs in all four panels. Semantic segmentation assigns the dog class to both animals; instance segmentation separates dog 1 from dog 2; panoptic segmentation keeps that instance distinction for things while also labeling stuff such as grass and sky.

The instance panel intentionally leaves the background generic: its focus is object instances. The panoptic panel assigns a region to every pixel. A list of boxes, a semantic mask and an instance mask are not interchangeable targets. Mask R-CNN addresses the instance case; the inherited FCN/U-Net slides primarily address semantic segmentation.

reproduction: <code>scripts/lecture10/make_figures.py</code>
-->
