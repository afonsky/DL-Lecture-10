---
zoom: 0.93
---

# Transposed Convolution Restores Shape — Not the Input

<figure>
  <img src="/img/lecture10/transposed_not_inverse.svg" class="h-66 mx-auto">
  <figcaption style="color:#b3b3b3ff; font-size: 11px; position: relative; top: 10px">Image source: recomputed from <a href="https://d2l.ai/chapter_computer-vision/transposed-conv.html#connection-to-matrix-transposition">d2l.ai §14.10.3, connection to matrix transposition</a></figcaption>
</figure>

<div class="grid grid-cols-2 gap-6 mt-4 text-sm">
<div class="rounded p-3 bg-blue-50 dark:bg-blue-900">

### **Shape debugging**  
#### If the output is short by less than one stride, check PyTorch's `output_padding` in `nn.ConvTranspose2d`.
</div>
<div class="rounded p-3 bg-green-50 dark:bg-green-900">

### **Checkerboard artifacts?**  
#### A common alternative is **resize + ordinary convolution**, rather than learned transposed convolution.
</div>
</div>

<span class="refs">Shape conventions: **See Lecture 4** · [PyTorch ConvTranspose2d](https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html) · [Odena et al., Deconvolution and Checkerboard Artifacts](https://distill.pub/2016/deconv-checkerboard/)</span>

<!--
Two minutes. The same 2x2 kernel used for a valid convolution and for transposed convolution gives Y = [[27, 37], [57, 67]] and Z = [[27, 91, 74], [138, 400, 282], [171, 429, 268]]. Z has X's 3x3 shape but is not X. "Transposed" refers to the transposed linear operator, not to undoing learned feature extraction or recovering lost information.

For practice, output_padding resolves the output-size ambiguity when stride > 1; it does not add learned content. The resize-plus-convolution pattern is a useful baseline when transposed convolution produces periodic checkerboard artifacts. Do not claim that one upsampling method is universally best.

reproduction: <code>scripts/lecture10/make_figures.py</code>
-->
