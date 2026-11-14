---
layout: section
---

# Neural Style Transfer
When CNNs become artists

---
zoom: 0.89
---

# Neural Style Transfer: The Idea

<div class="grid grid-cols-[3fr_2fr] gap-6">
<div>

### [Gatys et al. (2016)](https://arxiv.org/abs/1508.06576)

Combine the **content** of one image with the **style** of another.

### How it works
1. Take a pretrained CNN (VGG-19)
2. Define what "content" and "style" mean:
   - **Content** = activations in **deep layers**
   - **Style** = correlations between channels (**Gram matrix**) in **multiple layers**
3. Start with content image (or noise)
4. **Optimize the pixels** to minimize:

$$\mathcal{L} = \alpha \cdot \mathcal{L}_{\text{content}} + \beta \cdot \mathcal{L}_{\text{style}} + \gamma \cdot \mathcal{L}_{\text{TV}}$$

The **image** is the variable — not the network weights!

</div>
<div>

<figure>
  <img src="/img/style-transfer.svg" class="h-65 mx-auto">
  <figcaption class="text-xs text-gray-400 text-center mt-1">Style transfer (d2l.ai)</figcaption>
</figure>

<figure class="mt-2">
  <img src="/img/neural-style.svg" class="h-48 mx-auto">
  <figcaption class="text-xs text-gray-400 text-center mt-1">Process overview (d2l.ai)</figcaption>
</figure>

</div>
</div>

---
zoom: 0.9
---

# Style Transfer: Content & Style

<div class="grid grid-cols-2 gap-6">
<div>

### Content Loss

Compare feature maps at a deep layer:

$$\mathcal{L}_{\text{content}} = \frac{1}{2}\sum_{i,j}\left(F^l_{ij} - P^l_{ij}\right)^2$$

- $F^l$ = features of generated image at layer $l$
- $P^l$ = features of content image at layer $l$

**Deeper layers** capture higher-level content (objects, layout) without pixel-level detail.

### VGG-19 Layer Choices
- **Content**: `conv4_2` (deep → captures structure)
- **Style**: `conv1_1`, `conv2_1`, `conv3_1`, `conv4_1`, `conv5_1` (all scales)

</div>
<div>

### Style Loss (Gram Matrix)

The **Gram matrix** captures feature correlations:

$$G^l_{ij} = \sum_k F^l_{ik} F^l_{jk}$$

Style loss compares Gram matrices:

$$\mathcal{L}_{\text{style}} = \sum_l w_l \sum_{i,j}\left(G^l_{ij} - A^l_{ij}\right)^2$$

### Why Gram Matrix = Style?
- Captures **which features co-occur**
- Texture: "orange strokes with blue edges"
- Independent of spatial arrangement
- **Total Variation loss** smooths the output

</div>
</div>

---
zoom: 0.8
---

# Style Transfer: Results & Impact

<div class="grid grid-cols-[3fr_4fr] gap-16">
<div>

### What We Learn from Style Transfer

1. **Early layers** = low-level style (brushstrokes, colors)
2. **Deep layers** = high-level content (objects, layout)
3. CNNs learn a **hierarchy of representations**
4. These representations are **disentangled** (content vs style)

<br>

### Practical Applications
- **Photo filters** (Prisma, Artisto apps)
- **Artistic creation** tools
- **Video stylization**
- **Domain adaptation** (synthetic → real)

</div>
<div>

### From Slow to Real-Time

**Original** ([Gatys et al. (2016)](https://arxiv.org/abs/1508.06576)):
- Iterative optimization per image
- ~Minutes per image on GPU

**Fast Style Transfer** ([Johnson et al., 2016](https://arxiv.org/abs/1611.04558)):
- Train a **feedforward network** for each style
- ~15ms per image (real-time!)

**Arbitrary Style** ([Huang & Belongie, 2017](https://arxiv.org/abs/1703.06868)):
- **AdaIN** (Adaptive Instance Normalization)
- One network for **any** style
- Real-time

<div class="mt-3 bg-blue-50 dark:bg-blue-900 rounded p-3 text-sm">

### 💡 **Key insight**:<br>
#### Style transfer demonstrates that CNN features capture meaningful visual concepts at different levels of abstraction.

</div>

</div>
</div>
