---
layout: section
---

# Transposed Convolution
Upsampling feature maps

---
zoom: 0.94
---

# The Need for Upsampling

<div class="grid grid-cols-[5fr_4fr] gap-6">
<div>

### **The problem**
CNNs **downsample** through convolutions and pooling:

```
Input:   224 × 224
After conv+pool layers:
         112 → 56 → 28 → 14 → 7
```

For segmentation, we need output at **original resolution**.

### How to go back up?
- ❌ Nearest-neighbor upsampling — no learning
- ❌ Bilinear interpolation — no learning
- ✅ **Transposed convolution** — learnable upsampling!

</div>
<div>

### **Key Insight**

Regular convolution: **many-to-one** mapping
- Multiple input pixels → one output pixel

Transposed convolution: **one-to-many** mapping
- One input pixel → multiple output pixels

<div class="mt-4 bg-blue-50 dark:bg-blue-900 rounded p-3 text-sm">

#### Also called:
- #### Fractionally-strided convolution
- #### Deconvolution (⚠️ technically incorrect name)
</div>

</div>
</div>

---

# How Transposed Convolution Works

<div class="grid grid-cols-2 gap-6">
<div>

### Basic operation (stride=1, no padding)

#### Given **2×2 input** and **2×2 kernel**:
1. Each input element **multiplied** by entire kernel
2. Results **placed** at corresponding positions
3. Overlapping regions are **summed**

<br>

#### **Output size**: $(n + k - 1) \times (n + k - 1)$
- Input: 2×2, Kernel: 2×2 → Output: **3×3**
</div>
<div>
<br>
<br>
<figure>
  <img src="/img/trans_conv.svg" class="h-60 mx-auto">
  <figcaption class="text-xs text-gray-400 text-center mt-1">Transposed convolution (d2l.ai)</figcaption>
</figure>

</div>
</div>


#### Opposite of regular convolution:
- Conv 3×3 input, 2×2 kernel → 2×2 output
- **Transposed** 2×2 input, 2×2 kernel → 3×3 output

---

# Transposed Convolution: Stride

<div class="grid grid-cols-2 gap-6">
<div>

### Stride > 1 for larger upsampling

With **stride=2**: insert gaps between input elements before applying kernel.

- 2×2 input, 2×2 kernel, stride=2 → **4×4 output**

### General output size formula

$$o = (i - 1) \times s - 2p + k$$

where:
- $i$ = input size, $k$ = kernel size
- $s$ = stride, $p$ = padding

### Common recipe for 2× upsampling
Kernel: 4×4, Stride: 2, Padding: 1 → **doubles** spatial dimensions

</div>
<div>

<figure>
  <img src="/img/trans_conv_stride2.svg" class="h-55 mx-auto">
  <figcaption class="text-xs text-gray-400 text-center mt-1">Transposed conv with stride 2 (d2l.ai)</figcaption>
</figure>

<div class="mt-3 text-sm bg-green-50 dark:bg-green-900 rounded p-2">

**In FCN**: Use stride=32 with kernel 64×64 to upsample feature maps back to input resolution in one step!

</div>

</div>
</div>
