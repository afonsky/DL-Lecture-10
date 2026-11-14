---
layout: section
---

# Transfer Learning<br> &<br> Fine-Tuning
Standing on the shoulders of ImageNet

---
zoom: 0.9
---

# Why Transfer Learning?

<div class="grid grid-cols-2 gap-6">
<div>

### The Data Problem
- Deep nets need **millions** of labeled images
- Your dataset probably has **thousands** (or less)
- Collecting + labeling is **expensive**

### The Solution

> **Andrew Ng**: *"Transfer learning will be the next driver of ML commercial success — after supervised learning."*

A model trained on ImageNet (14M images, 1000 classes) learns **general visual features** that transfer well:
- **Early layers**: edges, textures, colors
- **Middle layers**: patterns, parts
- **Deep layers**: object parts, compositions

</div>
<div>

### When to Use Transfer Learning

<div class="text-sm">

| Your Dataset | Similarity to Source | Strategy |
|---|---|---|
| **Small** + Similar | Freeze backbone, train head | Linear probe |
| **Small** + Different | Fine-tune carefully, lower LR | Fine-tune top layers |
| **Large** + Similar | Fine-tune entire network | Full fine-tune |
| **Large** + Different | Consider training from scratch | Or fine-tune all |

</div>

<br>

<div class="mt-3 bg-blue-50 dark:bg-blue-900 rounded p-3 text-sm">

#### 💡 **Andrej Karpathy**:<br> *"Don't be a hero. Don't try to train from scratch if a pretrained model exists. This is the most common mistake I see."*

</div>

</div>
</div>

---
zoom: 0.85
---

# Fine-Tuning: Step by Step

<div class="grid grid-cols-[3fr_3fr] gap-6">
<div>

### The 4-Step Recipe

**Step 1**: Start with a **pretrained model** (ImageNet)

**Step 2**: **Replace** the output layer to match your classes

**Step 3**: **Freeze** early layers (optional but common for small datasets)

**Step 4**: **Train** on your dataset with a **lower learning rate**

```python
# PyTorch fine-tuning example
model = torchvision.models.resnet18(pretrained=True)

# Freeze all parameters
for param in model.parameters():
    param.requires_grad = False

# Replace final FC layer (trainable)
model.fc = nn.Linear(512, num_my_classes)

# Use smaller learning rate
optimizer = optim.SGD(model.fc.parameters(),
                      lr=0.001)
```

</div>
<div>

<figure>
  <img src="/img/finetune.svg" class="h-85 mx-auto">
  <figcaption class="text-xs text-gray-400 text-center mt-1">Fine-tuning process (d2l.ai)</figcaption>
</figure>

<div class="mt-3 text-sm">

### Practical Tips
- #### Use **smaller LR** for pretrained layers (10-100× smaller)
- #### **Gradually unfreeze** layers during training
- #### **Data augmentation** is critical for small datasets
- #### Start from a model trained on **similar domain**,<br> if possible

</div>

</div>
</div>

---
zoom: 0.95
---

# Fine-Tuning in Practice

<div class="grid grid-cols-2 gap-6">
<div>

### Example: Hot Dog Classification 🌭

From the d2l.ai textbook:
- **Dataset**: 1400 hot dog + 1400 other food images
- **Backbone**: ResNet-18 pretrained on ImageNet
- **Strategy**: Replace last FC, fine-tune

### Results:
- **Without** pretraining: ~60% accuracy
- **With** fine-tuning: **~93%** accuracy!

The pretrained features (edges, textures, shapes of food) transfer beautifully.

</div>
<div>

### What Features Transfer?

<div class="text-sm">

**Layer 1**: Edge detectors, color blobs
→ Transfer to **any** visual task ✅

**Layer 2-3**: Texture patterns, corners
→ Transfer to **most** visual tasks ✅

**Layer 4-5**: Object parts (wheels, eyes)
→ May need fine-tuning for different domains ⚠️

**Final layers**: Task-specific features
→ Must be **replaced** for new task ❌

</div>

<div class="mt-3 bg-green-50 dark:bg-green-900 rounded p-3 text-sm">

**Sebastian Raschka**: *"Fine-tuning pretrained models is one of the most impactful techniques in modern deep learning. It democratizes AI by making powerful models accessible even with limited data and compute."*

</div>

</div>
</div>
