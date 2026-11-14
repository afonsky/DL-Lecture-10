---
zoom: 0.88
---

# Fine-Tuning Today: Head First, Then Careful Unfreezing

<div class="grid grid-cols-[3fr_2fr] gap-7">
<div>

```python
import torch
from torch import nn
from torchvision import models

weights = models.ResNet18_Weights.DEFAULT
model = models.resnet18(weights=weights)
preprocess = weights.transforms()

for parameter in model.parameters():
    parameter.requires_grad = False
model.fc = nn.Linear(model.fc.in_features, 10)
optimizer = torch.optim.AdamW(model.fc.parameters(), lr=3e-4)

# Later: unfreeze one block with a smaller learning rate
for parameter in model.layer4.parameters():
    parameter.requires_grad = True
optimizer.add_param_group({
    "params": model.layer4.parameters(), "lr": 3e-5,
})
```

</div>
<div class="text-sm">

<div class="rounded p-3 bg-blue-50 dark:bg-blue-900">

### **1 · Linear probe**  
#### Learn only the new head first.
</div>
<div class="mt-3 rounded p-3 bg-green-50 dark:bg-green-900">

### **2 · Unfreeze gradually**  
#### Pretrained layers use a smaller LR.
</div>
<div class="mt-3 rounded p-3 bg-orange-50 dark:bg-orange-900">

### **3 · Validate**  
#### The printed rates are starting points, not laws. **See Lectures 2 and 8**
</div>

<div class="mt-4">

#### `weights.transforms()` matches preprocessing to the checkpoint.<br> At inference: `model.eval()` and `torch.inference_mode()`
</div>
</div>
</div>

<span class="refs">Source: [d2l.ai §14.2, Fine-Tuning](https://d2l.ai/chapter_computer-vision/fine-tuning.html) · [torchvision weights API](https://pytorch.org/vision/stable/models.html) · [A. Karpathy, CS231n Transfer Learning](https://cs231n.github.io/transfer-learning/) · [Training recipe](https://karpathy.github.io/2019/04/25/recipe/)</span>

<!--
Two minutes. This is the current weights-enum form of the inherited fine-tuning recipe, not a replacement for it. First train the newly initialized classification head. Then unfreeze the last residual block and give it a smaller learning rate; d2l similarly uses different rates for the new head and pretrained parameters.

The 3e-4 and 3e-5 values are illustrative starting points selected on validation data, not promised optima. The code adds a parameter group after optimizer creation; schedulers and checkpoint resumes need corresponding care. A frozen backbone's batch-normalization running statistics are a separate issue from requires_grad: inspect them rather than assuming freezing disables every state update.

PyTorch is unavailable in the authoring environment, so the API fragment is documentation-checked, not executed. No accuracy result is claimed.
-->
