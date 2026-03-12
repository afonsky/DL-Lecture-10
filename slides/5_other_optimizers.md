# The Problem with Vanilla SGD

### SGD struggles in "ravines" — narrow valleys that are common in loss surfaces

<br>

<div class="grid grid-cols-[6fr_4fr] gap-8">
<div>

<v-clicks>

* The gradient zigzags across the narrow direction
* Very slow progress along the important direction
* Think of a ball rolling down a tilted gutter — it bounces side to side

</v-clicks>

<v-click>

<br>

### We need smarter update rules!

Two key ideas:
1. **Momentum** — remember past gradients
2. **Adaptive learning rates** — different step size per parameter

</v-click>

</div>
<div>
  <figure>
    <img src="/sgd.png" style="width: 320px !important;">
    <figcaption style="color:#b3b3b3ff; font-size: 11px;"><br>SGD oscillates in ravines</figcaption>
  </figure>
</div>
</div>

---

# Momentum: The Ball Rolling Downhill

#### Intuition: give the optimizer "memory" of past gradients
<br>

<div class="grid grid-cols-[4fr_4fr] gap-8">
<div>

**SGD update** (no momentum):

$$\theta_{t+1} = \theta_t - \eta \cdot g_t$$

**SGD with Momentum**:

$$v_t = \beta \cdot v_{t-1} + g_t$$
$$\theta_{t+1} = \theta_t - \eta \cdot v_t$$

* $v_t$ is the **velocity** (accumulated gradient)
* $\beta$ is the **momentum coefficient** (typically 0.9)

</div>
<div>

### Physical analogy:
<v-clicks>

* Imagine a **ball rolling downhill**
* Without momentum:<br> ball stops at every bump
* With momentum: ball builds up speed and rolls through small bumps
* Consistent gradients → ball **accelerates**
* Oscillating gradients → they **cancel out**

</v-clicks>
<v-click>

> *"Momentum is almost always a good idea. Use 0.9 as default."*<br> — paraphrasing from Andrew Ng

</v-click>

</div>
</div>


---

# Momentum: Visual Comparison

<br>

<div class="grid grid-cols-[3fr_3fr] gap-20">
<div>

### Without Momentum (vanilla SGD)
* Oscillates back and forth
* Slow progress toward minimum
* Gets stuck in narrow valleys

<br>

```
Step 1: →↗   (overshoot)
Step 2: →↙   (correct back)
Step 3: →↗   (overshoot again)
Step 4: →↙   (still zigzagging)
```

</div>
<div>

### With Momentum ($\beta = 0.9$)
* Oscillations get dampened
* Accelerates in consistent direction
* Much faster convergence

<br>

```
Step 1: →↗   (overshoot)
Step 2: →→   (momentum dampens)
Step 3: →→→  (accelerating!)
Step 4: →→→→ (fast progress!)
```

</div>
</div>

<v-click>

<br>

#### Why $\beta = 0.9$?
#### It means: *"weight the current gradient 10% and the accumulated history 90%"*
#### Effectively averages over the ~last 10 gradients: $\frac{1}{1 - \beta} = \frac{1}{0.1} = 10$

</v-click>

---
zoom: 0.9
---

# Adagrad: Adapt Learning Rate Per Parameter

#### Problem: some parameters need big updates, others need small ones
<br>

<div class="grid grid-cols-[3fr_3fr] gap-8">
<div>

#### The idea:
<v-clicks>

* Track **how much each parameter has been updated**
* Parameters with **large past gradients** → **smaller** learning rate
* Parameters with **small past gradients** → **larger** learning rate
* Useful when features have very different scales

</v-clicks>

</div>
<div>

#### The update:

$$s_t = s_{t-1} + g_t^2$$
$$\theta_t = \theta_{t-1} - \frac{\eta}{\sqrt{s_t + \epsilon}} \cdot g_t$$

* $s_t$ accumulates squared gradients
* Dividing by $\sqrt{s_t}$ scales down frequent updates
* $\epsilon \approx 10^{-8}$ prevents division by zero

</div>
</div>

<v-click>

<br>

### Problem with Adagrad:
> $s_t$ only **grows** over time → learning rate shrinks to zero → training stops too early!<br>
> Works well for sparse data (NLP), but problematic for deep learning.

</v-click>

---
zoom: 0.9
---

# RMSProp: Fixing Adagrad's Problem

#### Idea by Geoffrey Hinton: use an **exponential moving average** instead of a sum
<br>

<div class="grid grid-cols-[4fr_3fr] gap-8">
<div>

### Adagrad (problematic):
$$s_t = s_{t-1} + g_t^2$$
$s_t$ keeps growing → LR → 0

<br>

### RMSProp (the fix):
$$s_t = \gamma \cdot s_{t-1} + (1-\gamma) \cdot g_t^2$$
$$\theta_t = \theta_{t-1} - \frac{\eta}{\sqrt{s_t + \epsilon}} \cdot g_t$$

* $\gamma = 0.9$ (typical) — it "forgets" old gradients
* $s_t$ stays bounded → learning rate stays healthy

</div>
<div>

### Why it works:

<v-clicks>

* **Recent gradients matter more** than old ones
* Adapts to the current curvature of the loss surface
* Effectively a "sliding window" over gradient magnitudes

</v-clicks>

<br>
<v-click>

> *"RMSProp was never published. It was proposed in<br> [slide #29 of Lecture 6 of Hinton's Coursera course](https://www.cs.toronto.edu/~hinton/coursera/lecture6/lec6.pdf#page=29)."*<br> — Deep learning folklore

</v-click>

</div>
</div>

---
zoom: 0.82
---

<style scoped>
  table {
    font-size: 16px;
  }
</style>

# Adam: The Best of Both Worlds

### **Adam** = **Ada**ptive **M**oment estimation = Momentum + RMSProp

<br>

<div class="grid grid-cols-[3fr_3fr] gap-8">
<div>

### Step 1: Momentum (1st moment)
$$m_t = \beta_1 \cdot m_{t-1} + (1-\beta_1) \cdot g_t$$

### Step 2: RMSProp (2nd moment)
$$v_t = \beta_2 \cdot v_{t-1} + (1-\beta_2) \cdot g_t^2$$

### Step 3: Bias correction
$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$

### Step 4: Update
$$\theta_t = \theta_{t-1} - \frac{\eta \cdot \hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

</div>
<div>

### Why bias correction?

<v-clicks>

* $m_0 = 0$ and $v_0 = 0$, so early estimates are biased toward zero
* Dividing by $(1 - \beta^t)$ corrects this — effect fades as $t$ grows

</v-clicks>

<v-click>

#### Default hyperparameters:
| Parameter | Default |
|-----------|---------|
| $\eta$ (learning rate) | 0.001 |
| $\beta_1$ (momentum) | 0.9 |
| $\beta_2$ (RMSProp) | 0.999 |
| $\epsilon$ | $10^{-8}$ |

</v-click>

<v-click>

> These defaults work well for **most** problems!

</v-click>

</div>
</div>

---
zoom: 0.95
---

# Adam: Why It's So Popular

<br>

<div class="grid grid-cols-[3fr_3fr] gap-8">
<div>

### Advantages:
<v-clicks>

* **Works well out of the box** — good defaults
* **Adapts per parameter** — no manual LR tuning per layer
* **Handles sparse gradients** — great for NLP/embeddings
* **Fast convergence** — often reaches good solutions quickly
* **Robust** to hyperparameter choices

</v-clicks>

</div>
<div>

### When to use what:

<v-clicks>

* **Adam**: your go-to default for most tasks
* **SGD + Momentum**: when you need best possible generalization (vision tasks, with careful LR tuning)
* **AdamW**: Adam with proper weight decay (preferred over Adam in modern practice)

</v-clicks>

</div>
</div>

<v-click>

<br>

> *"The most common optimization algorithm used in practice is Adam. It works well in most cases."*<br> — **Andrej Karpathy**

</v-click>

---
zoom: 0.85
---

<style scoped>
  table {
    font-size: 18px;
  }
</style>

# Optimizer Comparison: The Big Picture

<br>

| Feature | SGD | SGD+Momentum | Adagrad | RMSProp | Adam |
|---------|-----|-------------|---------|---------|------|
| Per-param LR | No | No | Yes | Yes | Yes |
| Momentum | No | Yes | No | No | Yes |
| Bias correction | - | - | - | No | Yes |
| Memory (extra vars) | 0 | 1x | 1x | 1x | 2x |
| Typical LR | 0.01 | 0.01 | 0.01 | 0.001 | 0.001 |
| Best for | Simple, tuned | Vision (SOTA) | Sparse data | General | **Default choice** |

<v-click>

<br>

### Evolution of optimizers:
```
SGD → SGD+Momentum → Adagrad → RMSProp → Adam → AdamW → NAdam
                                    ↓
                              (per-param LR idea)
```

</v-click>

---

# Learning Rate Scheduling

### The learning rate doesn't have to stay fixed!

<br>

<div class="grid grid-cols-[4fr_2fr] gap-8">
<div>

### Common schedules:

<v-clicks>

* **Step decay**: reduce LR by factor every N epochs
  * e.g., multiply by 0.1 every 30 epochs
* **Cosine annealing**: smooth decrease following a cosine curve
  * Very popular in modern training
* **Warmup + decay**: start low, increase, then decrease
  * Critical for Transformers!
* **Reduce on plateau**: reduce when validation loss stops improving

</v-clicks>

</div>
<div>

### PyTorch example:

```python
optimizer = torch.optim.Adam(
    model.parameters(), lr=0.001
)

# Step decay
scheduler = StepLR(
    optimizer, step_size=30, gamma=0.1
)

# Cosine annealing
scheduler = CosineAnnealingLR(
    optimizer, T_max=100
)

for epoch in range(100):
    train(model, optimizer)
    scheduler.step()
```

</div>
</div>

---

# Practical Tips: What the Experts Say

<br>

<div class="grid grid-cols-[3fr_3fr] gap-8">
<div>

### Andrej Karpathy's advice:

<v-clicks>

*  Start with **Adam**, lr=3e-4
*  If you want to squeeze out the last bit of performance, switch to **SGD+Momentum** with a tuned LR schedule
*  Always **monitor your loss curves** — they tell you everything
*  If loss explodes → **reduce learning rate**
*  If loss plateaus → try **increasing learning rate** briefly

</v-clicks>

</div>
<div>

### Sebastian Raschka's checklist:

<v-clicks>

*  **Overfit a single batch first** — verify the model can learn at all
*  Use **batch normalization** — makes optimization easier
*  Start with well-tested architectures and optimizers
*  **Gradient clipping** prevents exploding gradients (especially for RNNs)
*  **Weight decay** (L2 regularization) in the optimizer helps generalization

</v-clicks>

</div>
</div>

---

<style scoped>
  table {
    font-size: 18px;
  }
</style>

# Common Training Failures and Fixes


| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| Loss not decreasing | LR too low, or bug in code | Increase LR; verify data pipeline |
| Loss exploding (NaN) | LR too high | Decrease LR; add gradient clipping |
| Loss decreasing then plateaus | LR needs scheduling | Add LR decay or cosine schedule |
| Train loss ↓ but val loss ↑ | Overfitting | Add regularization, reduce model size |
| Training is very slow | LR too small, bad batch size | Increase LR; adjust batch size |
| Loss oscillates wildly | LR too high or batch too small | Decrease LR; increase batch size |

<v-click>

<br>

> *"Always plot your learning curves. If you're not looking at your loss curves, you're flying blind."*<br> — Andrej Karpathy

</v-click>

---

# Summary: The Optimization Toolkit

<br>

<v-clicks>

### 1. **Gradient Descent** is the foundation — compute gradient, take a step downhill

### 2. **Mini-batch SGD** makes it practical — use small batches for speed + noise benefits

### 3. **Momentum** adds memory — accelerates through consistent gradients, dampens oscillations

### 4. **Adaptive methods** (RMSProp, Adam) — adjust learning rate per parameter automatically

### 5. **Learning rate scheduling** — decay the LR over training for fine convergence

### 6. **Adam** is your default — start here, tune later if needed

</v-clicks>

<v-click>

<br>

> *"Don't be a hero. Use Adam with 3e-4."*<br> — Andrej Karpathy<br> ([A Recipe for Training Neural Networks, 2019](http://karpathy.github.io/2019/04/25/recipe/))

</v-click>

---

<style scoped>
  table {
    font-size: 14px;
  }
</style>

# Bonus: Optimizers × Architectures Cheat Sheet

| Architecture | Common Optimizer | Typical LR | LR Schedule | Notes |
|---|---|---|---|---|
| **MLP** | Adam | 1e-3 | Step decay | Good baseline for tabular data |
| **CNN** (ResNet, VGG) | SGD + Momentum (0.9) | 0.1 | Cosine / Step ×0.1 | Still SOTA for image classification |
| **RNN / LSTM** | Adam | 1e-3 | Reduce on plateau | + gradient clipping (max norm 1–5) |
| **Transformer** (BERT, GPT) | AdamW | 1e-4 – 5e-5 | Warmup + cosine decay | Warmup is critical (~1-5% of steps) |
| **GAN** | Adam ($\beta_1$=0.0, $\beta_2$=0.9) | 1e-4 – 2e-4 | None or linear decay | Lower $\beta_1$ stabilizes training |
| **Fine-tuning** (pretrained) | AdamW | 1e-5 – 5e-5 | Linear warmup + decay | Much smaller LR than from-scratch |
| **ViT** (Vision Transformer) | AdamW | 1e-3 | Warmup + cosine decay | Similar recipe to NLP Transformers |

<v-click>

<br>

> **Rule of thumb**: CNNs → SGD+Momentum; everything with attention → AdamW + warmup; unsure → Adam with LR 3e-4

</v-click>
