# Stochastic Gradient Descent (SGD)

### Core idea: approximate the gradient using a **single** random example

<div class="grid grid-cols-[4fr_3fr] gap-10">
<div>

**Full-batch GD** — uses all $n$ examples:

$$\theta_{t+1} = \theta_t - \eta \cdot \frac{1}{n}\sum_{i=1}^{n} \nabla \ell(\theta_t; x_i, y_i)$$

**SGD** — uses just **one** random example $i_t$:

$$\theta_{t+1} = \theta_t - \eta \cdot \nabla \ell(\theta_t; x_{i_t}, y_{i_t})$$

</div>
<div>
  <figure>
    <img src="/sgd.png" style="width: 290px !important;">
    <figcaption style="color:#b3b3b3ff; font-size: 11px;"><br>SGD path: noisy but makes progress</figcaption>
  </figure>
</div>
</div>

<v-click>

### Why does this work?
* On average, the single-example gradient points in the right direction
* $\mathbb{E}[\nabla \ell_{i_t}(\theta)] = \nabla \mathcal{L}(\theta)$ — it's an **unbiased estimate**
* Trade-off: **noisy** but **fast** per step

</v-click>

---

# Mini-batch SGD: The Practical Middle Ground

### Instead of 1 example or all examples, use a **mini-batch** of $b$ examples:

$$\theta_{t+1} = \theta_t - \frac{\eta}{b} \sum_{i \in B_t} \nabla \ell(\theta_t; x_i, y_i)$$

where $B_t$ is a random subset of size $b$

<br>

<div class="grid grid-cols-[1fr_1fr_1fr] gap-4">
<div>

#### Batch GD ($b = n$)
* Exact gradient
* 1 update per pass
* Very slow per update
* Stable convergence

</div>
<div>

#### SGD ($b = 1$)
* Very noisy gradient
* $n$ updates per pass
* Fast per update
* Very noisy convergence

</div>
<div>

#### Mini-batch ($b = 32...512$)
* **Good gradient estimate**
* **Good GPU utilization**
* **Fast and stable**
* **This is what everyone uses!**

</div>
</div>

---

<style scoped>
  table {
    font-size: 16px;
  }
</style>

# Terminology: Epochs, Iterations, Batches

#### Important vocabulary:

<v-clicks>

* **Epoch**: one full pass through the entire training dataset
* **Batch** (mini-batch): a subset of training examples used in one update
* **Iteration** (step): one gradient update using one batch

</v-clicks>

<v-click>

<br>

#### Example: CIFAR-10 (50,000 training images, batch size = 100)

| | |
|---|---|
| Examples per epoch | 50,000 |
| Batch size | 100 |
| **Iterations per epoch** | **500** |
| Training for 100 epochs | 50,000 total iterations |

</v-click>

<v-click>

<br>

> In PyTorch: `DataLoader(dataset, batch_size=100, shuffle=True)` handles batching for you!

</v-click>

---

# Batch Size: A Critical Hyperparameter

<br>

<div class="grid grid-cols-[3fr_3fr] gap-8">
<div>

### Small batch (e.g., 16-64)

<v-clicks>

* **More noise** → acts as regularization!
* Often **generalizes better**
* Lower GPU utilization
* More frequent updates

</v-clicks>

</div>
<div>

### Large batch (e.g., 512-4096)

<v-clicks>

* **Less noise** → more stable gradients
* Better **GPU utilization** (parallelism)
* Can lead to **sharp minima** (worse generalization)
* Fewer updates per epoch

</v-clicks>

</div>
</div>

<v-click>

#### Key insights:
</v-click>

<v-click>

> *"Powers of 2 work well because of GPU memory alignment: **32, 64, 128, 256, 512**. Start with 64 or 128."*<br> — Andrew Ng
</v-click>

<br>
<v-click>

> *"SGD noise isn't a bug — it's a **feature**! The noise helps escape sharp minima and find flatter, more generalizable solutions."*<br> — Sebastian Raschka
</v-click>

---

# SGD in Practice: Shuffling Matters

<br>

### Always **shuffle** your data each epoch!

<br>

<div class="grid grid-cols-[3fr_3fr] gap-8">
<div>

#### Without shuffling:
```python
# BAD: same order every epoch
for epoch in range(100):
    for batch in fixed_batches:
        train(batch)
```

* Model sees same patterns in same order
* Can lead to **biased gradients**

</div>
<div>

#### With shuffling:
```python
# GOOD: random order each epoch
for epoch in range(100):
    shuffle(dataset)
    for batch in create_batches(dataset):
        train(batch)
```

* Each epoch sees data in a new order
* Better gradient estimates

</div>
</div>

<v-click>

<br>

> PyTorch `DataLoader` does this automatically with `shuffle=True`

</v-click>
