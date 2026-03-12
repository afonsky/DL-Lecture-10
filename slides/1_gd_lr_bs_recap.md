# Recap: Training = Optimization

<div class="grid grid-cols-[5fr_3fr] gap-5">
<div>

### What are we optimizing?

<v-clicks>

* We have a model with **parameters** $\theta$<br> (weights & biases)
* We have a **loss function** $\mathcal{L}(\theta)$<br> that measures how bad the model is
* **Training = finding** $\theta$ **that minimizes** $\mathcal{L}$

</v-clicks>

<br>

<v-click>

### The challenge:
* Millions of parameters
* No closed-form solution
* Can't try all combinations
* **We need an iterative algorithm!**

</v-click>

</div>
<div>
  <figure>
    <img src="/ISLRv2_figure_10.4.png" style="width: 300px !important;">
    <figcaption style="color:#b3b3b3ff; font-size: 11px; position: absolute;"><br>NN with 2 hidden layers. Image source:
      <a href="https://hastie.su.domains/ISLR2/ISLRv2_website.pdf#page=416">ISLR Fig. 10.4</a>
    </figcaption>
  </figure>
</div>
</div>

---

# The Mountain Analogy

<div class="grid grid-cols-[3fr_4fr] gap-6">
<div>

### Imagine you're lost in the mountains in thick fog
<br>

<v-clicks>

* You want to get to the **lowest valley** (minimum loss)
* You can't see the whole landscape (too many dimensions)
* You **can feel the slope** under your feet (gradient)
* **Strategy**: always step downhill!
* This is **Gradient Descent**

</v-clicks>

</div>
<div>
  <figure>
    <img src="/Interlaken.jpg" style="width: 480px !important;">
    <figcaption style="color:#b3b3b3ff; font-size: 11px; position: absolute;"><br>Image source: A.B.
    </figcaption>
  </figure>   
</div>
</div>

---

# Gradient Descent: The Core Idea

<div class="grid grid-cols-[3fr_5fr] gap-10">
<div>
  <figure>
    <img src="/3d_loss.png" style="width: 270px !important;">
  </figure>
  <br>
  <figure>
    <img src="/2d_loss.png" style="width: 300px !important;">
    <figcaption style="color:#b3b3b3ff; font-size: 11px; position: absolute;"><br>Images source:
      <a href="https://d2l.ai/">d2l.ai</a>
    </figcaption>
  </figure>
</div>
<div>

### Algorithm:
1. Start with random parameters $\theta_0$
2. Compute the gradient: *"which direction is uphill?"*
3. Take a step in the **opposite** direction:

$$\theta_{t+1} = \theta_t - \eta \nabla \mathcal{L}(\theta_t)$$

4. Repeat until convergence

<br>

<v-click>

### Two key questions:
* **Which direction?** → The gradient $\nabla \mathcal{L}$ tells us
* **How big a step?** → The learning rate $\eta$ controls this

</v-click>

</div>
</div>

---

# Strategy 1: Random Search (Baseline)

```python {all}{maxHeight:'250px'}
# From Andrej Karpathy's CS231n
bestloss = float("inf")
for num in range(1000):
  W = np.random.randn(10, 3073) * 0.0001  # generate random parameters
  loss = L(X_train, Y_train, W)
  if loss < bestloss:
    bestloss = loss
    bestW = W
```

<v-click>

#### Result: **15.5% accuracy** (random chance = 10%)

</v-click>

<v-click>

#### This *guess-and-check* strategy is **horribly inefficient** — we're ignoring all information about the loss landscape!

</v-click>


<br>
<br>
<br>
<br>

##### Slide credit: [Andrej Karpathy, CS231n](https://cs231n.github.io/optimization-1/)

---

# Strategy 2: Random Local Search

```python {all}{maxHeight:'200px'}
W = np.random.randn(10, 3073) * 0.001
bestloss = float("inf")
for i in range(1000):
  Wtry = W + np.random.randn(10, 3073) * 0.0001  # try a small random step
  loss = L(X_train, Y_train, Wtry)
  if loss < bestloss:
    W = Wtry
    bestloss = loss
```

<v-click>

#### Result: **21.4% accuracy** — better, but still wasteful

</v-click>

<v-click>

#### Problem: we try random directions. But the gradient tells us **exactly** which direction to go!

</v-click>

<br>
<br>
<br>
<br>
<br>

##### Slide credit: [Andrej Karpathy, CS231n](https://cs231n.github.io/optimization-1/)

---

# Strategy 3: Following the Gradient ✓

### The gradient tells us the direction of **steepest ascent**

→ We go in the **opposite** direction to descend

<div class="grid grid-cols-[3fr_3fr] gap-6">
<div>

```python
# The actual gradient descent update
while True:
  grad = compute_gradient(loss_fn, data, W)
  W = W - learning_rate * grad
```

<v-click>

#### This is what **every** deep learning framework does under the hood!

</v-click>

</div>
<div>
  <figure>
    <img src="/Gradient_Descent.png" style="width: 300px !important;">
    <figcaption style="color:#b3b3b3ff; font-size: 11px; position: absolute;"><br>Image source:
      <a href="https://easyai.tech/en/ai-definition/gradient-descent/">easyai.tech</a>
    </figcaption>
  </figure>
</div>
</div>

---

# The Learning Rate: Most Important Hyperparameter

<div class="grid grid-cols-[3fr_3fr] gap-12">
<div>

  ## Too small → slow convergence
  <figure>
    <img src="/lr_1.png" style="width: 370px !important;">
  </figure>
</div>
<div>

  ## Too large → divergence!
  <figure>
    <img src="/lr_2.png" style="width: 370px !important;">
    <figcaption style="color:#b3b3b3ff; font-size: 11px; position: absolute;"><br>Images source:
      <a href="https://d2l.ai">d2l.ai</a>
    </figcaption>
  </figure>
</div>
</div>

<br>
<br>
<br>

<v-click>

> *"If you can only tune one hyperparameter, tune the learning rate."*<br> — Andrew Ng

</v-click>

---

<style scoped>
  table {
    font-size: 14px;
  }
</style>

# Learning Rate: Visual Guide

<div class="grid grid-cols-[5fr_4fr] gap-6">
<div>
    <img src="/Learning_Rate.png" style="width: 420px; position: relative">
</div>
<div>
  <figure>
    <img src="/Learning_Rate_Loss.png" style="width: 370px;">
    <figcaption style="color:#b3b3b3ff; font-size: 11px">Images source:
      <a href="https://www.jeremyjordan.me/nn-learning-rate/">jeremyjordan.me</a>
    </figcaption>
  </figure>
</div>
</div>

<v-click>

#### Practical starting points:
| Optimizer | Typical starting LR |
|-----------|---------------------|
| SGD | 0.1, 0.01 |
| SGD + Momentum | 0.01, 0.001 |
| Adam | 0.001, 3e-4 |

</v-click>
