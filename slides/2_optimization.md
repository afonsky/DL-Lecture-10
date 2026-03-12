# The Optimization Landscape

### What does the loss surface look like?

<div class="grid grid-cols-[3fr_3fr] gap-10">
<div>

<v-clicks>

* In deep learning, loss is a function of **millions** of parameters
* We can't visualize it directly<br> (too many dimensions!)
* But we can use PCA and project first components into 2D/3D

</v-clicks>

<br>

<v-click>

### Key insight:
> Deep learning loss surfaces are **non-convex**<br> — they have many hills, valleys, and flat regions

</v-click>

</div>
<div>
  <figure>
    <img src="/VGG_Resnet_loss_landscapes.png" style="width: 450px !important;">
    <figcaption style="color:#b3b3b3ff; font-size: 11px;"><br>VGG-56 (chaotic) vs ResNet-56 (smooth). Source:
      <a href="https://arxiv.org/abs/1712.09913">Li et al., 2018</a>
    </figcaption>
  </figure>
</div>
</div>

---
zoom: 0.95
---

# Convex vs Non-Convex

<div class="grid grid-cols-[3fr_3fr] gap-10">
<div>

### Convex (easy!)
* **Any local minimum = global minimum**
* Linear regression, logistic regression
* Gradient descent always finds the best solution

<br>
  <figure>
    <img src="/convex_function.png" style="width: 350px !important;">
  </figure>
</div>
<div>

### Non-convex (the real world)
* **Multiple local minima**
* All deep neural networks!
* No guarantee of finding the global minimum

<br>
  <figure>
    <img src="/global_and_local_minima.png" style="width: 350px !important;">
  </figure>
</div>
</div>

---

# Challenges in the Loss Landscape


#### Three obstacles that slow down or trap optimization:
<br>
<div class="grid grid-cols-[1fr_1fr_1fr] gap-4">
<div>

#### 1. Local Minima
* A valley that's not the deepest
* Gradient = 0, so GD stops
* Good news:
> *"In high dimensions, most local minima have similar loss to the global minimum"*<br> — Yann LeCun
</div>
<div>

#### 2. Saddle Points
* Gradient $= 0$, but it's neither a minimum nor maximum
* **Much more common** than local minima in high dimensions!
* > *"In high-dimensional spaces, local minima are not really the problem. Saddle points are the problem."*<br> — Yann LeCun

</div>
<div>

#### 3. Plateaus
* Regions where the gradient is *very* small
* Training appears to stall
* The optimizer makes tiny steps and seems stuck

<br>

> *"If you have $n$ parameters, you need all $n$ directions to curve up for a local min"*<br> — Yann LeCun

</div>
</div>

<v-click>


</v-click>

---
zoom: 0.95
---

# Why Non-Convexity Is Not as Scary as It Seems

<br>

<div class="grid grid-cols-[3fr_3fr] gap-10">
<div>

### The good news (empirically observed):

<v-clicks>

* Most local minima in deep networks have **similar loss values**
* Different local minima often give<br> **similar test accuracy**
* Overparameterized networks (more params than data) tend to have many good solutions
* The loss landscape becomes **smoother** with wider networks

</v-clicks>

</div>
<div>

### Practical implications:

<v-clicks>

* We don't need the global minimum!
* We just need a "good enough" minimum
* **The optimizer matters more for speed than final quality**
* Architecture choices (e.g., ResNet skip connections) can make the landscape smoother

</v-clicks>

</div>
</div>

<v-click>

<br>

> *"It's not about finding the global minimum. It's about finding a good solution fast."*<br> — Sebastian Raschka

</v-click>

---

# Convex and Non-convex: Examples from Your Courses

<div class="grid grid-cols-[3fr_4fr] gap-20">
<div>

### Convex (guaranteed optimal)
* Linear regression
* Logistic regression
* Softmax regression (cross-entropy loss)
* SVM (hinge loss)


</div>
<div>
<br>
  <figure>
    <img src="/convex_optimization.png" style="width: 400px !important;">
    <figcaption style="color:#b3b3b3ff; font-size: 11px;"><br>
      Convex: any local minimum is the global minimum
    </figcaption>
  </figure>
</div>
</div>

<br>

### Non-convex (need good optimizers!)
* Multi-layer perceptrons
* Convolutional neural networks
* Recurrent neural networks
* Basically anything with hidden layers