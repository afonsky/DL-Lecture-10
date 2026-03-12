# Gradient Descent: How It Works Step by Step

<div class="grid grid-cols-[4fr_3fr] gap-5">
<div>

### The Algorithm (pseudocode):

```python
1. Initialize parameters θ randomly
2. for t = 1, 2, ..., T:
     Compute gradient:  g = ∇L(θ)
     Update parameters: θ = θ - η · g
3. Return θ
```

<br>

<v-clicks>

* Each iteration uses the **entire dataset** to compute the gradient
* The gradient points to the steepest direction of **increase**
* We go in the **opposite** direction (hence the minus sign)

</v-clicks>

</div>
<div>
  <figure>
    <img src="/gradient_descent_2.png" style="width: 380px !important;">
  </figure>
</div>
</div>

---

# The Problem with Vanilla Gradient Descent

<br>

### For deep learning, vanilla GD has three major issues:

<br>

<div class="grid grid-cols-[1fr_1fr_1fr] gap-6">
<div>

#### 1. Computational Cost
* Must process **all** training data for **one** update
* ImageNet:<br> 1.2M images per step!
* GPT-scale:<br> billions of tokens

</div>
<div>

#### 2. Slow in Ravines
* Loss surface often has "elongated valleys"
* GD oscillates across the narrow dimension
* Very slow progress along the long dimension

</div>
<div>

#### 3. Memory
* Must store all gradients for all data simultaneously
* For large datasets, this doesn't fit in GPU memory

</div>
</div>

<v-click>

<br>

> *"Nobody uses batch gradient descent in deep learning. Not because it's wrong, but because it's too expensive."*<br> — Andrew Ng

</v-click>

---

# Applying GD to Deep Learning

<br>

### In deep learning, the loss is the average over all training examples:

$$\mathcal{L}(\theta) = \frac{1}{n} \sum_{i=1}^{n} \ell(\theta; x_i, y_i)$$

<v-clicks>

* Computing $\nabla \mathcal{L}(\theta)$ requires going through **all** $n$ examples
* For each example: forward pass + backward pass
* This is called **Batch Gradient Descent** (or "full-batch")

</v-clicks>

<br>

<v-click>


#### Key insight: do we really need the **exact** gradient?
<br>

> A **noisy approximation** computed on a small subset of data is often good enough!
>
> This leads us to **Stochastic Gradient Descent**...

</v-click>