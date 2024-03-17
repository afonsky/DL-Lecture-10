# Stochastic Gradient Descent

<div class="grid grid-cols-[3fr_4fr] gap-10">
<div>

### Algorithm
* At time (step) $t$, sample example $t_i$<br>
$\mathbf{x}_t = \mathbf{x}_{t-1} - \eta \nabla \ell_{t_i} (\mathbf{x}_{t-1})$

$f(\mathbf{x}) = \frac{1}{n}\sum\limits_{i=0}^n \ell_i (\mathbf{x})$
</div>
<div>
  <figure>
    <img src="/sgd.png" style="width: 290px !important;">
  </figure>
</div>
</div>
<br>

<div class="grid grid-cols-[3fr_4fr] gap-10">
<div>

* Compare to gradient descent<br>
$\mathbf{x}_t = \mathbf{x}_{t-1} - \eta \nabla f(\mathbf{x}_{t-1})$<br>

$f(\mathbf{x}) = \frac{1}{n}\sum\limits_{i=0}^n \ell_i (\mathbf{x})$
</div>
<div>
  <figure>
    <img src="/gradient_descent_2.png" style="width: 290px !important;">
  </figure>
</div>
</div>

---

# Sample Example

* Two rules to sample example it at time $t$:
  1. **Random rule**: choose $i_t \in \{1,..., n\}$ uniformly at random
  1. **Cyclic rule**: choose $i_t = 1, 2, ..., n, 1, 2, ..., n$
    * Often called **incremental gradient descent**

* **Randomized rule** is more common in practice<br><br>
$\mathbb{E} \big[ \nabla \ell_{t_i} (\mathbf{x}) \big] = \mathbb{E} \big[ \nabla f(\mathbf{x}) \big]$
  * An unbiased estimate of the gradient

---

# Convergence Rate

* Assume $f$ is convex with a diminishing $\eta$, e.g. $\eta = \mathcal{O} (1 / t)$
$$\mathbb{E} \big[ f(\mathbf{x}_T) \big] - f(\mathbf{x}^\ast) = \mathcal{O} (1 / \sqrt{T})$$

* Under the same assumption, for gradient descent
$$f(\mathbf{x}_T) - f(\mathbf{x}^\ast) = \mathcal{O} (1 / \sqrt{T})$$

* Assume gradient $L$-Lipschitz and fixed $\eta$<br>
$f(\mathbf{x}_T) - f(\mathbf{x}^\ast) = \mathcal{O} (1 / T)$
  * Does not improve for stochastic gradient descent

---

# Stochastic Gradient Descent in Practice

* Does not diminish the learning rate so dramatically
  * We don’t care about optimizing to high accuracy

* Despite converging slower, SGD is way faster on computing the gradient than GD in each iteration
  * Specially for deep learning with complex models and large-scale datasets

---

# Mini-batch Stochastic Gradient Descent

### Algorithm:

* At time $t$, sample a random subset $I_t \subset \{ 1, ..., n \}$ with $\lvert I_t \rvert = b$
$$\mathbf{x}_t = \mathbf{x}_{t-1} - \frac{\eta_t}{b} \sum_{i \in I_t} \nabla \ell_i (\mathbf{x}_{t-1})$$

* Again, it’s an unbiased estimate
$$\mathbb{E} \bigg[ \frac{1}{b} \sum_{i \in I_t} \nabla \ell_i (\mathbf{x}) \bigg] = \nabla f(\mathbf{x})$$

* It reduces variance by a factor of $1/b$ compared to SGD