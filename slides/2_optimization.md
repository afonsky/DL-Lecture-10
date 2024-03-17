# Optimization Problems

* General form:<br>
$\mathrm{minimize} f(\mathbf{x})~$ subject to $\mathbf{x} \in C$
  * Cost function $f$: $\R^n \rightarrow \R$

  * Constraint set example<br>
$C = \{ \mathbf{x} ~|~ h_1(\mathbf{x}) = 0, ..., h_m(\mathbf{x}) = 0, ~g_1(\mathbf{x}) \leq 0, ..., g_r(\mathbf{x}) \leq 0\}$

  * Unconstraint if $C = \R^n$

---

# Local Minima and Global Minima

* Most optimization problems have no close form solution

* We then aim to find a minima through iterative methods

<div class="grid grid-cols-[3fr_3fr] gap-20">
<div>

* Global minima $\mathbf{x}^\ast$:<br>
$f(\mathbf{x}^\ast) \leq f(\mathbf{x}),~ \forall \mathbf{x} \in C$
<br>
<br>

* Local minima $\mathbf{x}^\ast$:
$f(\mathbf{x}^\ast) \leq f(\mathbf{x}),~ \forall \mathbf{x}: \lVert \mathbf{x} - \mathbf{x}^\ast \rVert \leq \epsilon$
</div>
<div>
  <figure>
    <img src="/global_and_local_minima.png" style="width: 350px !important;">
  </figure>
</div>
</div>

---

# Convex Set

<br>
<br>
<div class="grid grid-cols-[3fr_3fr] gap-20">
<div>

* A subset $C$ of $\R^n$ is called **convex** if:

$\alpha \mathbf{x} = (1 - \alpha) \mathbf{y} \in C$

$\forall \alpha \in \lbrack 0, 1 \rbrack ~~~\forall \mathbf{x}, \mathbf{y} \in C$
</div>
<div>
  <figure>
    <img src="/convex_set.png" style="width: 350px !important;">
  </figure>
</div>
</div>

---

# Convex Function

<br>
<br>
<div class="grid grid-cols-[4fr_3fr] gap-10">
<div>

* $f: C \rightarrow \R$ is called **convex** if:

$f (\alpha \mathbf{x} + (1 - \alpha) \mathbf{y}) \leq \alpha f(\mathbf{x}) + (1 - \alpha) f(\mathbf{y})$

$\forall \alpha \in \lbrack 0, 1 \rbrack ~~~\forall \mathbf{x}, \mathbf{y} \in C$

* If the inequality is strict whenever $\alpha \in (0, 1)$ and $\mathbf{x} \neq \mathbf{y}$, then $f$ is called **strictly convex**

</div>
<div>
  <figure>
    <img src="/convex_function.png" style="width: 400px !important;">
  </figure>
</div>
</div>

---

# First-order condition

* $f$ is convex if and only if
$$f(\mathbf{y}) \geq f(\mathbf{x}) + \nabla f(\mathbf{x})^T (\mathbf{y} - \mathbf{x}), ~~~\forall \mathbf{x}, \mathbf{y} \in C$$

* If the inequality is strict, then $f$ is strictly convex
<br>
<br>
<div>
  <figure>
    <img src="/convex_function_2.png" style="width: 650px !important;">
  </figure>
</div>

---

# Second-order conditions

* $f$ is convex if and only if
$$\nabla^2 f(\mathbf{x}) \succeq 0, ~~~\forall \mathbf{x} \in C$$

* $f$ is strictly convex if and only if
$$\nabla^2 f(\mathbf{x}) \succ 0, ~~~\forall \mathbf{x} \in C$$

<br>
<br>

#### Note:<br> $\succeq$ and $\succ$ are **generalized inequalities**, they have been in use to represent partial orderings.<br> Read more here [https://math.stackexchange.com/questions/669085/what-does-curly-curved-less-than-sign-succcurlyeq-mean](https://math.stackexchange.com/questions/669085/what-does-curly-curved-less-than-sign-succcurlyeq-mean)

---

# Convex and Non-convex Examples

<div class="grid grid-cols-[3fr_4fr] gap-5">
<div>

* Convex
  * Linear regression<br> $f(\mathbf{x}) = \lVert \mathbf{Wx} - \mathbf{b} \rVert_2^2$
    * $\nabla f(\mathbf{x}) = 2 \mathbf{W}^T (\mathbf{Wx} - \mathbf{b})$
    * $\nabla^2 f(\mathbf{x}) = 2 \mathbf{W}^T \mathbf{W}$
  * Softmax regression
<br>

* Non-convex
  * Multi-layer perception
  * Convolution neural networks
  * Recurrent neural networks
</div>
<div>
  <figure>
    <img src="/VGG_Resnet_loss_landscapes.png" style="width: 550px !important;">
  </figure>
</div>
</div>

---

# Convex Optimization

* If $f$ is a convex function, and $C$ is a convex set,<br> then the problem is called a convex problem

* Any local minima is a global minima

* Unique global minima if strictly convex
<br>
<br>
<br>
<div>
  <figure>
    <img src="/convex_optimization.png" style="width: 450px !important;">
  </figure>
</div>

---

# Convex Optimization Proof

### Assume local minima $\mathbf{x}$, if exists a global minima $\mathbf{y}$

* Choose $\alpha \leq 1 - \frac{\epsilon}{\lvert \mathbf{x} + \mathbf{y} \rvert}$ and $\mathbf{z} = \alpha\mathbf{x} + (1 - \alpha)\mathbf{y}$

* Then $\lVert \mathbf{x} - \mathbf{z} \rVert = (1 - \alpha) \lVert \mathbf{x} + \mathbf{y} \rVert \leq \epsilon$

* Due to $\mathbf{y}$ is a global minima, so $f(\mathbf{y}) < f(\mathbf{x})$
$$f(\mathbf{z}) \leq \alpha f(\mathbf{x}) + (1 - \alpha) f(\mathbf{z}) < \alpha f(\mathbf{x}) + (1 - \alpha) f(\mathbf{x}) = \mathbf{x}$$

* It contradicts $\mathbf{x}$ is a local minima