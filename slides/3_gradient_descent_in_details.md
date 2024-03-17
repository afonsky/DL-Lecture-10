# Gradient Descent in Details
<br>


### Algorithm

<div class="grid grid-cols-[3fr_4fr] gap-5">
<div>
<br>

* Choose initial $\mathbf{x}_0$

* At time (step) $t = 1, ..., T$<br>
$\mathbf{x}_t = \mathbf{x}_{t-1} - \eta \nabla f(\mathbf{x}_{t-1})$
  * $\eta$ is learning rate
</div>
<div>
  <figure>
    <img src="/gradient_descent_2.png" style="width: 400px !important;">
  </figure>
</div>
</div>

---

# The Choice of Learning Rate

* Given $\lVert \mathbf{\Delta} \rVert < \epsilon$, for any $f$, by the Taylor expansion<br><br>
$$f(\mathbf{x} + \mathbf{\Delta}) \approx f(\mathbf{x}) + \mathbf{\Delta}^T \nabla f(\mathbf{x})$$

* Choose small enough learning rate $\eta \leq \frac{\epsilon}{\lVert \nabla f(\mathbf{x}) \rVert}$<br>
<br>

$$\lVert - \eta \nabla f(\mathbf{x}) \rVert \leq \epsilon$$
<br>

$$f(\mathbf{x} - \eta \nabla f(\mathbf{x})) \approx f(\mathbf{x}) - \eta \lVert \nabla f(\mathbf{x}) \rVert^2 \leq f(\mathbf{x})$$

---

# Convergence Rate

* Assume $f$ is convex, and its gradient is Lipschitz
continuous with constant $L$:<br>
$\lVert \nabla f(\mathbf{x}) - \nabla f(\mathbf{y}) \rVert \leq L \lVert \mathbf{x} - \mathbf{y} \rVert$
  * Gradient does not change dramatically


* If use learning rate $\eta \leq 1/L$, after $T$ steps<br>
$f(\mathbf{x}_T) - \mathbf{x}^\ast \leq \frac{\lVert \mathbf{x}_0 - \mathbf{x}^\ast \rVert^2}{2 \eta T}$
  * Convergence rate $\mathcal{O}(1/T)$
  * To get $f(\mathbf{x}_T) - f(\mathbf{x}^\ast) \leq \epsilon$, needs $\mathcal{O}(1/\epsilon)$ iterations

---

# Convergence Rate Proof (I)

* Gradient $L$-Lipschitz means
$$f(\mathbf{y}) \leq f(\mathbf{x}) + \nabla f(\mathbf{x})^T (\mathbf{y} - \mathbf{x}) + \frac{L}{2}\lVert \mathbf{y} - \mathbf{x} \rVert^2$$

* Plug in $\mathbf{y} = \mathbf{x} - \eta \nabla f(\mathbf{x})$
$$f(\mathbf{y}) \leq f(\mathbf{x}) - \bigg(1 - \frac{L\eta}{2} \bigg) \eta \lVert \nabla f(\mathbf{x}) \rVert^2$$

* Take $0 < \eta \leq 1/L$
$$f(\mathbf{y}) \leq f(\mathbf{x}) - \frac{\eta}{2} \lVert \nabla f(\mathbf{x}) \rVert^2$$

---

# Convergence Rate Proof (II)

* By the convexity: $f(\mathbf{x}) \leq f(\mathbf{x}^\ast) + \nabla f(\mathbf{x})^T (\mathbf{x} - \mathbf{x}^\ast)$

* Plug in to $f(\mathbf{y}) \leq f(\mathbf{x}) - \frac{\eta}{2} \lVert \nabla f(\mathbf{x}) \rVert^2$
$$f(\mathbf{y}) \leq f(\mathbf{x}^\ast) + \nabla f(\mathbf{x})^T (\mathbf{x} - \mathbf{x}^\ast) - \frac{\eta}{2} \lVert \nabla f(\mathbf{x}) \rVert^2$$

$$\begin{aligned} & f(\mathbf{y}) - f(\mathbf{x}^\ast) \leq \big(2 \eta \nabla f(\mathbf{x})^T (\mathbf{x} - \mathbf{x}^\ast) - \eta^2 \lVert \nabla f(\mathbf{x}) \rVert^2 \big)/2\eta\\[10pt]
=& \big( \lVert \mathbf{x} - \mathbf{x}^\ast \rVert^2 + 2 \eta \nabla f(\mathbf{x})^T (\mathbf{x} - \mathbf{x}^\ast) - \eta^2 \lVert \nabla f(\mathbf{x}) \rVert^2 - \lVert \mathbf{x} - \mathbf{x}^\ast \rVert^2 \big) / 2 \eta \\[10pt]
=& \big( \lVert \mathbf{x} - \mathbf{x}^\ast \rVert^2 - \lVert \mathbf{x} - \eta \nabla f(\mathbf{x}) - \mathbf{x}^\ast \rVert^2 \big) / 2 \eta\\[10pt]
=& \big( \lVert \mathbf{x} - \mathbf{x}^\ast \rVert^2 - \lVert \mathbf{y} - \mathbf{x}^\ast \rVert^2 \big) / 2 \eta \end{aligned}$$

---

# Convergence Rate Proof (III)

* Sum all $T$ steps
$$\sum_{t=1}^T f(\mathbf{x}) - f(\mathbf{x}^\ast) \leq \sum_{t=1}^T \big( \lVert \mathbf{x}_{t-1} - \mathbf{x}^\ast \rVert^2 - \lVert \mathbf{x}_{t} - \mathbf{x}^\ast \rVert^2 \big) / 2 \eta$$

$$= \big( \lVert \mathbf{x}_0 - \mathbf{x}^\ast \rVert^2 - \lVert \mathbf{x}_{T} - \mathbf{x}^\ast \rVert^2 \big) / 2 \eta \leq \lVert \mathbf{x}_0 - \mathbf{x}^\ast \rVert^2 / 2 \eta$$

* $f$ is decreasing every time:
$$f(\mathbf{x}_T) - f(\mathbf{x}^\ast) \leq \frac{1}{T} \sum_{t=1}^T f(\mathbf{x}_t) - f(\mathbf{x}^\ast) \leq \frac{\lVert \mathbf{x}_0 - \mathbf{x}^\ast \rVert^2}{2 \eta T}$$

---

# Apply to Deep Learning

* $f$ is the sum of loss over all training data, $\mathbf{x}$ is the learnable parameters
$$f(\mathbf{x}) = \frac{1}{n} \sum_{i=0}^n \ell_i (\mathbf{x}),$$

where $\ell_i (\mathbf{x})$ is the loss for the $i$-th example

* $f$ is often not convex, so the convergence analysis described before cannot be applied 😔