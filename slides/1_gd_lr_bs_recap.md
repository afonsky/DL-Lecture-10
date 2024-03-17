# Recap: Weights and Biases


<div class="grid grid-cols-[5fr_3fr]">
<div>

#### Consider a neural network with 2 hidden layers:
* The first hidden layer is as in single-layer NN:<br>
$A_k^{(1)} = g(w_{k0}^{(1)} + \sum\limits_{j=1}^p w_{kj}^{(1)}X_j)$
* The second hidden layer treats the activations<br>
from the first hidden layer:<br>
$A_l^{(2)} = g(w_{l0}^{(2)} + \sum\limits_{k=1}^{K_1} w_{lk}^{(2)}A_k^{(1)})$
* Output layer. We need to build different models:<br> $Z_m = \beta_{m0} + \sum\limits_{l=1}^{K_2} \beta_{ml} A_l^{(2)}$
</div>
<div>
  <figure>
    <img src="/ISLRv2_figure_10.4.png" style="width: 300px !important;">
    <figcaption style="color:#b3b3b3ff; font-size: 11px; position: absolute;"><br>NN with 2 hidden layers. Image source:
      <a href="https://hastie.su.domains/ISLR2/ISLRv2_website.pdf#page=416">ISLR Fig. 10.4</a>
    </figcaption>
  </figure>
<br>

##### Notation:<br> $W_i$ - **weights** (coefficients)<br> $B$ - **bias** (intercept)
</div>
</div>

* Class probability: $f_m (X) = \mathrm{Pr}(Y = m | X) = \frac{e^{Z_m}}{\sum_{l=0}^9 e^{Z_l}}$ (**softmax**)

---

# Recap: Fitting a Neural Network

<div class="grid grid-cols-[5fr_2fr]">
<div>

* The model parameters $\theta$ are:<br>
$\beta = (\beta_0, \beta_1, ..., \beta_K)$ and $w_k = (w_{k0}, w_{k1}, ..., w_{kp})$<br>
* We need to solve a nonlinear least squares problem:<br>
$\underset{\{w_k\}_1^K, \beta}{\mathrm{minimize}} \frac{1}{2}\sum\limits_{i=1}^n (y_i - f(x_i))^2$,<br>
where $f(x_i) = \beta_0 + \sum\limits_{k=1}^K \beta_k g(w_{k0} + \sum\limits_{j=1}^p w_{kj}x_{ij})$
</div>
<div>
  <figure>
    <img src="/ISLRv2_figure_10.1.png" style="width: 240px !important;">
    <figcaption style="color:#b3b3b3ff; font-size: 11px; position: absolute;"><br>Feed-forward NN. Image source:
      <a href="https://hastie.su.domains/ISLR2/ISLRv2_website.pdf#page=412">ISLR Fig. 10.1</a>
    </figcaption>
  </figure>
</div>
</div>

#### The problem is **nonconvex** in the parameters $\leadsto$ multiple solutions.

<div class="grid grid-cols-[3fr_5fr]">
<div>
  <figure>
    <img src="/ISLRv2_figure_10.17.png" style="width: 250px !important;">
    <figcaption style="color:#b3b3b3ff; font-size: 11px; position: absolute;"><br>Gradient descent for 1D θ. Image source:
      <a href="https://hastie.su.domains/ISLR2/ISLRv2_website.pdf#page=442">ISLR Fig. 10.17</a>
    </figcaption>
  </figure>
</div>

<div>

To overcome some of these issues we can use:
* **Slow Learning**
  * **Gradient Descent**
* **Regularization**
</div>
</div>

---

# Fitting a Neural Network: Gradient Descent

<div class="grid grid-cols-[3fr_6fr] gap-6">
<div>
  <figure>
    <img src="/ISLRv2_figure_10.17.png" style="width: 250px !important;">
    <figcaption style="color:#b3b3b3ff; font-size: 11px; position: absolute;"><br>Gradient descent for 1D θ. Image source:
      <a href="https://hastie.su.domains/ISLR2/ISLRv2_website.pdf#page=442">ISLR Fig. 10.17</a>
    </figcaption>
  </figure>
<br>
<br>
<br>

  <figure>
    <img src="/Gradient_Descent.png" style="width: 250px !important;">
    <figcaption style="color:#b3b3b3ff; font-size: 11px; position: absolute;"><br>Image source:
      <a href="https://easyai.tech/en/ai-definition/gradient-descent/">https://easyai.tech/en/ai-definition/gradient-descent</a>
    </figcaption>
  </figure>
</div>

<div>

Rewriting the least squares problem as:<br>
$R(\theta) = \frac{1}{2}\sum\limits_{i=1}^n (y_i - f_\theta(x_i))^2$

We can formulate the general **gradient descent algorithm**:
1. Start with a guess $\theta^0$ for all the parameters in $\theta$,<br> and set $t = 0$
1. Iterate until the objective $R(\theta)$ fails to decrease:
   1. Find a vector $\delta$ that reflects a small change in $\theta$, such that $\theta^{t+1}$ = $\theta^t + \delta$ reduces the objective;<br> i.e. such that $R(\theta^{t+1}) < R(\theta^t)$
   1. Set $t \leftarrow t + 1$
</div>
</div>

---

# Gradient Descent
<br>

How do we find the directions to move $\theta$ so as to decrease the objective $R(\theta)$?

One need to calculate **gradient** of $R(\theta)$ evaluated at some current value $\theta = \theta^m$:

$\nabla R(\theta^m) = \frac{\partial R(\theta)}{\partial\theta} \biggr\rvert_{\theta = \theta^m}$

The idea of gradient descent is to move $\theta$ a little in the opposite direction:

$\theta^{m+1} \leftarrow \theta^m - \eta \nabla R(\theta^m)$,

where $\eta$ is the **learning rate**.

If the gradient vector is zero, then we may have arrived at a minimum of the objective.

---

# Gradient Descent

<div class="grid grid-cols-[3fr_5fr] gap-10">
<div>
  <figure>
    <img src="/3d_loss.png" style="width: 270px !important;">
  </figure>
</div>
<div>
  <figure>
    <img src="/2d_loss.png" style="width: 330px !important;">
    <figcaption style="color:#b3b3b3ff; font-size: 11px; position: absolute;"><br>Images source:
      <a href="https://c.d2l.ai/berkeley-stat-157/units/linear.html">https://c.d2l.ai/berkeley-stat-157/units/linear.html</a>
    </figcaption>
  </figure>
</div>
</div>
<br>

* Choose a starting point $w_0$
* Repeat to update the weight for $t = 1, 2, 3$:<br> $\mathbf{w}_t = \mathbf{w}_{t-1} - \eta \frac{\partial\ell}{\partial\mathbf{w}_{t-1}}$
   * Gradient: a direction that increases the value
   * Learning rate $\eta$: a hyper-parameter specifies the step length

---

# Choose a Learning Rate

<br>
<br>
<div class="grid grid-cols-[3fr_3fr] gap-20">
<div>

  ## Not too small
  <br>
  <figure>
    <img src="/lr_1.png" style="width: 400px !important;">
  </figure>
</div>
<div>

  ## Not too big
  <br>
  <figure>
    <img src="/lr_2.png" style="width: 400px !important;">
    <figcaption style="color:#b3b3b3ff; font-size: 11px; position: absolute;"><br>Images source:
      <a href="https://c.d2l.ai/berkeley-stat-157/units/linear.html">https://c.d2l.ai/berkeley-stat-157/units/linear.html</a>
    </figcaption>
  </figure>
</div>
</div>

---

# Note on Learning Rate

<div>
    <img src="/Learning_Rate.png" style="width: 500px; position: relative">
</div>
<br>
<div>
  <figure>
    <img src="/Learning_Rate_Loss.png" style="width: 410px; position: absolute; right:150px">
    <figcaption style="color:#b3b3b3ff; font-size: 11px">Images source:
      <a href="https://www.jeremyjordan.me/nn-learning-rate/">https://www.jeremyjordan.me/nn-learning-rate</a>
    </figcaption>
  </figure>
</div>

---

# Recap: Mini-batch Stochastic Gradient Descent (SGD)

* Computing the gradient over the whole training data is too
expensive
  * Takes minutes to hours for DNN models
* Randomly sample $b$ examples $i_1, i_2, ..., i_b$ to approximate the loss<br>
$$\frac{1}{b}\sum_{i \in I_b} \ell(\mathbf{x}_i, y_i, \mathbf{w})$$
  * $b$ is the batch size, another important hyper-parameters

---

# Choose a Batch Size

<br>
<br>
<div class="grid grid-cols-[3fr_3fr] gap-20">
<div>

  ## Not too small
  <br>

  * Workload is too small, hard to fully utilize computation resources
  </div>
  <div>

  ## Not too big
  <br>

  * Memory issue
  * Waste computation, e.g. when all $x_i$ are identical
</div>
</div>
