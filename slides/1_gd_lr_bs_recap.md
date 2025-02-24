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

<div>
  <figure>
    <img src="/Interlaken.jpg" style="width: 700px !important;">
    <figcaption style="color:#b3b3b3ff; font-size: 11px; position: absolute;"><br>Image source: A.B.
    </figcaption>
  </figure>   
</div>

---

# Strategy 1: Random Search

```python {all}{maxHeight:'300px'}
# assume X_train is the data where each column is an example (e.g. 3073 x 50,000)
# assume Y_train are the labels (e.g. 1D array of 50,000)
# assume the function L evaluates the loss function

bestloss = float("inf") # Python assigns the highest possible float value
for num in range(1000):
  W = np.random.randn(10, 3073) * 0.0001 # generate random parameters
  loss = L(X_train, Y_train, W) # get the loss over the entire training set
  if loss < bestloss: # keep track of the best solution
    bestloss = loss
    bestW = W
  print 'in attempt %d the loss was %f, best %f' % (num, loss, bestloss)

# prints:
# in attempt 0 the loss was 9.401632, best 9.401632
# in attempt 1 the loss was 8.959668, best 8.959668
# in attempt 2 the loss was 9.044034, best 8.959668
# in attempt 3 the loss was 9.278948, best 8.959668
# in attempt 4 the loss was 8.857370, best 8.857370
# in attempt 5 the loss was 8.943151, best 8.857370
# in attempt 6 the loss was 8.605604, best 8.605604
# ... (truncated: continues for 1000 lines)
```
<br>

#### <v-click>This *guess-and-check* stategy is **not recommended**</v-click>
<br>

##### Slides credit: [Andrej Karpathy](https://cs231n.github.io/optimization-1/)

---

# Strategy 1: Random Search


#### We can take the best weights **W** found by this search and try it out on the test set:
```python {all}{maxHeight:'300px'}
# Assume X_test is [3073 x 10000], Y_test [10000 x 1]
scores = Wbest.dot(Xte_cols) # 10 x 10000, the class scores for all test examples
# find the index with max score in each column (the predicted class)
Yte_predict = np.argmax(scores, axis = 0)
# and calculate accuracy (fraction of predictions that are correct)
np.mean(Yte_predict == Yte)
# returns 0.1555
```

#### With the best **W** this gives an accuracy of about 15.5%.

---

# Strategy 2: Random Local Search

#### The first strategy you may think of is to try to extend one foot in a random direction and then take a step only if it leads downhill. Concretely, we will start out with a random **W**, generate random perturbations **δW** to it and if the loss at the perturbed **W+δW** is lower, we will perform an update. The code for this procedure is as follows:

```python {all}{maxHeight:'300px'}
W = np.random.randn(10, 3073) * 0.001 # generate random starting W
bestloss = float("inf")
for i in range(1000):
  step_size = 0.0001
  Wtry = W + np.random.randn(10, 3073) * step_size
  loss = L(Xtr_cols, Ytr, Wtry)
  if loss < bestloss:
    W = Wtry
    bestloss = loss
  print 'iter %d loss is %f' % (i, bestloss)
```

#### Using the same number of loss function evaluations as before (1000), this approach achieves test set classification accuracy of 21.4%. This is better, but still wasteful and computationally expensive.


---

# Strategy 3: Following the Gradient
<br>

#### In 1D, the derivative of a function:

$$\frac{df}{dx} = \lim\limits_{h\to0} \frac{f(x+h) - f(x)}{h}$$

<br>

#### In multiple dimensions, the **gradient** is the vector of (partial derivatives).

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
