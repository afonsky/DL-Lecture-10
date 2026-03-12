# Essentials of Artificial Neural Networks

### Building blocks:
<div class="grid grid-cols-[3fr_2fr_2fr] gap-3">
<div>

* Neuron
* Loss function
* Activation function
* <span style="color:#FA9370">**Optimizer**</span>
</div>

<div>

* Linear layer
* Convolution layer
* Pooling layer
* Recurrent layer
</div>

<div>
  <figure>
    <img src="/lego_A.jpg" style="width: 200px !important;">
    <figcaption style="color:#b3b3b3ff; font-size: 11px; position: absolute;"><br>Image source:
      <a href="http://sgaguilarmjargueso.blogspot.com/2014/08/de-lego.html">http://sgaguilarmjargueso.blogspot.com</a>
    </figcaption>
  </figure>   
</div>
</div>

### Today's focus:
<div class="grid grid-cols-[2fr_3fr_3fr] gap-8">
<div>

* Gradient Descent
* Learning Rate
* Mini-batch SGD

</div>
<div>

* Momentum
* Adam
* Learning Rate Scheduling
</div>

<div>
<br>

* Practical tips for training
</div>

</div>

---

# Why Should You Care About Optimization?

<div class="grid grid-cols-[3fr_2fr] gap-6">
<div>

### The same architecture can succeed or fail based on optimization choices

<v-clicks>

* The **architecture** defines *what* the model can learn
* The **optimizer** determines *whether* it actually learns it
* Bad optimization → model doesn't converge, overfits, or trains too slowly

</v-clicks>

<br>

<v-click>

> *"The most important hyperparameter to get right is the learning rate."*
>
> — **Andrew Ng**

</v-click>

</div>
<div>
  <figure>
    <img src="/VGG_Resnet_loss_landscapes.png" style="width: 350px !important;">
    <figcaption style="color:#b3b3b3ff; font-size: 11px;"><br>
      Loss landscapes of VGG and ResNet.<br>
      Source: <a href="https://arxiv.org/abs/1712.09913">Li et al., 2018</a>
    </figcaption>
  </figure>
</div>
</div>
