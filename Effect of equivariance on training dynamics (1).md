# Effect of equivariance on training dynamics

*Group Equivariant Convolutional Networks* (G-CNN) have gained significant traction in recent years owing to their ability to generalize the property of CNNs being equivariant to translations in convolutional layers. With equivariance, the network is able to exploit groups of symmetries and a direct consequence of this is that it generally needs less data to perform well. However, incorporating such inductive knowledge into the network may not always be advantageous, especially when the data itself does not exhibit full equivariance. To adress this issue, the concept of *relaxed equivariance* was introduced, offering a means to adjust the degree of equivariance imposed on the network, enabling it to operate on a level between full equivariance and no equivariance.

Interestingly, for rotational symmetries on fully equivariant data, [1] found that a fully equivariant network exhibits poorer performance compared to a relaxed equivariant network. One plausible rationale for this phenomenon is that the training dynamics benefit from relaxation of the equivariance constraint.  This proposition gains support from [3], which conducts an analysis on the spectra of maximum eigenvalues of the Hessian throughout training. This also calls for the question of how the amount of equivariance possessed by the data at hand influences the training dynamics. Equally intriguing, [2] shows that more equivariance imposed on a network does not necessarily imply more equivariance learned.

Inspired by the aforementioned observations, the purpose of this blog post is to investigate the following questions:

- How does (1) the amount of equivariance imposed on the network and (2) the amount of equivariance possessed by the data affect the training dynamics of the network?
- How do (1) and (2) affect the amount of equivariance learned by the network?


<!--Furthermore, to either substantiate some of the aforementioned observations or the effectiveness of approximate equivariance networks, we validate the following claims:

- Posed in [1], an approximate equivariant network outperforms an equivariant network on a fully equivariant dataset (isotropic flow).
- Posed in [5], an approximate equivariant network outperforms an equivariant network on non-equivariant smoke dataset. 
-->


## Reproduction Objective and Background

Our first objective is to reproduce the Super-resolution of 3D Turbulence experiment (Experiment 5.5) as presented in the paper "Discovering Symmetry Breaking in Physical Systems with Relaxed Group Convolution" [1].

In this study, the authors focus on understanding the asymmetries in physics data using relaxed equivariant neural networks. They employ various relaxed group convolution architectures to identify symmetry-breaking factors in different physical systems. 

The significance of this experiment lies in the discovery that relaxed equivariant models can outperform fully equivariant ones, even on data that fully exhibits equivariance.


### Original Experiment 5.5 Overview

In Experiment 5.5, the authors evaluate the performance of regular convolutional layers, group equivariant layers, and relaxed group equivariant layers integrated into a neural network. This network is tasked with upscaling 3D channel flow turbulence and isotropic turbulence data.

- **Isotropic Turbulence**: The statistical properties of the velocity fields are fully invariant with respect to rotations (?? and reflections ??).
- **Channel Flow Turbulence**: The flow is driven by a pressure difference between the channel ends and walls, which results in anisotropic turbulence, leading to partial loss of rotational (?reflexions?) equivariance.

#### Expectations Based on Equivariance

Given the data properties:
- For isotropic turbulence, networks with full equivariance should outperform those with relaxed equivariance, as the data fully adheres to symmetry principles.
- For channel flow turbulence, networks with relaxed equivariance should outperform fully equivariant networks due to the inherent symmetry breaking in the data.

#### Results and Observations

Their results indicate that incorporating both equivariance and relaxed equivariance enhances prediction performance. Interestingly, the relaxed group convolution outperformed even the fully equivariant network on isotropic turbulence data, which contradicts initial expectations.

This unexpected outcome may be attributed to the enhanced optimization process facilitated by the relaxed weights, and serves as the main motivating factor for our extension.
 
 ## Extension Objectives

To further explore the benefits of relaxing equivariance constraints, the primary objective of this extension is to analyze the effects of:
* The degree of data equivariance
* The degree of imposed model equivariance

on the training dynamics of relaxed equivariant models and a fully equivariant model.

### Data and Model

For this experiment, we use a synthetic $64 \times 64$ 2D smoke simulation dataset generated by PhiFlow [5] with varying boundary conditions and inflow positions. This dataset is rotationally equivariant if the buoyancy force remains constant while only the inflow positions vary. By altering the buoyancy force for simulations with different inflow positions, we can control the equivariance error in the data. Specifically, we generate three datasets with different levels of rotational equivariance.

The model employed is a segmentation network consisting exclusively of relaxed G-CNN layers with padding to preserve spatial size. It is trained to predict the next step of the simulation given the previous steps as input.

The model is tested under two distinct scenarios:
* **test-future**: The model is trained and tested on the same simulations but at different times.
* **test-domain**: The model is trained and tested on different simulations.

The first scenario evaluates the model's ability to extrapolate into the future for the same task, while the second scenario assesses the model's capacity to generalize across different simulations.

The degree of imposed model equivariance is controlled by the hyperparameter $\alpha$ introduced in the "Relaxed Equivariant Networks" section.

The training dynamics will be assessed using multiple metrics, all of which are explained in the "Training Dynamics Evaluation" section.

Our secondary objective is to assess the effects the degree of data equivariance and the degree of induced model equivariance have on the amount of learned equivariance.

TO DO: IMPORTANCE OF THIS QUESTION, HOW WE GET THE ANSWER




## Next Steps

Before presenting our reproduction and extension results, we will:
1. Recap the concepts of Group Convolutional Neural Networks (G-CNNs) and relaxed equivariant networks.
2. Outline the metrics employed to evaluate the relationships and performance.


## G-CNN

Consider the segmentation task depicted in the picture below.

![Alt text](https://analyticsindiamag.com/wp-content/uploads/2020/07/u-net-segmentation-e1542978983391.png)

Naturally, applying segmentation on a rotated or reflected image should give the same segmented image as applying such transformation after the segmentation. Mathematically, it means the neural network $f$ should satisfy: 

$$
f(gx) = gf(x)
$$

for all datapoints $x$ and transformations (in this case rotations and reflections) $g$.
A network that satisfies this property is considered to be equivariant w.r.t. the group of transformations comprised of 2D rotations and reflections.

To build such a network, it is sufficient that each of its layers is equivariant in the same sense. Recall in a CNN, its building block, the convolution layer, achieves equivariance to translations by means of weight sharing using kernels that are shifted along the image.
<!--
<span style="color:red;">Insert picture of convolution layer here</span>
-->

Formally, a group $(G,\cdot)$ is a set equipped with a closed and associative binary
operation, containing inverse elements and an identity. Adapting the idea of weight sharing to arbitrary groups of transformations leads to $G$-equivariant group convolution, defined between a kernel $\psi: G \rightarrow \mathbb{R}^{n\times m}$ and a function $f: G \rightarrow \mathbb{R}^m$ on element $g \in G$ as:
$$
    (\psi *_{G} f)(g) = \sum_{h \in G}\psi(g^{-1}h)f(h)
$$
When $G$ is the group of translations, this reduces to the regular convolution operation, which makes G-CNNs a generalization of CNNs, convolving over arbitrary groups.
<!--
LIFTING CONVOLUTION
The function $f$ represents a hidden layer of the G-CNN, and it's important to notice that its domain is the group $G$. The first layer of the neural network is usually an image, i.e. a function defined on $\mathbb{R}^2$...
-->

For the group convolution to be practically feasible, $G$ has to be finite and relatively small in size (roughly up to a hundred elements). 
However, if one is interested in equivariance w.r.t. an infinite group, e.g. all 2D rotations, $G$ would have to be a finite subset of those rotations, and it is unclear to what extent the network becomes truly rotationally equivariant.

## Relaxed Equivariant Networks

The desirability of equivariance in a network depends on the amount of equivariance possessed by the data of interest. To this end, relaxed equivariant networks are built on top of G-CNNs using a modified (relaxed) kernel consisting of a linear combination of standard G-CNN kernels.

$$ (\psi \hat{*}_{G} f)(g) = \sum_{h \in G}\psi(g,h)f(h) = \sum_{h \in G}\sum_{l=1}^L w_l(h) \psi_l(g^{-1}h)f(h) $$

The weights $w_l$ depend on the group elements $h$, leading to a loss of equivariance [5]. The equivariance error increases with the number of kernels $L$ and the variability of 
$w_l(h)$ over $h \in G$. This method allows the network to relax strict symmetry constraints, providing greater flexibility at the cost of reduced equivariance.

<!--Naturally, we would expect approximately equivariant networks to achieve better results than fully equivariant models on datasets which are themselves not perfectly equivariant.
[1] supports this intuition showing that an AENN yielded better results than a fully equivariant model on super-resolution tasks for partially-equivariant channel flow data.
Interestingly, the AENN prevailed even in the fully-equivariant isotropic flow dataset, which could potentially be explained by AENN weights enhancing optimisation.-->


## Measuring the Amount of Equivariance Learned by a Network
It is natural to measure the amount of equivariance a network $f$ has as the expected difference between the output of the transformed data and the transformed output of the original data.

$$ \mathbb{E}_{x \sim D}\left[\frac{1}{|G|}\sum_{g \in G}\|f(gx)-gf(x)\|\right]$$

We can estimate this expectation by computing the average for a series of batches from our test set. However this approach has downsides, which we can tackle using the Lie derivative. 

### Lie Derivative
<!--
<span style="color:red;">Add Lie group def and say that G' usually has that structure?</span>
-->


<!-- However, this approach is problematic as it only measures the amount of equivariance w.r.t. the finite group $G$. Instead, [2] proposed the use of (Lie) derivatives to evaluate the robustness of the network to infinitesimal transformations. For the notion of derivative to be defined, however, we need to assume the group to have a differential structure (Lie group). Since the space consisting of $G$ may be too peculiar to work in, we smoothly parameterize the representations of these transformations in the tangent space at the transformation that does nothing (identity element).  -->

In practice, even though we are imposing $G$-equivariance on a network, what we would like to achieve is $G'$-equivariance for an infinite (Lie) group $G'$ which contains $G$. The previous approach is problematic as it only measures the amount of acquired equivariance w.r.t. the finite group $G$, neglecting all other transformations, and thus doesn't give us the full picture of the network's equivariance.
 
 [2] proposed the use of Lie derivatives, which focus on the equivariance of the network thowards very small transformations in $G'$, and give us a way to measure $G'$-equivariance of the network. The intuitive idea is the following: Imagine a smooth path $p(t)$ traversing the group $G'$ that starts at the identity element (i.e. transformation that does nothing) of the group, $e_{G'}$. This means that at every time-point $t \geq 0$, $p(t)$ is an element of $G'$ (some transformation), and $p(0) = e_{G'}$. Then, we can define the function:
  $$\Phi_{p(t)}f(x) := p(t)^{-1}f(p(t)x)
  $$
  This function makes some transformation $p(t)$ on the data, applies $f$ to the transformed data, and finally applies the inverse transformation $p(t)^{-1}$ to the output. Notice that if $f$ is $G'$-equivariant this value is constantly equal to $f(x)$, and that $\Phi_{p(0)}f(x) = f(x)$. The Lie derivative of $f$ along the path $p(t)$ is the derivative 
  $$L_pf(x) := d\Phi_{p(t)}f(x)/dt = \lim_{t \to 0+} \frac{\Phi_{p(t)}f(x) - f(x)}{t}
  $$
at time $t=0$. One might note that this only measures the local equivariance around the identity element of the group. Luckily, it is shown in [2] that $L_{p}f(x) = 0$ for all $x$ and $d$ specific paths, where $d$ corresponds to the dimensionality of $G'$, is equivalent to $f$ being $G'$- equivariant.

<!-- 
### Equivariance error (EE)

 Another alternative for measuring equivariance relies on the variant of approximate equivariance network we consider. Recall that what broke equivariance therein are the weights used in the linear combination of kernels that constituted the modified kernel. Therefore, a proxy for the amount of equivariance is naturally the difference between the individual kernels used over all possible transformations.
$$\frac{1}{L|G|}\sum_{l=1}^L\sum_{g \in G} |w_l(g)-w_l(e)|$$ 

-->

## Training Dynamics Evaluation

To assess the training dynamics of a network, we quantify both the efficiency and efficacy of learning. 

To estimate the efficiency of learning, we analyze:
* convergence rate as measured by number of epochs trained until early stopping
* learning curve shape


For efficacy of learning, we are interested in the final performance and the generalizability of the learned parameters, which are quantified by the final RMSE, and sharpness of the loss landscape near the final weight-point [4]. 

### Sharpness

To measure the sharpness of the loss landscape after training, we consider changes in the loss averaged over random directions. Let $D$ denote a set of vectors randomly drawn from the unit sphere, and $T$ a set of displacements, i.e. real numbers. Then, the sharpness of the loss $\mathcal{L}$ at a point $w$ is: 

$$ \phi(w,D,T) = \frac{1}{|D||T|} \sum_{t \in T} \sum_{d \in D} |\mathcal{L}(w+dt)-\mathcal{L}(w)| 
$$

This definition is an adaptation from the one in [4] which does not normalize by $\mathcal{L}(w)$ inside the sum.
A sharper loss landscape around the model's final weights, usually implies a greater generalization gap.

Therefore, to estimate the efficacy of learning we use:
* final RMSE
* sharpness of the final point

### Hessian Eigenvalue Spectrum

Finally, the Hessian eigenvalue spectrum [3] sheds light on both the efficiency and efficacy of neural network training. Negative Hessian eigenvalues de-convexify the loss landscape disturbing the optimization process, whereas very large eigenvalues lead to training instability, sharp minima and consequently poor generalization.


## Reproduction Results
We aim to reproduce the experiment of [1] using a 64x64 syntetic smoke dataset which has rotational symmetries. Specifically the data contains 40 simulations varied by inflow positions and buoyant forces, which exhibit perfect C4 rotational symmetry. However, buoyancy factors change with inflow locations, disrupting this symmetry. We compare our results with those of the original paper for the rsteer and rgroup models, which are the ones the paper introduces. The reconstruction RMSE for both methods is found in the table below. 

<div style="display: flex;">
  <table border="0">
    <tr>
      <th colspan="3">Results from [1]</th>
    </tr>
    <tr>
      <td></td>
      <td>rgroup</td>
      <td>rsteer</td>
    </tr>
    <tr>
      <td>Domain</td>
      <td>0.73(0.02)</td>
      <td>0.67(0.01)</td>
    </tr>
    <tr>
      <td>Future</td>
      <td>0.82(0.01)</td>
      <td>0.80(0.00)</td>
    </tr>
  </table>

  <table border="0" > <!-- Adding space between the tables -->
    <tr>
      <th colspan="3">Reproduction Results</th>
    </tr>
    <tr>
      <td></td>
      <td>rgroup</td>
      <td>rsteer</td>
    </tr>
    <tr>
      <td>Domain</td>
      <td>wip</td>
      <td>0.63</td>
    </tr>
    <tr>
      <td>Future</td>
      <td>wip</td>
      <td>0.84</td>
    </tr>
  </table>
</div>

We see that the rsteer model performs similary to what was seen in [1], the results on in domain data are slighly better while on future data the performance is a bit worse. Possibly due to a difference in the early stopping metric used. Overal we can conclude that the results from the original paper reproduce.  


Our reproduction efford includes multiple improvements to the original codebase that make it easier to reuse the code in the future. First of all the code from [1] did not include the weight constraint on the relaxed weights that is shown in the paper, we added this to the codebase. The code and paper from [1] also did not include any hyperparameters for the rgroup model. Additionally we have uploaded the smoke datasets of [1] to huggingface for ease of use and we have updated the datageneration notebook to work with the most recent version of PhiFlow [6].  

## Extension Results
<div style="display: flex; justify-content: space-between;">
  <div style="flex: 1; margin-right: 10px;">
    <img src="https://hackmd.io/_uploads/H1tTMORQR.png" alt="Figure 1" style="max-width: 100%;">
    <p style="text-align: center;">Figure 1: Description</p>
  </div>
  <div style="flex: 1; margin-left: 10px;">
    <img src="https://hackmd.io/_uploads/rytpfdRmA.png" alt="Figure 2" style="max-width: 100%;">
    <p style="text-align: center;">Figure 2: Description</p>
  </div>
</div>





## References

[1] Discovering Symmetry Breaking in Physical Systems with Relaxed Group Convolution

[2] The Lie Derivative for Measuring Learned Equivariance

[3] How do vision transformers work?

[4] Improving Convergence and Generalization using Parameter Symmetries

[5] Approximately Equivariant Networks for Imperfectly Symmetric Dynamics

[6] PhiFlow: A differentiable PDE solving framework for deep learning via physical simulations.

Proposition 4.4, page 79