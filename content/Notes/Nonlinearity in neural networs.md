My understanding of the cutting edge of deep learning is that it is primarily heuristic driven--very often the reason xyz is used in the model is because a paper or an experiment showed that it was better over a couple data points. Indeed, generally there are no theoretical reasons for any one model architecture (e.g. transformers) to be better (with respect to some given metric) than another for a given task, and I wish there existed some mathematical or statistical framework for concrete guarantees on the effectiveness for a given neural network. An unfortunate side effect of this is that as models and data scale up, it will become increasingly difficult and expensive to understand what reasonable expectations for a new framework should be. Generally, most guarantees in the literature follow something like the [Universal approximation theorem](https://en.wikipedia.org/wiki/Universal_approximation_theorem#):
  

> For any function $f$ and $\varepsilon > 0$, there exists a sufficiently large neural network (e.g. single hidden layer MLP) such that $f$ can be approximated to within $\varepsilon$.
# High Level Overview

Heuristically, the reason why deep learning works in modelling this data is because there exist highly nonlinear correlations within the data. For example, the training objective for an autoregressive text model (e.g. ChatGPT) is to predict the next token conditional on the last $N$ tokens, where usually $N$ is a large number--as of 2025, cutting edge language models support anywhere from 32 thousand to 10 million tokens of context. Fortunately for us, human language exhibits patterns on a large range of scales, from grammar at the sentence level, to logical arguments at the paragraph level, etc. so it's reasonable to assume that there are many nonlinear correlations that a model potentially could learn in order to help it predict the next token. One only needs to take a look at the [Wikipedia page](https://en.wikipedia.org/wiki/Natural_language_processing#History) for natural language processing to see that it is for these reasons that people have always believed language modelling to be a solvable problem.

  

As another example, we can also look in the world of computer vision. Deep convolutional networks and vision transformers are trained to recognize patterns on pixel level data, whether it be for denoising or for image recognition. As with the case of language modelling, this has [historical roots](https://en.wikipedia.org/wiki/Neocognitron) based off of the belief that there are many local-to-global patterns, within image data. Common heuristics that get mentioned when people learn about computer vision are things like edge detection at the pixel level, to object and character detection at a larger but local level, to describing entire images at a more global scale.

  

The real breakthough of the last decade has been the tools to build the right kind of large, nonlinear model that can effectively build an internal representation of these highly nonlinear patterns. Unfortunately, getting the right framework to generate the nonlinearity isn't a given--stacking a bunch of layers with random initialization and training with the pytorch autograd can land you in tricky situations such as gradient vanishing or gradient exploding. There are various tricks used to control potentially bad behavior, such as residual connections, momentum, Xavier/Glorot/He initialization, training warmups, but I wonder if it is possible to formalize these tricks in formal framework (e.g. VC theory, fat shattering dimension). At the present moment I only have a surface level understanding of the theoretical foundations of machine learning, so there will be some perspectives missing, so this might be something to revisit in the future.

  

Mathematically, the ML model is a function from your data to the labels (e.g. next token, denoised image, ground truth, etc.)

$$

f_{model}: \mathbb{R}^{d_\text{data}} \rightarrow \mathbb{R}^{d_\text{labels}}

$$

whereas the loss landscape is the graph of the loss function $\ell$

$$

\ell : \mathbb{R}^{d_\text{model}} \rightarrow \mathbb{R}

$$

So that the model outputs will be always be a continuous, and in fact [Lipschitz](https://en.wikipedia.org/wiki/Lipschitz_continuity) function of the input data. The issue is that even though your model might theoretically be infinitely flexible and capable of modelling any function of the input space, there are no a priori guarantees that your model will have a favorable loss landscape for the problem that you are dealing with. Since the typical paradigm for training deep neural networks is some modification stochastic gradient descent, the gradient of the model parameters will be a random variable depending on the parameters themselves as well as the underlying data distribution. So a poorly designed model will have highly variable and fluctuating gradients between minibatches. Indeed, for any given minibatch, there could be many regions in parameter space with bad gradient behavior.

  

As an aside, a quick first test for your model design and initialization is to take a single minibatch and run gradient descent on only that minibatch. If your model converges reasonably quickly to an overfit, then you can be reasonably confident that you don't have issues within you minibatch. On the other hand if your model has performance issues, then it might be time to adjust some of your hyperparameters (e.g. learn rate, model design).

  
  

# Where Does Nonlinearity Live?

As a bit of an overgeneralization, deep neural networks alternate between nonlinear layers and linear layers--without the nonlinearity in between, the composition of two linear layers will simply be another linear layer. Of course, in practice modern architectures are more complicated than just a stack of layers.

  

## Activation Functions

The simplest way to have nonlinearity--we simply stick an elementwise activation function between 2 linear layers:

<pre class="mermaid">

flowchart LR

A["in"] -- Linear --> B("$$\mathbb{R}^{d_{mid}}$$")

B -- Nonlinear activation --> C("$$\mathbb{R}^{d_{mid}}$$")

C -- Linear --> D("out")

</pre>

There are a multitude of activation functions that we can choose from. During the early days of ML people preferred things like the sigmoid, but more modern architectures increasingly prefer the rectified linear unit (ReLU) and it's variants due to issues with vanishing gradients for very positive/negative input values.

- Sigmoid: $$\mathrm{sigmoid}(x) = \frac{1}{1 + e^{-x}}$$ (Note that up to scaling and shifting the input/output this is equivalent to $$\tanh$$)

- Softplus: $$\mathrm{softplus}(x) = \ln(1 + e^x)$$

- ReLU: $$\mathrm{ReLU}(x) = \max(x,0)$$

- GELU: $$\mathrm{GELU}(x) = x\Phi(x)$$, where $$\Phi(x) = \frac{1}{\sqrt{2\pi}}\int_{-\infty}^x e^{t^2/2} dt$$ is the CDF of the standard normal distribution.

- SiLU (swish): $$\mathrm{SiLU}_\beta(x) = x \cdot \mathrm{sigmoid}(\beta x)= \frac{x}{1 + e^{-\beta x}}$$

There are also gated linear unit (GLU) variants that use 2 different separate linear layers to create the nonlinearity:

<pre class="mermaid">

flowchart LR

A["in"] -- linear --> x["x"] & z["z"]

x --> GLU

z --> GLU

GLU --> out

</pre>

Heuristically, this allows the model to 'learn' the activation functions via the linear layer that outputs $$z$$, but it is hard to say whether we get a real performance increase. Indeed, the original GLU [paper](https://arxiv.org/pdf/2002.05202) attributes the success to 'divine benevolence'.

- GLU: $$\mathrm{GLU}_{z}(x) = (z) \cdot \mathrm{sigmoid}(x)$$

- ReGLU: $$\mathrm{ReGLU}_{z}(x) = (z) \cdot \mathrm{ReLU}(x)$$

- GEGLU: $$\mathrm{GEGLU}_{z}(x) = (\z) \cdot \mathrm{GELU}(x)$$

- SwiGLU: $$\mathrm{SwiGLU}_{z, \beta}(x) = (z) \cdot \mathrm{SiLU}_{\beta}(x)$$

Note that one can essentially define a GLU activation function for any given activation function above.

  
  

## Layer Norm

Another way nonlinearity is introduced into a model is via the layernorm operation. Suppose that $x \in \mathbb{R}^d$$ is some vector. Then the entry-wise mean $$\mu$$ and variance $$\sigma$$ can be found by the formulas

$$

\begin{split}

\mu &= \frac{1}{d} \sum x_i\\

\sigma^2 &= \frac{1}{d} \sum (x_i - \mu)^2

$$

Then we can write

$$

\mathrm{LayerNorm}(x)_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \varepsilon}}

$$

Geometrically, what we are doing is first projecting $$x$$ down to the codimension 1 linear subspace $$ \{\sum x_i = 0\}\subset \mathbb{R}^d $$ and then projecting the output of that part down onto the sphere $$ S^{d-2} = \{ \sum x_i^2 = d \} \subset \{\sum x_i = 0\}$$ with a $$\varepsilon$$ regularizer for numerical stability. The former operation is linear, but the latter operation is much more nonlinear.

  

As an aside, I wonder if the first linear projection is truly necessary. After all, we are paying for a $$d$$-dimensional residual stream; why are we arbitrarily projecting out one dimension? If anyone has thoughts on the matter, or has a high quality reference with an answer, please leave a comment down below. I'll gladly venmo 5 dollars to the first person to leave a high quality reply.

  

## Softmax

In self-attention, or when given a classification problem with $$N$$ potential choices to make, a model will use the softmax function over the logits

$$

\mathrm{Softmax}(x)_i = \frac{e^{x_i}}{\sum_j e^{x_j}}.

$$

Geometrically, we first apply element-wise exponentiation so that the resulting output is in the positive orthant and then project to the $$L^1$$ unit ball (i.e. the dimension $$N-1$$ simplex). In many ways, this is a straightforward generalization of the sigmoid function to higher dimensions and therefore carries the same downsides, such as potential gradient vanishing. Since it's applied over an entire sequence or all potential categories at once, this is potentially the most computationally expensive and least parallelizable step.

  

Some approaches to optimizing this step for inference/training include

- Changing exponentiation to a function with less extreme gradients at extreme values (e.g. using Softplus instead of exp)

- Using elementwise activation functions (e.g. [ReLU](https://arxiv.org/pdf/2309.08586), [tanh](https://arxiv.org/abs/2503.10622), [polynomials](https://arxiv.org/pdf/2410.18613), etc.)

- Enforcing sparsity in outputs

  
  

## Polynomial Layers

Another approach to nonlinearity is for layer outputs to be polynomial functions of the layer inputs. For example, consider the query-key matrix in self-attention. Suppose that $X \in \mathbb{R}^{N \times d_{model}}$ is a sequence of $N$ tokens, $Q, K \in \mathbb{R}^{d_{model} \times d_{head}}$ are the query and key matrices. Then then $qk$ matrix is given by

$$

\frac{XQK^TX^T}{\sqrt{d_{head}}}

$$

which will be a quadratic function of $X$ with coeffients given by $QK^T$. More generally, polynomials form the basis of [polynomial networks](https://arxiv.org/pdf/2003.03828), such as the mu-layer in the MONet paper. An advantage of using higher order polynomials is that they are generally more conducive for kernel fusion.

  
  

# Quantifying Nonlinearity

The goal I had in mind for myself when I started writing this was to better understand various ways the nonlinearity of a model can be quantified, and how one might detect issues with gradient exploding/vanishing, bad generalization, and inefficient/unstable training before training the model using stochastic gradient descent. There are a couple ways we can try to approach this:

- Measure layer-wise gradient norms on initialization

-

  
  

## Gradient Norms

  
  

## Jacobian Norms

  
  

## Model Generalizability

- [VC dimension](https://en.wikipedia.org/wiki/Vapnik%E2%80%93Chervonenkis_dimension)

- [Fat shattering dimension](https://mlweb.loria.fr/book/en/fatshattering.html)

  
  

##

  
  

# Some questions

  

## Nonlinearity of a Single Layer

What is a reasonable measure of nonlinearity in a single layer

  

- Neural networks struggle to learn extremely nonlinear functions https://arxiv.org/pdf/1806.08734

- More linear single layers makes models more robust https://arxiv.org/pdf/1910.08108

  
  

## Stacking Nonlinearity

What about the effects of stacking nonlinear blocks?

  

## Quantization

How much floating point precision do we even need?