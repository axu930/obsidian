Reading notes from https://arxiv.org/pdf/1601.00670
# Definitions/Notation
- $z$ : the latent variable
- $x$: observed data
- $p(z)$: Latent prior
- $p(x)$: Evidence
- $p(z,x)$: Joint marginal
- $p(z \vert x)$: Latent posterior
- $p(x \vert z)$: Sampling model

$$
\begin{array}{rll}
E \psi &= H\psi & \text{Expanding the Hamiltonian Operator} \\
&= -\frac{\hbar^2}{2m}\frac{\partial^2}{\partial x^2} \psi + \frac{1}{2}m\omega x^2 \psi & \text{Using the ansatz $\psi(x) = e^{-kx^2}f(x)$, hoping to cancel the $x^2$ term} \\
&= -\frac{\hbar^2}{2m} [4k^2x^2f(x)+2(-2kx)f'(x) + f''(x)]e^{-kx^2} + \frac{1}{2}m\omega x^2 f(x)e^{-kx^2} &\text{Removing the $e^{-kx^2}$ term from both sides} \\
& \Downarrow \\
Ef(x) &= -\frac{\hbar^2}{2m} [4k^2x^2f(x)-4kxf'(x) + f''(x)] + \frac{1}{2}m\omega x^2 f(x) & \text{Choosing $k=\frac{im}{2}\sqrt{\frac{\omega}{\hbar}}$ to cancel the $x^2$ term, via $-\frac{\hbar^2}{2m}4k^2=\frac{1}{2}m \omega$} \\
&= -\frac{\hbar^2}{2m} [-4kxf'(x) + f''(x)] \\
\end{array}
$$
# Objective
In general for high dimensional data, computing the latent posterior
$$ p(z \vert x) = \frac{p(z,x)}{p(x)}$$
is intractable since the data distribution $p(x)$ itself is either very expensive or impossible to compute (for example, via direct sampling or [[Markov Chain Monte-Carlo]]).  Therefore, we are interested in approximating the distribution $p(z\vert x)$ without necessarily needing to compute the marginal distribution for the evidence. 

The given example is a hierarchical mixture of Gaussians: 
$$ \begin{array}{rll}
\mu_k &\sim N(0,\sigma^2)\\
c_i &\sim \mathrm{Cat}(1/K, \cdots, 1/K)\\
x_i &\sim N(c_i^T \mu,0)
\end{array}$$
There are $K$ categories, with mean $\mu_k$, and the $i$th draw $x_i$ is from the $c_i$th mean. In this example, the latent vector is $z = (\mu, c)$ and the observations are $x$. While the posterior is theoretically computable, in practice it will have an exponential number of components scaling with the number of samples $N$.
# Approach
Variational inference starts with a family of probability densities $\mathcal{D}$ over the latent variables $z$. Given an observation $x$, the goal is to find the distribution $q^* \in \mathcal{D}$  that minimizes the KL-divergence with the posterior. 
$$
q^* = \arg\min_{} \{KL(q^*(z) \| p(z \vert x)) \mid q \in \mathcal{D}\}
$$
Unfortunately, the KL-divergence is not computationally tractable as it does depend on the marginal distribution for the evidence:
$$
KL(q(z) \| p(z \vert x)) = E_{z \sim q}[\log q(z)] - E_{z \sim q}[\log p(x,z)] + \log p(x)
$$
Therefore, we instead optimize for the argmax of the ELBO: 
$$
\begin{array}{rll}
ELBO(q) &=  E_{z \sim q}[\log p(x,z)] - E_{z \sim q}[\log q(z)]\\
&= E_{z \sim q}[\log p(x \vert z)] - KL(q(z) \| p(z \vert x))
\end{array}
$$
which is essentially just the KL-divergence without the dependency on the marginal evidence. Heuristically, we are finding the distribution within $\mathcal{D}$ that is closest to the posterior to $p(z\vert x)$ that also simultaneously maximizes the log-likelihoods of the sampling model. 

## Applications
This section ongoing TODO
- Variational bound gives selection criterion for models
- 
# Heuristics
Here are a couple examples of VI in practice, with the heuristics of how VI is applied. 
## Mean field variational family 
The mean-field variational family  the simplest setting for VI. It is a family of latent distributions where the latent variables are mutually independent; i.e
$$ 
q(z) = \prod_i q(z_i)
$$
## Structured variational family


## Mixture-based variational family
## Latent plan architectures
In architectures with latent plan backbones, e.g.:
- https://arxiv.org/abs/2402.04647
- https://arxiv.org/pdf/2505.03077
- https://arxiv.org/pdf/2502.01567


# Algorithms for VI

## Coordinate ascent variational inference (CAVI)
Inputs:
- 
```
while ELBO not converged:
	for i in {1, 2, ... , n}:
		update q(z_i)
	end
	compute ELBO(q) = 
end
```
