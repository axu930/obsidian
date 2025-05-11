Reading notes from https://arxiv.org/pdf/1601.00670
# Definitions/Notation
- $z$ : the latent variable
- $x$: observed data
- $p(z)$: Latent prior
- $p(x)$: Evidence
- $p(z,x)$: Joint marginal
- $p(z \vert x)$: Latent posterior
- $p(x \vert z)$: Sampling model

# Objective
In general for high dimensional data, computing the latent posterior
$$ 
p(z \vert x) = \frac{p(z,x)}{p(x)}
$$
is intractable since the data distribution $p(x)$ itself is either very expensive or impossible to compute (for example, via direct sampling or [[Markov Chain Monte-Carlo]]).  Therefore, we are interested in approximating the distribution $p(z\vert x)$ without necessarily needing to compute the marginal distribution for the evidence. 

The given example is a hierarchical mixture of Gaussians: 
$$
\begin{array}{rll}
\mu_k &\sim N(0,\sigma^2)\\
c_i &\sim \mathrm{Cat}(1/K, \cdots, 1/K)\\
x_i &\sim N(c_i^T \mu,0)
\end{array}
$$
There are $K$ categories, with mean $\mu_k$, and the $i$th draw $x_i$ is from the $c_i$th mean. In this example, the latent vector is $z = (\mu, c)$ and the observations are $x$. While the posterior is theoretically computable, in practice it will have an exponential number of components scaling with the number of samples $N$.

In general, for an autoregressive objective, such as language modeling, self-driving, etc., ....
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
which is the negative KL-divergence without the dependency on the marginal evidence (notice the missing $\log p(x)$ term. Heuristically, we are finding the distribution within $\mathcal{D}$ that is closest to the posterior to $p(z\vert x)$ that also simultaneously maximizes the log-likelihoods of the sampling model. 

Note that the $ELBO$ is in general a non-convex objective function. Therefore, one should be careful about the specific optimization algorithm and initialization when solving for the posterior estimate $q^*$. 
# Applications
This section is TODO
- Variational bound gives selection criterion for models
## Latent plan architectures
In architectures with latent plan backbones, e.g.:
- https://arxiv.org/abs/2402.04647
- https://arxiv.org/pdf/2505.03077
- https://arxiv.org/pdf/2502.01567

# Methods
Here are a couple approaches to VI with differing variational families and therefore have different tradeoffs for estimate fidelity and computational complexity. 
## Mean field variational family 
The mean-field variational family  the simplest setting for VI. It is a family of latent distributions where the latent variables are mutually independent; i.e
$$ 
q(z) = \prod_i q_i(z_i)
$$
The optimization problem associated with VI can be solved via [[#coordinate ascent variational inference (CAVI)]]. 

## Structured variational family
Dependencies within the latent variables. 

## Mixture-based variational family
A mixture of variational densities; i.e. a hierarchical model  model with specific 'mixture' latent variables. 



# Algorithms

##  Coordinate ascent variational inference (CAVI)
This is essentially a coordinate-wise hill climbing algorithm. Note that we can also do block ascent by simultaneously updating blocks of coordinates. The algorithm is as follows:
Inputs:
- Joint model $p(z,x)$ 
- Data set $x$ (i.e. samples)
Output:
- Joint distribution $q(z) = \prod_i q_i(z_i)$
```
while ELBO not converged:
	for i in {1, 2, ... , n}:
		update q_i(z_i) 
	end
	compute ELBO(q) = E[log(p(z,x))] - E[log(q(z))]
end
```
Here the update step iteratively updates each coordinate $i$ to maximize the ELBO specific to that coordinate, which will depend on the specific circumstances of the problem. For example, when we have access to the full conditionals, the optimal update for $q_i$ is
$$
q_i(z_i) \propto \exp E_{-i}[\log p(z_i \vert z_{-i} ,x)]
$$
where the $-i$ means to iterate over all the variables except $i$.  This can be derived from writing the ELBO specific to the $i$th coordinate: 
$$
ELBO(q_i) = E_i[E_{-i}[\log p(z_i \vert z_{-i} ,x)]] - E_i[\log q_i(z_i)] + \mathrm{const}

$$
and noting that the ELBO is maximized when the marginal distribution of $q_i$ is  proportional to $E_{-i}[\log p(z_i \vert z_{-i} ,x)]$. For example, when $p(z_i\vert z_{-i}, z)$ belongs to the [exponential family](https://en.wikipedia.org/wiki/Exponential_family), this is a tractable quantity. 

A priori CAVI can only find a local maxima, and convergence will generally be dependent on the initialization. And of course, when this distribution is intractable, we can still attempt other optimization methods to update $q_i$. 
## Stochastic variational inference (SVI)
Unfortunately, CAVI does not scale with massive data. SVI utilizes gradient ascent on subsampled data (similar to most deep learning algorithms). 

Inputs:
- Joint model $p(z,x)$ 
- Data set $x$ 
- Step sizes $\varepsilon_i > 0$, satisfying $\sum_i \varepsilon_i = \infty$ and  $\sum_i \varepsilon_i^2 < \infty$
Output:
- Joint distribution $q(z) = \prod_i q_i(z_i)$
```
while ELBO not converged:
	for i in {1, 2, ... , n}:
		update q_i(z_i) 
	end
	compute ELBO(q) = E[log(p(z,x))] - E[log(q(z))]
end
```




