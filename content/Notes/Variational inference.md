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
$$ p(z \vert x) = \frac{p(z,x)}{p(x)}$$
is intractable since the data distribution $p(x)$ itself is either very expensive or impossible to compute. The given example is a hierarchical mixture of Gaussians: 
$$ \begin{split}
\mu_k &\sim N(0,\sigma^2)\\
c_i &\sim \mathrm{Cat}(1/K, \cdots, 1/K)\\
x_i &\sim N(c_i^T \mu,0)
\end{split}$$
There are $K$ categories, with mean $\mu_k$, and the $i$th draw $x_i$ is from the $c_i$th mean. In this example, the latent vector is $z = (\mu, c)$ and the observations are $x$. While the posterior is theoretically computable, in practice it will have an exponential number of components scaling with the number of samples $N$. Therefore, we are interested in approximating the distribution $p(z\vert x)$ without necessarily needing to compute the marginal distribution for the evidence. 
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
\begin{split}
ELBO(q) &=  E_{z \sim q}[\log p(x,z)] - E_{z \sim q}[\log q(z)]\\
&= E_{z \sim q}[\log p(x \vert z)] - KL(q(z) \| p(z \vert x))
\end{split}
$$
which is essentially just the KL-divergence without the dependency on the marginal evidence. Heuristically, we are finding the distribution within $\mathcal{D}$ that is closest to the posterior to $p(z\vert x)$ that also simultaneously maximizes the log-likelihoods of the sampling model. 
# Heuristics

