---
layout: default
---
### Author
[Maziar Raissi](http://www.dam.brown.edu/people/mraissi/)

### Abstract

Modern datasets are rapidly growing in size and complexity, and there is a pressing need to develop new statistical methods and machine learning techniques to harness this wealth of data. This work presents a [novel regression framework](https://arxiv.org/abs/1704.03144) for encoding massive amount of data into a small number of "hypothetical" data points. While being effective, the resulting model is conceptually very simple and is built upon the seemingly self-contradictory idea of making [Gaussian processes](http://www.gaussianprocess.org/gpml/) "parametric". This simplicity is important specially when it comes to deploying machine learning algorithms on big data flow [engines](https://spark.apache.org/mllib/) such as [MapReduce](https://en.wikipedia.org/wiki/MapReduce) and [Apache Spark](https://spark.apache.org). Moreover, it is of great importance to devise models that are aware of their imperfections and are capable of properly quantifying the uncertainty in their predictions associated with such limitations.

**Methodology**

To address the most fundamental shortcoming of [Gaussian processes](http://www.gaussianprocess.org/gpml/), namely the lack of scalability to "big data", we propose to use **two** Gaussian processes rather than one;

1) A Gaussian process $$u(x)$$ in its classical sense whose hyper-parameters are trained using a "hypothetical dataset" and the corresponding negative log marginal likelihood. This Gaussian process is also used for prediction by conditioning on the hypothetical data.

2) A "parametric Gaussian process" $$f(x)$$ that is used to generate the hypothetical dataset consumed by $$u(x)$$.

One could think of $$f(x)$$ as the "producer" of the hypothetical data and $$u(x)$$ as the "consumer" of such data. Therefore, $$u(x)$$ never sees the real data, rendering the size of the real dataset irrelevant. In fact, $$f(x)$$ is the one that sees the real data and transforms it into the hypothetical data consumed by $$u(x)$$.

* * * * 

**Hypothetical Data**

At iteration $$n$$ of the algorithm, let us postulate the existence of some *hypothetical dataset* $$\{\mathbf{z},\mathbf{u}_n\}$$ with

$$
\mathbf{u}_n \sim \mathcal{N}(\mathbf{m}_n,\mathbf{S}_n).
$$

Here, $$\mathbf{m}_n$$ is the mean of the hypothetical data and $$\mathbf{S}_n$$ is the covariance matrix. Moreover, $$\mathbf{z} = \{z^i\}_{i=1}^M$$, $$\mathbf{u}_n = \{u^i_n\}_{i=1}^M$$, and $$M$$ is the size of the hypothetical data. The size $$M$$ of the hypothetical data is assumed to be much smaller than the size $$N$$ of the real dataset $$\{\mathbf{x},\mathbf{y}\}$$. The locations $$\mathbf{z}$$ of the hypothetical dataset are obtained by employing the [k-means clustering](https://en.wikipedia.org/wiki/K-means_clustering) algorithm on $$\mathbf{x}$$ (or a smaller subset of it) and are fixed throughout the algorithm.

* * * * 

**Consumer of Hypothetical Data**

Let us start by making the prior assumption that

$$
u_{n-1}(x) \sim \mathcal{GP}\left(0, k(x,x';\theta_{n-1})\right),
$$

is a zero mean [Gaussian process](http://www.gaussianprocess.org/gpml/) with covariance function $$k(x,x';\theta_{n-1})$$ which depends on the hyper-parameters $$\theta_{n-1}$$. The hyper-parameters can be updated from $$\theta_{n-1}$$ to $$\theta_{n}$$ by taking a step proportional to the gradient of the *negative log marginal likelihood*

$$
\mathcal{NLML}(\theta_{n-1}) := \frac{1}{2} \mathbf{m}^T_n \mathbf{K}_{n-1}^{-1} \mathbf{m}_n + \frac{1}{2} \log |\mathbf{K}_{n-1}| + \frac{M}{2} \log (2 \pi),
$$

where $$\mathbf{K}_{n-1} = k(\mathbf{z}, \mathbf{z}; \theta_{n-1})$$. It is worth highlighting that we are using the mean $$\mathbf{m}_n$$ of the hypothetical data $$\mathbf{u}_n$$ in the formula for the negative log marginal likelihood rather than the actual hypothetical data $$\mathbf{u}_n$$. Moreover, predictions can be made by conditioning on the hypothetical data and obtaining

$$
u_n(x)| \mathbf{u}_n \sim \mathcal{GP}\left(\mathbf{q}_n^T \mathbf{K}_n^{-1}\mathbf{u}_n, k(x,x';\theta_n) - \mathbf{q}_n^T \mathbf{K}_n^{-1}\mathbf{q}_n\right).
$$

where $$\mathbf{q}^T_n = k(x,\mathbf{z};\theta_n)$$ and $$\mathbf{K}_n = k(\mathbf{z},\mathbf{z};\theta_n)$$. In fact, for prediction purposes, we need to maginalize $$\mathbf{u}_n$$ out and use

$$
u_n(x)|\mathbf{m}_n,\mathbf{S}_n \sim \mathcal{GP}\left(\mu_n(x), \Sigma_n(x,x')\right),
$$

where

$$
\mu_n(x) = \mathbf{q}_n^T \mathbf{K}_n^{-1}\mathbf{m}_n,
$$

and

$$
\Sigma_n(x,x') = k(x,x';\theta_n) - \mathbf{q}_n^T \mathbf{K}_n^{-1}\mathbf{q}_n + \mathbf{q}_n^T \mathbf{K}_n^{-1} \mathbf{S}_n \mathbf{K}_n^{-1}\mathbf{q}_n.
$$

* * * * 

**Producer of Hypothetical Data**


Let us define a [parametric Gaussian process](https://arxiv.org/abs/1704.03144) by the resulting conditional distribution

$$
f_n(x) := u_n(x)|\mathbf{m}_n,\mathbf{S}_n \sim \mathcal{GP}\left(\mu_n(x), \Sigma_n(x,x')\right).
$$

The mean $$\mathbf{m}_n$$ and the covariance matrix $$\mathbf{S}_n$$ of the hypothetical dataset can be updated by employing the posterior distribution

$$
f_n(\mathbf{z})|\mathbf{x}_{n+1},\mathbf{y}_{n+1}
$$

resulting from conditioning on the observed mini-batch of data $$\{\mathbf{x}_{n+1},\mathbf{y}_{n+1}\}$$ of size $$N_{n+1}$$; i.e.,

$$
\mathbf{m}_{n+1} = \mu_n(\mathbf{z}) + \Sigma_n(\mathbf{z},\mathbf{x}_{n+1}) \Sigma_n(\mathbf{x}_{n+1},\mathbf{x}_{n+1})^{-1}\left[\mathbf{y}_{n+1} - \mu_n(\mathbf{x}_{n+1})\right],
$$

$$
\mathbf{S}_{n+1} = \Sigma_n(\mathbf{z},\mathbf{z}) - \Sigma_n(\mathbf{z},\mathbf{x}_{n+1}) \Sigma_n(\mathbf{x}_{n+1},\mathbf{x}_{n+1})^{-1} \Sigma_n(\mathbf{x}_{n+1},\mathbf{z}).
$$

It is worth mentioning that $$\mu_n(\mathbf{z}) = \mathbf{m}_n$$ and $$\Sigma_n(\mathbf{z},\mathbf{z}) = \mathbf{S}_n$$. The information corresponding to the mini-batch $$\{\mathbf{x}_{n+1}, \mathbf{y}_{n+1}\}$$ is now distilled in the parameters $$\mathbf{m}_{n+1}$$ and $$\mathbf{S}_{n+1}$$. The algorithm is initialized by setting $$\mathbf{m}_0 = \mathbf{0}$$ and $$\mathbf{S}_0 = k(\mathbf{z},\mathbf{z};\theta_0)$$ where $$\theta_0$$ is some initial set of hyper-parameters. Therefore, initially $$f_0(x) = u_0(x)$$.


* * * * *

**Illustrative Example**

![](http://www.dam.brown.edu/people/mraissi/assets/img/OneDimensional.png)
> _Illustrative example:_ (A) Plotted are 6000 training data generated by random perturbations of a one dimensional function. (B) Depicted is the resulting prediction of the model. The blue solid line represents the true data generating function, while the dashed red line depicts the predicted mean. The shaded orange region illustrates the two standard deviations band around the mean. The red circles depict the resulting mean values for the 8 hypothetical data points after a single pass through the entire dataset while mini-batches of size one are employed per each iteration of the training algorithm. It is remarkable how the training procedure places the mean of the hypothetical dataset on top of the underlying function.

* * * * *

**Conclusions**

This work introduced the concept of [parametric Gaussian processes](https://arxiv.org/abs/1704.03144) (PGPs), which is built upon the seemingly self-contradictory idea of making [Gaussian processes](http://www.gaussianprocess.org/gpml/) "parametric". Parametric Gaussian processes, by construction, are designed to operate in "big data" regimes where one is interested in *quantifying the uncertainty* associated with noisy data. The effectiveness of the proposed approach was demonstrated using an illustrative example with simulated data and a benchmark dataset in the airline industry with approximately $$6$$ million records (see [here](https://arxiv.org/abs/1704.03144)).

* * * * *

**Acknowledgements**

This work received support by the DARPA EQUiPS grant N66001-15-2-4055 and the AFOSR grant FA9550-17-1-0013. All data and codes are publicly available on [GitHub](https://github.com/maziarraissi/ParametricGP).

* * * * *

	@article{raissi2017parametric,
	  title={Parametric Gaussian Process Regression for Big Data},
	  author={Raissi, Maziar},
	  journal={arXiv preprint arXiv:1704.03144},
	  year={2017}
	}

