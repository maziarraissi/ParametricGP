# Parametric Gaussian Processes in Python

This work introduces the concept of parametric Gaussian processes (PGPs), which is built upon the seemingly self-contradictory idea of making Gaussian processes parametric. Parametric Gaussian processes, by construction, are designed to operate in "big data" regimes where one is interested in quantifying the uncertainty associated with noisy data. The proposed methodology circumvents the well-established need for stochastic variational inference, a scalable algorithm for approximating posterior distributions. The effectiveness of the proposed approach is demonstrated using an illustrative example with simulated data and a benchmark dataset in the airline industry with approximately 6 million records.

For more details, please refer to the following: (https://arxiv.org/abs/1704.03144)

  - Raissi, Maziar. "Parametric Gaussian Process Regression for Big Data." arXiv preprint arXiv:1704.03144 [stat.ML] (2017).

## Citation

    @article{raissi2017parametric,
      title={Parametric Gaussian Process Regression for Big Data},
      author={Raissi, Maziar},
      journal={arXiv preprint arXiv:1704.03144},
      year={2017}
    }

## Example

To see an example, please run the `example.py` file.

    # let X, y be the loaded data
    # Model creation:
    pgp = PGP(X, y, M = 10, max_iter = 6000, N_batch = 1)
    
    # Training
    pgp.train()
    
    # Prediction
    mean_star, var_star = pgp.predict(X_star)

## Installing Dependencies

This code depends on `autograd` (https://github.com/HIPS/autograd), `scikit-learn` (http://scikit-learn.org/stable/index.html), and `pyDOE` (https://pythonhosted.org/pyDOE/) which can be installed using

    pip install autograd
    pip install -U scikit-learn
    pip install --upgrade pyDOE
