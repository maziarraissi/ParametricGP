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

To see an example, please run the `one_dimensional_example.py` file in the folder `/PGPs_autograd/Examples/` or `/PGPs_tensorflow/Examples/`.

    # let X, y be the loaded data
    # Model creation:
    pgp = PGP(X, y, M = 8, max_iter = 6000, N_batch = 1)
    
    # Training
    pgp.train()
    
    # Prediction
    mean_star, var_star = pgp.predict(X_star)

## Installing Dependencies

This code depends on `autograd` (https://github.com/HIPS/autograd), `tensorflow` (https://www.tensorflow.org/), `numpy` (http://www.numpy.org/), `scikit-learn` (http://scikit-learn.org/stable/index.html), `matplotlib` (https://matplotlib.org/), `pyDOE` (https://pythonhosted.org/pyDOE/), and `pandas` (http://pandas.pydata.org/) which can be installed using

    pip install autograd
    pip install tensorflow
    pip install numpy
    pip install -U scikit-learn
    pip install matplotlib
    pip install --upgrade pyDOE
    pip install pandas
