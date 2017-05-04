#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Maziar Raissi
"""

import sys
sys.path.insert(0, '../PGP/')

import numpy as np
import matplotlib.pyplot as plt
from pyDOE import lhs
from parametric_GP import PGP
from Utilities import Normalize

if __name__ == "__main__":
    
    np.random.seed(12345)
    # Setup
    N = 6000
    D = 1
    lb = 0.0*np.ones((1,D))
    ub = 1.0*np.ones((1,D))    
    noise = 0.1

    Normalize_input_data = 1
    Normalize_output_data = 1
    
    # Generate traning data
    def f(x):
        return x*np.sin(4*np.pi*x)    
    X = lb + (ub-lb)*lhs(D, N)
    y = f(X) + noise*np.random.randn(N,1)
    
    # Generate test data
    N_star = 400
    X_star = lb + (ub-lb)*np.linspace(0,1,N_star)[:,None]
    y_star = f(X_star)
    
    # Normalize Input Data
    if Normalize_input_data == 1:
        X_m = np.mean(X, axis = 0)
        X_s = np.std(X, axis = 0)
        X = Normalize(X, X_m, X_s)
        
        X_star = Normalize(X_star, X_m, X_s)

    # Normalize Output Data
    if Normalize_output_data == 1:
        y_m = np.mean(y, axis = 0)
        y_s = np.std(y, axis = 0)   
        y = Normalize(y, y_m, y_s)
        
        y_star = Normalize(y_star, y_m, y_s)
    
    # Model creation
    M = 8
    pgp = PGP(X, y, M, max_iter = 6000, N_batch = 1,
              monitor_likelihood = 10, lrate = 1e-3)
        
    # Training
    pgp.train()
    
    # Prediction
    mean_star, var_star = pgp.predict(X_star)
    
    # Plot Results
    Z = pgp.sess.run(pgp.Z)
    m = pgp.sess.run(pgp.m)
    plt.figure(figsize=(10,10))
    plt.rc('text', usetex=True)
    plt.rcParams.update({'font.size': 15})
    plt.subplot(2, 1, 1)
    plt.plot(X,y,'b+',alpha=1)
    plt.xlabel('$x$')
    plt.ylabel('$f(x)$')
    plt.title('(A)')
    plt.legend(['%d traning Data' % N], loc='lower left')

    
    plt.subplot(2, 1, 2)
    plt.plot(Z,m, 'ro', alpha=1, markersize=14)
    plt.plot(X_star, y_star, 'b-', linewidth=2)
    plt.plot(X_star, mean_star, 'r--', linewidth=2)
    lower = mean_star - 2.0*np.sqrt(var_star)
    upper = mean_star + 2.0*np.sqrt(var_star)
    plt.fill_between(X_star.flatten(), lower.flatten(), upper.flatten(), facecolor='orange', alpha=0.5)
    plt.xlabel('$x$')
    plt.ylabel('$f(x), \overline{f}(x)$')
    plt.title('(B)')
    plt.tight_layout()
    plt.legend(['%d hypothetical data' % M, '$f(x)$', '$\overline{f}(x)$', 'Two standard deviations'], loc='lower left')

    
    plt.savefig('../Fig/OneDimensional.eps', format='eps', dpi=1000)