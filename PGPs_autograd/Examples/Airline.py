#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Maziar Raissi
"""

import sys
sys.path.insert(0, '../PGP/')

import autograd.numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from parametric_GP import PGP

if __name__ == "__main__":
        
    # Import the data
    data = pd.read_pickle('airline.pickle')

    # Convert time of day from hhmm to minutes since midnight
    data.ArrTime = 60*np.floor(data.ArrTime/100)+np.mod(data.ArrTime, 100)
    data.DepTime = 60*np.floor(data.DepTime/100)+np.mod(data.DepTime, 100)

    # Pick out the data
    Y = data['ArrDelay'].values
    names = ['Month', 'DayofMonth', 'DayOfWeek', 'plane_age', 'AirTime', 'Distance', 'ArrTime', 'DepTime']
    X = data[names].values
    
    N = len(data)
    np.random.seed(N)

    # Shuffle the data and only consider a subset of it
    perm = np.random.permutation(N)
    X = X[perm]
    Y = Y[perm]
    XT = X[int(2*N/3):N]
    YT = Y[int(2*N/3):N]
    X = X[:int(2*N/3)]
    Y = Y[:int(2*N/3)]

    # Normalize Y scale and offset
    Ymean = Y.mean()
    Ystd = Y.std()
    Y = (Y - Ymean) / Ystd
    Y = Y.reshape(-1, 1)
    YT = (YT - Ymean) / Ystd
    YT = YT.reshape(-1, 1)

    # Normalize X on [0, 1]
    Xmin, Xmax = X.min(0), X.max(0)
    X = (X - Xmin) / (Xmax - Xmin)
    XT = (XT - Xmin) / (Xmax - Xmin)
    
    # Model creation
    M = 500
    pgp = PGP(X, Y, M, max_iter = 10000, N_batch = 1000,
              monitor_likelihood = 10, lrate = 1e-3)
        
    # Training
    pgp.train()
    
    # Prediction
    mean_star, var_star = pgp.predict(XT)
    
    # MSE
    print('MSE: %f' % ((mean_star-YT)**2).mean())
    print('MSE_mean: %f' % ((Y.mean()-YT)**2).mean())
    
    # ARD
    ARD = 1/np.sqrt(np.exp(pgp.hyp.value[1:-1]))
    ARD_x = np.arange(len(ARD))
    
    # Plot ARD
    fig, ax = plt.subplots(figsize=(10,5))
    plt.rcParams.update({'font.size': 16})
    ax.barh(ARD_x,ARD)
    ax.set_yticks(ARD_x)
    ax.set_yticklabels(names)
    ax.set_xlabel('ARD weights')
    
    plt.savefig('../Fig/Flights.eps', format='eps', dpi=1000)
    
    #####
    # MSE: 0.832810
    # MSE_mean: 0.999799