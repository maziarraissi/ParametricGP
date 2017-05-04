#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Maziar Raissi
"""

import numpy as np
import tensorflow as tf

def square_dist(X, Xp, lengthscales):
    X = X / lengthscales
    Xs = tf.reduce_sum(tf.square(X), 1)

    Xp = Xp / lengthscales
    Xps = tf.reduce_sum(tf.square(Xp), 1)
    
    return -2 * tf.matmul(X, Xp, transpose_b=True) + \
           tf.reshape(Xs, (-1, 1)) + tf.reshape(Xps, (1, -1))

def kernel_tf(X, Xp, hyp):
    variance = tf.exp(hyp[0])
    lengthscales = tf.sqrt(tf.exp(hyp[1:]))
    return variance * tf.exp(-square_dist(X, Xp, lengthscales) / 2)

def kernel(X, Xp, hyp):
    output_scale = np.exp(hyp[0])
    lengthscales = np.sqrt(np.exp(hyp[1:]))
    X = X/lengthscales
    Xp = Xp/lengthscales
    X_SumSquare = np.sum(np.square(X),axis=1);
    Xp_SumSquare = np.sum(np.square(Xp),axis=1);
    mul = np.dot(X,Xp.T);
    dists = X_SumSquare[:,np.newaxis]+Xp_SumSquare-2.0*mul
    return output_scale * np.exp(-0.5 * dists)

def Normalize(X, X_m, X_s):
    return (X-X_m)/(X_s)
     
def Denormalize(X, X_m, X_s):    
    return X_s*X + X_m

def fetch_minibatch(X,y,N_batch):
    N = X.shape[0]
    idx = np.random.permutation(N)
    X_batch = X[idx[0:N_batch],:]
    y_batch = y[idx[0:N_batch]]
    return X_batch, y_batch