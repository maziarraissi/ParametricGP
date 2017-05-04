#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Maziar Raissi
"""

import numpy as np
import tensorflow as tf
from sklearn.cluster import KMeans
from Utilities import kernel, kernel_tf, fetch_minibatch
import timeit


class PGP:
    def __init__(self, X, y, M=10, max_iter = 2000, N_batch = 1, 
                 monitor_likelihood = 10, lrate = 1e-3):
        (N,D) = X.shape
        
        # kmeans on a subset of data
        N_subset = min(N, 10000)
        idx = np.random.choice(N, N_subset, replace=False)
        kmeans = KMeans(n_clusters=M, random_state=0).fit(X[idx,:])
        Z = kmeans.cluster_centers_
        
        hyp = np.log(np.ones(D+1))
        logsigma_n = np.array([-4.0])
        hyp = np.concatenate([hyp, logsigma_n])
        
        m = np.zeros((M,1))
        S = kernel(Z,Z,hyp[:-1])

        self.X = X
        self.y = y
        
        self.M = M
        self.Z = tf.Variable(Z,dtype=tf.float64,trainable=False)
        self.K_u_inv = tf.Variable(np.eye(M),dtype=tf.float64,trainable=False)
                
        
        self.m = tf.Variable(m,dtype=tf.float64,trainable=False)
        self.S = tf.Variable(S,dtype=tf.float64,trainable=False)
        
        self.nlml = tf.Variable(0.0, dtype=tf.float64, trainable=False)
                       
        self.hyp = hyp
        
        self.max_iter = max_iter
        self.N_batch = N_batch
        self.monitor_likelihood = monitor_likelihood
        self.jitter = 1e-8
        self.jitter_cov = 1e-8
        
        self.lrate = lrate
        self.optimizer = tf.train.AdamOptimizer(self.lrate)
        
        # Tensor Flow Session
        # self.sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
        self.sess = tf.Session()
        
    def train(self):
        print("Total number of parameters: %d" % (self.hyp.shape[0]))
        
        X_tf = tf.placeholder(tf.float64)
        y_tf = tf.placeholder(tf.float64)
        hyp_tf = tf.Variable(self.hyp, dtype=tf.float64)
        
        train = self.likelihood(hyp_tf, X_tf, y_tf)
        
        init = tf.global_variables_initializer()
        self.sess.run(init)
        
        start_time = timeit.default_timer()
        for i in range(1,self.max_iter+1):
            # Fetch minibatch
            X_batch, y_batch = fetch_minibatch(self.X,self.y,self.N_batch)
            self.sess.run(train, {X_tf:X_batch, y_tf:y_batch})
            
            if i % self.monitor_likelihood == 0:
                elapsed = timeit.default_timer() - start_time
                nlml = self.sess.run(self.nlml)
                print('Iteration: %d, NLML: %.2f, Time: %.2f' % (i, nlml, elapsed))
                start_time = timeit.default_timer()

        self.hyp = self.sess.run(hyp_tf)
            
    def likelihood(self, hyp, X_batch, y_batch, monitor=False): 
        M = self.M
        
        Z = self.Z
        
        m = self.m
        S = self.S
        
        jitter = self.jitter
        jitter_cov = self.jitter_cov
        
        N = tf.shape(X_batch)[0]
        
        logsigma_n = hyp[-1]
        sigma_n = tf.exp(logsigma_n)
        
        # Compute K_u_inv
        K_u = kernel_tf(Z, Z, hyp[:-1])    
        L = tf.cholesky(K_u + np.eye(M)*jitter_cov)        
        K_u_inv = tf.matrix_triangular_solve(tf.transpose(L), tf.matrix_triangular_solve(L, np.eye(M), lower=True), lower=False)
                
        K_u_inv_op = self.K_u_inv.assign(K_u_inv)
          
        # Compute mu
        psi = kernel_tf(Z, X_batch, hyp[:-1])    
        K_u_inv_m = tf.matmul(K_u_inv, m)   
        MU = tf.matmul(tf.transpose(psi), K_u_inv_m)
        
        # Compute cov
        Alpha = tf.matmul(K_u_inv, psi)
        COV = kernel_tf(X_batch, X_batch, hyp[:-1]) - tf.matmul(tf.transpose(psi), tf.matmul(K_u_inv,psi)) + \
                tf.matmul(tf.transpose(Alpha), tf.matmul(S,Alpha))
        
        # Compute COV_inv
        LL = tf.cholesky(COV  + tf.eye(N, dtype=tf.float64)*sigma_n + tf.eye(N, dtype=tf.float64)*jitter) 
        COV_inv = tf.matrix_triangular_solve(tf.transpose(LL), tf.matrix_triangular_solve(LL, tf.eye(N, dtype=tf.float64), lower=True), lower=False)
        
        # Compute cov(Z, X)
        cov_ZX = tf.matmul(S,Alpha)
        
        # Update m and S
        alpha = tf.matmul(COV_inv, tf.transpose(cov_ZX))
        m_new = m + tf.matmul(cov_ZX, tf.matmul(COV_inv, y_batch-MU))
        S_new = S - tf.matmul(cov_ZX, alpha)
        
        if monitor == False:
            m_op = self.m.assign(m_new)
            S_op = self.S.assign(S_new)
        
        # Compute NLML
        K_u_inv_m = tf.matmul(K_u_inv, m_new)
        
        NLML = 0.5*tf.matmul(tf.transpose(m_new), K_u_inv_m) + tf.reduce_sum(tf.log(tf.diag_part(L))) + 0.5*np.log(2.*np.pi)*tf.cast(M, tf.float64)
        
        train = self.optimizer.minimize(NLML)
        
        nlml_op = self.nlml.assign(NLML[0,0])
        
        return tf.group(*[train, m_op, S_op, nlml_op, K_u_inv_op])
    

    def predict(self, X_star):
        Z = self.sess.run(self.Z)
        m = self.sess.run(self.m)
        S = self.sess.run(self.S)
        hyp = self.hyp
        K_u_inv = self.sess.run(self.K_u_inv)
        
        N_star = X_star.shape[0]
        partitions_size = 10000
        (number_of_partitions, remainder_partition) = divmod(N_star, partitions_size)
        
        mean_star = np.zeros((N_star,1));
        var_star = np.zeros((N_star,1));
        
        for partition in range(0,number_of_partitions):
            print("Predicting partition: %d" % (partition))
            idx_1 = partition*partitions_size
            idx_2 = (partition+1)*partitions_size
            
            # Compute mu
            psi = kernel(Z, X_star[idx_1:idx_2,:], hyp[:-1])    
            K_u_inv_m = np.matmul(K_u_inv,m)   
            mu = np.matmul(psi.T,K_u_inv_m)
            
            mean_star[idx_1:idx_2,0:1] = mu;        
        
            # Compute cov  
            Alpha = np.matmul(K_u_inv,psi)
            cov = kernel(X_star[idx_1:idx_2,:], X_star[idx_1:idx_2,:], hyp[:-1]) - \
                    np.matmul(psi.T, np.matmul(K_u_inv,psi)) + np.matmul(Alpha.T, np.matmul(S,Alpha))
            var = np.abs(np.diag(cov))# + np.exp(hyp[-1])
            
            var_star[idx_1:idx_2,0] = var
    
        print("Predicting the last partition")
        idx_1 = number_of_partitions*partitions_size
        idx_2 = number_of_partitions*partitions_size + remainder_partition
        
        # Compute mu
        psi = kernel(Z, X_star[idx_1:idx_2,:], hyp[:-1])    
        K_u_inv_m = np.matmul(K_u_inv,m)   
        mu = np.matmul(psi.T,K_u_inv_m)
        
        mean_star[idx_1:idx_2,0:1] = mu;        
    
        # Compute cov  
        Alpha = np.matmul(K_u_inv,psi)
        cov = kernel(X_star[idx_1:idx_2,:], X_star[idx_1:idx_2,:], hyp[:-1]) - \
                np.matmul(psi.T, np.matmul(K_u_inv,psi)) + np.matmul(Alpha.T, np.matmul(S,Alpha))
        var = np.abs(np.diag(cov))# + np.exp(hyp[-1])
        
        var_star[idx_1:idx_2,0] = var
        
        
        return mean_star, var_star
