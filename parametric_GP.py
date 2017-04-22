#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Maziar Raissi
"""

import autograd.numpy as np
from autograd import value_and_grad
from sklearn.cluster import KMeans
from Utilities import kernel, fetch_minibatch, stochastic_update_Adam

class PGP:
    def __init__(self, X, y, M=10, max_iter = 2000, N_batch = 1, 
                 monitor_likelihood = 10, lrate = 1e-3):
        (N,D) = X.shape
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
        self.Z = Z
        self.m = m
        self.S = S
        
        self.hyp= hyp
        
        self.max_iter = max_iter
        self.N_batch = N_batch
        self.monitor_likelihood = monitor_likelihood
        self.jitter = 1e-8
        self.jitter_cov = 1e-8
        
        # Adam optimizer parameters
        self.mt_hyp = np.zeros(hyp.shape)
        self.vt_hyp = np.zeros(hyp.shape)
        self.lrate = lrate
        
        
    def train(self):
        print("Total number of parameters: %d" % (self.hyp.shape[0]))
        
        # Gradients from autograd 
        UB = value_and_grad(self.likelihood_UB)
        
        for i in range(1,self.max_iter+1):
            # Fetch minibatch
            self.X_batch, self.y_batch = fetch_minibatch(self.X,self.y,self.N_batch) 
            
            # Compute likelihood_UB and gradients 
            NLML, D_NLML = UB(self.hyp)
            
            # Update hyper-parameters
            self.hyp, self.mt_hyp, self.vt_hyp = stochastic_update_Adam(self.hyp, D_NLML, self.mt_hyp, self.vt_hyp, self.lrate, i)
            
            if i % self.monitor_likelihood == 0:
                print("Iteration: %d, likelihood_UB: %.2f" % (i, NLML))
        
        NLML, D_NLML = UB(self.hyp)
    
    def likelihood_UB(self, hyp): 
        M = self.M 
        Z = self.Z
        m = self.m
        S = self.S 
        X_batch = self.X_batch
        y_batch = self.y_batch 
        jitter = self.jitter 
        jitter_cov = self.jitter_cov
        N = X_batch.shape[0]
        
        
        logsigma_n = hyp[-1]
        sigma_n = np.exp(logsigma_n)
        
        # Compute K_u_inv
        K_u = kernel(Z, Z, hyp[:-1])    
        K_u_inv = np.linalg.solve(K_u + np.eye(M)*jitter_cov, np.eye(M))
    #    L = np.linalg.cholesky(K_u  + np.eye(M)*jitter_cov)    
    #    K_u_inv = np.linalg.solve(np.transpose(L), np.linalg.solve(L,np.eye(M)))
        
        self.K_u_inv = K_u_inv
          
        # Compute mu
        psi = kernel(Z, X_batch, hyp[:-1])    
        K_u_inv_m = np.matmul(K_u_inv,m)   
        MU = np.matmul(psi.T,K_u_inv_m)
        
        # Compute cov
        Alpha = np.matmul(K_u_inv,psi)
        COV = kernel(X_batch, X_batch, hyp[:-1]) - np.matmul(psi.T, np.matmul(K_u_inv,psi)) + \
                np.matmul(Alpha.T, np.matmul(S,Alpha))
        
        COV_inv = np.linalg.solve(COV  + np.eye(N)*sigma_n + np.eye(N)*jitter, np.eye(N))
    #    L = np.linalg.cholesky(COV  + np.eye(N)*sigma_n + np.eye(N)*jitter) 
    #    COV_inv = np.linalg.solve(np.transpose(L), np.linalg.solve(L,np.eye(N)))
        
        # Compute cov(Z, X)
        cov_ZX = np.matmul(S,Alpha)
        
        # Update m and S
        alpha = np.matmul(COV_inv, cov_ZX.T)
        self.m = m + np.matmul(cov_ZX, np.matmul(COV_inv, y_batch-MU))    
        self.S = S - np.matmul(cov_ZX, alpha)
                
        # Compute NLML        
        Beta = y_batch - MU
        NLML_1 = np.matmul(Beta.T, Beta)/(2.0*sigma_n*N)
        
        NLML_2 = np.trace(COV)/(2.0*sigma_n)
        NLML_3 = N*logsigma_n/2.0 + N*np.log(2.0*np.pi)/2.0
        NLML = NLML_1 + NLML_2 + NLML_3
        
        return NLML[0,0]

    def predict(self, X_star):
        Z = self.Z
        m = self.m.value
        S = self.S.value
        hyp = self.hyp
        K_u_inv = self.K_u_inv
        
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
            var = np.abs(np.diag(cov)) + np.exp(hyp[-1])
            
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
        var = np.abs(np.diag(cov)) + np.exp(hyp[-1])
        
        var_star[idx_1:idx_2,0] = var
        
        
        return mean_star, var_star
