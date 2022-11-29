# -*- coding: utf-8 -*-

import sys
import time
import torch
import gpytorch



def get_nan_entries(mu):
    buf = ~torch.isnan(mu)   
    ibuf = torch.where(buf)[0]
    N = torch.sum(buf)
    
    return buf, ibuf, N

def omit_nan_covar(mu, cov):
    buf, ibuf, N = get_nan_entries(mu)
    cov2 = torch.zeros(N,N)
    for i in range(N):
        cov2[i,:] = cov[ibuf[i],buf] 
        
    return cov2

def omit_nan_mu_covar_v0(mu, cov, N, buf, ibuf):
    cov2 = torch.zeros(N,N)
    for i in range(N):
        cov2[i,:] = cov[ibuf[i],buf] 
        
    mu2 = mu[ibuf]
    return mu2, cov2

def omit_nan_mu_covar(mu, cov):
    buf, ibuf, N = get_nan_entries(mu)
    return omit_nan_mu_covar_v0(mu, cov, N, buf, ibuf)
    
    
    
    
    
    
    
    
