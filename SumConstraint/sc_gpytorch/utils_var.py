# -*- coding: utf-8 -*-

import sys
import time
import torch
import gpytorch
import numpy as np

from .utils import get_nan_entries, omit_nan_mu_covar_v0

#initialize variational parameters
def init_var_pars(model, train_x, train_y, num_tasks):
    ndim = train_x.shape[0]
    model.muq = torch.nn.Parameter(train_y.clone().reshape(-1), requires_grad = True)
    
    buf = torch.tril(0.001*torch.ones(ndim*num_tasks,ndim*num_tasks))
    buf += 0.05*torch.diag(torch.ones(ndim*num_tasks))
    model.Lbuf = torch.nn.Parameter(buf, requires_grad = True)

#sum of 2 functions
def get_f_sum(f1, f2):
    def f_sum(f):
        return f1(f) + f2(f)
    return f_sum

#sum of log of 2 functions
def get_f_lsum(f1, f2):
    def f_lsum(f):
        return torch.log(torch.exp(f1(f)) + torch.exp(f2(f)))
    return f_lsum

#apply vector of functions to entries of array
def combine_highD(fvec): #should be iterable!
    def f_ges(f):
        res = 0
        for j in range(len(fvec)):
            res += fvec[j](f[:,j])
        return res
    return f_ges



#calculate Kullback-Leibler divergence for two Gaussians
def KL_2Gaussians(p, q, dim):
    
    if dim == 1:
        mup = p.loc
        sigp = p.scale
        muq = q.loc
        sigq = q.scale
        
        return torch.log(sigq/sigp) + (sigp**2 + (mup - muq)**2)/(2*sigq**2) - 0.5
    else:
        mup = p.loc
        Sigp = p.covariance_matrix
        muq = q.loc
        Sigq = q.covariance_matrix + 1e-3*torch.eye(dim)
        
        #dropout
        buf, ibuf, N = get_nan_entries(muq)
        muq, Sigq = omit_nan_mu_covar_v0(muq, Sigq, N, buf, ibuf)
        mup, Sigp = omit_nan_mu_covar_v0(mup, Sigp, N, buf, ibuf)
        
        res = 0
        Sigq_inv = torch.inverse(Sigq)
        res += torch.logdet(Sigq) - torch.logdet(Sigp) - dim
        res += torch.trace(Sigq_inv@Sigp)
        mudiff = (muq - mup).unsqueeze(1)
        res += (mudiff.T@Sigq_inv@mudiff).squeeze()
        return 0.5*res
        

#calculate evidence lower bound
def get_ELBO(qf, pf, f_lpyf_vec, indices):
    dim = 1 if pf.loc.ndim == 0 else pf.loc.shape[0]
    
    #fvec = torch.linspace(0,10,101)
    #fvec = torch.linspace(-5,15,201)
    if dim == 1:
        fint = get_fint(f_lpyf_vec, qf)
        buf, fvec = fint()
        ELBO1 = torch.trapezoid(buf, fvec)
    else:
        inan, ivm, i0t, i1t, i2t, i3t = indices
        iloc = torch.where(~torch.isnan(qf.loc))
        qf_buf = torch.distributions.normal.Normal(qf.loc[iloc], qf.covariance_matrix.diag()[iloc])
        fint_ges, fvec_ges = get_fint_ges(f_lpyf_vec, qf_buf, indices)
        ELBO1 = torch.sum(torch.trapezoid(fint_ges, fvec_ges, dim=1))
            
                
    ELBO2 = KL_2Gaussians(qf, pf, dim)
    return ELBO1 - ELBO2


    
#calculate integral of f
def get_fint(f_lpyf, qf):
    def fint(): #function inside function necessary?
        fmin = qf.loc - qf.scale*5
        fmax = qf.loc + qf.scale*5
        fvec = torch.linspace(fmin.item(), fmax.item(), 100)
        return f_lpyf(fvec)*torch.exp(qf.log_prob(fvec)), fvec
    return fint   

#calculate integrals of all f at once
def get_fint_ges(f_lpyf_ges, qf, indices):
    fmin = qf.loc - qf.scale*5
    fmax = qf.loc + qf.scale*5
    Nf = qf.loc.shape[0]
    
    fvec_ges = torch.tensor(np.linspace(fmin.detach().numpy(), fmax.detach().numpy(), 100, axis=1))
    
    inan, ivm, i0t, i1t, i2t, i3t = indices
    lpyf_buf = torch.zeros(fvec_ges.shape)
    lpyf_buf[i0t,:] = f_lpyf_ges[0](fvec_ges[i0t,:].T).T
    lpyf_buf[i1t,:] = f_lpyf_ges[1](fvec_ges[i1t,:].T).T
    lpyf_buf[i2t,:] = f_lpyf_ges[2](fvec_ges[i2t,:].T).T
    lpyf_buf[i3t,:] = f_lpyf_ges[3](fvec_ges[i3t,:].T).T
    
    fint_ges = lpyf_buf*torch.exp(qf.log_prob(fvec_ges.T).T)
    return fint_ges, fvec_ges

#get vector of functions for likelihoods of different components
def get_f_lpyf_vec(target, noise, opt_L_vec, Ns, it, vm_val):
    Ndata = Ns[0]
    Ntask = Ns[1]
    N_virtual = Ns[2]
    
    inan = torch.where(torch.isnan(target))[0]
    inotnan = torch.where(~torch.isnan(target))[0]
    
    opt_L_vec = torch.tensor(opt_L_vec*Ndata)
    opt_L_vec[inan] = -1*torch.ones(inan.shape[0]).long()
    i0 = torch.where(opt_L_vec == 0)[0]
    i1 = torch.where(opt_L_vec == 1)[0]
    i2 = torch.where(opt_L_vec == 2)[0]
    i3 = torch.where(opt_L_vec == 3)[0]
    
    i0t = torch.where(opt_L_vec[inotnan] == 0)[0]
    i1t = torch.where(opt_L_vec[inotnan] == 1)[0]
    i2t = torch.where(opt_L_vec[inotnan] == 2)[0]
    i3t = torch.where(opt_L_vec[inotnan] == 3)[0]
    
    noise_vm = get_noise_vm_var(it)    
    ivm_buf = torch.where(target == vm_val)[0]
    ivm = ivm_buf[torch.where(ivm_buf > (Ndata-N_virtual)*Ntask)]
    
    noise_vec = noise*torch.ones(target.shape[0])
    noise_vec[i1] = noise_vec[i1]**2
    noise_vec[ivm] = noise_vm
    noise0 = noise_vec[i0]
    noise1 = noise_vec[i1] 
    noise2 = noise_vec[i2]  
    noise3 = noise_vec[i3]     
    
    f_lpyf0 = get_f_G_lpdf(target[i0], noise0)
    f_lpyf1 = get_f_chi2_lpdf(target[i1], noise1)
    f_lpyf2 = get_f_chi2_lpdf(target[i2], noise2)
    f_lpyf3 = get_f_chi2_lpdf(target[i3], noise3)
    
    return [f_lpyf0, f_lpyf1, f_lpyf2, f_lpyf3], (inan, ivm, i0t, i1t, i2t, i3t)
    
    
def get_noise_vm_var(it):    
    if it < 100:
        noise_vm = torch.tensor(1e-1)
    else:
        noise_vm = torch.max(torch.tensor(1e-1)/(1 + (it - 99)*10), torch.tensor(1e-6))
        
    return noise_vm


###############
############### dist utils

#log pdf of chi-squared distribution
def chi2_log_pdf(x2vec, f, sig, iv):
    if x2vec.ndim == 0:
        x2vec = x2vec.unsqueeze(0)
    Nx = x2vec.shape[0]
    Nf = f.shape[0]
    if x2vec.shape[0] == 1:
        x2vec = x2vec*torch.ones(f.shape[0])
    else:
        x2vec = torch.kron(x2vec, torch.ones(f.shape[0]))
        
    if sig.numel() == 1:
        sig = sig*torch.ones(x2vec.shape[0])
    else:
        sig = torch.kron(sig, torch.ones(f.shape[0]))
        
    f = torch.reshape(f.T, (1,-1)).flatten()

    lqpdf = torch.zeros(f.shape[0])
    ifpos = f >= 0
    
    xvec = torch.sqrt(x2vec[ifpos])
    xvec = torch.max(xvec, torch.tensor(1e-5))
    
    mu = torch.sqrt(f[ifpos])
    sig = torch.sqrt(sig[ifpos])
    qbuf1 = torch.distributions.normal.Normal(mu, sig)
    #qpdf = torch.exp(qbuf1.log_prob(xvec))/(2*xvec) + torch.exp(qbuf1.log_prob(-xvec))/(2*xvec)    
    #lqpdf[ifpos] = torch.log(qpdf)    
    lqpdf[ifpos] = qbuf1.log_prob(xvec) - torch.log(2*xvec) #approximate lqpdf of chi2
    
    return lqpdf.reshape(Nx,Nf).T.squeeze()

#get function that returns log pdf of chi-squared distribution
def get_f_chi2_lpdf(x2vec, sig, iv=([], [], 0, 0, 0, '')):
    def chi2_lpdf(f):
        return chi2_log_pdf(x2vec, f, sig, iv)
    return chi2_lpdf

#get function that returns log pdf of Gaussian
def get_f_G_lpdf(xvec, sig):
    Gdist = torch.distributions.normal.Normal(xvec, sig)
    return Gdist.log_prob
