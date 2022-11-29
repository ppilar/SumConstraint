#!/usr/bin/env python3

import torch
from torch.nn import ModuleList

from typing import Optional

from gpytorch.lazy import KroneckerProductLazyTensor, lazify
from gpytorch.priors import Prior
from gpytorch.kernels.index_kernel import IndexKernel
from gpytorch.kernels.kernel import Kernel

#from ..constraint.sum_x_conditional import *
import time
import numpy as np

#modified version of the MultitaskKernel from gpytorch; the sum constraint has been incorporated
class sc_MultitaskKernel(Kernel):
    r"""
    Kernel supporting Kronecker style multitask Gaussian processes (where every data point is evaluated at every
    task) using :class:`gpytorch.kernels.IndexKernel` as a basic multitask kernel.

    Given a base covariance module to be used for the data, :math:`K_{XX}`, this kernel computes a task kernel of
    specified size :math:`K_{TT}` and returns :math:`K = K_{TT} \otimes K_{XX}`. as an
    :obj:`gpytorch.lazy.KroneckerProductLazyTensor`.

    :param ~gpytorch.kernels.Kernel data_covar_module: Kernel to use as the data kernel.
    :param int num_tasks: Number of tasks
    :param int rank: (default 1) Rank of index kernel to use for task covariance matrix.
    :param ~gpytorch.priors.Prior task_covar_prior: (default None) Prior to use for task kernel.
        See :class:`gpytorch.kernels.IndexKernel` for details.
    :param dict kwargs: Additional arguments to pass to the kernel.
    """

    def __init__(
        self,
        data_covar_module: Kernel,
        mean_module, 
        num_tasks: int,
        rank: Optional[int] = 1,
        task_covar_prior: Optional[Prior] = None,
        C  = 0,
        F = 0,
        constrain = 1,
        **kwargs,
    ):
        """"""
        super(sc_MultitaskKernel, self).__init__(**kwargs)
        self.task_covar_module = IndexKernel(
            num_tasks=num_tasks, batch_shape=self.batch_shape, rank=rank, prior=task_covar_prior
        )
        self.data_covar_module = data_covar_module
        self.num_tasks = num_tasks
        
        self.C = C  
        self.varC = 0 if type(C) == torch.Tensor else 1 #non-constant constraint y/n
        self.mean_module = mean_module #mean module of GP
        self.mu2 =  torch.zeros(num_tasks) #constrained mean 
        self.F = F #matrix F containing coefficients of constraint
        self.constrain = constrain #determine whether to constrain or not
        self.finv = 100 ##heuristic factor to improve conditioning in inverse when constraining the GP


    def forward(self, x1, x2, diag=False, last_dim_is_batch=False, **params):
        if last_dim_is_batch:
            raise RuntimeError("MultitaskKernel does not accept the last_dim_is_batch argument.")
         
        #covar_i ... task covariance
        covar_i = self.task_covar_module.covar_matrix
        if len(x1.shape[:-2]):
            covar_i = covar_i.repeat(*x1.shape[:-2], 1, 1)
        if self.varC == 1:
            covar_i = covar_i.evaluate()
                
        
        #covar_x ... data covariance
        covar_x = self.data_covar_module.forward(x1, x2, **params)
        if self.varC == 0:
            mean_x = self.mean_module(x1)[0,:]
        else:
            mean_x = self.mean_module(x1)
            
        if not torch.is_tensor(covar_x):
            covar_x = covar_x.evaluate()
        
            
        #complete covar matrix
        if self.constrain == 1: #constrained 
            eps = 1e-4
            if self.varC == 0:
                res, mu2 = ConstructRes(covar_x,covar_i,mean_x,self.num_tasks,self.C,self.F,self.finv)
            elif self.varC == 1:
                res, mu2 = ConstructResVarC(covar_x,covar_i,mean_x,self.num_tasks,self.C,self.F,x1, self.finv)       
            res = res + eps*torch.eye(res.shape[0])     #add small number to diagonal since matrix singular           
        else: #unconstrained
            res = KroneckerProductLazyTensor(lazify(covar_x), covar_i)
            mu2 = mean_x
            
        
        self.mu2 = mu2.float()
        if self.varC == 0:
            self.mu2 = self.mu2.repeat(x1.size(),1)
            
        return res.diag() if diag else res#.detach()

    def num_outputs_per_input(self, x1, x2):
        """
        Given `n` data points `x1` and `m` datapoints `x2`, this multitask
        kernel returns an `(n*num_tasks) x (m*num_tasks)` covariance matrix.
        """
        return self.num_tasks


    
    
    
#construct full covariance matrix for constant C
def ConstructRes(covar_x, covar_i, mean_x, num_tasks, C, F,finv):
    mu2, Sigma2 = get_mu2_Sigma2(mean_x,covar_i,C,F,finv)
    #res = KroneckerProductLazyTensor(lazify(covar_x),lazify(Sigma2))
    res = torch.kron(covar_x,Sigma2)

    return lazify(res), mu2
    
#construct full covariance matrix for non-constant C
def ConstructResVarC(covar_x, covar_i, mean_x, num_tasks, fC, F, x1, finv):
    #initialize quantities
    lx1 = covar_x.shape[0]
    mu2 = torch.zeros(mean_x.shape)    
    mean_x_ges = torch.reshape(mean_x,[lx1*num_tasks])
    res = torch.kron(covar_x, covar_i)
    
    #construct matrix Fges, containing constraints F for each datapoint
    NF = F.shape[0]
    Fges = torch.zeros(lx1*NF,lx1*num_tasks)
    for i in range(lx1):                
        Fges[i*NF:(i+1)*NF,num_tasks*i:num_tasks*i+num_tasks] = torch.tensor(F)
    
    #construct constrained mean, covar
    Cvec = fC(x1.squeeze()).reshape(-1)
    mean_x_ges, res = get_mu2_Sigma2(mean_x_ges,res,Cvec,Fges, finv)
    mu2 = torch.reshape(mean_x_ges,[-1,num_tasks])
    
    return lazify(res), mu2


# construct mean and covar of constrained Gaussian
def get_mu2_Sigma2(mu, Sigma, S, F=0, finv=100):
    #initialize quantities
    d = mu.size()[0]    
    if S.ndim == 0:
        S = S.unsqueeze(0)
    if type(F) == list:
        F = torch.tensor(F).unsqueeze(0).float()

    if not torch.is_tensor(Sigma):
        Sigma = Sigma.evaluate()
    SU = Sigma@F.T
    Cq = F@SU
    
    # add small value to diagonal if smallest eigenvalue very small/negative    
    leig = torch.linalg.eigvals(Cq).detach()
    lmin = torch.min(torch.real(leig))
    lmax = torch.max(torch.real(leig))
    
    if lmin < 1e-6:
        Cq += finv*torch.abs(lmin)*torch.eye(Cq.shape[0])
    
    #calculate constrained mu2, Sigma2
    C = torch.inverse(Cq)@SU.T
    A = torch.eye(d) - C.T@F
    mu2 = A@mu + C.T@S     
    Sigma2 = A@Sigma@A.T
    
    return mu2, Sigma2



