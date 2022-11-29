#!/usr/bin/env python3

import numpy as np
import torch
import time

from gpytorch.distributions import MultivariateNormal
from gpytorch.likelihoods import _GaussianLikelihoodBase
from gpytorch.mlls.marginal_log_likelihood import MarginalLogLikelihood

from .sc_multitask_multivariate_normal import *
from .utils import *
from .utils_Laplace import get_f_lpfX, construct_f_lpyf, get_f_sum, max_Newton
from .utils_var import get_f_lpyf_vec, get_ELBO
from .sc_multitask_multivariate_normal import MultitaskMultivariateNormalDropout

#SC changes are marked by comment 'SC:'
class sc_ExactMarginalLogLikelihood(MarginalLogLikelihood):
    """
    The exact marginal log likelihood (MLL) for an exact Gaussian process with a
    Gaussian likelihood.

    .. note::
        This module will not work with anything other than a :obj:`~gpytorch.likelihoods.GaussianLikelihood`
        and a :obj:`~gpytorch.models.ExactGP`. It also cannot be used in conjunction with
        stochastic optimization.

    :param ~gpytorch.likelihoods.GaussianLikelihood likelihood: The Gaussian likelihood for the model
    :param ~gpytorch.models.ExactGP model: The exact GP model

    Example:
        >>> # model is a gpytorch.models.ExactGP
        >>> # likelihood is a gpytorch.likelihoods.Likelihood
        >>> mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
        >>>
        >>> output = model(train_x)
        >>> loss = -mll(output, train_y)
        >>> loss.backward()
    """

    def __init__(self, likelihood, model):
        if not isinstance(likelihood, _GaussianLikelihoodBase):
            raise RuntimeError("Likelihood must be Gaussian for exact inference")
        super(sc_ExactMarginalLogLikelihood, self).__init__(likelihood, model)

    def _add_other_terms(self, res, params):
        # Add additional terms (SGPR / learned inducing points, heteroskedastic likelihood models)
        for added_loss_term in self.model.added_loss_terms():
            res = res.add(added_loss_term.loss(*params))

        # Add log probs of priors on the (functions of) parameters
        for name, module, prior, closure, _ in self.named_priors():
            res.add_(prior.log_prob(closure(module)).sum())

        return res

    def forward(self, function_dist, target, fhat_prev, dlabel, it, opt_L, opt_L_vec, L_Newton_pars, var_pars,  N_virtual, vm_val, rs_vec, rs_jtrans_vec, omit_inds=[], Jterm = 0, *params):
        r"""
        Computes the MLL given :math:`p(\mathbf f)` and :math:`\mathbf y`.

        :param ~gpytorch.distributions.MultivariateNormal function_dist: :math:`p(\mathbf f)`
            the outputs of the latent function (the :obj:`gpytorch.models.ExactGP`)
        :param torch.Tensor target: :math:`\mathbf y` The target values
        :rtype: torch.Tensor
        :return: Exact MLL. Output shape corresponds to batch shape of the model/input data.
        """
        if not isinstance(function_dist, MultivariateNormal):
            raise RuntimeError("ExactMarginalLogLikelihood can only operate on Gaussian random variables")
        
        
        #initialize some parameters
        No = 0
        fhat = torch.tensor(-1.)
        sig = torch.tensor(-1.)
        W = torch.tensor([-1.])
        
        if opt_L == 'L': #Laplace approximation
        
            noise = self.likelihood.noise            
            f_lpfX, mu, K = get_f_lpfX(function_dist)
            
            f_lpyf = construct_f_lpyf(opt_L_vec, target, fhat_prev, N_virtual, noise, rs_vec, rs_jtrans_vec, it, vm_val, dlabel)
            f_psi = get_f_sum(f_lpfX, f_lpyf)
            
            fhat, sig, W, xges = max_Newton(f_lpyf, f_lpfX, mu, K, fhat_prev, L_Newton_pars)
            res = f_lpyf(fhat) + f_lpfX(fhat) + 0.5*torch.logdet(sig) + 0.5*torch.log(torch.tensor(2.)*torch.pi)
            
        elif opt_L == 'var': #variational approach
            Ndata = target.shape[0]
            Ntask = 1 if target.ndim == 1 else target.shape[1]
            target = target.reshape(-1)
            
            f_lpyf_ges, indices = get_f_lpyf_vec(target, self.likelihood.noise, opt_L_vec, (Ndata, Ntask, N_virtual), it, vm_val)
                
            #construct variational distribution
            muq = var_pars[0]
            Lbuf = torch.tril(var_pars[1])
            buf = 1e-4*torch.eye(Lbuf.shape[0])
            sigq = Lbuf@Lbuf.T + buf
            qf =  MultitaskMultivariateNormalDropout(muq.reshape(Ndata,Ntask), sigq)
            
            #calculate ELBO
            res = get_ELBO(qf, function_dist, f_lpyf_ges, indices)
            
        else:
            # Get the log prob of the marginal distribution
            output = self.likelihood(function_dist, *params)
            
            #SC: omit specific outputs when calculating mll
            MT_dim = target.shape[1]
            No = len(omit_inds)
            if No > 0:
                (Nd, Nf)  = target.shape
                
                #SC: omit aux vars from mll calculation
                buf = torch.tensor([]).long()
                inds = []
                for i in range(Nf):
                    if not i in omit_inds:
                        inds.append(i)
                        #b1 = torch.tensor(range(i*Nd,(i+1)*Nd))                    
                        b1 = torch.tensor(range(i,Nf*Nd,Nf))
                        buf = torch.cat((buf,b1))
                        
                buf = torch.sort(buf)[0]
                target_buf = target[np.ix_(range(Nd),inds)]
                loc2 = output.loc.reshape(-1,Nf)[np.ix_(range(Nd),inds)]
                cov2 = output.covariance_matrix[np.ix_(buf,buf)]
                MT_dim2 = MT_dim - No
                
                #SC: omit virtual measurements from mll calculation
                if N_zc > 0:
                    target_buf = target_buf[:-N_zc,:]
                    loc2 = loc2[:-N_zc,:]
                    cov2 = cov2[:-MT_dim2*N_zc,:-MT_dim2*N_zc]
                
                
                output2 = MultitaskMultivariateNormalDropout(loc2, lazify(cov2))            
                res = output2.log_prob(target_buf)
            else:
                res = output.log_prob(target)
                
            
        

        res = self._add_other_terms(res, params)
        #Scale by the amount of data we have SC: modified
        if No > 0:
            num_data = loc2.shape[1]
        else:
            num_data = function_dist.event_shape.numel()
            
        return res.div_(num_data), fhat, sig, W
