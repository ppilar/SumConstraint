#!/usr/bin/env python3

import numpy as np
import torch

from gpytorch.distributions import MultivariateNormal
from gpytorch.likelihoods import _GaussianLikelihoodBase
from gpytorch.mlls.marginal_log_likelihood import MarginalLogLikelihood

from .sc_multitask_multivariate_normal import *


#SC changes are marked by comment 'SC:'
class ExactMarginalLogLikelihood(MarginalLogLikelihood):
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
        super(ExactMarginalLogLikelihood, self).__init__(likelihood, model)

    def forward(self, function_dist, target, N_zc = 0, omit_inds=[], Jterm = 0, *params):
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

        # Get the log prob of the marginal distribution
        output = self.likelihood(function_dist, *params)
        
        #omit specific outputs when calculating mll
        MT_dim = target.shape[1]
        No = len(omit_inds)
        if No > 0:
            (Nd, Nf)  = target.shape
            #target_buf = target[np.ix_(range(Nd),omit_inds)]
            
            
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
            
        if Jterm == 1: #add Jacobi term corresponding to trafo, to obtain comparable mll
            #print(target_buf.shape)
            term = torch.sum(torch.log(2*torch.sqrt(torch.abs(target_buf))))
            #print(term)
            res = res.add(term)
            

        # Add additional terms (SGPR / learned inducing points, heteroskedastic likelihood models)
        for added_loss_term in self.model.added_loss_terms():
            res.add(added_loss_term.loss(*params))

        # Add log probs of priors on the (functions of) parameters
        for _, prior, closure, _ in self.named_priors():
            res.add_(prior.log_prob(closure()).sum())
            
        # Scale by the amount of data we have
        if No > 0:
            num_data = loc2.shape[1]
            return res.div_(num_data)
        else:
            num_data = target.size(-1)
            return res.div_(num_data)

    def pyro_factor(self, output, target, *params):
        import pyro

        mll = self(output, target, *params)
        pyro.factor("gp_mll", mll)
        return mll
