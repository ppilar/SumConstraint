# -*- coding: utf-8 -*-
"""
Created on Thu May 13 19:27:41 2021

@author: Admin
"""
import warnings
from typing import Any

import torch
from torch import Tensor


from gpytorch.constraints import GreaterThan
from gpytorch.distributions import base_distributions
from gpytorch.functions import add_diag
from gpytorch.lazy import (
    BlockDiagLazyTensor,
    DiagLazyTensor,
    KroneckerProductLazyTensor,
    MatmulLazyTensor,
    RootLazyTensor,
    lazify,
)

from gpytorch.likelihoods import Likelihood, _GaussianLikelihoodBase
from gpytorch.utils.warnings import OldVersionWarning
from gpytorch.likelihoods.noise_models import MultitaskHomoskedasticNoise
from gpytorch.likelihoods.multitask_gaussian_likelihood import _MultitaskGaussianLikelihoodBase


#SC changes are marked by comment 'SC:'
class sc_MultitaskGaussianLikelihood(_MultitaskGaussianLikelihoodBase):
    """
    A convenient extension of the :class:`gpytorch.likelihoods.GaussianLikelihood` to the multitask setting that allows
    for a full cross-task covariance structure for the noise. The fitted covariance matrix has rank `rank`.
    If a strictly diagonal task noise covariance matrix is desired, then rank=0 should be set. (This option still
    allows for a different `log_noise` parameter for each task.). This likelihood assumes homoskedastic noise.

    Like the Gaussian likelihood, this object can be used with exact inference.
    """

    def __init__(
        self,
        num_tasks,
        N_virtual = 0,
        rank=0,
        task_correlation_prior=None,
        batch_shape=torch.Size(),
        noise_prior=None,
        noise_constraint=None,
    ):
        """
        Args:
            num_tasks (int): Number of tasks.

            rank (int): The rank of the task noise covariance matrix to fit. If `rank` is set to 0,
            then a diagonal covariance matrix is fit.

            task_correlation_prior (:obj:`gpytorch.priors.Prior`): Prior to use over the task noise correlaton matrix.
            Only used when `rank` > 0.

        """
        if noise_constraint is None:
            noise_constraint = GreaterThan(1e-4)

        noise_covar = MultitaskHomoskedasticNoise(
            num_tasks=num_tasks, noise_prior=noise_prior, noise_constraint=noise_constraint, batch_shape=batch_shape
        )
        super().__init__(
            num_tasks=num_tasks,
            noise_covar=noise_covar,
            rank=rank,
            task_correlation_prior=task_correlation_prior,
            batch_shape=batch_shape,
        )

        self.register_parameter(name="raw_noise", parameter=torch.nn.Parameter(torch.zeros(*batch_shape, 1)))
        self.register_constraint("raw_noise", noise_constraint)
        self.N_virtual = N_virtual #SC: save number of virtual measurements

    @property
    def noise(self):
        return self.raw_noise_constraint.transform(self.raw_noise)

    @noise.setter
    def noise(self, value):
        self._set_noise(value)

    def _set_noise(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_noise)
        self.initialize(raw_noise=self.raw_noise_constraint.inverse_transform(value))

    def _shaped_noise_covar(self, base_shape, *params):
        noise_covar = super()._shaped_noise_covar(base_shape, *params)
        noise = self.noise
        noise_covar = noise_covar.add_diag(noise)
        
        #SC: reduce noise for virtual measurements to enforce values
        if self.N_virtual != 0:
            noise_covar2 = noise_covar.evaluate()
            N_ges = noise_covar2.shape[0]
            N_modify = self.N_virtual*self.num_tasks
            noise_covar2[N_ges-N_modify:,N_ges-N_modify:] = 1e-4*torch.eye(N_modify)
            noise_covar = lazify(noise_covar2)
        return noise_covar