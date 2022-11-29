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



class sc_MultitaskGaussianLikelihood(_MultitaskGaussianLikelihoodBase):
    """
    A convenient extension of the :class:`gpytorch.likelihoods.GaussianLikelihood` to the multitask setting that allows
    for a full cross-task covariance structure for the noise. The fitted covariance matrix has rank `rank`.
    If a strictly diagonal task noise covariance matrix is desired, then rank=0 should be set. (This option still
    allows for a different `noise` parameter for each task.)

    Like the Gaussian likelihood, this object can be used with exact inference.

    """

    def __init__(
        self,
        num_tasks,
        N_virtual = 0,
        rank=0,
        task_prior=None,
        batch_shape=torch.Size(),
        noise_prior=None,
        noise_constraint=None,
        has_global_noise=True,
        has_task_noise=True,
    ):
        """
        Args:
            num_tasks (int): Number of tasks.

            rank (int): The rank of the task noise covariance matrix to fit. If `rank` is set to 0,
            then a diagonal covariance matrix is fit.

            task_prior (:obj:`gpytorch.priors.Prior`): Prior to use over the task noise covariance matrix if
            `rank` > 0, or a prior over the log of just the diagonal elements, if `rank` == 0.

            has_global_noise (bool): whether to include a \\sigma^2 I_{nt} term in the noise model.

            has_task_noise (bool): whether to include task-specific noise terms, which add I_n \kron D_T
            into the noise model.

            At least one of has_global_noise or has_task_noise should be specified.

        """
        super(Likelihood, self).__init__()
        if noise_constraint is None:
            noise_constraint = GreaterThan(1e-4)

        if not has_task_noise and not has_global_noise:
            raise ValueError(
                "At least one of has_task_noise or has_global_noise must be specified. "
                "Attempting to specify a likelihood that has no noise terms."
            )

        if has_task_noise:
            if rank == 0:
                self.register_parameter(
                    name="raw_task_noises", parameter=torch.nn.Parameter(torch.zeros(*batch_shape, num_tasks))
                )
                self.register_constraint("raw_task_noises", noise_constraint)
                if noise_prior is not None:
                    self.register_prior("raw_task_noises_prior", noise_prior, lambda m: m.task_noises)
                if task_prior is not None:
                    raise RuntimeError("Cannot set a `task_prior` if rank=0")
            else:
                self.register_parameter(
                    name="task_noise_covar_factor",
                    parameter=torch.nn.Parameter(torch.randn(*batch_shape, num_tasks, rank)),
                )
                if task_prior is not None:
                    self.register_prior("MultitaskErrorCovariancePrior", task_prior, lambda m: m._eval_covar_matrix)
        self.num_tasks = num_tasks
        self.rank = rank

        if has_global_noise:
            self.register_parameter(name="raw_noise", parameter=torch.nn.Parameter(torch.zeros(*batch_shape, 1)))
            self.register_constraint("raw_noise", noise_constraint)
            if noise_prior is not None:
                self.register_prior("raw_noise_prior", noise_prior, lambda m: m.noise)

        self.has_global_noise = has_global_noise
        self.has_task_noise = has_task_noise       
        self.N_virtual = N_virtual #SC: save number of virtual measurements

    @property
    def noise(self):
        return self.raw_noise_constraint.transform(self.raw_noise)

    @noise.setter
    def noise(self, value):
        self._set_noise(value)

    @property
    def task_noises(self):
        if self.rank == 0:
            return self.raw_task_noises_constraint.transform(self.raw_task_noises)
        else:
            raise AttributeError("Cannot set diagonal task noises when covariance has ", self.rank, ">0")

    @task_noises.setter
    def task_noises(self, value):
        if self.rank == 0:
            self._set_task_noises(value)
        else:
            raise AttributeError("Cannot set diagonal task noises when covariance has ", self.rank, ">0")

    def _set_noise(self, value):
        self.initialize(raw_noise=self.raw_noise_constraint.inverse_transform(value))

    def _set_task_noises(self, value):
        self.initialize(raw_task_noises=self.raw_task_noises_constraint.inverse_transform(value))

    @property
    def task_noise_covar(self):
        if self.rank > 0:
            return self.task_noise_covar_factor.matmul(self.task_noise_covar_factor.transpose(-1, -2))
        else:
            raise AttributeError("Cannot retrieve task noises when covariance is diagonal.")

    @task_noise_covar.setter
    def task_noise_covar(self, value):
        # internally uses a pivoted cholesky decomposition to construct a low rank
        # approximation of the covariance
        if self.rank > 0:
            self.task_noise_covar_factor.data = pivoted_cholesky(value, max_iter=self.rank)
        else:
            raise AttributeError("Cannot set non-diagonal task noises when covariance is diagonal.")

    def _eval_covar_matrix(self):
        covar_factor = self.task_noise_covar_factor
        noise = self.noise
        D = noise * torch.eye(self.num_tasks, dtype=noise.dtype, device=noise.device)
        return covar_factor.matmul(covar_factor.transpose(-1, -2)) + D

    def _shaped_noise_covar(self, base_shape, add_noise=True, *params, **kwargs):
        noise_covar = super()._shaped_noise_covar(base_shape, *params, **kwargs)
        #noise = self.noise
        #noise_covar = noise_covar.add_diag(noise)
        
        #SC: reduce noise for virtual measurements to enforce values
        if self.N_virtual != 0:
            noise_covar2 = noise_covar.evaluate()
            N_ges = noise_covar2.shape[0]
            N_modify = self.N_virtual*self.num_tasks
            noise_covar2[N_ges-N_modify:,N_ges-N_modify:] = 1e-4*torch.eye(N_modify)
            noise_covar = lazify(noise_covar2)
        return noise_covar

