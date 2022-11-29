import numpy as np
import time
import dill
import sys

import matplotlib.pyplot as plt

import torch
import gpytorch
from gpytorch.lazy import lazify #, KroneckerProductLazyTensor

from .sc_gpytorch.sc_multitask_multivariate_normal import MultitaskMultivariateNormalDropout
from .sc_gpytorch.sc_MultitaskKernel import sc_MultitaskKernel
from .sc_gpytorch.sc_exact_gp import sc_ExactGP

from .sc_gpytorch.sc_exact_marginal_log_likelihood import sc_ExactMarginalLogLikelihood
from .sc_gpytorch.sc_multitask_gaussian_likelihood import sc_MultitaskGaussianLikelihood

from .utils import *
from .sc_results import sc_results

from .plots import plot_results, conf_plot

from .sc_gpytorch.utils import get_nan_entries, omit_nan_mu_covar_v0

#define multitask GP model
class MultitaskGPModel(sc_ExactGP):
    
    #set parameters of model
    def __init__(self, train_x, train_y, likelihood, dropper, MT_GP_kernel, C, F, mode):
        #dropper: handles missing measurements
        #MT_GP_kernel: selects what kernel to use
        #MT_dim: the multitask-dimension of the GP
        #C: the constraint; either constant or function C(x)
        #F: matrix F containing coefficients of constraint        
               
        self.mode = mode
        super(MultitaskGPModel, self).__init__(train_x, train_y, likelihood, self.mode)
        
        MT_dim = train_y.shape[1]
        self.MT_GP_kernel = MT_GP_kernel        
        self.constrain = 0
        self.dropper = dropper
        self.rank = MT_dim
        
        self.pred_drop_index = 0
        self.finv = 100 #heuristic factor to improve conditioning in inverse in constrained GP
        
        set_model_mean_covar_modules(self, MT_GP_kernel, MT_dim, C, F, self.constrain)

    #forward pass
    def forward(self, x):
        if self.MT_GP_kernel != 1:
            mean_x = self.mean_module(x)
        else:
            mean_x = self.covar_module.mean_module(x)
        covar_x = self.covar_module(x,x).evaluate()
            
        #update mean to include constraint information
        if self.MT_GP_kernel == 1:
            mean_x = self.covar_module.mu2
            
        #dropout data points
        if self.pred_drop_index < 2:          
            mean_x, covar_x = self.dropper.mean_covar_dropout(mean_x,covar_x)
            
        if self.pred_drop_index in [1,2]:
            self.pred_drop_index += 1
        if self.pred_drop_index == 3:
            self.pred_drop_index = 1
            
        return MultitaskMultivariateNormalDropout(mean_x, lazify(covar_x))
        
    #select whether constrained GP should constrain or not; allows e.g. to start constraining only after some iterations
    def set_constrain(self, constrain):
        self.constrain = constrain
        if self.MT_GP_kernel == 1:
            self.covar_module.constrain = constrain
            
    #finv is a small number added to diagonal when calculating the constrained kernel to avoid numerical artefacts/singular matrices
    def set_finv(self, finv):
        self.finv = finv
        if self.MT_GP_kernel == 1:
            self.covar_module.finv = finv
        
        
        
#get mean and covar modules for GP model
def set_model_mean_covar_modules(model, MT_GP_kernel, MT_dim, C, F, constrain):    
    cm = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
    
    if MT_GP_kernel == 1: #constrained GP        
        buf_mean = gpytorch.means.MultitaskMean(gpytorch.means.ConstantMean(), num_tasks=MT_dim)
        model.covar_module = sc_MultitaskKernel(cm, buf_mean, num_tasks = MT_dim, rank = model.rank, C = C, F=F, constrain=constrain)
    elif MT_GP_kernel == 0  or MT_GP_kernel == 2: #unconstrained GP
        model.mean_module = gpytorch.means.MultitaskMean(gpytorch.means.ConstantMean(), num_tasks=MT_dim)    
        model.covar_module = gpytorch.kernels.MultitaskKernel(cm,num_tasks=MT_dim,rank=model.rank)



#train GP and collect predictions
def evaluate_model(jkernel, dataset, N_iter, constrain_start, use_approximation, early_stopping):    
    x, y, xs, MT_dim, dropper, F, C, transform_yn = dataset.load(jkernel)    
    
    # remove auxiliary outputs from transformed vector, if they are learned separately (by unconstrained GP)
    if dataset.drop_aux == 1 and jkernel > 0: #check for dropper clone
        y, MT_dim, dropper, F = drop_auxvar(dataset.ilist_drop_aux_trans, y, MT_dim, dropper, F)
    
    Ntry = 0 #number of overall attempts
    Ndnc = 0 #number of failures to converge
    nc = 40 #number of steps for convergence criterion
    for itry in range(200): #loop to catch numerical instabilities and retry with newly initialized hyperparameters
        try: #retry with different initialization of hyperaparameters, when cholesky fails
              #initialize model
                     
                    
            Ntry += 1
            likelihood = sc_MultitaskGaussianLikelihood(num_tasks = MT_dim, N_virtual = dataset.N_virtual)
            opt_approx = get_opt_approx(transform_yn, use_approximation)
            
            model = MultitaskGPModel(x, y, likelihood, dropper, jkernel, C, F, opt_approx)
            fhat, var_pars = init_approx_pars(opt_approx, model, x, y, MT_dim)
        
            
            if Ndnc >= 10: #if convergence is failing, add higher value to matrix to be inverted during constraining
                model.set_finv(model.finv+int(Ndnc/5)*100)
                print('f+',int(Ndnc/5)*100)
            
            # enter training mode
            model.train()
            likelihood.train()
            #SC - Laplace (1)
            mll = sc_ExactMarginalLogLikelihood(likelihood, model) #log-marginal likelihood; loss function
            optimizer, scheduler, N_iter0, N_iter_max, L_Newton_pars = dataset.get_optimizer(jkernel, model, use_approximation)    #optimizer and scheduler
            if N_iter == -1: N_iter = N_iter0
            if N_iter > N_iter_max: N_iter_max = N_iter
                
                
                
            mllges, mllges2, mllges3 = np.zeros([3, N_iter])
            titer = np.zeros([N_iter])
            tmllges = []
            for i in range(N_iter):
                start = time.time() 
                
                lbuf = round(model.covar_module.data_covar_module.base_kernel.lengthscale.item(),2)
                nbuf = round(model.likelihood.noise.item(),2)
                sys.stdout.write("\r iteration %i: , ls: %f, sig: %f" % (i, lbuf, nbuf))
                sys.stdout.flush()
                
                if i == constrain_start:
                    model.set_constrain(1)            
                    
                optimizer.zero_grad()
                output = model(x)
                
                #SC-Laplace(2)
                t0 = time.time()
                loss_buf, fhat, sig, W = mll(output, y, fhat, dataset.dlabel, i, opt_approx, dataset.Laplace_opt_vec,  L_Newton_pars, var_pars, dataset.N_virtual, dataset.vm_val, dataset.rescale_trans_vec, dataset.rescale_jtrans_vec)
                loss = -loss_buf
                tmllges.append(time.time() - t0)
                #if jkernel > 0 and dataset.dlabel != 'mesh':
                #    mllges2[i] = mll(output, y, dataset.N_virtual, omit_inds = dataset.ilist_aux_trans, Jterm=0)
                #    mllges3[i] = mll(output, y, dataset.N_virtual, omit_inds = dataset.ilist_trans_aux, Jterm=1)
                
                
                loss.backward()
                mllges[i] = -loss.item()
                #print('loss:',mllges[i])
                
                optimizer.step()
                scheduler.step()
                #print('ls:',model.covar_module.base_kernel.lengthscale.item())
                titer[i] = time.time() - start
                #print('t:',titer[i])
                
                if i > nc and early_stopping == 1: #stop early, if mll has converged
                    if np.std(mllges[i-nc:i+1]) < 0.1:
                        break
                del loss
        
            ls = model.covar_module.data_covar_module.base_kernel.lengthscale.item()
            if np.std(mllges[i-nc:i]) > 0.2 or np.sum(np.isnan(mllges[i-nc:i])) > 10:  #check variance of recent losses; criterion for early stopping
                print('mu:',np.mean(np.abs(mllges[i-nc:i+1])),'std:',np.std(mllges[i-nc:i+1]))
                print('did not converge; try again')
                Ndnc += 1
            elif model.covar_module.data_covar_module.base_kernel.lengthscale.item() <= dataset.lsmin:   #check if lengthscale is not unreasonable
                print('\nls:', model.covar_module.data_covar_module.base_kernel.lengthscale.item(),'\n')
                print('unreasonable lengthscale; try again')
            else:
                break
                
                        
        except Exception:            
            print('error in cholesky decomposition; try again, i:', i,', Ntry:',Ntry)
        
    #titer_avg = np.sum(titer)/training_iter
    #print('average step time: ' + str(titer_avg))
    #print('\naverage tmll:', str(np.array(tmllges).mean()))
    
    
    if jkernel == 0 and dataset.dlabel == 'dp':
        dataset.ls0 = model.covar_module.data_covar_module.base_kernel.lengthscale.item()
    
    #enter evaluation (predictive posterior) mode
    model.eval()
    likelihood.eval()
    model.pred_drop_index = 1
    
    #model prediction on test points
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        set_approx(model, opt_approx, fhat, sig, W, y)        
        predictions = model(xs)
    
    #extract quantities of interest from predictions
    with torch.no_grad():
        lower_trans, upper_trans = predictions.confidence_region() #2sd above and below
        pmean_trans = predictions.mean
        
    #extract zero crossings to create virtual measurements
    if dataset.virtual_measurements == 1 and jkernel == 0:
        if dataset.dlabel == 'logsin':
            dataset.zc_lists, dataset.N_virtual = get_zero_crossings(xs, pmean_trans[:,dataset.ilist_vm]-dataset.vm_crit)
        else:
            dataset.zc_lists, dataset.N_virtual = get_zero_crossings(xs, pmean_trans[:,dataset.ilist_vm])
        dataset.insert_virtual_measurements()
        
    #get pmean_trans, ... into correct shape before re-inserting aux outputs
    if dataset.drop_aux == 1 and jkernel > 0:
        pmean_trans, lower_trans, upper_trans = prepare_aux(dataset.ilist_drop_aux_trans, pmean_trans, lower_trans, upper_trans)
        
    #insert auxiliary outputs that have been learned before
    if dataset.unconstrained_as_aux == 1 and transform_yn == 1:
        pmean_trans, lower_trans, upper_trans = dataset.fill_in_aux(pmean_trans, lower_trans, upper_trans)
        
    
    #backtransform outputs
    pmean, lower, upper = dataset.backtransform(transform_yn, pmean_trans, lower_trans, upper_trans)
    
    #save predictions
    dataset.models.append(model)
    dataset.set_prediction(jkernel, transform_yn, pmean, pmean_trans, lower, lower_trans, upper, upper_trans, mllges[:i+1], mllges2[:i+1], mllges3[:i+1], N_iter_max, Ntry)
    
    print('')
    return dataset



def get_results(MT_GP_kernels, dlabel, noise, f_dropout, N_datasets=1, N_iter=-1, Ntrain=-1, constrain_start=0, use_approximation = 1, early_stopping=0, sopt=0, same_data=0):

    pars = {"Ntrain": Ntrain, "noise": noise, "fdr": f_dropout, "kernels": MT_GP_kernels, "sopt": sopt}
    
    results = sc_results(MT_GP_kernels) #initialize results
    for jN in range(N_datasets):    #loop over datasets
        print('\njN=' + str(jN))
        ds = get_data(dlabel, pars, same_data) #load dataset
        for jkernel in MT_GP_kernels:   #loop over kernels
            ds = evaluate_model(jkernel, ds, N_iter, constrain_start, use_approximation, early_stopping) #train model and make predictions
        results.add(ds) #add dataset to results
    #%%     
                
    #make plots
    print('\n\n')
    plot_results(ds)
    conf_plot(ds)    
    #%% 
    
    #print and save results        
    results.get_statistics()
    fname = f'{dlabel}Nd{N_datasets}Nit{N_iter}fd{f_dropout}nf{noise}'
    with open('results/data/output_' + fname + '.txt', "w") as ftxt:
        collect_parameters(ftxt, ds, MT_GP_kernels, ds.Ntrain, N_datasets, N_iter, constrain_start, use_approximation, early_stopping, same_data)        
        results.print_statistics(ftxt)
    
    with open('results/data/results_' + fname + '.dill','wb') as f:
                dill.dump(results,f)

    return ds, results