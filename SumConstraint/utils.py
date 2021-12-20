import dill
import numpy as np
import gpytorch
import torch
import time
import sys

from .datasets.HO_dataset import HO_dataset
from .datasets.damped_HO_dataset import damped_HO_dataset
from .datasets.free_fall_dataset import free_fall_dataset
from .datasets.dp_dataset import dp_dataset
from .datasets.mesh_dataset import mesh_dataset
from .datasets.logsin_dataset import logsin_dataset

#from .sc_gpytorch.sc_exact_marginal_log_likelihood import ExactMarginalLogLikelihood
#from .sc_gpytorch.sc_multitask_gaussian_likelihood import MultitaskGaussianLikelihoodZN



#load dataset
def get_data(dlabel, pars, same_data = 0):
    if same_data == 1: #load dataset from previous run
        with open('data_previous.dill','rb') as f:
            dataset = dill.load(f)
    elif same_data == -1:
        with open('results/plot_data/'+dlabel+'_plot_data.dill','rb') as f:
            dataset = dill.load(f)
    else: #create new dataset, selected via dlabel
        if dlabel == 'HO':
            dataset = HO_dataset(pars)
        elif dlabel == 'dHO':
            dataset = damped_HO_dataset(pars)
        elif dlabel == 'ff':
            dataset = free_fall_dataset(pars)
        elif dlabel == 'mesh':
            dataset = mesh_dataset(pars)
        elif dlabel == 'dp':
            dataset = dp_dataset(pars)
        elif dlabel == 'logsin':
            dataset = logsin_dataset(pars)
        else:
            return -1
        with open('data_previous.dill','wb') as f: #save dataset to enable reloading
            dill.dump(dataset,f)
    return dataset

      
#print both to console and file
def print2(tstr,ftxt):
    print(tstr)
    ftxt.write(tstr+'\n')           
                
#list parameters of dataset and learning process
def collect_parameters(ftxt, ds, MT_GP_kernels, Ndata, N_datasets, N_iter, constrain_start, early_stopping, same_data):
    print2('dlabel:' + str(ds.dlabel), ftxt)
    print2('xmax:' + str(ds.xmax), ftxt)
    print2(' ', ftxt)
    print2('MT_GP_kernels' + str(MT_GP_kernels), ftxt)
    print2('Ndata:' + str(Ndata), ftxt)
    print2('N_datasets:' + str(N_datasets), ftxt)
    print2('dropout_mode:' + str(ds.dropout_mode), ftxt)
    print2('f_dropout:' + str(ds.f_dropout), ftxt)    
    print2('N_iterations:' + str(N_iter), ftxt)
    print2('constrain_start:' + str(constrain_start), ftxt)
    print2('early_stopping:' + str(early_stopping), ftxt)
    print2('same_data:' + str(same_data), ftxt)
    print2('noise:' + str(ds.noise), ftxt)
    print2('virtual_measurements:' + str(ds.virtual_measurements), ftxt)
    print2('use_2_as_auxvar:' + str(ds.unconstrained_as_aux), ftxt)
    print2('drop_aux:' + str(ds.drop_aux), ftxt)
    print2(' ', ftxt)
    print2(' ', ftxt)
                
    
            
#get list of zero crossings of outputs pmean
def get_zero_crossings(xs, pmean, tol = 1e-3):
    Nf = pmean.shape[1]
    Nx = pmean.shape[0]
    x_zero_lists = []
    N_artificial = 0
    for i in range(Nf):
        x_zero_list = []
        for j in range(1,Nx):                
            if torch.sign(pmean[j-1,i]) != torch.sign(pmean[j,i]):
                x_zero_list.append((xs[j-1]+xs[j])/2)
            elif j == 1 and torch.abs(pmean[j-1,i]) < tol:
                x_zero_list.append(xs[j])
            elif j == Nx-1 and torch.abs(pmean[j,i]) < tol:
                x_zero_list.append(xs[j])                
        x_zero_lists.append(x_zero_list)           
        N_artificial += len(x_zero_list)
        
    return x_zero_lists, N_artificial



#omit auxiliary outputs from GP
def drop_auxvar(ilist,y,MT_dim,dropper,F):
    ilist2 = [i for i in range(MT_dim) if i not in ilist]
    
    MT_dim = MT_dim - len(ilist)
    y = y[:,ilist2]
    dropper.drop_ind = dropper.drop_ind[:,ilist2]
    F = F[:,ilist2]
    return y, MT_dim, dropper, F

#create empty entries for auxiliary outputs in tensors ptrans, ...
def prepare_aux(ilist, ptrans, ltrans, utrans):
    MT_dim = ptrans.shape[1] + len(ilist)
    ilist2 = [i for i in range(MT_dim) if i not in ilist]
    
    pt, lt, ut = torch.zeros(3,ptrans.shape[0],MT_dim)
    pt[:,ilist2] = ptrans
    lt[:,ilist2] = ltrans
    ut[:,ilist2] = utrans
    
    return pt, lt, ut
    
    
    
    
    
    
    
    
    
    