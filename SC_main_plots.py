import torch
import numpy as np
import dill
import SumConstraint as sc
import matplotlib.pyplot as plt

#%%
#fixed seeds for testing
#s = np.random.randint(10000)
s = 42
torch.manual_seed(s)
np.random.seed(s)

#%% inputs
MT_GP_kernels = [0,1]  #options: 0 .. unconstrained, 1 .. constrained, 2.. unconstrained on transformed outputs
es = 0 #early_stopping
same_data = -1 #-1 ... load datasets used for plots
sopt = 1 #special option
N_iter = -1
noise = -1
f_dropout = -1
use_approximation = 1

dlabel = 'HO'
sc.train_GP.get_results(MT_GP_kernels, dlabel, noise, f_dropout, use_approximation = use_approximation, early_stopping=es, sopt=sopt, same_data=same_data)

dlabel = 'dHO'
sc.train_GP.get_results(MT_GP_kernels, dlabel, noise, f_dropout, use_approximation = use_approximation, early_stopping=es, sopt=sopt, same_data=same_data)

dlabel = 'ff' # f_dropout = 0.2 # noise = 0.1
sc.train_GP.get_results(MT_GP_kernels, dlabel, noise, f_dropout, use_approximation = use_approximation, early_stopping=es, sopt=sopt, same_data=same_data)

dlabel = 'logsin' # f_dropout = 0.2 # noise = 0.1
sc.train_GP.get_results(MT_GP_kernels, dlabel, noise, f_dropout, use_approximation = use_approximation, early_stopping=es, sopt=sopt, same_data=same_data)

# dlabel = 'dp' # f_dropout = 0 # noise = 0
# sc.train_GP.get_results(MT_GP_kernels, dlabel, noise, f_dropout, use_approximation = use_approximation, early_stopping=es, sopt=sopt, same_data=same_data)

