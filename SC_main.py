import torch
import numpy as np
import dill
import SumConstraint as sc
import matplotlib.pyplot as plt

#fixed seeds for testing
#s = np.random.randint(10000)
# s = 1
# torch.manual_seed(s)
# np.random.seed(s)

#%% inputs

dlabel = 'mesh' #options: HO, dHO, ff, mesh, dp, logsin
MT_GP_kernels = [0, 1, 2]  #options: 0 .. unconstrained, 1 .. constrained, 2.. unconstrained on transformed outputs
f_dr_vec = [0] #f_dropout ... fraction of datapoints that are dropped
noise_vec = [0.0001] #noise ... defines noise level
Nds = 50    #number of datasets
es = 0 #early stopping
same_data = 0
#%% training and results
#loop over parameter configurations
for f_dropout in f_dr_vec:
    for noise in noise_vec:
        
        ds, results = sc.train_GP.get_results(MT_GP_kernels, dlabel, noise, f_dropout, N_datasets=Nds, N_iter=-1, Ntrain=-1, constrain_start=0, early_stopping=es, sopt=0, same_data=same_data)
