import torch
import numpy as np
import dill
import SumConstraint as sc
import matplotlib.pyplot as plt

#fixed seeds for testing
#s = np.random.randint(10000)
#s = 7010
#torch.manual_seed(s)
#np.random.seed(s)

#%% inputs

dlabel = 'HO' #options: HO, dHO, ff, mesh, dp, logsin
Ntrain = -1  #-1 for standard value
N_datasets = 1 #number of different datasets
MT_GP_kernels = [0,1]  #options: 0 .. unconstrained, 1 .. constrained, 2.. unconstrained on transformed outputs

f_dr_vec = [0] #f_dropout ... fraction of datapoints that are dropped
noise_vec = [0.1] #noise ... defines noise level

N_iter = 300 #total number of iterations in training process
constrain_start = 0 #at what iteration to start constraining
same_data = 0 #1 ... use same data as during last run




#%% training and results
#loop over parameter configurations
for f_dropout in f_dr_vec:
    for noise in noise_vec:
        pars = {"Ntrain": Ntrain, "noise": noise, "fdr": f_dropout, "kernels": MT_GP_kernels}
        
        results = sc.sc_results.sc_results(MT_GP_kernels) #initialize results
        for jN in range(N_datasets):    #loop over datasets
            print('\njN=' + str(jN))
            ds = sc.utils.get_data(dlabel, pars, same_data) #load dataset
            for jkernel in MT_GP_kernels:   #loop over kernels
                ds = sc.train_GP.evaluate_model(jkernel, ds, N_iter, constrain_start) #train model and make predictions
            results.add(ds) #add dataset to results
        
        #%%     
                    
        #make plots
        print('\n\n')
        sc.plots.plot_results(ds)
        sc.plots.conf_plot(ds)    
        #%% 
        
        #print and save results        
        results.get_statistics()
        fname = str(dlabel) + 'Nd' + str(N_datasets) + 'Nit' + str(N_iter) + 'fd' + str(f_dropout) + 'nf' + str(noise)
        ftxt = open('results/data/output_' + fname + '.txt', "w")
        sc.utils.collect_parameters(ftxt, ds, MT_GP_kernels, ds.Ntrain, N_datasets, N_iter, constrain_start, same_data)        
        results.print_statistics(ftxt)
        ftxt.close()
        
        with open('results/data/results_' + fname + '.dill','wb') as f:
                    dill.dump(results,f)