import numpy as np
from .utils import print2

#object to store learned models and prediction results
class sc_results():
    def __init__(self, kernels):
        self.ds_list = [] #list of datasets
        self.N_ds = 0 #number of datasets
        self.kernels = kernels #different kernels that were used during training
        self.N_kernel = 3#len(self.kernels)
        
    def add(self, dataset): #add dataset to results
        self.ds_list.append(dataset)
        self.N_ds += 1
        
    #calculate statistics of results over multiple datasets
    def get_statistics(self):
        self.erms_ges, self.deltaC_ges, self.mll_ges, self.mll2_ges, self.mll3_ges, self.unc_ges = np.zeros([6, self.N_ds, self.N_kernel])
        for i in range(self.N_ds):
            self.erms_ges[i,:] = self.ds_list[i].calculate_erms()
            self.deltaC_ges[i,:] = self.ds_list[i].calculate_deltaC()
            self.unc_ges[i,:] = self.ds_list[i].calculate_uncertainty()
            for j in range(self.N_kernel):
                self.mll_ges[i,j] = self.ds_list[i].mll_ges[j,self.ds_list[i].imll[j]]
                self.mll2_ges[i,j] = self.ds_list[i].mll2_ges[j,self.ds_list[i].imll[j]]
                self.mll3_ges[i,j] = self.ds_list[i].mll3_ges[j,self.ds_list[i].imll[j]]
            
        self.erms_avg = np.mean(self.erms_ges,0)
        self.erms_std = np.std(self.erms_ges,0)
        
        self.deltaC_avg = np.nanmean(self.deltaC_ges,0)
        self.deltaC_std = np.nanstd(self.deltaC_ges,0)
        
        self.unc_avg = np.nanmean(self.unc_ges,0)
        self.unc_std = np.nanstd(self.unc_ges,0)
        
        self.mll_avg = np.mean(self.mll_ges,0)
        self.mll_std = np.std(self.mll_ges,0)
        
        self.mll2_avg = np.mean(self.mll2_ges,0)
        self.mll2_std = np.std(self.mll2_ges,0)
        
        self.mll3_avg = np.mean(self.mll3_ges,0)
        self.mll3_std = np.std(self.mll3_ges,0)
        
        
    #print statistics
    def print_statistics(self, ftxt, ndec = 6):
        print2('', ftxt)
        print2('\nmll:', ftxt)
        print2(str(np.round(self.mll_avg,ndec)),ftxt)
        
        print2('sigma-mll:', ftxt)
        print2(str(np.round(self.mll_std,ndec)),ftxt)
        
        # print2('\nmll2:', ftxt)
        # print2(str(np.round(self.mll2_avg,ndec)),ftxt)
        
        # print2('sigma-mll2:', ftxt)
        # print2(str(np.round(self.mll2_std,ndec)),ftxt)
        
        # print2('\nmll3:', ftxt)
        # print2(str(np.round(self.mll3_avg,ndec)),ftxt)
        
        # print2('sigma-mll3:', ftxt)
        # print2(str(np.round(self.mll3_std,ndec)),ftxt)
        
        print2('\noutside:', ftxt)
        print2(str(np.round(self.unc_avg,ndec)),ftxt)
        
        print2('sigma-out:', ftxt)
        print2(str(np.round(self.unc_std,ndec)),ftxt)
        
        print2('\nerms:', ftxt)
        print2(str(np.round(self.erms_avg,ndec)),ftxt)
        
        print2('sigma-erms:', ftxt)
        print2(str(np.round(self.erms_std,ndec)),ftxt)
        
        print2('\ndeltaC:', ftxt)
        print2(str(np.round(self.deltaC_avg,ndec)),ftxt)
        
        print2('sigma-deltaC:', ftxt)
        print2(str(np.round(self.deltaC_std,ndec)),ftxt)
      



