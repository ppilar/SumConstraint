import numpy as np
import torch
import copy

from ..dropper import dropper


#class with basic functionalities for datasets to which sum constraint can be applied
class sc_dataset():
    def __init__(self, pars):
        self.N_virtual = 0 #number of virtual measurements
        
        self.kernels = pars.get("kernels",[0,1]) #number of different kernels
        self.N_kernel = 3#len(self.kernels)
        
        
        self.models = [] #list of models learned by different GPs
        self.Ntrain = pars.get("Ntrain",-1)
        self.noise = pars.get("noise",-1)
        self.dropout_mode = pars.get("drm",-1)
        self.f_dropout = pars.get("fdr", -1)
        
        #parameters for virtual measurements
        self.vm_crit = 0
        self.vm_val = 0
        
        self.xmin_off = 0
        self.xmax_off = 0
        
        self.lsmin = 0.001

    #add random noise to data
    def add_noise(self, noise):
        if self.dlabel == 'mesh':
            buf = self.train_f
            buf[:,:-2] += np.random.normal(0,noise,[self.Ntrain,self.MT_dim-2])
            return buf
        else:
            return self.train_f + np.random.normal(0, noise, [self.Ntrain, self.MT_dim])
        
    #create (untransformed) training and test data
    def train_and_test(self):
        self.train_x = torch.linspace(self.xmin,self.xmax,self.Ntrain).float()
        self.test_x = torch.linspace(self.xmin-self.xmin_off, self.xmax+self.xmax_off, self.Ntest).float()
        
        #add noise
        self.train_f = self.f(self.train_x)
        self.train_y = self.add_noise(self.noise)        
        self.test_y = self.f(self.test_x)
    
    #dropout data points and transform data
    def dropout_and_trans(self): #initialize dropout and transformed outputs
        self.init_dropout()
        self.init_trans()
        
    #dropout data points
    def init_dropout(self):
        self.dropper = dropper(self.f_dropout, self.dropout_mode)
        self.train_y = self.dropper.dropout(self.train_y)

    #transform data
    def init_trans(self):
        self.train_x_trans = self.train_x
        self.train_y_trans = self.ftrans(self.train_y)
        self.test_y_trans = self.ftrans(self.test_y)
        self.dropper_trans = dropper(self.f_dropout, self.dropout_mode, self.drop_ind_trans(self.dropper.drop_ind))
    
        
    #load relevant data for GP training from sc_dataset
    def load(self, jkernel):
        transform_yn = 0 if jkernel == 0 else 1        
        if transform_yn == 0:
            return self.train_x.float(), self.train_y.float(), self.test_x.float(), self.MT_dim, copy.copy(self.dropper), self.F, self.get_C(), transform_yn
        else:
            return self.train_x_trans, self.train_y_trans, self.test_x, self.MT_dim_trans, copy.copy(self.dropper_trans), self.F, self.get_C(), transform_yn

    #initialize arrays to store results
    def init_predictions(self, N_iter): #N_iter attribute of ds?
        d0 = self.test_y.shape[0]
        d1 = self.test_y.shape[1]
        self.pmean_ges = torch.zeros([self.N_kernel,d0,d1])
        self.lower_ges = torch.zeros([self.N_kernel,d0,d1])
        self.upper_ges = torch.zeros([self.N_kernel,d0,d1])

        d0 = self.test_y_trans.shape[0] #+ self.N_virtual #consider virtual zero crossings
        d1 = self.test_y_trans.shape[1]
        self.pmean_trans_ges = torch.zeros([self.N_kernel,d0,d1])
        self.lower_trans_ges = torch.zeros([self.N_kernel,d0,d1])
        self.upper_trans_ges = torch.zeros([self.N_kernel,d0,d1])
        
        self.imll = torch.zeros(self.N_kernel).int()
        self.mll_ges = torch.zeros([self.N_kernel, N_iter])
        self.mll2_ges = torch.zeros([self.N_kernel, N_iter])
        self.mll3_ges = torch.zeros([self.N_kernel, N_iter])
        
    #store predictions after training
    def set_prediction(self, jkernel, transform_yn, pmean, pmean_trans, lower, lower_trans, upper, upper_trans, mll_ges, mll2_ges, mll3_ges, N_iter):
        if jkernel == 0:
            self.init_predictions(N_iter)
        
        self.pmean_ges[jkernel,:,0:pmean.shape[1]] = pmean
        self.lower_ges[jkernel,:,0:pmean.shape[1]] = lower
        self.upper_ges[jkernel,:,0:pmean.shape[1]] = upper
        
        if transform_yn == 1:            
            self.pmean_trans_ges[jkernel,:,0:pmean_trans.shape[1]] = pmean_trans
            self.lower_trans_ges[jkernel,:,0:lower_trans.shape[1]] = lower_trans
            self.upper_trans_ges[jkernel,:,0:upper_trans.shape[1]] = upper_trans
        
        self.imll[jkernel] = mll_ges.size-1
        self.mll_ges[jkernel,:mll_ges.size] = torch.tensor(mll_ges)
        self.mll2_ges[jkernel,:mll2_ges.size] = torch.tensor(mll2_ges)
        self.mll3_ges[jkernel,:mll3_ges.size] = torch.tensor(mll3_ges)
         
    #use auxiliary variables from unconstrained GP in current GP
    def fill_in_aux(self, pmean_trans, lower_trans, upper_trans):
        for i in range(len(self.ilist_drop_aux)):
            pmean_trans[:,self.ilist_drop_aux_trans[i]] = self.pmean_ges[0,:,self.ilist_drop_aux[i]]
            lower_trans[:,self.ilist_drop_aux_trans[i]] = self.lower_ges[0,:,self.ilist_drop_aux[i]]
            upper_trans[:,self.ilist_drop_aux_trans[i]] = self.upper_ges[0,:,self.ilist_drop_aux[i]]
            
        return pmean_trans, lower_trans, upper_trans
        
    #backtransform transformed outputs
    def backtransform(self, transform_yn, pmean_trans, lower_trans, upper_trans):
        if transform_yn == 1:
            lower_trans_buf = lower_trans.clone()
            upper_trans_buf = upper_trans.clone()
            for i in self.ilist_aux_trans:    #only use mean for backtransformations
                lower_trans_buf[:,i] = pmean_trans[:,i]
                upper_trans_buf[:,i] = pmean_trans[:,i]
            
            pmean = self.fback(pmean_trans)
            lower = self.fback(lower_trans_buf)
            upper = self.fback(upper_trans_buf)
            
            return pmean, lower, upper
        else:
            return pmean_trans, lower_trans, upper_trans
    
    #return constraint C (constant) or fC (function)
    def get_C(self):
        if self.constant_C == 1:
            return self.C
        else:
            return self.fC
    
    ##### more polish needed below
    
    #insert virtual measurements into data
    def insert_virtual_measurements(self):
        N_virtual = 0
        
        ix = 0
        for i in self.ilist_vm_trans:
            x_zero_list = self.zc_lists[ix]
            ix += 1
            
            for j in range(len(x_zero_list)):
                y_new = self.dropper_trans.dvalue*torch.ones([1,self.MT_dim_trans])
                y_new[0][i] = self.vm_val
                drop_new = torch.ones([1,self.MT_dim_trans])
                drop_new[0][i] = 0
                self.dropper_trans.drop_ind = torch.cat((self.dropper_trans.drop_ind,drop_new))
                if self.dropper_trans.dropout_mode == 0:
                    self.dropper_trans.dropout_mode = 1                
                    
                self.train_x_trans = torch.cat((self.train_x_trans,x_zero_list[j].unsqueeze(0)))
                self.train_y_trans = torch.cat((self.train_y_trans,y_new))
                
    #standard optimizer parameters
    def get_optimizer_pars(self, jkernel):
        lr = 0.3
        s1 = 200
        s2 = 0.5
        
        return lr, s1, s2
        
    #get optimizer and scheduler
    def get_optimizer(self, jkernel, model):
        lr, s1, s2 = self.get_optimizer_pars(jkernel)
        
        optimizer = torch.optim.Adam([
            {'params': model.parameters()},  # Includes GaussianLikelihood parameters
        ],lr = lr)                
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,s1,s2)
        print(s1)
        return optimizer, scheduler
    
    
    #get first and last index for error calculation
    def get_i1i2(self, opt='all'):
        if opt=='int': #interpolation error
            i1 = (np.abs(self.test_x - min(self.train_x))).argmin()+1
            i2 = (np.abs(self.test_x - max(self.train_x))).argmin()-1
        else: #all testpoints
            i1 = 0
            i2 = self.pmean_ges.shape[1]
        return i1, i2
    
    #calculate root mean squared error
    def calculate_erms(self, opt='all'):
        i1, i2 = self.get_i1i2(opt)
        
        erms = torch.zeros(self.N_kernel)
        for i in range(self.N_kernel):
            Ydiff = self.test_y[i1:i2,:] - self.pmean_ges[i][i1:i2,:]
            erms[i] = torch.sqrt(torch.mean(Ydiff**2))
        
        return erms
    
    #calculate absolute violation of constraint
    def calculate_deltaC(self, opt='all'): #TODO: include case of non-constant C
        i1, i2 = self.get_i1i2(opt)
        
        deltaC = torch.zeros(self.N_kernel)
        for i in range(self.N_kernel): 
            buf = self.ftrans(self.pmean_ges[i,i1:i2,:])
            deltaC[i] = torch.mean(torch.abs(torch.matmul(self.F,buf.T) - self.get_Cbuf(i1,i2,buf)))
            
        return deltaC
    
    #get correct C for calculation of delta C for various cases
    def get_Cbuf(self,i1,i2,buf):
        if self.constant_C == 1:
            if self.dlabel == 'dp':   #calculate error with respect to better estimate C0   
                Cbuf = self.C0
            elif self.C.ndim == 0: #constant C; 1 constraint
                Cbuf = self.C
            else: #constant C; multiple constraints           
                Cbuf = self.C.unsqueeze(1).repeat(1,buf.shape[0])
        else:   #variable C; 1 constraint
            Cbuf = self.fC(self.test_x[i1:i2])
            
        return Cbuf
        
    
    
    
    
    
    
    
    