from .sc_dataset import sc_dataset
from ..dropper import dropper
import numpy as np
import torch



class logsin_dataset(sc_dataset):
    def __init__(self, pars):
        sc_dataset.__init__(self, pars)        
        self.dlabel = 'logsin'
        
        #outside inputs
        if self.noise == -1: self.noise = 1
        if self.Ntrain == -1: self.Ntrain = 20        
        if self.dropout_mode == -1: self.dropout_mode = 1
        if self.f_dropout == -1: self.f_dropout = 0
        
        self.Ntest = 100
        self.lsmin = 0.1
        
        self.MT_dim = 2
        self.MT_dim_trans = 3
        
        #parameters of data generating process
        self.constant_C = 0
        self.init_fpars()
        
        
        #define inputs
        self.xmin = -1.2
        self.xmax = 2
        
        self.train_and_test()
        
        self.init_dropout()        
        #omit invalid measurements (log(<0) is undefined)
        i_neg = torch.where(self.train_y[:,0]<=0) 
        self.dropper.drop_ind[:,0][i_neg] = 1
        self.train_y[:,0][i_neg] = self.dropper.dvalue
        
        self.init_trans()
        
        #meta data for trainig process
        self.unconstrained_as_aux = 1
        self.ilist_aux = [1]
        self.ilist_aux_trans = [1]
        
        self.virtual_measurements = 1
        self.ilist_vm = [1]
        self.ilist_vm_trans = [2]        
        self.vm_crit = torch.tensor(-np.pi/2)
        self.vm_val = torch.tensor(-1.)
        
        self.drop_aux = 1
        self.ilist_drop_aux = self.ilist_aux
        self.ilist_drop_aux_trans = self.ilist_aux_trans
        
        self.Laplace_opt_vec = [2,0,3] if self.drop_aux == 0 else [2,3]
        
       
    def init_fpars(self):        
        self.a = 5
        self.b = 1        
        
        self.F = torch.tensor([1.,0.,1.]).unsqueeze(0)    
    
    def fC(self, t):
        Fbuf = self.f(t)
        Cbuf = torch.log(Fbuf[:,0]) + torch.sin(Fbuf[:,1]) #adapt to avoid nan!
        return Cbuf
    
    def f(self, x): #calculate (true) outputs from inputs
        F = torch.zeros([x.shape[0], self.MT_dim ])
        F[:,0] = (2*torch.exp(-self.a*(x-1)**2) + 1*torch.exp(-self.a*(x+1)**2) + 0.2)
        F[:,1] = -0.5*x**3
        return F
        
    def ftrans(self, F): #calculate transformed outputs Ftrans from F
        Nt = F.shape[0]
        Ftrans = torch.zeros([Nt,self.MT_dim_trans])
        Ftrans[:,0] = torch.log(F[:,0])
        Ftrans[:,1] = F[:,1]#torch.log(F[:,0]) + torch.sin(F[:,1])
        Ftrans[:,2] = torch.sin(F[:,1])
        return Ftrans
        
    def fback(self, Ftrans): #transform transformed outputs back, via help of auxiliary variables
        Nt = Ftrans.shape[0]
        F = torch.zeros([Nt,self.MT_dim])
        F[:,0] = torch.exp(Ftrans[:,0])
        Ft2buf = torch.minimum(torch.maximum(Ftrans[:,2],torch.tensor(-1)),torch.tensor(1))
        F[:,1] = (np.abs(Ftrans[:,1])<=np.pi/2)*torch.arcsin(Ft2buf) + (np.abs(Ftrans[:,1])>np.pi/2)*(-np.pi - torch.arcsin(Ft2buf))# + torch.floor((Ftrans[:,1]+np.pi/2)/np.pi)*np.pi# - np.pi/2 #(Ftrans[:,1]+np.pi/2)%(np.pi)
        return F
        
    def drop_ind_trans(self, drop_ind): #get dropout indices for Y_trans
        Nt = drop_ind.shape[0]
        drop_ind_trans = torch.zeros([Nt, self.MT_dim_trans])
        drop_ind_trans[:,0] = drop_ind[:,0]
        drop_ind_trans[:,1] = drop_ind[:,1]
        drop_ind_trans[:,2] = drop_ind[:,1]
        return drop_ind_trans
    
            
