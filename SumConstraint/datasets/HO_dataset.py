from .sc_dataset import sc_dataset
from ..dropper import dropper
import numpy as np
import torch



class HO_dataset(sc_dataset):
    def __init__(self, pars, dlabel = 'HO'):
        sc_dataset.__init__(self, pars)        
        self.dlabel = dlabel
        
        #outside inputs
        if self.noise == -1: self.noise = 1
        if self.Ntrain == -1: self.Ntrain = 20        
        if self.dropout_mode == -1: self.dropout_mode = 1
        if self.f_dropout == -1: self.f_dropout = 0
        if self.sopt == -1: self.sopt = 0
        
        self.Ntest = 100                
        self.MT_dim = 2
        self.MT_dim_trans = 4       
        self.constant_C = 1
        
        #initialize dataset specific parameter for curves and constraint
        self.init_pars()
        
        #define inputs
        self.xmin = 0
        self.xmax = 10
        self.xmin_off = -0.1
        
        #generate data
        self.train_and_test()
        self.init_dropout()
        if self.sopt == 1 and self.dlabel == 'HO': #drop out some specific points for demonstrative purposes
            self.dropper.drop_ind[:3,1] = torch.ones(3)
            self.train_y = self.dropper.remove_dropout(self.train_y)
        self.init_trans()
        #self.dropout_and_trans()  #initialize dropout and transformed outputs
        
        #meta data for trainig process
        self.unconstrained_as_aux = 1
        self.ilist_aux = [0,1]
        self.ilist_aux_trans = [0,2]
        
        self.virtual_measurements = 1
        self.ilist_vm = [0,1]
        self.ilist_vm_trans = [1,3]
        
        self.drop_aux = 0
        self.ilist_drop_aux = [0,1]
        self.ilist_drop_aux_trans = [0,2]
        
    def init_pars(self):       
        
        fnorm = 5        
        C0 = 20
        self.C = torch.tensor(C0/fnorm**2)
        self.x0 = torch.sqrt(2*self.C)
        self.m = 1
        self.w0 = 1
        self.D = self.m*self.w0**2
        
        self.F = torch.tensor([0,self.D/2,0,self.m/2]).unsqueeze(0)
        
    def f(self, t): #calculate (true) outputs from inputs
        F = torch.zeros([t.shape[0], self.MT_dim ])
        F[:,0] = self.x0*torch.sin(self.w0*t)#/fnorm
        F[:,1] = self.x0*self.w0*torch.cos(self.w0*t)#/fnorm
        return F
        
    def ftrans(self, F): #calculate transformed outputs Ftrans from F
        Nt = F.shape[0]
        Ftrans = torch.zeros([Nt,self.MT_dim_trans])
        Ftrans[:,0] = F[:,0]
        Ftrans[:,1] = F[:,0]**2
        Ftrans[:,2] = F[:,1]
        Ftrans[:,3] = F[:,1]**2
        return Ftrans
        
    def fback(self, Ftrans): #transform transformed outputs back, via help of auxiliary variables
        Nt = Ftrans.shape[0]
        F = torch.zeros([Nt,self.MT_dim])
        F[:,0] = torch.sign(Ftrans[:,0])*torch.sqrt(torch.abs(torch.maximum(Ftrans[:,1],torch.tensor(0.))))  #put negative values to 0
        F[:,1] = torch.sign(Ftrans[:,2])*torch.sqrt(torch.abs(torch.maximum(Ftrans[:,3],torch.tensor(0.))))
        return F
        
    def drop_ind_trans(self, drop_ind): #get dropout indices for Y_trans
        Nt = drop_ind.shape[0]
        drop_ind_trans = torch.zeros([Nt, self.MT_dim_trans])
        drop_ind_trans[:,0] = drop_ind[:,0]
        drop_ind_trans[:,1] = drop_ind[:,0]
        drop_ind_trans[:,2] = drop_ind[:,1]
        drop_ind_trans[:,3] = drop_ind[:,1]
        return drop_ind_trans
        
            
