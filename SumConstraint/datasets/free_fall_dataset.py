# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 12:00:30 2020

@author: phipi206
"""
from .sc_dataset import sc_dataset
from ..dropper import dropper
import numpy as np
import torch



class free_fall_dataset(sc_dataset):
    def __init__(self, pars):
        sc_dataset.__init__(self, pars)
        
        self.dlabel = 'ff'
        #outside inputs
        if self.noise == -1: self.noise = 1
        if self.Ntrain == -1: self.Ntrain = 20        
        if self.dropout_mode == -1: self.dropout_mode = 1
        if self.f_dropout == -1: self.f_dropout = 0
        
        self.Ntest = 100
        self.MT_dim = 2
        self.MT_dim_trans = 3
        
        #parameters of data generating process        
        self.constant_C = 1
        self.init_pars()
        
        
        
        #define inputs
        self.xmin = 0
        self.xmax = 6
        self.xmin_off = 0.1
        
        self.train_and_test()   
        
        #scale
        fnorm = 20
        self.train_f = self.train_f/fnorm
        self.train_y = self.train_y/fnorm
        self.test_y = self.test_y/fnorm
        self.C = self.C/fnorm
        self.F[0,2] = self.F[0,2]*fnorm
        
        self.init_dropout()
        if self.sopt == 1: #drop out some specific points for demonstrative purposes
            self.dropper.drop_ind[4:8,0] = torch.ones(4)
            self.dropper.drop_ind[10:12,0] = torch.ones(2)           
            self.train_y = self.dropper.remove_dropout(self.train_y)
        self.init_trans()
        
        #meta data
        self.unconstrained_as_aux = 1
        self.ilist_aux = [1]
        self.ilist_aux_trans = [1]
        
        
        self.virtual_measurements = 1
        self.ilist_vm = [1]
        self.ilist_vm_trans = [2]
        
        self.drop_aux = 0        
        self.ilist_drop_aux = [1]
        self.ilist_drop_aux_trans = [1]
        
    def init_pars(self):      
        self.C = torch.tensor(200.0)
        self.m = 1
        
        self.g = 9.81
        self.v0 = torch.sqrt(torch.tensor(2*self.C/self.m))
        self.F = torch.tensor([self.m*self.g,0,self.m*0.5]).unsqueeze(0)        
        
        
        
        
    def f(self, t): #calculate (true) outputs from inputs
        F = torch.zeros([t.shape[0], self.MT_dim ])
        F[:,0] =  (self.v0*t - self.g/2*t**2)
        F[:,1] = (self.v0 - self.g*t)
        return F
        
    def ftrans(self, F): #calculate transformed outputs Ftrans from F
        Nt = F.shape[0]
        Ftrans = torch.zeros([Nt,self.MT_dim_trans])
        Ftrans[:,0] = F[:,0]
        Ftrans[:,1] = F[:,1]
        Ftrans[:,2] = F[:,1]**2
        return Ftrans
        
    def fback(self, Ftrans): #transform transformed outputs back, via help of auxiliary variables
        Nt = Ftrans.shape[0]
        F = torch.zeros([Nt,self.MT_dim])
        F[:,0] = Ftrans[:,0]  #put negative values to 0
        F[:,1] = torch.sign(Ftrans[:,1])*torch.sqrt(torch.abs(torch.maximum(Ftrans[:,2],torch.tensor(0.))))
        return F
        
    def drop_ind_trans(self, drop_ind): #get dropout indices for Y_trans
        Nt = drop_ind.shape[0]
        drop_ind_trans = torch.zeros([Nt, self.MT_dim_trans])
        drop_ind_trans[:,0] = drop_ind[:,0]
        drop_ind_trans[:,1] = drop_ind[:,1]
        drop_ind_trans[:,2] = drop_ind[:,1]
        return drop_ind_trans
        
            
