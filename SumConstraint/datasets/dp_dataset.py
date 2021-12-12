# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 12:00:30 2020

@author: phipi206
"""
from .sc_dataset import sc_dataset
from ..dropper import dropper
import numpy as np
import torch
import csv


class dp_dataset(sc_dataset):
    def __init__(self, pars):
        sc_dataset.__init__(self, pars)          
        self.dlabel = 'dp'
        
        #outside inputs
        #if self.noise == -1: self.noise = 0 #no additional noise for real data
        if self.Ntrain == -1: self.Ntrain = 15
        if self.dropout_mode == -1: self.dropout_mode = 1
        if self.f_dropout == -1: self.f_dropout = 0
                
        self.MT_dim = 8
        self.MT_dim_trans = 12
        self.constant_C = 1
        
        #load dp data
        self.xv, self.Eges = self.load_dp_data()
        
        #split into train and test set        
        Ndp = 200
        step= int(np.ceil((Ndp-1)/(self.Ntrain)))
        itrain = np.array(range(2,Ndp,step))
        itest = np.array(range(0,1+step*(itrain.size-1)))
        itest = np.array([i for i in itest if i not in itrain ])
        self.Ntest = len(itest)
        
        #energy
        self.C = torch.mean(torch.tensor(self.Eges[itrain])).float() #estimate on training data
        self.C0 = torch.mean(torch.tensor(self.Eges)).float() #estimate on all 200 dps
        
        
        #initialize remaining parameters and rescale data
        self.init_pars()
        self.rescale_data()
        
        
        #define inputs
        self.xmin = 0
        self.xmax = 2
        self.train_x = torch.tensor(itrain/100).float()
        self.test_x = torch.tensor(itest/100).float()
                
        #define outputs
        self.train_y = torch.tensor(self.xv[:,itrain].T)
        self.test_y =  torch.tensor(self.xv[:,itest].T)
        
        #dropout and transformed outputs
        self.dropout_and_trans()
        
        
        
        #meta data
        self.unconstrained_as_aux = 1 
        self.ilist_aux = [4,5,6,7]
        self.ilist_aux_trans = [4,6,8,10]
        
        self.virtual_measurements = 1
        self.ilist_vm = [4,5,6,7]
        self.ilist_vm_trans = [5,7,9,11]
        
        self.drop_aux = 1
        self.ilist_drop_aux = [1,3,4,5,6,7]
        self.ilist_drop_aux_trans = [1,3,4,6,8,10]
        
    def load_dp_data(self):        
        itr = np.random.randint(0,21) #randomly pick one out of the 21 different trajectories
        ltr = 200 #length of trajectory
        tr_max = 17000 #max dp. to be considered as start of trajectory
        tr_start = np.random.randint(10000,tr_max) #pick starting point somewhere in second half of motion
        tr_end = tr_start + ltr
        
        xv, Eges = load_dp_trajectory(itr,tr_start,tr_end)
        return xv, Eges
        
        
    def init_pars(self):
        self.m1 = 6.5
        self.m2 = 1
        self.g = 9.81
        
        self.F = torch.tensor([self.m1*self.g,0.,self.m2*self.g,0.,0,self.m1/2,0,self.m1/2,0,self.m2/2,0,self.m2/2]).unsqueeze(0)
        
    def rescale_data(self): #rescale data for output values to have similar magnitude
        self.xv[:4,:] *= 20
        self.xv[4:,:] *= np.sqrt(10)
        self.F[:,:4] *= 10/20 
        self.C *= 10
        self.C0 *= 10
        
        
    def ftrans(self, F): #calculate transformed outputs Ftrans from F
        Nt = F.shape[0]
        Ftrans = torch.zeros([Nt,self.MT_dim_trans])
        Ftrans[:,0] = F[:,0]
        Ftrans[:,1] = F[:,1]
        Ftrans[:,2] = F[:,2]
        Ftrans[:,3] = F[:,3]
        Ftrans[:,4] = F[:,4]
        Ftrans[:,5] = F[:,4]**2
        Ftrans[:,6] = F[:,5]
        Ftrans[:,7] = F[:,5]**2
        Ftrans[:,8] = F[:,6]
        Ftrans[:,9] = F[:,6]**2
        Ftrans[:,10] = F[:,7]
        Ftrans[:,11] = F[:,7]**2
        return Ftrans
        
    def fback(self, Ftrans): #transform transformed outputs back, via help of auxiliary variables
        Nt = Ftrans.shape[0]
        F = torch.zeros([Nt,self.MT_dim])
        F[:,0] = Ftrans[:,0]
        F[:,1] = Ftrans[:,1]
        F[:,2] = Ftrans[:,2]
        F[:,3] = Ftrans[:,3]
        F[:,4] = torch.sign(Ftrans[:,4])*torch.sqrt(torch.abs(torch.maximum(Ftrans[:,5],torch.tensor(0.))))
        F[:,5] = torch.sign(Ftrans[:,6])*torch.sqrt(torch.abs(torch.maximum(Ftrans[:,7],torch.tensor(0.))))  #put negative values to 0
        F[:,6] = torch.sign(Ftrans[:,8])*torch.sqrt(torch.abs(torch.maximum(Ftrans[:,9],torch.tensor(0.))))
        F[:,7] = torch.sign(Ftrans[:,10])*torch.sqrt(torch.abs(torch.maximum(Ftrans[:,11],torch.tensor(0.))))
        return F
        
    def drop_ind_trans(self, drop_ind): #get dropout indices for Y_trans
        Nt = drop_ind.shape[0]
        drop_ind_trans = torch.zeros([Nt, self.MT_dim_trans])
        drop_ind_trans[:,0] = drop_ind[:,0]
        drop_ind_trans[:,1] = drop_ind[:,1]
        drop_ind_trans[:,2] = drop_ind[:,2]
        drop_ind_trans[:,3] = drop_ind[:,3]
        drop_ind_trans[:,4] = drop_ind[:,4]
        drop_ind_trans[:,5] = drop_ind[:,4]
        drop_ind_trans[:,6] = drop_ind[:,5]
        drop_ind_trans[:,7] = drop_ind[:,5]
        drop_ind_trans[:,8] = drop_ind[:,6]
        drop_ind_trans[:,9] = drop_ind[:,6]
        drop_ind_trans[:,10] = drop_ind[:,7]
        drop_ind_trans[:,11] = drop_ind[:,7]
        return drop_ind_trans
    
    def get_optimizer_pars(self, jkernel):
        lr = 0.1
        if jkernel == 0:
            s1 = 500
            s2 = 0.5
        else:
            s1 = 800
            s2 = 0.2
        
        return lr, s1, s2



#load random section of dp trajectory
def load_dp_trajectory(ds,i0,i1):
    dp_raw = load_dp_data(ds)
    xv = get_cartesian_xv(dp_raw,i0,i1)
    E,_,_ = get_energy_cartesian(xv)
    
    return xv, E


def load_dp_data(ndp = 0): #load dp trajectroy
    path = '../double_pendulum/data/original/dpc_dataset_csv/'
    with open(path+str(ndp)+'.csv', newline='') as csvfile:
        dpreader = csv.reader(csvfile,delimiter=',')
        vdp = []
        for row in dpreader:
            vdp.extend([float(i) for i in row])
        
    return np.reshape(np.array(vdp),[-1,len(row)])

def get_cartesian_xv(dp_raw,i0,i1): #extract cartesian positions x and velocities v from data
    x = dp_raw[:,2:]
    
    #use fixation of double pendulum as origin:
    x[:,0] = -(x[:,0] - dp_raw[:,0]) #y1
    x[:,2] = -(x[:,2] - dp_raw[:,0]) #y2
    x[:,1] = -(x[:,1] - dp_raw[:,1]) #x1
    x[:,3] = -(x[:,3] - dp_raw[:,1]) #x2    
    v = np.gradient(x,1/500,axis=0) #choose 500 instead of 400 Hz; see appendix
    
    #l1 = 589.31 px = 0.091m -> 1m=6475.9px
    #l2 = 450.57 px = 0.07m  -> 1m=6436.71px
    pxfactor = 6450 #convert from pixels to meters
    x = x/pxfactor#*20
    v = v/pxfactor
    
    xv = np.zeros([8,i1-i0])
    xv[:4,:] = x[i0:i1,:].T
    xv[4:,:] = v[i0:i1,:].T
    
    return xv

def get_energy_cartesian(xv,m1=6.5): #calculate energy from data
    m2 = 1    
    g = 9.81#/20
    Epot = m1*g*xv[0,:] + m2*g*xv[2,:]
    Ekin = m1/2*(xv[4,:]**2 + xv[5,:]**2) + m2/2*(xv[6,:]**2 + xv[7,:]**2)
    E = Epot + Ekin
    
    return E, Ekin, Epot


