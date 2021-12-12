# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 12:00:30 2020

@author: phipi206
"""
from .sc_dataset import sc_dataset
from ..dropper import dropper
import numpy as np
import torch



class mesh_dataset(sc_dataset):
    def __init__(self, pars):
        sc_dataset.__init__(self, pars)        
        self.dlabel = 'mesh'
        
        #outside inputs
        if self.noise == -1: self.noise = 1
        if self.Ntrain == -1: self.Ntrain = 20        
        if self.dropout_mode == -1: self.dropout_mode = 1
        if self.f_dropout == -1: self.f_dropout = 0
        
        self.Ntest = 100
        self.MT_dim = 8
        self.MT_dim_trans = 10     
        self.constant_C = 1
        
        #parameters of data generating process   
        self.init_pars()
        
        #define inputs
        self.xmin = 0
        self.xmax = 5
        
        self.train_and_test()
        self.dropout_and_trans()
        
        #meta data        
        self.unconstrained_as_aux = 0        
        self.ilist_aux = [1]
        self.ilist_aux_trans = range(self.MT_dim_trans)
        
        self.virtual_measurements = 0        
        self.drop_aux = 0
        
    def init_pars(self):      
        self.C = 16*torch.tensor([1.,1.46,0.26,2])#.repeat(len(x),1).T 
        self.F = torch.tensor([[1,-2,0,0,1,0,0,0,0,0],[1,0,-2,0,0,0,0,1,0,0],[0,0,0,0,1,-2,0,1,0,0],[0,0,0,0,0,0,0,0,0,1]]).float()
             
        
    def f(self, x): #calculate (true) outputs from inputs
        f0 = 4*torch.tensor([[1,1],[2,1],[2.1,1.5],[0,0]]).float()#.double()
        F = torch.zeros(len(x),self.MT_dim)
        for j in range(x.shape[0]):
            fj = f0 + cf(x[j])
            R = get_R(beta(x[j]))
            fj2 = (R@fj.T).T + cf(x[j])
            F[j,:] = fj2.reshape(-1)
            F[j,-2:] = 4*torch.tensor([1,1])
        return F
        
    def ftrans(self, F): #calculate transformed outputs Ftrans from F
        Z = get_Z_2D(F)
        Ftrans = get_ytrans(Z)
        return Ftrans
        
    def fback(self, Ftrans): #transform transformed outputs back, via help of auxiliary variables
        Q = get_Qt(Ftrans)
        Z1 = restore_Z_2D(Q)
        Z1[:,0,:] = -Z1[:,0,:] #find way to detect sign problem!
        A1,B1 = get_origin_angles_2D(Z1)
        alpha_R = (A1-np.pi/4)[:,3]
        Z2 = rotate_Z_back_simple(Z1,alpha_R)
        F = get_F_2D(Z2)
        return F
        
    def drop_ind_trans(self, drop_ind): #get dropout indices for Y_trans
        drop_ind_trans = np.zeros([drop_ind.shape[0],10])
        drop_ind_trans[:,0] = np.logical_or(drop_ind[:,0],drop_ind[:,1])
        drop_ind_trans[:,1] = np.logical_or(drop_ind[:,0],np.logical_or(drop_ind[:,1],np.logical_or(drop_ind[:,2],drop_ind[:,3])))
        drop_ind_trans[:,2] = np.logical_or(drop_ind[:,0],np.logical_or(drop_ind[:,1],np.logical_or(drop_ind[:,4],drop_ind[:,5])))
        drop_ind_trans[:,3] = np.logical_or(drop_ind[:,0],np.logical_or(drop_ind[:,1],np.logical_or(drop_ind[:,6],drop_ind[:,7])))
        drop_ind_trans[:,4] = np.logical_or(drop_ind[:,2],drop_ind[:,3])
        drop_ind_trans[:,5] = np.logical_or(drop_ind[:,2],np.logical_or(drop_ind[:,3],np.logical_or(drop_ind[:,4],drop_ind[:,5])))
        drop_ind_trans[:,6] = np.logical_or(drop_ind[:,2],np.logical_or(drop_ind[:,3],np.logical_or(drop_ind[:,6],drop_ind[:,7])))
        drop_ind_trans[:,7] = np.logical_or(drop_ind[:,4],drop_ind[:,5])
        drop_ind_trans[:,8] = np.logical_or(drop_ind[:,4],np.logical_or(drop_ind[:,5],np.logical_or(drop_ind[:,6],drop_ind[:,7])))            
        drop_ind_trans[:,9] = np.logical_or(drop_ind[:,6],drop_ind[:,7])
        return torch.tensor(drop_ind_trans).float()
    
    def get_optimizer_pars(self, jkernel):
        lr = 0.1
        s1 = 800
        s2 = 0.2
        
        return lr, s1, s2
    
      
#helper functions to generate triangle trajectory     
def cf(x):
    return 0.5*torch.cos(torch.tensor(2*x))
        
def beta(x):
    return 0.5*x

#rotation matrix
def get_R(alpha):
    if type(alpha) != torch.Tensor:
        alpha = torch.tensor(alpha)
    return torch.tensor([[torch.cos(alpha),-torch.sin(alpha)],[torch.sin(alpha),torch.cos(alpha)]])#.double()

#construct matrix Z from F
def get_Z_2D(F):
    if F.ndim == 1:
        F = F.unsqueeze(0)
        
    N = F.shape[0]
    return F.reshape(N,-1,2).permute([0,2,1])

#get outputs F from Z
def get_F_2D(Z):
    N = Z.shape[0]
    return Z.permute([0,2,1]).reshape(N,-1)

#calculate entries y from Z
def get_ytrans(Zges):
    N = Zges.shape[0]
    dy = int(Zges.shape[2]*(Zges.shape[2]+1)/2)
    yges = torch.zeros(N,dy)
    for j in range(N):
        Z = Zges[j,:,:]
        ZTZ = Z.T@Z
        for i in range(ZTZ.shape[0]):
            if i == 0:
                y = ZTZ[i,i:]
            else:
                y = torch.cat((y,ZTZ[i,i:]))
        yges[j,:] = y
    return yges         

#arrange y in matrix Q
def get_Qt(y):
    N = y.shape[0]    
    d = int(-1/2+np.sqrt(1/4+2*y[0,:].numel()))
    Qtges = torch.zeros(N,d,d)
    for j in range(N):
        Qt = torch.zeros(d,d)
        yind = 0
        for i in range(d):
            Qt[i,i:] = y[j,yind:yind+d-i]
            Qt[i+1:,i] = Qt[i,i+1:]            
            yind += d-i
        Qtges[j,:,:] = Qt
    return Qtges


#use SVD to backtransform
def restore_Z_2D(Q):
    N = Q.shape[0]
    for j in range(N):
        U,S,V = torch.svd(Q[j,:,:])
        Z = (torch.sqrt(torch.diag(S))@V.T)[:2,:]
        dZ = Z.shape[0]
        if j == 0:
            Zges = Z
        else:
            Zges = torch.cat((Zges,Z))
    return Zges.reshape(N,dZ,-1)

#get angles with respect to origin
def get_origin_angles_2D(Z):   
    if Z.ndim == 2:
        Z = Z.unsqueeze(0)
    
     
    N = Z.shape[0]
    Nangles = Z.shape[2]
    alphages = torch.zeros(N,Nangles)
    betages = torch.zeros(N,Nangles)
    for j in range(N):
        alpha = torch.zeros(Nangles)
        beta = torch.zeros(Nangles)
        for i in range(Nangles):
            #angle via cos
            alpha[i] = torch.acos(Z[j,0,i]/torch.sqrt(Z[j,0,i]**2 + Z[j,1,i]**2))
            if Z[j,1,i] < 0:
                alpha[i] = 2*np.pi - alpha[i]
                
            #angle via tan
            beta[i] = torch.atan(Z[j,1,i]/(Z[j,0,i]+1e-6))
            if Z[j,0,i] < 0 and Z[j,1,i] > 0:
                beta[i] += np.pi
            if Z[j,0,i] < 0 and Z[j,1,i] < 0:
                beta[i] += np.pi
            if Z[j,0,i] > 0 and Z[j,1,i] < 0:
                beta[i] = 2*np.pi - beta[i]
            
        alphages[j,:] = alpha
        betages[j,:] = beta
        
    return alphages, betages

#rotate corner points according to angle mismatch
def rotate_Z_back_simple(Zges,alpha): #same angle for all points
    if Zges.ndim == 2:
        Zges = Zges.unsqueeze(1)
    N = Zges.shape[0]
    Nnode = Zges.shape[2]
    Zges_new = torch.zeros(Zges.shape)
    for j in range(N):
        Z = Zges[j,:,:]
        Znew = torch.zeros(Z.shape)
        #R = get_R(A1[j,0]-A[j,0])
        R = get_R(alpha[j])
        Znew = R.T@Z#.double()
        Zges_new[j,:,:] = Znew
    return Zges_new

