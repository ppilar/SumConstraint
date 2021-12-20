import torch
import numpy as np

#object that handles incomplete measurements
class dropper:    
    def __init__(self, f_dropout, dropout_mode, drop_ind = 0):#, Y = 0):
        #f_dropout ... probability for point to be dropped
        #dropout_mode:
        #   0 ... no dropout
        #   1 ... random points dropped
        #   2 ... entire output dropped
        
        self.f_dropout = f_dropout        
        self.drate = 1 - self.f_dropout
        self.dropout_mode = dropout_mode
        self.drop_ind = drop_ind
        self.dvalue = float("nan")
        
        #if type(Y) != int:
        #    _,_ = self.dropout(Y)
            
    def dropout(self,Y):
        if isinstance(self.drop_ind, int):
            self.drop_ind = self.get_dropout_ind(Y.shape)
        Y_new = self.remove_dropout(Y)
        #return self.drop_ind, Y_new
        return Y_new
        
    def get_dropout_ind(self,Yshape):
        N = Yshape[0]
        MT_dim = Yshape[1]
        dropout_ind = np.zeros([N,MT_dim])    
        Ynew = np.array([])
        if self.dropout_mode == 1:
            for j in range(N):
                tflist = np.random.binomial(1,self.drate,MT_dim).astype(bool)        
                if np.sum(tflist) == 0:
                    tflist[np.random.randint(0,MT_dim)] == True
                dropout_ind[j,:] = (~tflist).astype(int)
                #Ynew = np.concatenate((Ynew,Y[j,tflist]))
        
        if self.dropout_mode == 2:
            dropout_ind[:,0] = np.random.binomial(1,self.f_dropout,N).astype(bool) 
            #dropout_ind[:-1:4,1] = np.zeros(int(N/3))
        
        if self.dropout_mode == 3:
            dropout_ind[:,1] = np.ones(N)
            dropout_ind[:,0] = np.kron(np.ones(int(N/2)),np.array([0,1]))
            dropout_ind[:,2] = np.kron(np.ones(int(N/2)),np.array([1,0]))
            
        #drop out some specific points for HO-plot:
        #dropout_ind[:3,1] = np.ones(3)
        
        return torch.tensor(dropout_ind)
        
    def remove_dropout(self,Y):
        for i in range(Y.shape[0]):
            for j in range(Y.shape[1]):
                if self.drop_ind[i,j]==1:
                    Y[i,j] = self.dvalue
        return Y
    
    def mean_covar_dropout(self,mean_x2,covar_x):
        if torch.sum(self.drop_ind) == 0:
            covar_x2 = covar_x
            mean_x3 = mean_x2
        else: 
            covar_x2 = self.covar_dropout(covar_x)
            mean_x3 = self.mean_dropout(mean_x2)
        
        return mean_x3, covar_x2


    def covar_dropout(self,covar):
        dim = covar.shape[0]
        ind = np.reshape(self.drop_ind,dim)
        indeq0 = ind==0
        covar_new = torch.zeros([dim,dim])
        for i in range(dim):
            if indeq0[i]==True:
                covar_new[i,indeq0] = covar[i,indeq0]
            else:
                covar_new[i,i] = 1e-5
                
        return covar_new
    
    def mean_dropout(self,mean):
        mean_new = torch.zeros([mean.shape[0],mean.shape[1]])
        for i in range(mean.shape[0]):
            for j in range(mean.shape[1]):
                if self.drop_ind[i,j] == 0:
                    mean_new[i,j] = mean[i,j]
                else:
                    mean_new[i,j] = self.dvalue
            
        return mean_new