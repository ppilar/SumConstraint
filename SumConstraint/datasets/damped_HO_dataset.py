from .HO_dataset import HO_dataset
import torch

class damped_HO_dataset(HO_dataset):
    def __init__(self, pars):
        HO_dataset.__init__(self, pars, 'dHO')
        self.dlabel = 'dHO'
        self.constant_C = 0
       
    def init_pars(self):
        super().init_pars()
        self.b = 0.1
        self.w = torch.sqrt(torch.tensor(self.w0**2-(self.b/(2*self.m)**2)))
        
    def fC(self, t):
        Fbuf = self.f(t)
        Cbuf = 0.5*self.D*Fbuf[:,0]**2 + 0.5*self.m*Fbuf[:,1]**2 
        return Cbuf
    
    def f(self, t):
        F = torch.zeros([t.shape[0], self.MT_dim ])
        self.x0d = self.x0*torch.exp((-self.b*t/(2*self.m)))
        F[:,0] = self.x0d*torch.sin(self.w*t)
        F[:,1] = self.x0d*self.w*torch.cos(self.w*t) - self.x0d*self.b/(2*self.m)*torch.sin(self.w*t)
        return F