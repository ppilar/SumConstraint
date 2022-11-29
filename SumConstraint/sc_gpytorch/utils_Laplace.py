# -*- coding: utf-8 -*-

import torch
import gpytorch
from .utils import omit_nan_mu_covar

#calculate gradient and Hessian of flpdf in Laplace approximation
def get_grad_H2(flpdf, x, mu, Kinv):
    d = x.shape[0]
    xl = flpdf(x)
    xdot0 = torch.autograd.grad(xl, x, create_graph=True, retain_graph = True)[0]
    W = torch.autograd.grad(torch.sum(xdot0), x, create_graph=True, retain_graph = True)[0]
    W = torch.diag(-W)
    
    xdot = -(xdot0 - Kinv@(x - mu))
    xdot2 = -(-W - Kinv)  
    
    return xdot, xdot2, W, xdot0


#Newton algorithm for Laplace approximation
def max_Newton(flpdf, f_lpfX, mu, K, x0, pars = [1, 1, 1, 2]):#step_size = 1, it_decay = 5, fd = 2, Ndec = 5):
    
    it_decay, step_size, Ndec, fd = pars    
    x = torch.autograd.Variable(torch.clone(x0).detach().float(), requires_grad = True)
    d = x.shape[0]


    xges = []
    xges.append(x.detach())
    xbuf = torch.zeros(d)
    
    delta_x = 10
    delta_target = 1e5
    target = 1e5
    it = 0
    
    inv_off = 1e-7*torch.eye(x.shape[0]) #1e-7 for HO
    Kinv_off = 1e-5*torch.eye(x.shape[0]) #1e-5 for HO
    Kinv = torch.inverse(K + Kinv_off)
    #print('')
    while(delta_target > 1e-8):#delta_x > 1e-6):
        xdot, xdot2, W, xdot0 = get_grad_H2(flpdf, x, mu, Kinv)  
        flag = -1
        if flag == -1:
            #straightforward
            inv_xdot2 = torch.inverse(xdot2 + inv_off)        
            x = x - step_size*inv_xdot2@xdot
        
        
        delta_x = torch.norm(x.detach() - xges[-1])
        xges.append(x.detach())  
        
        target0 = target
        target = flpdf(x) + f_lpfX(x)
        if it > 0:
            delta_target = torch.abs(target-target0)
             
        it += 1
        if it%it_decay == 0 and it > 0:
            step_size = step_size/fd
        if torch.sum(torch.isnan(x)) > 0:
            print('nan!')   
        if it > it_decay*Ndec:
            #print('delta_t:', delta_target)
            break
        
    xdot, xdot2, W, xdot0 = get_grad_H2(flpdf, x, mu, Kinv)   
    flag = -1
    if flag == -1:
        inv_xdot2 = torch.inverse(xdot2 + inv_off) 
        sig = sig_check(inv_xdot2)
        sig = torch.inverse(xdot2 + inv_off) 
        
    x = x.detach()
    return x, sig, W, xges


#####################
#####################



def sig_check(sig):
    sigsum = torch.sum(sig - sig.T).item()    
    if sigsum != 0:
        sig = 0.5*(sig + sig.T)
        #print('sigsum:', sigsum)
    return sig


#sum of two functions
def get_f_sum(f1, f2):
    def f_sum(f):
        return f1(f) + f2(f)
    return f_sum

#get function that calculates log of prior probability
def get_f_lpfX(function_dist):
    mu = function_dist.loc
    sig = function_dist.covariance_matrix
    mu, sig = omit_nan_mu_covar(mu, sig)
    function_dist2 = gpytorch.distributions.MultivariateNormal(mu, sig)

    def f_lpfX(f):
        return function_dist2.log_prob(f)
    return f_lpfX, mu, sig


#get function that calculates logpdf for Gaussian noise
def get_f_G_lpdf(xvec, sig):
    if sig.numel() == 1:
        sig = sig*torch.eye(xvec.shape[0])
    elif sig.ndim != 2:
        sig = torch.diag(sig)

    dist = gpytorch.distributions.MultivariateNormal(xvec, sig)
    fdist, _, _ = get_f_lpfX(dist)
    return fdist


#get function that calculates logpdf for squared nonlinearity
def get_f_chi2_lpdf(xvec, sig, iv=([], [], 0, 0, 0, '')):
    def chi2_lpdf(f):
        return chi2_log_pdf(xvec, f, sig, iv)
    return chi2_lpdf


#get function that calculates logpdf for log nonlinearity
def get_f_log_lpdf(xvec, sig, iv=([], [], 0, 0, 0, '')):
    def log_lpdf(f):
        return log_log_pdf(xvec, f, sig, iv)
    return log_lpdf

#get function that calculates logpdf for sin nonlinearity
def get_f_sin_lpdf(xvec, sig, iv=([], [], 0, 0, 0, '')):
    def sin_lpdf(f):
        return sin_log_pdf(xvec, f, sig, iv)
    return sin_lpdf


#construct vector of functions to calculate log(p(y|f)) for measurements
def construct_f_lpyf(optl_vec, target, fhat_prev, N_virtual, noise, rs_vec, rs_jtrans_vec, it, vm_val, dlabel):
    Ndata = target.shape[0]
    Ntask = target.shape[1]
    noise_ges = noise*torch.ones(target.shape)
    vm_start = 0

    for i, j in zip(range(len(rs_jtrans_vec)), rs_jtrans_vec):
        noise_ges[:, j] *= rs_vec[i]
        if j in [0,2]:
            noise_ges[:,j] /= torch.tensor(min(max(it/30, 1),20))

    target = target.reshape(-1)
    noise_ges = noise_ges.reshape(-1)
    Ntask = len(optl_vec)
    optl_vec_ges = torch.kron(torch.ones(Ndata), torch.tensor(optl_vec))

    # omit nan entries - dropout
    inan = torch.where(~torch.isnan(target))
    target = target[inan]
    noise_ges = noise_ges[inan]
    optl_vec_ges = optl_vec_ges[inan]

    i_target_vec = []
    for i in range(4):
        i_target_vec.append(torch.where(optl_vec_ges == i)[0])

    f_lpyf_vec = []
    jopt_vec = []
    for j in range(4):
        jopt = 0
        buf = i_target_vec[j]

        if j == 0 and len(buf) != 0:
            f_lpyf = get_f_G_lpdf(target[buf], noise_ges[buf])  # include iv
        elif j == 1 and len(buf) != 0:
            if it >= vm_start:                
                iiv, iiv_f = get_iiv(len(i_target_vec[j]), N_virtual, fhat_prev, buf)
                f_lpyf = get_f_chi2_lpdf(
                    target[buf], noise_ges[buf], (iiv, iiv_f, it, vm_val, vm_start, dlabel))  # (iiv,iv[1],iv[2]))
            else:
                f_lpyf = get_f_chi2_lpdf(target[buf], noise_ges[buf])
        elif j == 2 and len(buf) != 0:
            f_lpyf = get_f_log_lpdf(target[buf], noise_ges[buf])
        elif j == 3 and len(buf) != 0:
            if it >= vm_start:
                iiv, iiv_f = get_iiv(len(i_target_vec[j]), N_virtual, fhat_prev, buf)
                f_lpyf = get_f_sin_lpdf(
                    target[buf], noise_ges[buf], (iiv, iiv_f, it, vm_val, vm_start, dlabel))
            else:
                f_lpyf = get_f_sin_lpdf(target[buf], noise_ges[buf])
        else:
            jopt = -1

        if jopt != -1:
            f_lpyf_vec.append(f_lpyf)
        else:
            f_lpyf_vec.append([])

        jopt_vec.append(jopt)

    def lpyf_ges_lpdf(f):
        res = 0

        for j in range(4):
            if jopt_vec[j] != -1:
                res = res + f_lpyf_vec[j](f[i_target_vec[j]])
        return res

    return lpyf_ges_lpdf

#get indices for virtual measurements
def get_iiv(lt, N_virtual, fhat_prev, buf):
    iiv = torch.tensor(range(lt-N_virtual, lt))
    if len(iiv) > 0:
        iiv_f = fhat_prev[buf][iiv].detach()
    else:
        iiv_f = []
    return iiv, iiv_f


#log pdf of squared nonlinearity
def chi2_log_pdf(x2vec, f, sig, iv):
    if sig.numel() == 1:
        sig = sig*torch.ones(x2vec.shape[0])

    xvec = torch.sqrt(x2vec)
    # omit nan values (dropout)
    inan = torch.where(~torch.isnan(xvec))[0]
    xvec = xvec[inan]

    # do not discard f<0 but penalize instead, to obtain gradient
    mu = torch.sign(f)*torch.sqrt(torch.abs(f))
    sig = torch.sqrt(sig)

    nibuf = torch.where(f < 0)
    xvec, sig2 = adjust_noise_for_vm(sig, xvec, iv[0], iv[1], iv[2], iv[3], iv[4], iv[5])
    xvec = torch.maximum(xvec, torch.tensor(1e-5))
    qbuf1 = torch.distributions.normal.Normal(mu, sig2)

    # qpdf = torch.exp(qbuf1.log_prob(xvec))/(2*xvec) + torch.exp(qbuf1.log_prob(-xvec))/(2*xvec)
    qpdf = torch.exp(qbuf1.log_prob(xvec))/(2*xvec) #approximate chi2 pdf
    lqpdf = torch.log(qpdf)    
    return torch.sum(lqpdf)

#log pdf of log nonlinearity
def log_log_pdf(x2vec, f, sig, iv=0):
    xvec = torch.exp(x2vec)
    mu = torch.exp(f)

    qbuf = torch.distributions.normal.Normal(mu, sig)
    # qpdf = torch.exp(qbuf.log_prob(xvec))*xvec
    lqpdf = qbuf.log_prob(xvec) + x2vec

    return torch.sum(lqpdf)

#log pdf of sin nonlinearity
def sin_log_pdf(x2vec, f, sig, iv=0):
    x2vec, _, _, _ = arcsin_check(x2vec)
    xvec = torch.arcsin(x2vec)

    f2, ibuf_g, ibuf_l, ibuf = arcsin_check(f)
    fdiff = f - f2
    mu = torch.arcsin(f2) + fdiff

    xvec, sig2 = adjust_noise_for_vm(sig, xvec,  iv[0], iv[1], iv[2], iv[3], iv[4], iv[5])

    qbuf = torch.distributions.normal.Normal(mu, sig2)
    qpdf = torch.exp(qbuf.log_prob(xvec))#/torch.sqrt(1-x2vec**2 + 1e-4)

    buf = torch.ones(qpdf.shape[0])
    buf[ibuf] = 0
    buf = buf/torch.sqrt(torch.abs(1-x2vec**2) + 1e-4)
    qpdf = qpdf*buf

    lqpdf = torch.log(qpdf)
    return torch.sum(lqpdf)

#check for valid arguments in arcsin
def arcsin_check(x2vec):
    x2vec = torch.maximum(torch.tensor(-1.), x2vec)
    x2vec = torch.minimum(torch.tensor(1.), x2vec)

    i_greater = torch.where(x2vec > 1)[0]
    i_lower = torch.where(x2vec < -1)[0]
    ibuf = torch.cat((i_greater, i_lower))
    return x2vec, i_greater, i_lower, ibuf


#adjust noise and xvec to enforce virtual measurements
def adjust_noise_for_vm(sig, xvec, iiv, iiv_f, it, vm_val, vm_start, dlabel):
    op = [1e-2, 30, 30] #adjustment parameters
    
    if len(iiv) > 0 and it >= vm_start:  # virtual measurements
        noise = torch.ones(xvec.shape)
        noise[iiv] = torch.tensor(0.)
        noise = noise*sig
        buf = torch.zeros(xvec.shape)

        #slowly decay noise towards small value
        buf[iiv] = sig[iiv]*torch.exp(-torch.tensor(it/op[2])) + 1e-4
        sig2 = noise + buf
        
        # slowly let value of vm go towards the correct value, otherwise Laplace does not converge
        off_buf = torch.tensor(it/1.5)
        off = op[0]*torch.exp(-off_buf/op[1])  
        xvec[iiv] = xvec[iiv] + off
    else:
        sig2 = sig

    return xvec, sig2
