import matplotlib.pyplot as plt
import torch

#simple plot of means and learning curves 
def plot_results(ds):
    fig0, ax0 = plt.subplots(1,2,figsize=(16,8))
    fig1, ax1 = plt.subplots(1,2,figsize=(16,8))
    
    #plot of original outputs
    llist = ['true model']*ds.MT_dim
    plot_loop(ax0[0], ds.train_x, ds.train_y, 'scatter', cmode=0)
    plot_loop(ax0[0], ds.test_x, ds.test_y, cstr=':', cmode=1)
    for jk in ds.kernels:
        plot_loop(ax0[0], ds.test_x, ds.pmean_ges[jk], cstr=get_cstr(jk), cmode = 0)
        llist += get_label(jk)*ds.MT_dim    
        ax0[1].semilogy(abs(-ds.mll_ges[jk,:ds.imll[jk]]),get_cstr(jk)+'k')
    ax0[0].axhline(0,ds.xmin,ds.xmax,linestyle=':',color='gray')
    ax0[0].legend(llist)
    
    
    #plot of transformed outputs
    llist2 = ['true model']*ds.MT_dim_trans
    plot_loop(ax1[0], ds.train_x_trans, ds.train_y_trans, 'scatter',cmode=1)
    plot_loop(ax1[0], ds.test_x, ds.test_y_trans, cstr=':',cmode=1)
    for jk in ds.kernels[1:]:        
        plot_loop(ax1[0], ds.test_x, ds.pmean_trans_ges[jk], cstr=get_cstr(jk), cmode = 1)
        llist2 += get_label(jk)*ds.MT_dim_trans   
        ax1[1].semilogy(abs(-ds.mll_ges[jk,:ds.imll[jk]]),get_cstr(jk)+'k')
    if ds.dlabel in ['HO', 'dHO', 'logsin']:
        draw_C_curve(ax1[0],ds)
    llist2 += ['constraint']
    ax1[0].axhline(0,ds.xmin,ds.xmax,linestyle=':',color='gray')
    ax1[0].legend(llist2)
    
        
    fig0.savefig('results/plots/pmean.pdf', format='pdf')
    fig1.savefig('results/plots/pmean_trans.pdf', format='pdf')
        
    #ax0[1]

def get_label(jk):
    labels = [['MT_GP'],['constrained GP'],['MT_GP_trans']]
    return labels[jk]
      
def get_cstr(jk):
    cstr = ['--','-','--']
    return cstr[jk]
        
        
def plot_loop(ax,X,Yges,pmode='',cstr = '',cmode = 0): #maybe make property of ds?
    #define colors
    if cmode == 0:
        lcc = ['b', 'r', 'g', 'c', 'k', 'm']
    if cmode == 1:
        lcc = ['b', 'g', 'r', 'c', 'k', 'm']
        
    #plot curves
    for j in range(Yges.shape[1]):
        cc = lcc[j%6]            
        if pmode == 'scatter':
            ax.scatter(X,Yges[:,j],color = cc)            
        else:
            ax.plot(X,Yges[:,j],cstr+cc)
            
def draw_C_curve(ax,ds):    
    if ds.constant_C == 0:
        Cplot = ds.get_C()(ds.test_x)
        ax.plot(ds.test_x,2*Cplot,linestyle='--',color = 'gray')
    else:
        ax.axhline(2*ds.C,ds.xmin,ds.xmax,linestyle='--',color='gray')            
            
######
######

#plot for unconstrained and constrained GP, together with confidence intervals
def conf_plot(ds, conf_yn = 1):
    Nzc = ds.N_virtual
    
    if ds.dlabel == 'HO' or ds.dlabel == 'dHO' or ds.dlabel == 'ff':
        fig, axs = plt.subplots(2,2,figsize = (14,12))
    elif ds.dlabel =='dp' or ds.dlabel == 'mesh':        
        fig, axs = plt.subplots(2,2,figsize = (10,14))
    else:
        fig, axs = plt.subplots(2,2,figsize = (14,14))
    
    xlabel = 't'
    if ds.dlabel == 'HO' or ds.dlabel == 'dHO':
        ovec = [0,1]
        covec = ['m','c']
        tvec = [1,3]
        ctvec = ['g', 'b']
        mvec = ['o','X']
        l0vec = [r'$z_{\rm aux}$',r'$v_{\rm aux}$']
        ltvec = ['$z^2$','$v^2$','$2E$']
        lbvec = ['$z$','$v$']
    elif ds.dlabel == 'ff':
        ovec = [0,1]
        covec = ['m','c']
        tvec = [0,2]
        ctvec = ['g', 'b']
        mvec = ['o','X']
        l0vec = [r'$z_0$',r'$v_{\rm aux}$']
        ltvec = ['$z^2$','$v^2$','$2E$']
        lbvec = ['$z$','$v$']
    elif ds.dlabel == 'dp':
        ovec = range(8)
        covec = ['c']*4 + ['m']*4
        tvec = [0,1,2,3,5,7,9,11]
        ctvec = ['b']*4 + ['r']*4
        mvec = ['o']*8
        l0vec = ['p']*4 + ['v']*4
        ltvec = ['p']*4 + ['v']*4 + ['2E']
        lbvec = ['p']*4 + ['v']*4
    elif ds.dlabel == 'mesh':
        ovec = range(8)
        tvec= range(10)
        covec = ['b']*8
        ctvec = ['g']*10
        mvec = ['o']*10
        l0vec = ['x']*8
        ltvec = ['aux']*9
        lbvec = ['x']*8
        xlabel = r'$\alpha$'
    elif ds.dlabel == 'logsin':
        ovec = [0,1]
        covec = ['c', 'm']
        tvec = [0,2]
        ctvec = ['g','b']
        mvec = ['o','x']
        #lvec = ['$f_1$','$f_2$']
        l0vec = [r'$f_1^0$',r'$f_2^0$']
        ltvec = ['log($f_1$)','sin($f_2$)', '$2C$']
        lbvec = ['$f_1}$','$f_2$']
        xlabel = 'x'
        
    else:
        ovec = range(2)
        tvec= range(3)
        covec = ['m','c']*3
        ctvec = ['b','r','g']*3
        mvec = ['o']*10
        l0vec = ['x']*10
    
    
    #plot of original outputs (unconstrained GP)
    for i, c, m in zip(ovec,covec,mvec): #vector of indices for original curves
        axs[0,0].scatter(ds.train_x,ds.train_y[:,i],color=c, marker = m, zorder=4, label='_nolegend_')
        axs[0,0].plot(ds.test_x,ds.pmean_ges[0,:,i],'--'+c)
        axs[0,0].plot(ds.test_x,ds.test_y[:,i],':k',zorder=-1, label='_nolegend_')
        if conf_yn == 1:    
            axs[0,0].fill_between(ds.test_x,ds.lower_ges[0,:,i],ds.upper_ges[0,:,i],alpha=0.2,color=c)
    #axs[0,0].legend(['$z_0$','$v_0$'])
    axs[0,0].legend(l0vec)
    axs[0,0].set(xlabel=xlabel)
    #axs[0,0].set(ylim=[-1.8,2.0])
    #axs[0,0].set(ylim=[-1.3,1.7])
    #axs[0,0].set(ylim=[-3.5,1.3])
    #axs[0,0].set(ylim=[-4.2,2.5])
    
    
    #plot of transformed outputs (constrained GP)
    for i, c, co, m in zip(tvec,ctvec,covec,mvec):
        if Nzc > 0:
            axs[0,1].scatter(ds.train_x_trans[:-Nzc],ds.train_y_trans[:-Nzc,i],color=c, marker=m, zorder=4, label='_nolegend_')
            axs[0,1].scatter(ds.train_x_trans[-Nzc:],ds.train_y_trans[-Nzc:,i],color=co, marker='s', zorder=5, label='_nolegend_')
        else:
            axs[0,1].scatter(ds.train_x_trans[:],ds.train_y_trans[:,i],color=c, marker=m, zorder=4, label='_nolegend_')
        axs[0,1].plot(ds.test_x,ds.pmean_trans_ges[1,:,i],'-'+c)
        if conf_yn == 1:    
            axs[0,1].fill_between(ds.test_x,ds.lower_trans_ges[1,:,i],ds.upper_trans_ges[1,:,i],alpha=0.2,color=c)
            
        axs[0,1].plot(ds.test_x,ds.test_y_trans[:,i],':k',zorder=-1, label='_nolegend_')
    
    if ds.dlabel in ['HO', 'dHO', 'logsin']:
        draw_C_curve(axs[0,1],ds)
    axs[0,1].legend(ltvec)
    axs[0,1].set(xlabel=xlabel)
    #axs[0,1].set(ylim=[-0.25,2.25])
    #axs[0,1].set(ylim=[-0.25,1.9])


    #plot of backtransformed outputs (constrained GP)    
    for i, c, m in zip(ovec,ctvec,mvec):
        axs[1,1].scatter(ds.train_x,ds.train_y[:,i],color=c,marker=m, zorder=4, label='_nolegend_')
        axs[1,1].plot(ds.test_x,ds.pmean_ges[1,:,i],'-'+c)
        if conf_yn == 1:    
            axs[1,1].fill_between(ds.test_x,ds.lower_ges[1,:,i],ds.upper_ges[1,:,i],alpha=0.2,color=c)    
        axs[1,1].plot(ds.test_x,ds.test_y[:,i],':k',zorder=-1, label='_nolegend_')
        
    #axs[1,1].legend(['$z$','$v$'])
    axs[1,1].legend(lbvec)
    axs[1,1].set(xlabel=xlabel)
    #axs[1,1].set(ylim=[-1.8,2.0])
    #axs[0,0].set(ylim=[-1.3,1.7])
    #axs[0,0].set(ylim=[-3.5,1.3])
    #axs[0,0].set(ylim=[-4.2,2.5])

    
    #plot of original outputs - both constraind and unconstrained GP
    for i, co, ct, m in zip(ovec,covec,ctvec, mvec):
        axs[1,0].scatter(ds.train_x,ds.train_y[:,i],color=ct,marker=m, zorder=4, label='_nolegend_')
        axs[1,0].plot(ds.test_x,ds.pmean_ges[0,:,i],'--'+co)
        axs[1,0].plot(ds.test_x,ds.pmean_ges[1,:,i],'-'+ct)
        if conf_yn == 1:    
            axs[1,0].fill_between(ds.test_x,ds.lower_ges[1,:,i],ds.upper_ges[1,:,i],alpha=0.2,color=ct)    
            axs[1,0].fill_between(ds.test_x,ds.lower_ges[0,:,i],ds.upper_ges[0,:,i],alpha=0.2,color=co)
        axs[1,0].plot(ds.test_x,ds.test_y[:,i],':k',zorder=-1, label='_nolegend_')
    lbuf = [None]*(len(l0vec)+len(ltvec)-1)
    lbuf[::2] = l0vec
    lbuf[1::2] = lbvec
    axs[1,0].legend(lbuf)
    axs[1,0].set(xlabel=xlabel)
    #axs[1,0].set(ylim=[-1.5,2])

    if conf_yn == 1:
        plt.savefig('results/plots/'+ds.dlabel+'_conf_illustration.pdf', format='pdf')
    else:
        plt.savefig('results/plots/'+ds.dlabel +'_illustration.pdf', format='pdf')

