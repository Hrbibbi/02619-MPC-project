import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from pathlib import Path

import src.constants as c
import src.parameters as p

FIGDIR = Path('./figs')

plt.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    'font.size': 14,
})

def mass_to_height_zbar(m):
        # Convert mass to height
        h = m / (c.rho * c.A[:2])
        return h

def plot_hist(model, title, plot_disturbance=True, filename=None,step=True,dots=True,targets = True, grid=False, grid_axes=None, col=None,legend=True,NL=False,state_est=False,econ=False):
    hist = model.hist
    #zbar = model.zbar

    nrows = 3 if plot_disturbance or econ else 2

    if grid:
        if grid_axes is None or col is None:
            raise ValueError("When grid=True, you must pass grid_axes and col.")
        ax_col = grid_axes[:, col]          
        ax = ax_col[:nrows]                
        fig = ax[0].figure
    else:
        fig, ax = plt.subplots(nrows, 1, figsize=(9, 3*nrows), sharex=True)

    t0 = hist['t'][0]
    tf = hist['t'][-1]

    ax[0].step(hist['t'], hist['u'][:,0], where="post", label=r'$u_1$')
    ax[0].step(hist['t'], hist['u'][:,1], where="post", label=r'$u_2$')
    ax[0].set_ylabel("Inputs")

    if dots:
        if state_est:
            ax[1].plot(hist['t'], hist['zhat'][:,0],"-",color="C0", label=r'$\hat{x}_1$', lw=2,alpha=1)
            ax[1].plot(hist['t'], hist['zhat'][:,1],"-",color="C1", label=r'$\hat{x}_2$', lw=2,alpha=1)
        else:
            if NL:
                ax[1].plot(hist['t'], hist['y_nl'][:, 0], ".", color="lightskyblue", alpha=0.5, label=r'$y_1$')
                ax[1].plot(hist['t'], hist['y_nl'][:, 1], ".", color="moccasin", alpha=0.5, label=r'$y_2$')
                ax[1].plot(hist['t'], hist['h_nl'][:,0],color="C0", label=r'$h_1$', lw=2)
                ax[1].plot(hist['t'], hist['h_nl'][:,1],color="C1", label=r'$h_2$', lw=2)
            else:
                ax[1].plot(hist['t'], hist['y'][:, 0], ".", color="lightskyblue", alpha=0.5, label=r'$y_1$')
                ax[1].plot(hist['t'], hist['y'][:, 1], ".", color="moccasin", alpha=0.5, label=r'$y_2$')
                ax[1].plot(hist['t'], hist['h'][:,0],color="C0", label=r'$h_1$', lw=2)
                ax[1].plot(hist['t'], hist['h'][:,1],color="C1", label=r'$h_2$', lw=2)
        
        
        #ax[1].plot(hist['t'], 19.98401978+hist['xhat'][:, 0]/ (c.rho * c.A[0]), "-", color="blue", alpha=0.5, label=r'$xhat_1$')
        #ax[1].plot(hist['t'], 22.35303769+hist['xhat'][:, 1]/ (c.rho * c.A[1]), "-", color="orange", alpha=0.5, label=r'$xhat_2$')
        
        
        ax[1].set_ylabel("Heights")
    else:
        # --- heights row ---
        
        ax[1].plot(hist['t'], hist['y'][:,0], label=r'$y_1$', lw=2,color="tab:blue")
        ax[1].plot(hist['t'], hist['y'][:,1], label=r'$y_2$', lw=2,color='tab:orange')
        ymu = hist['mean_path']['y']           
        ysd = np.sqrt(hist['var']['y'])       
        
        ax[1].plot(hist['t'], ymu[:, 0], '-',  lw=2, label=r'$\mu_t(y_1)$',color='black',alpha=0.4)
        ax[1].plot(hist['t'], ymu[:, 0] + 3*ysd[:, 0], '--', label=r'$\mu_t(y_1)\pm 3\sigma_t$',color='black',alpha=0.4)
        ax[1].plot(hist['t'], ymu[:, 0] - 3*ysd[:, 0], '--',color='black',alpha=0.4)

        ax[1].plot(hist['t'], ymu[:, 1], '-',  lw=2, label=r'$\mu_t(y_2)$',color='black',alpha=0.4)
        ax[1].plot(hist['t'], ymu[:, 1] + 3*ysd[:, 1], '--', label=r'$\mu_t(y_2)\pm3\sigma_t$',color='black',alpha=0.4)
        ax[1].plot(hist['t'], ymu[:, 1] - 3*ysd[:, 1], '--',color='black',alpha=0.4)
        ax[1].set_ylabel("Heights")
        
    if targets:
        ax[1].plot(hist['t'], hist['zbar'][:,0], '--', label=r'$\overline{z}_1$')
        ax[1].plot(hist['t'], hist['zbar'][:,1], '--', label=r'$\overline{z}_2$')
        ax[1].set_ylabel("Heights")
        
        
    if plot_disturbance:
        if step:
            ax[2].step(hist['td'], hist['d'][:,0], where='post', label=r'$d_1$')
            ax[2].step(hist['td'], hist['d'][:,1], where='post', label=r'$d_2$')
        else:
            ax[2].plot(hist['td'], hist['d'][:,0], label=r'$d_1$')
            ax[2].plot(hist['td'], hist['d'][:,1], label=r'$d_2$')
        ax[2].set_ylabel("Disturbances")
    elif econ:
        ax[2].step(hist['td'],hist['c'][:,0], where='post',label=r'$c_1$')
        ax[2].step(hist['td'],hist['c'][:,1], where='post',label=r'$c_2$')
        ax[2].set_ylabel("Pumping Cost")
    
    ax[-1].set_xlabel("time [s]")
    ax[-1].set_xlim(t0, tf)
    if legend:
        for i in range(nrows):
            if grid:
                if i == 1:
                    ax[i].legend(loc="upper right", bbox_to_anchor=(1.35, 1.0),frameon=False)
                    #1.59
                    ax[i].grid(True, alpha=0.3)
                else:
                    ax[i].legend(loc="upper right", bbox_to_anchor=(1.35, 1.0),frameon=False)
                    ax[i].grid(True, alpha=0.3)
                
            else:
                ax[i].legend(loc="upper right", bbox_to_anchor=(1.15, 1.0))
                ax[i].grid(True, alpha=0.3)

    for i in range(nrows):
        ax[i].grid(True, alpha=0.3)
               
    if grid:
        ax[0].set_title(title)
        return fig
    
    fig.suptitle(title)
    plt.tight_layout()
    if filename is not None:
        filepath = FIGDIR / Path(filename)
        plt.savefig(filepath.with_suffix('.pdf'), format='pdf')
    plt.show()
    
def add_ensemble_to_last(hists, keys=('y',)):
    """
    Compute mean/variance for selected keys across runs and stash in last hist.
    keys can be any subset of: 'y', 'h', 'u', 'd', 'zbar'
    """
    if len(hists) == 0:
        return

    t0 = hists[0]['t']
    if not all(np.array_equal(t0, H['t']) for H in hists):
        raise ValueError("Time vectors differ across runs; align/resample first.")

    if 'd' in keys and 'td' in hists[0]:
        td0 = hists[0]['td']
        if not all(('td' in H) and np.array_equal(td0, H['td']) for H in hists):
            raise ValueError("Disturbance time vectors differ across runs.")

    mean_path, var_path = {}, {}

    for k in keys:
        if all(k in H for H in hists):
            A = np.stack([H[k] for H in hists], axis=0)  
            mean_path[k] = A.mean(axis=0)
            var_path[k]  = A.var(axis=0, ddof=0)

    hists[-1]['mean_path'] = mean_path
    hists[-1]['var']       = var_path
    hists[-1]['nsim']      = len(hists)
