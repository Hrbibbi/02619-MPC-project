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

def plot_hist(model, title, plot_disturbance=True, filename=None):
    hist = model.hist
    zbar = model.zbar

    nrows = 3 if plot_disturbance else 2
    fig, ax = plt.subplots(nrows, 1, figsize=(9, 3*nrows), sharex=True)

    t0 = hist['t'][0]
    tf = hist['t'][-1]

    ax[0].step(hist['t'], hist['u'][:,0], where="post", label=r'$u_1$')
    ax[0].step(hist['t'], hist['u'][:,1], where="post", label=r'$u_2$')
    ax[0].set_ylabel("Inputs")

    ax[1].plot(hist['t'], hist['y'][:, 0], ".", color="lightskyblue", alpha=0.5, label=r'$y_1$')
    ax[1].plot(hist['t'], hist['y'][:, 1], ".", color="moccasin", alpha=0.5, label=r'$y_2$')
    line1, = ax[1].plot(hist['t'], hist['h'][:,0], label=r'$h_1$', lw=2)
    line2, = ax[1].plot(hist['t'], hist['h'][:,1], label=r'$h_2$', lw=2)
    ax[1].axhline(zbar[0], ls='--', color=line1.get_color(), label=r'$\overline{z}_1$')
    ax[1].axhline(zbar[1], ls='--', color=line2.get_color(), label=r'$\overline{z}_2$')
    ax[1].set_ylabel("Heights")

    if plot_disturbance:
        ax[2].step(hist['td'], hist['d'][:,0], where='post', label=r'$d_1$')
        ax[2].step(hist['td'], hist['d'][:,1], where='post', label=r'$d_2$')
        ax[2].set_ylabel("Disturbances")
    
    ax[-1].set_xlabel("time [s]")
    ax[-1].set_xlim(t0, tf)
    for i in range(nrows):
        ax[i].legend(loc="upper right", bbox_to_anchor=(1.15, 1.0))
        ax[i].grid(True, alpha=0.3)

    fig.suptitle(title)
    plt.tight_layout()
    if filename is not None:
        filepath = FIGDIR / Path(filename)
        plt.savefig(filepath.with_suffix('.pdf'), format='pdf')
    plt.show()