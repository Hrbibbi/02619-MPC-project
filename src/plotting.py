import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

import src.constants as c
import src.parameters as p

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    'font.size': 14,
})

def plot_hist(model, title):
    hist = model.hist
    hbar = model.hbar
    fig, ax = plt.subplots(3, 1, figsize=(9, 9), sharex=True)

    ax[0].step(hist['t'], hist['u'][:,0], where="post", label=r'$u_1$')
    ax[0].step(hist['t'], hist['u'][:,1], where="post", label=r'$u_2$')
    ax[0].set_xlabel("time [s]")
    ax[0].set_ylabel("Inputs")
    ax[0].legend()
    ax[0].grid(True, alpha=0.3)

    ax[1].plot(hist['t'], hist['y'][:, 0], ".", color="lightskyblue", alpha=0.5, label=r'$y_1$')
    ax[1].plot(hist['t'], hist['y'][:, 1], ".", color="moccasin", alpha=0.5, label=r'$y_2$')
    line1, = ax[1].plot(hist['t'], hist['h'][:,0], label=r'$h_1$', lw=2)
    line2, = ax[1].plot(hist['t'], hist['h'][:,1], label=r'$h_2$', lw=2)
    ax[1].axhline(hbar[0], ls='--', color=line1.get_color(), label=r'$\overline{h}_1$')
    ax[1].axhline(hbar[1], ls='--', color=line2.get_color(), label=r'$\overline{h}_2$')
    ax[1].set_xlabel("time [s]")
    ax[1].set_ylabel("Heights")
    ax[1].legend()
    ax[1].grid(True, alpha=0.3)

    ax[2].step(hist['td'], hist['d'][:,0], where='post', label=r'$d_1$')
    ax[2].step(hist['td'], hist['d'][:,1], where='post', label=r'$d_2$')
    # ax[2].scatter(hist['td'], hist['d'][:,0])
    # ax[2].scatter(hist['td'], hist['d'][:,1])
    ax[2].set_xlabel("time [s]")
    ax[2].set_ylabel("Disturbances")
    ax[2].legend()
    ax[2].grid(True, alpha=0.3)

    fig.suptitle(title)
    plt.tight_layout()
    plt.show()