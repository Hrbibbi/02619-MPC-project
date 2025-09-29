import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

import src.constants as c
import src.parameters as p

def plot_hist(hist, title):
    fig, ax = plt.subplots(3, 1, figsize=(9, 9), sharex=True)

    ax[0].step(hist['t'], hist['u'][:,0], where="post", label="u1")
    ax[0].step(hist['t'], hist['u'][:,1], where="post", label="u2")
    ax[0].set_xlabel("time [s]")
    ax[0].set_ylabel("Inputs")
    ax[0].legend()
    ax[0].grid(True, alpha=0.3)

    ax[1].plot(hist['t'], hist['h'][:,0], label="h1")
    ax[1].plot(hist['t'], hist['h'][:,1], label="h2")
    ax[1].set_xlabel("time [s]")
    ax[1].set_ylabel("Heights")
    ax[1].legend()
    ax[1].grid(True, alpha=0.3)

    ax[2].step(hist['td'], hist['d'][:,0], where='post', label="d1")
    ax[2].step(hist['td'], hist['d'][:,1], where='post', label="d2")
    # ax[2].scatter(hist['td'], hist['d'][:,0])
    # ax[2].scatter(hist['td'], hist['d'][:,1])
    ax[2].set_xlabel("time [s]")
    ax[2].set_ylabel("Disturbances")
    ax[2].legend()
    ax[2].grid(True, alpha=0.3)

    fig.suptitle(title)
    plt.tight_layout()
    plt.show()