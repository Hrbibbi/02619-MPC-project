import numpy as np
import matplotlib.pyplot as plt
from src.models import Deterministic, StochasticPiecewise, StochasticBrownian
from src.plotting import plot_hist, add_ensemble_to_last
import src.parameters as p
from pathlib import Path
FIGDIR = Path("C:/Users/magnu/Desktop/Niende_semester/02619/Exam_project/02619-MPC-project/figs")
Y_transform = lambda F: np.log(np.maximum(F, 1e-12))
F_transform = lambda Y: np.exp(Y)
nsim = 10
det_hists = []
stoch_hists = []
sde_hists = []
seed = 0
for i in range(nsim):
    determ = Deterministic(p.dt, p.zbar, p.ubar_segments, p.d_determ)
    stoch_piece = StochasticPiecewise(p.dt, p.zbar, p.ubar_segments, p.sig_v, p.mu_d, p.sig_d, p.t_d,seed=seed+i*10**7)
    #stoch_brown = StochasticBrownian(p.dt, p.zbar, p.ubar_segments, p.sig_v, p.sig_sde,p.F3,p.F4,p.Fbar)
    stoch_brown = StochasticBrownian(p.dt, p.zbar, p.ubar_segments, p.sig_v,p.D_state,p.S_state,p.F0,Y_transform,F_transform,seed=seed+i*10**7)

    determ.simulate(p.t0, p.tf, p.h0, ctrl_type="")
    stoch_piece.simulate(p.t0, p.tf, p.h0, ctrl_type="")
    stoch_brown.simulate(p.t0, p.tf, p.h0, ctrl_type="")
    det_hists.append(determ.hist)
    stoch_hists.append(stoch_piece.hist)
    sde_hists.append(stoch_brown.hist)

add_ensemble_to_last(sde_hists)
add_ensemble_to_last(stoch_hists)
add_ensemble_to_last(det_hists)
stoch_brown.hist = sde_hists[-1]
stoch_piece.hist = stoch_hists[-1]
determ.hist      = det_hists[-1]



#plot_hist(determ, "Open-loop - Deterministic", filename='p2_determ',targets=False)
#plot_hist(stoch_piece, "Open-loop - Stochastic piecewise", filename='p2_stoch_piece',targets=False)
#plot_hist(stoch_brown, "Open-loop - Stochastic brownian", plot_disturbance=True,step=False,targets=False, filename='p2_stoch_brown')


fig, axes = plt.subplots(3, 3, figsize=(14, 10), sharex='col')

plot_hist(determ, "Deterministic",
          filename=None, targets=False,
          plot_disturbance=True,step=False,dots=False,
          grid=True, grid_axes=axes, col=0,legend=False)

plot_hist(stoch_piece, "Stochastic piecewise",
          filename=None, targets=False,
          plot_disturbance=True,step=True,dots=False,
          grid=True, grid_axes=axes, col=1,legend=False)

plot_hist(stoch_brown, "Stochastic brownian",
          filename=None, targets=False, plot_disturbance=True, step=False,dots=False,
          grid=True, grid_axes=axes, col=2,legend=True)

#axes[2, 0].axis('off')
#axes[2, 1].axis('off')

plt.tight_layout()
#fig.savefig(FIGDIR / 'p2_grid.pdf', format='pdf')
plt.show()