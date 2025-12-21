import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import src.parameters as p
from src.models import StochasticBrownian, Deterministic, StochasticPiecewise
from src.plotting import plot_hist
from optimize_controller_inputs import optimize_pid, run_expected, optimize_pid_expected
import matplotlib.pyplot as plt
from pathlib import Path
FIGDIR = Path("C:/Users/magnu/Desktop/Niende_semester/02619/Exam_project/02619-MPC-project/figs")
zbar = (np.array([20,20]),np.array([5,5]),400)
delta = 0.3
def loss(e,delta=0.3):
    ae = np.abs(e)
    huber = np.where(ae <= delta, 0.5*ae**2, delta*(ae - 0.5*delta))
    loss_f = huber.sum()
    return loss_f

Y_transform = lambda F: np.log(np.maximum(F, 1e-12))
F_transform = lambda Y: np.exp(Y)

def opt_env(seed):
    return StochasticBrownian(p.opt_dt, zbar, p.ubar, p.sig_v,
                              p.D_state_opt, p.S_state, p.F0,Y_transform,F_transform, seed=seed)
#KP,KI,KD,LOSS = optimize_pid(model_factory=opt_env,loss_fn=loss, maxiter=10)   
#KP,KI,KD,LOSS = run_expected(opt_env,loss,maxiter=10, runs=10)
KP, KI, KD, LOSS = optimize_pid_expected(opt_env,loss,maxiter=10,n_realizations=10)
#KP = np.diag([9.62131974, 9.53210738])
#KI = np.diag([0.37880018 ,0.62824258])
#KD = np.diag([6.99353109,5.20565595])

#Best so far:
#KP = np.diag([9.77035178, 9.53988767])
#KI = np.diag([0.23372376 ,0.49258047])
#KD = np.diag([6.41633324,3.91678005])

#Maybe
#KP = np.diag([14.50993611, 14.2086796])
#KI = np.diag([0.70606334 ,0.87047718])
#KD = np.diag([6.90143581,9.02315348])

print(f"Best found KP: {KP}, Best found KI: {KI}, Best found KD: {KD}")
#print(f"Best found controller param loss: {LOSS}" )

seed = 0
for ctrl_type in ['P', 'PI', 'PID']:
    #stoch_brown = StochasticBrownian(p.dt, p.zbar, p.ubar, p.sig_v,p.D_state,p.S_state,p.F0,Y_transform,F_transform)
    #stoch_brown.simulate(p.t0, p.tf, p.h0, ctrl_type=ctrl_type)
    #plot_hist(stoch_brown, f"Closed-loop {ctrl_type}-controller - Stochastic brownian", plot_disturbance=True, filename=f'p3_stoch_brown_{ctrl_type}',step=False)   
    determ = Deterministic(p.dt, p.zbar, p.ubar, p.d_determ)
    stoch_piece = StochasticPiecewise(p.dt, p.zbar, p.ubar, p.sig_v, p.mu_d, p.sig_d, p.t_d,seed=seed)
    stoch_brown = StochasticBrownian(p.dt, p.zbar, p.ubar, p.sig_v,p.D_state,p.S_state,p.F0,Y_transform,F_transform,seed=seed)
    stoch_brown.KP, stoch_brown.KI, stoch_brown.KD = KP,KI,KD  
    determ.KP, determ.KI, determ.KD = KP,KI,KD  
    stoch_piece.KP, stoch_piece.KI, stoch_piece.KD = KP,KI,KD  
    determ.simulate(p.t0, p.tf, p.h0, ctrl_type=ctrl_type)
    stoch_piece.simulate(p.t0, p.tf, p.h0, ctrl_type=ctrl_type)
    stoch_brown.simulate(p.t0, p.tf, p.h0, ctrl_type=ctrl_type)
    
    fig, axes = plt.subplots(3, 3, figsize=(14, 10), sharex='col')

    plot_hist(determ, f"Deterministic - {ctrl_type}",
            filename=None, targets=True,
            plot_disturbance=True,step=False,dots=True,
            grid=True, grid_axes=axes, col=0,legend=False)

    plot_hist(stoch_piece, f"Stochastic piecewise - {ctrl_type}",
            filename=None, targets=True,
            plot_disturbance=True,step=True,dots=True,
            grid=True, grid_axes=axes, col=1,legend=False)

    plot_hist(stoch_brown, f"Stochastic brownian - {ctrl_type}",
            filename=None, targets=True, plot_disturbance=True, step=False,dots=True,
            grid=True, grid_axes=axes, col=2,legend=True)

    #axes[2, 0].axis('off')
    #axes[2, 1].axis('off')

    plt.tight_layout()
    fig.savefig(FIGDIR / f'GOOD2_{ctrl_type}_grid.pdf', format='pdf')
    plt.show()