import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from src.models import Deterministic, StochasticPiecewise, StochasticBrownian
from src.plotting import plot_hist, add_ensemble_to_last
from src.MPC_model import FourTank_MPC
import src.parameters as p
import src.constants as c
def compute_ss(model,x0):
    def wrap_rhs(m):
        d = model.get_disturbance(0) 
        return model.get_rhs(0,m,d)
    sol = sp.optimize.root(wrap_rhs,x0)
    h_ss = sol.x/(c.rho*c.A)
    return sol.x, h_ss
determ = Deterministic(p.dt, p.zbar, p.ubar, p.d_determ)
determ.u = np.array([50,50])
Y_transform = lambda F: np.log(np.maximum(F, 1e-12))
F_transform = lambda Y: np.exp(Y)
stoch_brown = StochasticBrownian(p.dt, p.zbar, p.ubar, p.sig_v,p.D_state,p.S_state,p.F0,Y_transform,F_transform)
Fbar = np.array([p.F3bar[0],p.F4bar[0]])
d_ss = np.array([p.beta3(0),p.beta4(0)])

x0 = np.zeros(4)
m_ss, h_ss = compute_ss(determ, x0)
n = m_ss.size
sigma_v = 1e+0
R = (sigma_v**2) * np.eye(n)
#In the SDE case we have a larger state vector since disturbance
m_ss = np.concatenate((m_ss,Fbar))
h_ss = np.concatenate((h_ss,Fbar))  
#m_ss = m_ss
I_C = True
O_C = True
ez1 = 1   
ez2 = 1         
Wz = np.diag([1,1])
Wu = np.diag([0,0])
Wdu = np.diag([0.5,0.5])
Ws = np.array([np.diag([10.0,10.0]),np.diag([10.0,10.0])])
Wt = np.array([np.diag([10.0,10.0]),np.diag([10.0,10.0])])
u_set = np.array([25,25])
stoch_brown.u = u_set
MPC = FourTank_MPC(stoch_brown, m_ss, determ.u, d_ss,R,Wz,Wu,Wdu,Ws,Wt,I_C,O_C,static_kf=True,MPC_type="linear",use_kf=True,NL_sim=True)
T = 3
h0 = np.concatenate((p.h0, Fbar))
MPC.simulate(p.t0,p.tf,h0,T)
plot_hist(MPC, f"Closed-loop MPC-controller - SDE", filename=f'p9_stochpiece_MPC',step=False)


from src.models import FourTank

t       = MPC.hist['t']
m_true  = MPC.hist['m']         
h_true  = MPC.hist['h']          
xhat    = MPC.hist['xhat']       
y_meas  = MPC.hist['y']         
y_nl_meas = MPC.hist['y_nl']
m_ss    = MPC.m_ss.ravel()       
m_hat   = xhat + m_ss[:4]           
h_hat   = FourTank.mass_to_height(m_hat)
h_nl  = MPC.hist['h_nl']      # true nonlinear heights

n_tanks = 4

fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex=True)
axes = axes.ravel()

for k in range(n_tanks):
    ax = axes[k]
    ax.plot(t[:-1], h_true[:-1, k], label="Linear height", linewidth=2)
    #ax.plot(t, h_nl[:, k],   label="Nonlinear height")
    ax.plot(t[:-1], h_hat[:-1, k],  '--', lw=2, label="KF estimate")
    ax.plot(t[:-1], y_meas[:-1, k], 'o', markersize=3, alpha=0.1, label="Measurement")

    ax.set_title(f"Tank {k+1}")
    ax.set_ylabel("Height [cm]")
    ax.grid(True)

axes[-1].set_xlabel("Time [s]")
axes[-2].set_xlabel("Time [s]")

fig.suptitle("State estimation")

# leave room on the right for the legend
fig.subplots_adjust(right=0.8)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles, labels,
    loc="center left",
    bbox_to_anchor=(0.82, 0.5),
    frameon=False,
    prop={'size': 9},
)
plt.savefig("./figs/KF_lin_jump.pdf")
plt.show()



fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex=True)
axes = axes.ravel()

for k in range(n_tanks):
    ax = axes[k]
    #ax.plot(t, h_true[:, k], label="Linear height", linewidth=2)
    ax.plot(t[:-1], h_nl[:-1, k],   label="Nonlinear height")
    ax.plot(t[:-1], h_hat[:-1, k],  '--', lw=2, label="KF estimate")
    ax.plot(t[:-1], y_nl_meas[:-1, k], 'o', markersize=3, alpha=0.1, label="Measurement")

    ax.set_title(f"Tank {k+1}")
    ax.set_ylabel("Height [cm]")
    ax.grid(True)

axes[-1].set_xlabel("Time [s]")
axes[-2].set_xlabel("Time [s]")

fig.suptitle("State estimation")

# leave room on the right for the legend
fig.subplots_adjust(right=0.8)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles, labels,
    loc="center left",
    bbox_to_anchor=(0.82, 0.5),
    frameon=False,
    prop={'size': 9},
)
plt.savefig("./figs/KF_nonlin_jump.pdf")
plt.show()




fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex=True)
axes = axes.ravel()

for k in range(n_tanks):
    ax = axes[k]
    #ax.plot(t, h_true[:, k], label="Linear height", linewidth=2)
    ax.plot(t[:-1], h_true[:-1, k]-h_hat[:-1,k],   label="Abs. diff. linear")
    ax.plot(t[:-1], h_nl[:-1, k]-h_hat[:-1, k],   label="Abs. diff. nonlinear")

    ax.set_title(f"Tank {k+1}")
    ax.set_ylabel("Height [cm]")
    ax.grid(True)

axes[-1].set_xlabel("Time [s]")
axes[-2].set_xlabel("Time [s]")

fig.suptitle("State estimation diff.")

# leave room on the right for the legend
fig.subplots_adjust(right=0.8)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles, labels,
    loc="center left",
    bbox_to_anchor=(0.82, 0.5),
    frameon=False,
    prop={'size': 9},
)
plt.savefig("./figs/KF_all_difference_jump.pdf")
plt.show()


