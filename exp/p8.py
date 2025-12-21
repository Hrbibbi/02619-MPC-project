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
stoch_piece = StochasticPiecewise(p.dt, p.zbar, p.ubar, p.sig_v, p.mu_d, p.sig_d, p.t_d)
stoch_piece.u = determ.u
d_ss = stoch_piece.mu_d
x0 = np.zeros(4)
m_ss, h_ss = compute_ss(determ, x0)
I_C = True
O_C = False
#R = (p.sig_v**2) * np.eye(4)
#For the deterministic model we have no measurement noise and constant disturbance
n = m_ss.size
Q = 1e-10 * np.eye(n)
sigma_v = 1e+0
R = (sigma_v**2) * np.eye(n)
Wz = np.diag([1,1])
Wu = np.diag([0,0])
Wdu = np.diag([0.5,0.5])
Ws = np.diag([1.0,1.0])
Wt = np.diag([1.0,1.0])
MPC = FourTank_MPC(stoch_piece, m_ss, stoch_piece.u, d_ss,R,Wz,Wu,Wdu,Ws,Wt,I_C,O_C,static_kf=True,MPC_type="",use_kf=True,log_NL=True)
T = 10
MPC.simulate(p.t0,p.tf,p.h0,T)
plot_hist(MPC, f"Closed-loop MPC-controller - Stoch piecewise", filename=f'p8_stochpiece_MPC')

states = MPC.hist['xhat']
t = MPC.hist['t']
m_true = MPC.hist['m']
m_ss = MPC.m_ss.ravel()
m_hat = states + m_ss
n = m_true.shape[1]
for k in range(states.shape[1]):
    plt.plot(t, m_true[:,k], label=f"True mass state {k}")
    plt.plot(t, m_hat[:, k], '--', label=f"KF estimate state {k}")
    plt.grid(True)
    plt.xlabel("Time")
    plt.ylabel("State value")
    plt.legend()
plt.show()



from src.models import FourTank


t       = MPC.hist['t']
m_true  = MPC.hist['m']         
h_true  = MPC.hist['h']          
xhat    = MPC.hist['xhat']       
y_meas  = MPC.hist['y']         

m_ss    = MPC.m_ss.ravel()       
m_hat   = xhat + m_ss[:4]           
h_hat   = FourTank.mass_to_height(m_hat)
h_nl  = MPC.hist['h_nl']      # true nonlinear heights

n_tanks = 4

fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex=True)
axes = axes.ravel()

for k in range(n_tanks):
    ax = axes[k]
    ax.plot(t, h_true[:, k], label="Linear height", linewidth=2)
    ax.plot(t, h_nl[:, k],   label="Nonlinear height")
    ax.plot(t, h_hat[:, k],  '--', lw=2, label="KF (linear) estimate")
    ax.plot(t, y_meas[:, k], 'o', markersize=3, alpha=0.1, label="Measurement")

    ax.set_title(f"Tank {k+1}")
    ax.set_ylabel("Height [m]")
    ax.grid(True)

axes[-1].set_xlabel("Time [s]")
axes[-2].set_xlabel("Time [s]")

fig.suptitle("State estimation")

fig.subplots_adjust(right=0.8)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles, labels,
    loc="center left",
    bbox_to_anchor=(0.82, 0.5),
    frameon=False,
    prop={'size': 9},
)

plt.show()
