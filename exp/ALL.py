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
determ.u = np.array([100,100])
Y_transform = lambda F: np.log(np.maximum(F, 1e-12))
F_transform = lambda Y: np.exp(Y)
stoch_brown_L = StochasticBrownian(p.dt, p.zbar, p.ubar, p.sig_v,p.D_state,p.S_state,p.F0,Y_transform,F_transform)
stoch_brown_NL = StochasticBrownian(p.dt, p.zbar, p.ubar, p.sig_v,p.D_state,p.S_state,p.F0,Y_transform,F_transform)
stoch_brown_PID = StochasticBrownian(p.dt, p.zbar, p.ubar, p.sig_v,p.D_state,p.S_state,p.F0,Y_transform,F_transform)
Fbar = np.array([p.F3bar[0],p.F4bar[0]])
d_ss = np.array([p.beta3(0),p.beta4(0)])
x0 = np.zeros(4)
m_ss, h_ss = compute_ss(determ, x0)

n = m_ss.size
sigma_v = 1e+0/2
R = (sigma_v**2) * np.eye(n)
m_ss = np.concatenate((m_ss,Fbar))
h_ss = np.concatenate((h_ss,Fbar))  
#m_ss = m_ss
I_C = True
O_C = True
KP = np.diag([9.62131974, 9.53210738])
KI = np.diag([0.37880018 ,0.62824258])
KD = np.diag([6.99353109,5.20565595])


u_set = np.array([25,25])
stoch_brown_L.u = u_set
stoch_brown_NL.u = u_set
stoch_brown_PID.u = u_set
MPC_L = FourTank_MPC(stoch_brown_L, m_ss, determ.u, d_ss,R,I_C,O_C,static_kf=False,MPC_type="linear",use_kf=True,NL_sim=True)
T = 5
h0 = np.concatenate((stoch_brown_L.mass_to_height(m_ss[:4]), Fbar))+np.array([15,15,0,0,0,0])
stoch_brown_PID.KP, stoch_brown_PID.KI, stoch_brown_PID.KD = KP,KI,KD
stoch_brown_PID.simulate(p.t0, p.tf, h0[:4], ctrl_type="PID")
h0 = np.concatenate((stoch_brown_L.mass_to_height(m_ss[:4]), Fbar))+np.array([15,15,0,0,0,0])
MPC_L.simulate(p.t0,p.tf,h0,T)
p.EKF = True
MPC_NL = FourTank_MPC(stoch_brown_NL, m_ss, determ.u, d_ss,R,I_C,O_C,static_kf=False,MPC_type="nonlinear",use_kf=True,NL_sim=True)
h0 = np.concatenate((stoch_brown_L.mass_to_height(m_ss[:4]), Fbar))+np.array([15,15,0,0,0,0])
MPC_NL.simulate(p.t0,p.tf,h0,T)



def _get_hist_key(hist, base):
    """Return 'base_nl' if present, otherwise 'base'."""
    nl_key = f"{base}_nl"
    return nl_key if nl_key in hist else base

hist_pid = stoch_brown_PID.hist
hist_l   = MPC_L.hist
hist_nl  = MPC_NL.hist


def _get_y(hist):
    return hist["y_nl"] if "y_nl" in hist else hist["y"]

t      = hist_pid["t"]
y_pid  = _get_y(hist_pid)
y_l    = _get_y(hist_l)
y_nl   = _get_y(hist_nl)
h_pid  = hist_pid["h"]
h_l    = hist_l["h_nl"]
h_nl   = hist_nl["h_nl"]


u_pid  = hist_pid["u"]
u_l    = hist_l["u"]
u_nl   = hist_nl["u"]

zbar = hist_l.get("zbar", None)

fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex=True)
(ax11, ax12), (ax21, ax22) = axes

colors = {"PID": "C0", "LMPC": "C1", "NMPC": "C2"}

ax11.plot(t, y_pid[:, 0], label="PID",   color=colors["PID"],  linewidth=2)
ax11.plot(t, y_l[:, 0],   label="LMPC",  color=colors["LMPC"], linewidth=2)
ax11.plot(t, y_nl[:, 0],  label="NMPC",  color=colors["NMPC"], linewidth=2)

if zbar is not None:
    ax11.plot(t, zbar[:, 0], "k--", label="reference")

ax11.set_title("Tank 1 level")
ax11.set_ylabel("Level [cm]")
ax11.grid(True, alpha=0.3)

ax12.plot(t, y_pid[:, 1], label="PID",   color=colors["PID"],  linewidth=2)
ax12.plot(t, y_l[:, 1],   label="LMPC",  color=colors["LMPC"], linewidth=2)
ax12.plot(t, y_nl[:, 1],  label="NMPC",  color=colors["NMPC"], linewidth=2)
if zbar is not None:
    ax12.plot(t, zbar[:, 1], "k--", label="reference")

ax12.set_title("Tank 2 level")
ax12.set_ylabel("Level [cm]")
ax12.grid(True, alpha=0.3)

ax21.step(t, u_pid[:, 0], where="post", label="PID",   color=colors["PID"],  linewidth=2)
ax21.step(t, u_l[:, 0],   where="post", label="LMPC",  color=colors["LMPC"], linewidth=2)
ax21.step(t, u_nl[:, 0],  where="post", label="NMPC",  color=colors["NMPC"], linewidth=2)

ax21.set_title("Pump 1 flow")
ax21.set_ylabel(r"$u_1$ [cm$^3$/s]") 
ax21.set_xlabel("Time [s]")
ax21.grid(True, alpha=0.3)

ax22.step(t, u_pid[:, 1], where="post", label="PID",   color=colors["PID"],  linewidth=2)
ax22.step(t, u_l[:, 1],   where="post", label="LMPC",  color=colors["LMPC"], linewidth=2)
ax22.step(t, u_nl[:, 1],  where="post", label="NMPC",  color=colors["NMPC"], linewidth=2)

ax22.set_title("Pump 2 flow")
ax22.set_ylabel(r"$u_2$ [cm$^3$/s]")
ax22.set_xlabel("Time [s]")
ax22.grid(True, alpha=0.3)

handles, labels = ax11.get_legend_handles_labels()
fig.legend(handles, labels,
           loc="upper center", ncol=4,
           bbox_to_anchor=(0.5, 1.02),
           frameon=False)

fig.suptitle("PID vs LMPC vs NMPC – Levels and Flows", y=1.08)
fig.tight_layout(rect=[0, 0, 1, 0.97])

fig.savefig("comparison_2x2_levels_flows_far_from_SS.pdf")
plt.show()
