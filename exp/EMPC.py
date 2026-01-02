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

u_set = np.array([25,25])
stoch_brown.u = u_set
MPC = FourTank_MPC(stoch_brown, m_ss, determ.u, d_ss,R,I_C,O_C,static_kf=False,MPC_type="economic",use_kf=True,NL_sim=True)
T = 5
#h0 = np.concatenate((p.h0, Fbar))
h0 = np.concatenate((stoch_brown.mass_to_height(m_ss[:4]), Fbar))
#h0 = np.concatenate((stoch_brown.mass_to_height(m_ss[:4]), Fbar))
MPC.simulate(p.t0,p.tf,h0,T)
plot_hist(MPC, f"Closed-loop EMPC-controller - SDE", filename=f'EMPC_low_xi',step=False,NL=True,state_est=False,plot_disturbance=False,econ=True)