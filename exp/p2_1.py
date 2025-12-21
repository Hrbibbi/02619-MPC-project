import numpy as np
from src.models import StochasticBrownian
from src.plotting import plot_hist
import src.parameters as p
#Remember transforms
Y_transform = lambda F: np.log(np.maximum(F, 1e-12))
F_transform = lambda Y: np.exp(Y)
#stoch_brown = StochasticBrownian(p.dt, p.zbar, p.ubar_segments, p.sig_v,p.D_state,p.S_state,p.F0)
stoch_brown = StochasticBrownian(p.dt, p.zbar, p.ubar_segments, p.sig_v,p.D_state,p.S_state,p.F0,Y_transform,F_transform)
stoch_brown.simulate(p.t0, p.tf, p.h0, ctrl_type="")

plot_hist(stoch_brown, "Open-loop - Stochastic brownian", plot_disturbance=True, filename='p2_stoch_brown',step=False)