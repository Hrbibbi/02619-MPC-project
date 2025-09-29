import numpy as np

from src.models import Deterministic, StochasticPiecewise, StochasticBrownian
from src.plotting import plot_hist
import src.parameters as p

determ = Deterministic(p.dt, p.zbar, p.ubar_segments, p.d_determ)
stoch_piece = StochasticPiecewise(p.dt, p.zbar, p.ubar_segments, p.sig_v, p.mu_d, p.sig_d, p.t_d)
stoch_brown = StochasticBrownian(p.dt, p.zbar, p.ubar_segments, p.sig_v, p.sig_sde)

determ.simulate(p.t0, p.tf, p.h0, ctrl_type="")
stoch_piece.simulate(p.t0, p.tf, p.h0, ctrl_type="")
stoch_brown.simulate(p.t0, p.tf, p.h0, ctrl_type="")

plot_hist(determ, "Open-loop - Deterministic", filename='p2_determ')
plot_hist(stoch_piece, "Open-loop - Stochastic piecewise", filename='p2_stoch_piece')
plot_hist(stoch_brown, "Open-loop - Stochastic brownian", plot_disturbance=False, filename='p2_stoch_brown')