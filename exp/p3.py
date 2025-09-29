import numpy as np

from src.models import Deterministic, StochasticPiecewise, StochasticBrownian
from src.plotting import plot_hist
import src.parameters as p

determ = Deterministic(p.dt, p.hbar, p.ubar, p.d_determ)
stoch_piece = StochasticPiecewise(p.dt, p.hbar, p.ubar, p.sig_v, p.mu_d, p.sig_d, p.t_d)

determ.simulate(p.t0, p.tf, p.m0, ctrl_type=p.ctrl_type)
stoch_piece.simulate(p.t0, p.tf, p.m0, ctrl_type=p.ctrl_type)

plot_hist(determ, f"Closed-loop {p.ctrl_type}-controller - Deterministic")
plot_hist(stoch_piece, f"Closed-loop {p.ctrl_type}-controller - Stochastic piecewise")