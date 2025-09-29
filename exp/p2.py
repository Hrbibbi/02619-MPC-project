import numpy as np

from src.models import Deterministic, StochasticPiecewise, StochasticBrownian
from src.plotting import plot_hist
import src.parameters as p

determ = Deterministic(p.dt, p.hbar, p.ubar_segments, p.d_determ)
stoch_piece = StochasticPiecewise(p.dt, p.hbar, p.ubar_segments, p.sig_v, p.mu_d, p.sig_d, p.t_d)
# stoch_brown = StochasticBrownian(p.dt, p.hbar, p.ubar)

determ.simulate(p.t0, p.tf, p.m0, ctrl_type="")
stoch_piece.simulate(p.t0, p.tf, p.m0, ctrl_type="")
# T_SDE, M_SDE, U_SDE = run_piecewise_open_loop(stoch_brown, segments, p.m0, p.dt)


plot_hist(determ, "Open-loop - Deterministic")
plot_hist(stoch_piece, "Open-loop - Stochastic piecewise")
# step_plot_inputs_outputs(T_SDE, U_SDE, H_SDE, "Open-loop - SDE")
