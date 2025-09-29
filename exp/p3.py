import numpy as np

from src.models import Deterministic, StochasticPiecewise, StochasticBrownian
from src.plotting import plot_hist
import src.parameters as p

for ctrl_type in ['P', 'PI', 'PID']:
    determ = Deterministic(p.dt, p.hbar, p.ubar, p.d_determ)
    stoch_piece = StochasticPiecewise(p.dt, p.hbar, p.ubar, p.sig_v, p.mu_d, p.sig_d, p.t_d)
    stoch_brown = StochasticBrownian(p.dt, p.hbar, p.ubar, p.sig_v, p.sig_sde)

    determ.simulate(p.t0, p.tf, p.m0, ctrl_type=ctrl_type)
    stoch_piece.simulate(p.t0, p.tf, p.m0, ctrl_type=ctrl_type)
    stoch_brown.simulate(p.t0, p.tf, p.m0, ctrl_type=ctrl_type)

    plot_hist(determ, f"Closed-loop {ctrl_type}-controller - Deterministic", filename=f'p3_determ_{ctrl_type}')
    plot_hist(stoch_piece, f"Closed-loop {ctrl_type}-controller - Stochastic piecewise", filename=f'p3_stoch_piece_{ctrl_type}')
    plot_hist(stoch_brown, f"Closed-loop {ctrl_type}-controller - Stochastic brownian", plot_disturbance=False, filename=f'p3_stoch_brown_{ctrl_type}')