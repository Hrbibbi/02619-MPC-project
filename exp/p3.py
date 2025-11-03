import numpy as np

from src.models import Deterministic, StochasticPiecewise, SDE
from src.plotting import plot_hist
import src.parameters as p

for ctrl_type in ['P', 'PI', 'PID']:
    determ = Deterministic(p.dt, p.zbar, p.ubar, p.d_determ)
    stoch_piece = StochasticPiecewise(p.dt, p.zbar, p.ubar, p.sig_v, p.mu_d, p.sig_d, p.t_d)
    stoch_brown = SDE(p.dt, p.zbar, p.ubar, p.sig_v, p.sig_sde)

    determ.simulate(p.t0, p.tf, p.h0, ctrl_type=ctrl_type)
    stoch_piece.simulate(p.t0, p.tf, p.h0, ctrl_type=ctrl_type)
    stoch_brown.simulate(p.t0, p.tf, p.h0, ctrl_type=ctrl_type)

    plot_hist(determ, f"Closed-loop {ctrl_type}-controller - Deterministic", filename=f'p3_determ_{ctrl_type}')
    plot_hist(stoch_piece, f"Closed-loop {ctrl_type}-controller - Stochastic piecewise", filename=f'p3_stoch_piece_{ctrl_type}')
    plot_hist(stoch_brown, f"Closed-loop {ctrl_type}-controller - SDE", plot_disturbance=False, filename=f'p3_sde_{ctrl_type}')