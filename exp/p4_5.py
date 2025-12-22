import numpy as np
import matplotlib.pyplot as plt

from src.models import Deterministic, StochasticPiecewise, SDE
import src.parameters as p

tf = 1000

# determ = Deterministic(p.dt, p.zbar, p.ubar, p.d_determ)
# _ = determ.lin_valid_region(
#     p.h0, p.ubar, p.d_determ, tf=tf, N=10, span=0.5, filename='figs/p4/valid_region_determ.pdf'
# )

# stoch_piece = StochasticPiecewise(p.dt, p.zbar, p.ubar, p.sig_v, p.mu_d, p.sig_d, p.t_d)
# _ = stoch_piece.lin_valid_region(
#     p.h0, p.ubar, p.d_determ, tf=tf, N=40, span=0.5, filename='figs/p4/valid_region_stoch_piece.pdf'
# )

stoch_brown = SDE(p.dt, p.zbar, p.ubar, p.sig_v, p.mu_log_OU1, p.sig_OU, p.coef_OU)
print(stoch_brown.mu_log_to_mu_OU(p.mu_log_OU1))
_ = stoch_brown.lin_valid_region(
    p.h0, p.ubar, p.mu_log_OU1, tf=tf, N=40, span=0.5, filename='figs/p4/valid_region_stoch_brown.pdf'
)