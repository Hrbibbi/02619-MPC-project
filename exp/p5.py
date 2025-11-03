import numpy as np
import matplotlib.pyplot as plt

from src.models import Deterministic, StochasticPiecewise, SDE
import src.parameters as p

tf = 500

determ = Deterministic(p.dt, p.zbar, p.ubar, p.d_determ)
determ.plot_lin(p.h0, p.ubar, p.d_determ, tf=tf, plot_title='Deterministic -- Continuous linearization')


noise = 5.0
sig_d = np.array([noise, noise])
stoch_piece = StochasticPiecewise(p.dt, p.zbar, p.ubar, p.sig_v, p.mu_d, sig_d, p.t_d)
plot_title = 'Stochastic piecewise -- Continuous linearization'
stoch_piece.plot_lin(p.h0, p.ubar, p.mu_d, tf=tf, plot_title=plot_title)