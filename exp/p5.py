import numpy as np
import matplotlib.pyplot as plt

from src.models import Deterministic, StochasticPiecewise, SDE
import src.parameters as p

tf = 1000

determ = Deterministic(p.dt, p.zbar, p.ubar, p.d_determ)
determ.plot_lin(
    p.h0, p.ubar, p.d_determ, tf=tf,
    plot_title='Continuous linearization -- deterministic model',
    filename='figs/p5/cont_lin_determ.pdf'
)

determ = Deterministic(p.dt, p.zbar, p.ubar, p.d_determ)
determ.step_response_lin(
    p.h0, p.ubar, p.d_determ, tf=tf,
    plot_title='Continuous linearization -- deterministic model',
    filename='figs/p5/cont_lin_determ.pdf',
    normalized=True
)


noise = 5.0
sig_d = np.array([noise, noise])
stoch_piece = StochasticPiecewise(p.dt, p.zbar, p.ubar, p.sig_v, p.mu_d, sig_d, p.t_d)
plot_title = 'Continuous linearization -- stochastic piecewise model'
stoch_piece.plot_lin(p.h0, p.ubar, p.mu_d, tf=tf, plot_title=plot_title,
                     filename='figs/p5/cont_lin_stoch_piece.pdf')


stoch_brown = SDE(p.dt, p.zbar, p.ubar, p.sig_v, p.mu_log_OU1, p.sig_OU, p.coef_OU)
plot_title = 'Continuous linearization -- stochastic SDE model'
stoch_brown.plot_lin(p.h0, p.ubar, p.mu_log_OU1, tf=tf, plot_title=plot_title,
                     filename='figs/p5/cont_lin_stoch_brown.pdf')