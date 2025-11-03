import numpy as np

from src.models import Deterministic, StochasticPiecewise, SDE
import src.parameters as p

tf = 800

determ = Deterministic(p.dt, p.zbar, p.ubar, p.d_determ)
for idx in range(2):
    plot_title = 'Deterministic -- step response' + f' -- $u_{idx+1}$'
    determ.step_response(idx, p.h0, p.ubar, p.d_determ, tf, normalized=True, plot_title=plot_title, filename=f'figs/p4/determ_u{idx+1}.pdf')

sig_d_base = 2.0
sig_v_base = 1.0
noise_levels = [
    (1, 1, 'low noise'),
    (2, 2, 'medium noise'),
    (5, 3, 'high noise')
]
for i, (f1, f2, name) in enumerate(noise_levels):
    sig_d = f1 * np.array([sig_d_base, sig_d_base])
    sig_v = f2 * sig_v_base
    stoch_piece = StochasticPiecewise(p.dt, p.zbar, p.ubar, sig_v, p.mu_d, sig_d, p.t_d)
    for idx in range(2):
        plot_title = 'Stochastic piecewise -- step response -- ' + name + f'-- $u_{idx+1}$'
        stoch_piece.step_response(idx, p.h0, p.ubar, p.mu_d, tf, plot_title=plot_title, measurements=True, filename=f'figs/p4/stoch_piece_u{idx+1}_{name.replace(" ","_")}.pdf')