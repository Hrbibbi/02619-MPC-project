import numpy as np

from src.models import Deterministic, StochasticPiecewise, SDE
import src.parameters as p

tf = 800

determ = Deterministic(p.dt, p.zbar, p.ubar, p.d_determ)
plot_title = 'Step response -- deterministic model'
determ.step_response(
    p.h0, p.ubar, p.d_determ, tf,
    normalized=True, plot_title=plot_title,
    filename=f'figs/p4/determ.pdf'
)

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
    plot_title = 'Step response -- stochastic piecewise model -- ' + name
    stoch_piece.step_response(
        p.h0, p.ubar, p.mu_d, tf, plot_title=plot_title,
        measurements=True, normalized=False,
        filename=f'figs/p4/stoch_piece_{name.replace(" ","_")}.pdf'
    )