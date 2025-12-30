import numpy as np
import matplotlib.pyplot as plt

from src.models import SDE
import src.parameters as p

tf = 2000
mu_log_OU = np.full((2,), 200.0)
ubar_segments = [
    (0, [100.0, 100.0]),
    (500.0, [400.0, 400.0]),
    (1000.0, [100.0, 100.0]),
    (1500.0, [400.0, 400.0]),
]
stoch_brown = SDE(p.dt, p.zbar, ubar_segments, p.sig_v, mu_log_OU, p.sig_OU, p.coef_OU)
stoch_brown.extended_kalman_NMPC(0,tf,p.h0, plot=True)

sigma_grid = np.linspace(0.5*p.sig_v, 2.0*p.sig_v, 50)
stoch_brown.pem_sweep_R_scalar(0,tf,p.h0, sigma_grid=sigma_grid, plot=True, filename='figs/p12/PEM_sig_v.pdf')
None