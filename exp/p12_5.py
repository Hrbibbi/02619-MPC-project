import numpy as np
import matplotlib.pyplot as plt

from src.models import SDE
import src.parameters as p

t0 = 0
tf = 1200
Nh = 6
r = [
    (300, [40.0, 40.0]),
    (600, [50.0, 30.0]),
    (900, [25.0, 45.0]),
    (1200, [60.0, 60.0]),
]
Wz = np.eye(2)
Wu = 0.01*np.eye(2)
Wdu = 0.01*np.eye(2)
stoch_brown = SDE(p.dt, p.zbar, p.ubar, p.sig_v, p.mu_log_OU1, p.sig_OU, p.coef_OU)
filename = r'figs/p12/boundNMPC_high_Wz.pdf'
stoch_brown.bound_constrained_NMPC(t0, tf, p.h0, Nh, r, Wz, Wu, Wdu, filename=filename)

Wz = np.eye(2)
Wu = 0.01*np.eye(2)
Wdu = 0.1*np.eye(2)
filename = r'figs/p12/boundNMPC_high_Wdu.pdf'
stoch_brown.bound_constrained_NMPC(t0, tf, p.h0, Nh, r, Wz, Wu, Wdu, filename=filename)

Wz = np.eye(2)
Wu = 0.04*np.eye(2)
Wdu = 0.01*np.eye(2)
filename = r'figs/p12/boundNMPC_high_Wu.pdf'
stoch_brown.bound_constrained_NMPC(t0, tf, p.h0, Nh, r, Wz, Wu, Wdu, filename=filename)