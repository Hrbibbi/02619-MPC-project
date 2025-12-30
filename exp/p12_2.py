import numpy as np
import matplotlib.pyplot as plt

from src.models import SDE
import src.parameters as p

tf = 3600
ubar_segments = [(10*(int(t)-60), [u,u]) for t,u in zip(np.cumsum([60,60,60,60,60,60]),[50,100,150,200,250,300])]
mu_log_OU = np.full((2,), 200.0)
print(ubar_segments)
# ubar_segments = [
#     (0, [100.0, 100.0]),
#     (500.0, [400.0, 400.0]),
#     (1000.0, [100.0, 100.0]),
#     (1500.0, [400.0, 400.0]),
# ]
stoch_brown = SDE(p.dt, p.zbar, ubar_segments, p.sig_v, mu_log_OU, p.sig_OU, p.coef_OU)
stoch_brown.extended_kalman_NMPC(0,tf,p.h0, plot=True, filename='figs/p12/cont-disc-EKF.pdf')
None