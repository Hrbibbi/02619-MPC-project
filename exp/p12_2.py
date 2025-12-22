import numpy as np
import matplotlib.pyplot as plt

from src.models import SDE
import src.parameters as p

tf = 2000
stoch_brown = SDE(p.dt, p.zbar, p.ubar, p.sig_v, p.mu_log_OU, p.sig_OU, p.coef_OU)
stoch_brown.extended_kalman_NMPC(0,tf,p.h0, filename='figs/p12/cont-disc-EKF.pdf')