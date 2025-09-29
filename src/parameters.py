import numpy as np
import src.constants as c

n = 4
h0 = np.zeros((4,))
m0 = c.rho * c.A * h0

dt = 0.1
t0 = 0.0
tf = 20.0 * 60.0
nt = int((tf - t0)/dt) + 1

# Controller parameters
ubar = np.array([300.0, 300.0])
hbar = np.array([15.0, 25.0, 15.0,20.0])

umin = np.array([0.0, 0.0])
umax = np.array([500.0, 500.0])

KP = np.zeros((2, n)); KP[0,0]=2.0; KP[1,1]=2.0
KI = np.zeros((2, n)); KI[0,0]=0.05; KI[1,1]=0.05
KD = np.zeros((2, n)); KD[0,0]=1.0;  KD[1,1]=1.0

ubar_segments = [
    (0, [100.0, 300.0]),
    (400.0, [400.0, 600.0]),
    (800.0, [250.0, 400.0]),
]

### Disturbance and measurements
d_determ = np.full((2,), 100)

sig_v = 1.0

mu_d = np.array([100.0, 100.0])
sig_d = np.array([50.0, 50.0])
t_d = 10

sig_sde = 50.0