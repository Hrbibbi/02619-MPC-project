import numpy as np
import src.constants as c

n = 4
# h0 = np.zeros((4,))
h0 = np.full((4,), 5)

dt = 10.0
t0 = 0.0
tf = 70.0 * 60.0
nt = int((tf - t0)/dt) + 1

# Controller parameters
ubar = np.array([300.0, 300.0])
zbar = np.array([15.0, 25.0])

umin = np.array([0.0, 0.0])
umax = np.array([500.0, 500.0])

KP = np.zeros((2, 2)); KP[0,0]=2.0; KP[1,1]=2.0
KI = np.zeros((2, 2)); KI[0,0]=0.05; KI[1,1]=0.05
KD = np.zeros((2, 2)); KD[0,0]=1.0;  KD[1,1]=1.0

ubar_segments = [
    (0, [100.0, 300.0]),
    (400.0, [400.0, 600.0]),
    (800.0, [250.0, 400.0]),
]

### Disturbance and measurements
d_determ = np.full((2,), 100.0)

sig_v = 1.5

mu_d = d_determ
sig_d = np.array([10.0, 10.0])
t_d = 100

### SDE disturbance params
# Coefficients are for Y = log F
sig_OU = 1/5*np.full((2,), 0.05) # 0.01 to 0.05 reasonable
coef_OU = np.full((2,), 1/(5*dt))
# mu_log_OU = np.full((2,), 100.0) # desired mean in log domain
mu_log_OU1 = np.full((2,), 100.0)
mu_log_OU = [
    (0.0,  [100.0, 100.0]),
    (1000.0, [30.0,  30.0]),
]
