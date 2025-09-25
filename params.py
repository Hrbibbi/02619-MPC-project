import numpy as np
# ----------------- parameters -----------------
n   = 4
rho = 1.0
a   = np.array([1.2272, 1.2272, 1.2272, 1.2272])
A   = np.array([380.1327, 380.1327, 380.1327, 380.1327])
gamma = np.array([0.58, 0.72])
F0 = np.array([300.0, 300.0])

h0 = np.zeros(n)
x0 = rho * A * h0

dt  = 30.0
t0  = 0.0
te  = 20.0 * 60.0
nt  = int((te - t0)/dt) + 1

# controller params 
ubar = np.array([300.0, 300.0])
I    = np.zeros_like(ubar)
KP   = np.zeros((2, n));  KP[0,0]=2.0; KP[1,1]=2.0
KI   = np.zeros((2, n));  KI[0,0]=0.05; KI[1,1]=0.05
KD   = np.zeros((2, n));  KD[0,0]=1.0;  KD[1,1]=1.0
umin = np.array([0.0,   0.0])
umax = np.array([500.0, 500.0])
hbar = np.array([15.0, 25.0, 15.0,20.0])

# Constant noise for deterministic model
d_c = np.zeros((2, nt))
d_c[:, :] = 5  # simple step disturbance

# We need to generate nt random points centered around some value?
Q = np.array([[2,1],[1,10]])
d_p = np.random.multivariate_normal(np.array([0,0]),Q,nt).T

# noise covariances (as in your SDE/S models)
Q   = np.diag([50.0**2, 30.0**2])      # SDE diffusion (tanks 3–4)
Rvv = np.diag([1.0, 1.0, 0.0, 0.0])    # measurement noise on h1, h2


#Poisson process
exp_times = []
t_dummy = t0
#while t_dummy <= te:
#    pass
    


