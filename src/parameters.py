import numpy as np
import src.constants as c
import optimize_controller_inputs as o_c
n = 4
h0 = np.zeros((4,))

dt = 10
t0 = 0.0
tf = 20.0 * 60.0
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
d_determ = np.full((2,), 100)

sig_v = 1.0

mu_d = np.array([100.0, 100.0])
sig_d = np.array([10.0, 10.0])
t_d = 100

sig_sde = 30

Fbar = np.ones(nt)*10

#Disturbance models:
#Determnistic part
aF3= aF4 = 0.003
F3bar,F4bar = 60, 60
#F3 = lambda F3,t: aF3*(F3bar-F3) 
#F4 = lambda F4,t: aF4*(F4bar-F4) 
F3 = lambda F3,t: 0
F4 = lambda F4,t: 0 

#Stochastic part
sig_sdeF3 = 0.005
sig_sdeF4 = 0.006
#dwF3 = lambda F3,t: sig_sdeF3*F3
#dwF4 = lambda F4,t: sig_sdeF4*F4
dwF3 = lambda F3,t: sig_sde
dwF4 = lambda F4,t: sig_sde

sig_sde = 5
#aF3, aF4 = 0.005, 0.005   # small damping constants
aF3 = aF4 = 0.01
F3bar = np.concatenate([np.full(nt//2, 50), np.full(nt - nt//2, 70)])
F4bar = np.concatenate([np.full(nt//2, 50), np.full(nt - nt//2, 70)])
F3 = lambda F3,t: aF3*(F3bar[t] - F3)
F4 = lambda F4,t: aF4*(F4bar[t] - F4)
dwF3 = lambda F3,t: sig_sde
dwF4 = lambda F4,t: sig_sde

D_state = np.array([F3,F4])
S_state = np.array([dwF3,dwF4])
F0 = np.array([50,50])
#Optimization state parameters
opt_dt = 10
nt_opt = int((tf - t0)/opt_dt) + 1
F3bar_opt = np.concatenate([np.full(nt_opt//2, 50), np.full(nt_opt - nt_opt//2, 70)])
F4bar_opt = np.concatenate([np.full(nt_opt//2, 50), np.full(nt_opt - nt_opt//2, 70)])
F3_opt = lambda F3,t: aF3*(F3bar_opt[t] - F3)
F4_opt = lambda F4,t: aF4*(F4bar_opt[t] - F4)
D_state_opt = np.array([F3_opt,F4_opt])
#F0 = np.zeros(2)

#1/2 * (F(t)-self.Fbar[t]/F_max-F_min)**2 - tau * np.log(F(t)-F_min/F_max-F_min)*np.log(F_max-F(t)/F_max-F_min