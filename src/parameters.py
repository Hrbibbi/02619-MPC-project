import numpy as np
import src.constants as c
import optimize_controller_inputs as o_c
n = 4
h0 = np.zeros((4,))

dt = 10
t0 = 0.0
#tf = 20.0 * 60.0
tf = 60.0*60.0
nt = int((tf - t0)/dt) + 1




# Controller parameters
#ubar = np.array([100.0, 100.0])
ubar = np.array([0.0, 0.0])
#zbar = np.array([15.0, 25.0])
#zbar = [
#    (0, [20,15]),
#    (600.0, [25, 20])
#]
#THis is good
#zbar = (np.array([24,24]),np.array([2,2]),800)
#zbar = (np.array([30,30]),np.array([5,5]),800)
#zbar = (np.array([27+20,33+20]),np.array([8,8]),800)

#zbar = (np.array([60,60]),np.array([5,5]),800)
#zbar = (np.array([19.98,22.35]),np.array([0,0]),800)

#Economic
#zbar = (np.array([19.98+10,22.35+7]),np.array([5,5]),2500)
zbar = (np.array([60,60]),np.array([5,5]),2500)


#zbar = (np.array([27,27]),np.array([5,5]),1800)

umin = np.array([0.0, 0.0])
umax = np.array([500.0, 500.0])
#Useually i have used 30 here but for the final plots in the end i have used 30. and in economic i have used 50
du_lim = 50
dumin = np.array([-du_lim, -du_lim])
dumax = np.array([+du_lim, +du_lim])
#ocmax = 0.1
#ocmin = 0.1
ocmax = 1
#ocmin = 1
# for economic 
ocmin = 0

#Weight matrices for LMPC
Wz  = np.diag([10,10])                
Wu  = np.diag([0.01,0.01]) 
#Wu = np.diag([0,0])           
#Wdu = np.diag([1,1])  
Wdu = np.diag([1,1])              
Ws = np.array([np.diag([1.0,1.0]),np.diag([1.0,1.0])])
Wt = np.array([np.diag([1.0,1.0]),np.diag([1.0,1.0])])
#Ws = np.array([np.diag([100.0,100.0]),np.diag([100.0,100.0])])
#Wt = np.array([np.diag([100.0,100.0]),np.diag([100.0,100.0])])

#Cost for Economic MPC
#c = np.full((nt,2), 1)
mu_c = np.array([35,35])
sig_c = np.array([20,20])
t_c = 100
#0
cU = c.construct_cost_poisson(nt, dt, mu_c, sig_c, t_c, seed=0)

#xi_v = 1000*2  
xi_v = 100
xi = np.full((nt,2),xi_v)


#Extended kalman filter
EKF = False



#KP = np.zeros((2, 2)); KP[0,0]=2.0; KP[1,1]=2.0
#KI = np.zeros((2, 2)); KI[0,0]=0.05; KI[1,1]=0.05
#KD = np.zeros((2, 2)); KD[0,0]=1.0;  KD[1,1]=1.0

KP = np.zeros((2, 2)); KP[0,0]=1/3; KP[1,1]=1/3
KI = np.zeros((2, 2)); KI[0,0]=1/3; KI[1,1]=1/3
KD = np.zeros((2, 2)); KD[0,0]=1/3;  KD[1,1]=1/3

#KP = np.diag([10,10])
#KI = np.diag([0.31588595,0.47722896])
#KD = np.diag([5.40119979,6.5867388])

ubar_segments = [
    (0, [100.0, 300.0]),
    (400.0, [400.0, 600.0]),
    (800.0, [250.0, 400.0]),
    (1600.0, [100, 300]),
    (2800.0, [500, 400])
]

### Disturbance and measurements
d_determ = np.full((2,), 200)
#d_determ = np.concatenate([np.full(nt//2, 200), np.full(nt - nt//2, 250)])
#d_determ = np.concatenate()
#d_determ = np.vstack([d_determ,d_determ])

sig_v = 1.0/2

mu_d = np.array([200.0, 200.0])
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


#Feller condition 2*a*Fbar >= sigma^2 then F_t > 0
#sig_sde = 0.1*1/2
#sig_sde = 0.1*1/2
#This is what is used to produce open loop sims for SDE 
#sig_sde = 0.1*1/4
sig_sde = 0.1*0.1
aF3 = aF4 = 0.1
        
#THis is what we experiment with on KF and state estimation
#sig_sde = 0.001
#aF3 = aF4 = 0.05*0.1

#*1/10
F3bar = np.concatenate([np.full(150, 200), np.full(100, 230),np.full(111, 200)])
F4bar = np.concatenate([np.full(150, 200), np.full(100, 230),np.full(111, 200)])

F3bar = np.concatenate([np.full(160, 200), np.full(50, 250),np.full(151, 200)])
F4bar = np.concatenate([np.full(160, 200), np.full(50, 180),np.full(151, 200)])
sl = np.arange(int(180),int(180)+2)
F3bar[sl] = 260
F4bar[sl] = 170

F3bar = np.concatenate([np.full(nt//2, 200), np.full(nt - nt//2, 200)])
F4bar = np.concatenate([np.full(nt//2, 200), np.full(nt - nt//2, 200)])
#sl = np.arange(int(nt//2),int(nt//2)+2)
#F3bar[sl] = 300
#F4bar[sl] = 300
def piecewise_signal(lengths, values):
    assert len(lengths) == len(values)
    return np.concatenate([np.full(L, v) for L, v in zip(lengths, values)])
lengths = [60, 120, 80, 100, 101]
#lengths = [120, 80, 60, 100, 101] 
F3_vals = [200, 220, 190, 150, 200]
F4_vals = [200, 190, 220, 170, 200]

F3bar = piecewise_signal(lengths, F3_vals)
F4bar = piecewise_signal(lengths, F4_vals)

F3bar = np.concatenate([np.full(nt//2, 200), np.full(nt - nt//2, 200)])
F4bar = np.concatenate([np.full(nt//2, 200), np.full(nt - nt//2, 200)])

lengths = [60,60,60,60,60,61]
u_vals = [50,100,150,200,250,300]
u_prefixed = piecewise_signal(lengths, u_vals)
u_prefixed = np.vstack([u_prefixed, u_prefixed])

#F3 = lambda F3,t: aF3*(F3bar[t] - F3)
#F4 = lambda F4,t: aF4*(F4bar[t] - F4)
beta3 = lambda k: np.log(max(F3bar[k], 1e-12)) - (sig_sde**2)/(4.0*aF3)
beta4 = lambda k: np.log(max(F4bar[k], 1e-12)) - (sig_sde**2)/(4.0*aF4)
Y3 = lambda Y3,t: aF3*(beta3(t) - Y3)
Y4 = lambda Y4,t: aF4*(beta4(t) - Y4)

dwF3 = lambda F3,t: sig_sde
dwF4 = lambda F4,t: sig_sde

D_state = np.array([Y3,Y4])
S_state = np.array([dwF3,dwF4])
#F0 = np.array([20,20])*1/2
F0 = np.array([200,200])
#Optimization state parameters
opt_dt = 10
nt_opt = int((tf - t0)/opt_dt) + 1
F3bar_opt = np.concatenate([np.full(nt_opt//2,100), np.full(nt_opt - nt_opt//2, 120)])
F4bar_opt = np.concatenate([np.full(nt_opt//2, 100), np.full(nt_opt - nt_opt//2, 120)])
F3_opt = lambda F3,t: aF3*(F3bar_opt[t] - F3)
F4_opt = lambda F4,t: aF4*(F4bar_opt[t] - F4)
#D_state_opt = np.array([F3_opt,F4_opt])
print(beta3,beta4)
Y3_opt = lambda Y3,t: aF3*(beta3(t) - Y3)
Y4_opt = lambda Y4,t: aF4*(beta4(t) - Y4)
D_state_opt = np.array([Y3_opt,Y4_opt])
#F0 = np.zeros(2)