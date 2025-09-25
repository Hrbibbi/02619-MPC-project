import scipy as sp
import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
#np.array([15,15,15,15])
class four_tank_D:
    def __init__(self,gamma,F0,a,A,rho,ubar,KP,KI,I,KD,umin,umax,hbar,d,dt=0.1,g=981):
        self.gamma = gamma
        self.u = F0
        self.d = d
        self.ubar = ubar
        self.a = a
        self.A = A
        self.rho = rho
        self.g = g
        self.n = len(self.A)
        self.sol = None
        self.dt = dt
        self.hist_t = []
        self.hist_m = []
        self.hist_u = []
        self.KP = KP
        self.I = I
        self.KI = KI
        self.KD = KD
        self.umin = umin
        self.umax = umax
        self.ybar = self.rho*self.A*hbar
        self.hbar = hbar
        self.t0 = 0
    
    def flows(self, h):
        #Return outflows qi
        h = np.maximum(h, 0.0)
        return self.a * np.sqrt(2.0 * self.g * h)
        
    def get_param(self,h):
        #Get q parameters for ODE
        q_in = np.zeros(self.n)
        q = np.zeros(self.n)
        q_in[0] = self.gamma[0]*self.u[0]
        q_in[1] = self.gamma[1]*self.u[1]
        q_in[2] = (1-self.gamma[1])*self.u[1]
        q_in[3] = (1-self.gamma[0])*self.u[0]
        q = self.a*np.sqrt(2*self.g*h)
        return q_in, q
    def get_disturbance(self, t):
        k = int((t - self.t0) // self.dt)
        k = 0 if k < 0 else min(k, self.d.shape[1] - 1)
        return self.d[:, k]
    
    def f(self,t,x):
        #ODE function 
        m = x
        h = self.get_heights(m)
        q_in, q = self.get_param(h)
        d = self.get_disturbance(t)
        return np.array([self.rho*q_in[0]+self.rho*q[2]-self.rho*q[0], self.rho*q_in[1]+self.rho*q[3]-self.rho*q[1], 
                         self.rho*q_in[2]-self.rho*q[2]+self.rho*d[0], self.rho*q_in[3]-self.rho*q[3]+self.rho*d[1]])
        
    def SISOPID(self,mk,mk_1,dt_k,flag="PID"):
        #PID controller
        h = self.get_heights(mk)
        h_old = self.get_heights(mk_1)
        e = self.hbar-h
        P = self.KP @ e
        if flag == "P":
            v = self.ubar+P
        elif flag == "PI":
            v = self.ubar+P+self.I
        elif flag == "PID":
            D = - self.KD @ ((h-h_old)/dt_k)
            v = self.ubar+P+self.I+D
        else:
            print("No valid controller chosen")
            return
        free = (v > self.umin) & (v < self.umax) 
        self.u = np.clip(v, self.umin, self.umax)  
        if flag in ("PI", "PID"):
            self.I += (self.KI @ e) * free * dt_k 
        
    def get_heights(self,m):
        return m/(self.rho*self.A)   
    
    def get_outputs(self,m):
        return m[0:2]/(self.rho*self.A[0:2])        
    
    def simulate(self,t0,te,x0,flag):
        #Simulation loop
        N = int(np.ceil((te-t0) / self.dt))
        t = t0
        m = np.asarray(x0, dtype=float).copy()
        nx = m.size
        u0 = np.atleast_1d(self.u).astype(float)
        nu = u0.size
        t_hist = np.empty(N + 1, dtype=float)
        m_hist = np.empty((N + 1, nx), dtype=float)
        u_hist = np.empty((N + 1, nu), dtype=float)

        t = float(t0)
        mk_1 = m.copy()
        t_hist[0] = t
        m_hist[0] = m
        u_hist[0] = u0
        
        k = 0
        while (k < N) and (t < te):
            
            dt_k = min(self.dt, te - t)
            self.SISOPID(m,mk_1,dt_k,flag)
            t_end = t + dt_k
            self.sol = solve_ivp(self.f,[t,t_end],m,t_eval=[t_end],rtol=1e-6,atol=1e-8)
            if not self.sol.success:
                raise RuntimeError(f"ode solver failed at step {k}:-> {self.sol.message}")
            mk_1 = m
            m = self.sol.y[:,-1]
            t = self.sol.t[-1]
            
            k += 1
            t_hist[k] = t
            m_hist[k] = m
            u_hist[k] = np.atleast_1d(self.u).astype(float)

        self.hist_t = t_hist[:k+1]
        self.hist_m = m_hist[:k+1, :]
        self.hist_u = u_hist[:k+1, :]
        return self.hist_t, self.hist_m, self.hist_u
