import numpy as np
import scipy as sp

import src.constants as c
import src.parameters as p

class FourTank:
    """Class modelling the modified four tank system"""
    def __init__(self, dt, zbar, ubar, seed=0):
        self.rng = np.random.default_rng(seed)
        
        if isinstance(ubar, list) and isinstance(ubar[0], tuple):
            # looks like [(t0, u0), (t1, u1), ...]
            self.ubar_type = 'piecewise'
            self.set_ubar_piecewise(ubar)
            self.u = np.array(ubar[0][1])
        else:
            self.ubar_type = 'constant'
            self.ubar = np.asarray(ubar, float).copy()
            self.u = self.ubar.copy()
            
        if isinstance(zbar,list) and isinstance(zbar[0], tuple):
            self.zbar_type = 'piecewise'
            self.set_zbar_piecewise(zbar)
        elif isinstance(zbar,tuple) and len(zbar) == 3:
            #if zbar is just a tuple its a possion process
            self.zbar_type = 'poisson'
            mu_z, sig_z,t_z = zbar
            self.pois_seed = np.random.default_rng(int(seed+1e7))
            self.set_zbar_poisson(mu_z,sig_z,t_z)
        else:
            self.zbar_type = 'constant'
            self.zbar = np.asarray(zbar,float).copy()
            
        
        self.n = 4
        self.dt = dt
        self.nu = 2
        self.I = np.zeros((2,))
        self.KP = p.KP.copy()
        self.KI = p.KI.copy()
        self.KD = p.KD.copy()
        self.hist = {}

       

    @staticmethod
    def mass_to_height(m):
        # Convert mass to height
        h = m / (c.rho * c.A)
        return h
    
    @staticmethod
    def height_to_mass(h):
        # Convert height to mass
        m = c.rho * c.A * h
        return m

    @staticmethod
    def get_output(m):
        # Get outputs h1 and h2
        out = m[:2] / (c.rho * c.A[:2])
        return out
    @staticmethod
    def linearize_fourtank(m_ss,u_ss,F=False,Fbar=None):
        h_ss = FourTank.mass_to_height(m_ss[:4])
        T = (c.A/c.a)*np.sqrt(2*h_ss/c.g)
        
        if F:
            sigma = np.array([p.sig_sdeF3,p.sig_sdeF4])
            a = np.array([p.aF3,p.aF4])
            if Fbar is None:
                Fbar = p.F0
                Ybar = np.array([p.beta3(0),p.beta4(0)])
            else:
                Ybar = Fbar
                Fbar = np.exp(Fbar)
            #p.F0
            #Ybar = np.array([p.beta3(0),p.beta4(0)])
            V = lambda F, Ybar : 0.5*F*sigma**2 - F*a*np.log(F)+F*a*Ybar
            VF3 = lambda F3 : 0.5*sigma[0]**2 - a[0] * np.log(F3)-a[0]+a[0]*Ybar[0]
            VF4 = lambda F4 : 0.5*sigma[1]**2 - a[1] * np.log(F4)-a[1]+a[1]*Ybar[1]
            VYbar3 = lambda F3: F3*a[0]
            VYbar4 = lambda F4 : F4*a[1]
            A = np.array([
                        [-1/T[0],   0.0,      1/T[2],   0.0, 0.0, 0.0],
                        [0.0,     -1/T[1],    0.0,      1/T[3], 0.0, 0.0],
                        [0.0,      0.0,     -1/T[2],    0.0, 1.0, 0.0],
                        [0.0,      0.0,      0.0,     -1/T[3], 0.0, 1.0],
                        [0.0,      0.0,      0.0,     0.0, VF3(Fbar[0]), 0.0],
                        [0.0,      0.0,      0.0,    0.0, 0.0, VF4(Fbar[1])]
                        ])
            #C = np.concatenate(np.diag(1.0/(c.rho*c.A)),np.zeros((2,FourTank.n+2)))
            C_heights = np.diag(1.0/(c.rho*c.A))        # 4x4 on the mass states
            C = np.hstack([C_heights, np.zeros((4, 2))])# 4x6 (no direct measurement of extra states)

            #Cz = C[0:2,:]
            Gb = np.array([[0.0,0.0],
                           [0.0,0.0],
                           [0.0,0.0],
                           [0.0,0.0],
                           [Fbar[0]*sigma[0],0.0],
                           [0.0,Fbar[1]*sigma[1]],
                           ])
            
            Gd = np.array([[0.0,0.0],
                           [0.0,0.0],
                           [0.0,0.0],
                           [0.0,0.0],
                           [VYbar3(Fbar[0]),0.0],
                           [0.0,VYbar4(Fbar[1])]
                           ])
            #This needs fixing.
            
            B = np.array([
                
                [c.rho*c.gamma[0],             0.0],
                [0.0,                    c.rho*c.gamma[1]],
                [0.0,                    c.rho*(1-c.gamma[1])],
                [c.rho*(1-c.gamma[0]),         0.0],
                [0.0,         0.0],
                [0.0,         0.0]
            ])
            #B = np.concatenate([B,Gd],axis=1)
        else:
            A = np.array([
                        [-1/T[0],   0.0,      1/T[2],   0.0],
                        [0.0,     -1/T[1],    0.0,      1/T[3]],
                        [0.0,      0.0,     -1/T[2],    0.0],
                        [0.0,      0.0,      0.0,     -1/T[3]]
                        ])
            C = np.diag(1.0/(c.rho*c.A))
            #Cz = C[0:2, :]
            Gd = np.array([[0.0,0.0],
                           [0.0,0.0],
                           [c.rho,0.0],
                           [0.0,c.rho]
                           ])
            B = np.array([
                [c.rho*c.gamma[0],             0.0],
                [0.0,                    c.rho*c.gamma[1]],
                [0.0,                    c.rho*(1-c.gamma[1])],
                [c.rho*(1-c.gamma[0]),         0.0]  
                ])
            #B = np.concatenate([B,Gd],axis=1)
            Gb = None
            #E = ???
        D = np.zeros((4,1))
        return A,B,C,Gd,Gb,D

    def get_flows(self, h):
        # Get flows q and q_in
        q = np.zeros(self.n)
        q_in = np.array(
            [c.gamma[0]*self.u[0], c.gamma[1]*self.u[1], (1-c.gamma[1])*self.u[1], (1-c.gamma[0])*self.u[0]]
        )
        q = c.a * np.sqrt(2 * c.g * h)
        return q_in, q
    
    def get_rhs(self, t, m, d):
        # ODE RHS dm/dt = f
        h = FourTank.mass_to_height(m)
        q_in, q = self.get_flows(h)
        f = c.rho * (q_in + np.array([q[2]-q[0],q[3]-q[1],-q[2]+d[0],-q[3]+d[1]]))
        return f
    
    def get_disturbance(self, t):
        raise NotImplementedError
    
    
    def get_disturbance_hist(self):
        raise NotImplementedError
    
    def get_measurement_noise(self, t):
        raise NotImplementedError
    
    def construct_measurement_noise_disc(self,N):
        raise NotImplementedError
    
    def construct_disturbance_disc(self,N):
        raise NotImplementedError

    def solve_step(self, t, m, d, dt):
        sol = sp.integrate.solve_ivp(
            self.get_rhs, (t, t + dt), m, t_eval=[t + dt],
            args=(d,), rtol=1e-6, atol=1e-8
        )
        if not sol.success:
            raise RuntimeError(f"ODE solver failed: {sol.message}")
        return sol.y[:, -1]
    
    def simulate(self, t0, tf, h0, ctrl_type="PID"):
        """
        Simulation of the four tank system in the time-interval [t0, tf]
        Choose ctrl_type from ["", "P", "PI", "PID] ("" corresponds to open loop)
        """
        self.t0 = t0
        self.tf = tf
        self.N = int(np.ceil((tf-t0) / self.dt))
        t = float(t0)

        m0 = FourTank.height_to_mass(h0)
        m = np.asarray(m0, dtype=float).copy()
        m_prev = m.copy()

        u0 = np.atleast_1d(self.get_ubar(t0)).astype(float)
        nx, nu = m.size, u0.size
        
        self.hist['zbar'] = np.empty((self.N + 1, 2), dtype=float)
        self.hist['t'] = np.empty(self.N + 1, dtype=float)
        self.hist['m'] = np.empty((self.N + 1, nx), dtype=float)
        self.hist['y'] = np.empty((self.N + 1, nx), dtype=float) # y is h + noise
        self.hist['u'] = np.empty((self.N + 1, nu), dtype=float)

        self.hist['t'][0] = t
        self.hist['m'][0] = m
        self.hist['u'][0] = u0
        self.hist['zbar'][0] = np.atleast_1d(self.get_zbar(t))[:2]
        
        self.k = 0
        v = None
        while (self.k < self.N) and (t < tf):
            dt = min(self.dt, tf - t)
            
            v_prev = v
            v = self.get_measurement_noise(t)
            if self.k == 0:
                v_prev = v

            h_prev = FourTank.mass_to_height(m_prev)
            h = FourTank.mass_to_height(m)

            y_prev = np.clip(h_prev + v_prev, 0, None)
            y = np.clip(h + v, 0, None)
            self.hist['y'][self.k] = np.clip(FourTank.mass_to_height(m) + v, 0, None)
            ubar = self.get_ubar(t)
            zbar = self.get_zbar(t)
            self.hist['zbar'][self.k] = np.atleast_1d(zbar)[:2] 
            self.controller(zbar,ubar, y[:2], y_prev[:2], dt, ctrl_type)
            
            d = self.get_disturbance(t)
            
            m_prev = m
            m = self.solve_step(t, m, d, dt)
            if np.any(m < 0):
                print('hello')
            t = t + dt

            self.k += 1
            self.hist['t'][self.k] = t
            self.hist['m'][self.k] = m
            self.hist['u'][self.k] = np.atleast_1d(self.u).astype(float)
        
        self.hist['h'] = FourTank.mass_to_height(self.hist['m'])
        self.hist['y'][-1] = np.clip(self.hist['h'][-1] + self.get_measurement_noise(tf), 0, None)
        self.hist['zbar'][-1] = np.atleast_1d(self.get_zbar(self.tf))[:2]
        td_hist, d_hist = self.get_disturbance_hist()
        self.hist['td'] = td_hist
        self.hist['d'] = d_hist
    
    def controller(self, zbar,ubar, y, y_prev, dt, ctrl_type):
        # PID controller (2 variable input y1 and y2, 2 variable output u1 and u2)
        if not ctrl_type in ["", "P", "PI", "PID"]:
            raise ValueError("No valid control type chosen")
        
        e = zbar - y
        u_cand = ubar.copy()
        if "P" in ctrl_type:
            P = self.KP @ e
            u_cand += P
        if "I" in ctrl_type:
            u_cand += self.I
        if "D" in ctrl_type:
            D = - self.KD @ ((y - y_prev) / dt)
            u_cand += D
            
        free = (u_cand > p.umin) & (u_cand < p.umax) 
        self.u = np.clip(u_cand, p.umin, p.umax)

        if "I" in ctrl_type:
            self.I += (self.KI @ e) * free * dt
    
    def set_ubar_piecewise(self, segments):
        self.ubar_piecewise = True
        self.ubar_times = [t for (t, _) in segments]
        self.ubar_vals = [np.asarray(u, float) for (_, u) in segments]
        self.ubar_idx = 0 # index of segment
        
    def set_zbar_piecewise(self,segments):
        self.zbar_piecewise = True
        self.zbar_times = [t for (t, _) in segments]
        self.zbar_vals = [np.asarray(u, float) for (_, u) in segments]
        self.zbar_idx = 0 # index of segment
    def set_zbar_poisson(self,mu,sig,t_z):
        self.mu_z = mu
        self.sig_z = sig
        self.t_z = t_z
        z0 = self.mu_z + self.pois_seed.normal(scale=self.sig_z)
        self.zbar_vals_hist = [np.clip(z0, 0.0, None)]
        self.zbar_times_hist = [0.0]
    
    def get_zbar(self, t):
        if self.zbar_type == 'constant':
            return self.zbar
        elif self.zbar_type == 'piecewise':
            while self.zbar_idx + 1 < len(self.zbar_times) and t >= self.zbar_times[self.zbar_idx+1]:
                self.zbar_idx += 1
            return self.zbar_vals[self.zbar_idx]
        elif self.zbar_type == 'poisson':
            while t >= self.zbar_times_hist[-1]:
                next_t = self.zbar_times_hist[-1] + self.pois_seed.exponential(scale=self.t_z)
                self.zbar_times_hist.append(next_t)
                z_new = self.mu_z + self.pois_seed.normal(scale=self.sig_z)
                self.zbar_vals_hist.append(z_new)
            if t < self.zbar_times_hist[-1]:
                arr = np.array(self.zbar_times_hist)
                idx = np.searchsorted(arr, t)
                return self.zbar_vals_hist[idx]
            else:
                return self.zbar_vals_hist[-1]
        else:
            raise ValueError('Unknown zbar_type')
    
    def get_ubar(self, t):
        if self.ubar_type == 'constant':
            return self.ubar
        elif self.ubar_type == 'piecewise':
            while self.ubar_idx + 1 < len(self.ubar_times) and t >= self.ubar_times[self.ubar_idx+1]:
                self.ubar_idx += 1
            return self.ubar_vals[self.ubar_idx]
        else:
            raise ValueError('Unknown ubar_type')
        

class Deterministic(FourTank):
    def __init__(self, dt, zbar, ubar, d):
        """d: constant disturbance"""
        super().__init__(dt, zbar, ubar)
        self.d = d
    
    def get_disturbance(self, t):
        return self.d
    
    def get_disturbance_hist(self):
        td_hist = np.array([self.t0, self.tf])
        d_hist = np.array([self.d, self.d])
        #d_hist = self.d
        return td_hist, d_hist

    def get_measurement_noise(self, t):
        # No noise in deterministic
        return np.zeros((4,))
    
    def construct_measurement_noise_disc(self, N):
        return np.zeros((2,N))
    
    def construct_disturbance_disc(self,N):
        return np.arange(self.t0, self.tf + self.dt, self.dt),np.reshape(np.repeat(self.d[None,:], N+1, axis=1),(N+1,2)) 
    #self.d.T
    #

class StochasticPiecewise(FourTank):
    def __init__(self, dt, zbar, ubar, sig_v, mu_d, sig_d, t_d, seed=0):
        """
        sig_v (scalar): std. dev. for independent measurement noise
        mu_d (length 2): mean disturbance
        sig_d (length 2): std. dev. for disturbances
        t_d (scalar): mean time between disturbances
        """
        super().__init__(dt, zbar, ubar, seed)

        self.sig_v = sig_v
        self.mu_d = mu_d
        self.sig_d = sig_d
        self.t_d = t_d

        self.td_hist = [0.]
        self.d_hist = []
        
    
    def get_disturbance(self, t):
        """Poisson intervalled distrubances"""
        while t >= self.td_hist[-1]:
            d_cand = self.mu_d + self.rng.normal(scale=self.sig_d)
            self.d_hist.append(np.clip(d_cand, p.umin, p.umax))
            self.td_hist.append(self.td_hist[-1] + self.rng.exponential(scale=self.t_d))
        return self.d_hist[-1]
    
    def get_disturbance_hist(self):
        td_hist = np.array(self.td_hist[:-1] + [self.tf])
        d_hist = np.vstack((self.d_hist, self.d_hist[-1]))
        return td_hist, d_hist

    def get_measurement_noise(self, t):
        return self.rng.normal(scale=self.sig_v, size=self.n)
    
    def construct_measurement_noise_disc(self, N):
        return self.rng.normal(scale=self.sig_v, size=(self.n,N))
    
    def construct_disturbance_disc(self,N):
        T = (N - 1) * self.dt
        event_times = [0.0]
        t = 0.0
        while t <= T:
            t += self.rng.exponential(scale=self.t_d)
            event_times.append(t)
        event_times = np.array(event_times)
        eps = self.rng.normal(size=(self.sig_d.shape[0], event_times.size))
        d_vals = self.mu_d[:, None] + self.sig_d[:, None] * eps
        #d_vals = self.mu_d + self.rng.normal(scale=self.sig_d, size=(self.sig_d.shape[0],event_times.size))
        d_vals = np.clip(d_vals, p.umin[:, None], p.umax[:, None])
        #self.umin[:,None], self.umax[:,None])
        #
        t_grid = np.arange(N) * self.dt
        idx = np.searchsorted(event_times, t_grid, side='right') - 1
        series = d_vals[:, idx]
        
        #return event_times,np.transpose(series)
        td_hist = np.concatenate((t_grid,np.array([self.tf])))
        d_hist = np.vstack((np.transpose(series), series[:,-1]))
        return td_hist, d_hist

class StochasticBrownian(FourTank):
    def __init__(self, dt, zbar, ubar, sig_v,Fd,Fs,F0,Y_transform = None, F_transform=None, seed=0):
        super().__init__(dt, zbar, ubar, seed)
        self.Y_t = Y_transform
        self.F_t = F_transform
        self.sig_v = sig_v
        self.n_F = len(Fd)
        self.Fd3,self.Fd4 = Fd[0],Fd[1]
        self.Fs3,self.Fs4 = Fs[0],Fs[1]
        self.F = np.asarray(F0,float).copy()
        if not True:
            ap = self.F.copy()
            heights = self.mass_to_height(np.array([0,0,ap[0],ap[1]]))
            self.d_hist = [heights[2:].copy()]
        else:
            self.d_hist = [self.F.copy()]
        self.td_hist = [0.]
    
    def get_disturbance(self, t):
        return self.F.copy()
    
    def get_disturbance_hist(self):
        return np.array(self.td_hist), np.vstack(self.d_hist)

    def get_rhs_sde(self, t, m,d):
        # ODE RHS dm/dt = f
        h = FourTank.mass_to_height(m)
        q_in, q = self.get_flows(h)
        f = c.rho * (q_in + np.array([q[2]-q[0],q[3]-q[1],-q[2]+d[0],-q[3]+d[1]]))
        return f
    
    def get_measurement_noise(self, t):
        return self.rng.normal(scale=self.sig_v, size=self.n)
    
    def construct_measurement_noise_disc(self, N):
        return self.rng.normal(scale=self.sig_v[:,None], size=(self.n,N))
    
    def construct_disturbance_disc(self,N):
        ts = np.arange(N+1)
        beta3s = np.zeros(N+1)
        beta4s = np.zeros(N+1)
        for t in ts:
            beta3s[t] = p.beta3(t)
            beta4s[t] = p.beta4(t)
        return np.arange(self.t0, self.tf + self.dt, self.dt), np.stack((beta3s,beta4s),axis=1)

    def get_sde(self,F,t):
        F3, F4 = F
        drift = np.array([self.Fd3(F3,t),self.Fd4(F4,t)])
        sigma = np.array([self.Fs3(F3,t), self.Fs4(F4,t)])
        return drift,sigma
    
    def construct_step(self,m,f,D,S,dW,Y):
        return np.concatenate([m, Y]),np.concatenate([f,D]),np.pad(S,(self.n,0),'constant'),np.pad(dW,(self.n,0),'constant')

    def solve_step(self, t, m, d, dt):
        f = self.get_rhs_sde(t, m,d)
        dW = np.sqrt(dt) * self.rng.normal(size=self.n_F)
        Y = self.Y_t(self.F) if self.Y_t else self.F
        D,S = self.get_sde(Y,self.k)
        m_s,f_s,sig_s,dW_s = self.construct_step(m,f,D,S,dW,Y)
        sol_s = m_s + f_s*dt+sig_s*dW_s
        #sol_s = np.clip(sol_s,0,None)
        sol_m = sol_s[:self.n]
        sol_m = np.clip(sol_m,0,None)
        Y_new = sol_s[self.n:]                                 
        self.F = self.F_t(Y_new) if self.F_t else Y_new
        self.td_hist.append(t+dt)
        if not True:
            ap = self.F.copy()
            heights = self.mass_to_height(np.array([0,0,ap[0],ap[1]]))
            self.d_hist.append(heights[2:])
        else:
            self.d_hist.append(self.F.copy())
        return sol_m
    
   