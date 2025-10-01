import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

import src.constants as c
import src.parameters as p

class FourTank:
    """Class modelling the modified four tank system"""
    def __init__(self, dt, zbar, ubar, seed=0):
        self.zbar = zbar.copy()

        if isinstance(ubar, list) and isinstance(ubar[0], tuple):
            # looks like [(t0, u0), (t1, u1), ...]
            self.ubar_type = 'piecewise'
            self.set_ubar_piecewise(ubar)
        else:
            self.ubar_type = 'constant'
            self.ubar = np.asarray(ubar, float).copy()
        
        self.n = 4
        self.dt = dt

        self.I = np.zeros((2,))
        self.KP = p.KP.copy()
        self.KI = p.KI.copy()
        self.KD = p.KD.copy()
        self.hist = {}

        self.rng = np.random.default_rng(seed)

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
    
    def get_output(m):
        # Get outputs h1 and h2
        out = m[:2] / (c.rho * c.A[:2])
        return out

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
        N = int(np.ceil((tf-t0) / self.dt))
        t = float(t0)

        m0 = FourTank.height_to_mass(h0)
        m = np.asarray(m0, dtype=float).copy()
        m_prev = m.copy()

        u0 = np.atleast_1d(self.get_ubar(t0)).astype(float)
        nx, nu = m.size, u0.size

        self.hist['t'] = np.empty(N + 1, dtype=float)
        self.hist['m'] = np.empty((N + 1, nx), dtype=float)
        self.hist['y'] = np.empty((N + 1, nx), dtype=float) # y is h + noise
        self.hist['u'] = np.empty((N + 1, nu), dtype=float)

        self.hist['t'][0] = t
        self.hist['m'][0] = m
        self.hist['u'][0] = u0
        
        k = 0
        v = None
        while (k < N) and (t < tf):
            dt = min(self.dt, tf - t)

            v_prev = v
            v = self.get_measurement_noise(t)
            if k == 0:
                v_prev = v

            h_prev = FourTank.mass_to_height(m_prev)
            h = FourTank.mass_to_height(m)

            y_prev = np.clip(h_prev + v_prev, 0, None)
            y = np.clip(h + v, 0, None)
            self.hist['y'][k] = np.clip(FourTank.mass_to_height(m) + v, 0, None)

            ubar = self.get_ubar(t)
            self.controller(ubar, y[:2], y_prev[:2], dt, ctrl_type)
            
            d = self.get_disturbance(t)
            
            m_prev = m
            m = self.solve_step(t, m, d, dt)
            if np.any(m < 0):
                print('hello')
            t = t + dt

            k += 1
            self.hist['t'][k] = t
            self.hist['m'][k] = m
            self.hist['u'][k] = np.atleast_1d(self.u).astype(float)
        
        self.hist['h'] = FourTank.mass_to_height(self.hist['m'])
        self.hist['y'][-1] = np.clip(self.hist['h'][-1] + self.get_measurement_noise(tf), 0, None)
        td_hist, d_hist = self.get_disturbance_hist()
        self.hist['td'] = td_hist
        self.hist['d'] = d_hist
    
    def reset(self):
        """Reset for simulate."""
        pass
    
    def controller(self, ubar, y, y_prev, dt, ctrl_type):
        # PID controller (2 variable input y1 and y2, 2 variable output u1 and u2)
        if not ctrl_type in ["", "P", "PI", "PID"]:
            raise ValueError("No valid control type chosen")
        
        e = self.zbar - y
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
    
    def get_ubar(self, t):
        if self.ubar_type == 'constant':
            return self.ubar
        elif self.ubar_type == 'piecewise':
            while self.ubar_idx + 1 < len(self.ubar_times) and t >= self.ubar_times[self.ubar_idx+1]:
                self.ubar_idx += 1
            return self.ubar_vals[self.ubar_idx]
        else:
            raise ValueError('Unknown ubar_type')
    
    def get_steady_state(self, m0, us, ds):
        self.u = us
        rhs_wrap = lambda m,d: self.get_rhs(0, m, d)
        ms = sp.optimize.fsolve(rhs_wrap, m0, args=(ds,))
        return ms
    
    def step_response(self, h0, us, ds, tf):
        m0 = FourTank.height_to_mass(h0)
        ms = self.get_steady_state(m0, us, ds)
        hs = FourTank.mass_to_height(ms)
        
        labels = [r"$h_1$", r"$h_2$", r"$h_3$", r"$h_4$"]
        linestyles = ["-", "--", ":"]

        plt.figure(figsize=(10,6))
        incs = [0.10, 0.25, 0.50]
        for j, inc in enumerate(incs):
            us_inc = (1 + inc) * us
            self.ubar = us_inc
            self.simulate(0, tf, hs, ctrl_type="")
            for i in range(2):
                plt.plot(
                    self.hist['t'], self.hist['h'][:,i],
                    color=f"C{i}",
                    linestyle=linestyles[j],
                    label=labels[i] if j==0 else None
                )
        
        plt.xlabel("Time")
        plt.ylabel("Height")
        plt.legend()

        # First legend (tanks)
        leg1 = plt.legend(title="Tanks", loc="upper left")

        # Second legend (increments / linestyles)
        from matplotlib.lines import Line2D
        handles = [Line2D([0], [0], color="k", linestyle=ls, label=f"{100*inc:.0f}\%")
                for ls, inc in zip(linestyles, incs)]
        plt.legend(handles=handles, title="Inflow increase", loc="upper right")
        plt.grid(True)
        plt.gca().add_artist(leg1)  # keep both legends
        plt.show()
        

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
        return td_hist, d_hist

    def get_measurement_noise(self, t):
        # No noise in deterministic
        return np.zeros((4,))

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

class StochasticBrownian(FourTank):
    def __init__(self, dt, zbar, ubar, sig_v, sig_sde, seed=0):
        super().__init__(dt, zbar, ubar, seed)
        self.sig_v = sig_v
        self.sig_sde = sig_sde
    
    def get_disturbance(self, t):
        # This eliminates d in the RHS and extracts f
        return np.zeros(2)
    
    def get_disturbance_hist(self):
        return None, None

    def get_measurement_noise(self, t):
        return self.rng.normal(scale=self.sig_v, size=self.n)
    
    def get_sde_sigma(self):
        # TODO: implement this using potentials instead of constant
        sigma = np.array([0,0,self.sig_sde, self.sig_sde])
        return sigma

    def solve_step(self, t, m, d, dt):
        f = self.get_rhs(t, m, d)
        dW = np.sqrt(dt) * self.rng.normal(size=self.n)
        g = self.get_sde_sigma()
        sol = m + f * dt + g * dW
        sol = np.clip(sol, 0, None)
        return sol