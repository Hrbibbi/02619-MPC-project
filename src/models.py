import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

import src.constants as c
import src.parameters as p
import src.plotting

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

    def generate_disturbances(self):
        raise NotImplementedError

    def get_disturbance(self, t):
        raise NotImplementedError
    
    def get_disturbance_hist(self):
        raise NotImplementedError
    
    def generate_measurement_noise(self):
        raise NotImplementedError
    
    def get_measurement_noise(self, t):
        raise NotImplementedError

    def reset_noise(self):
        self.generate_disturbances()
        self.generate_measurement_noise()

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
    
    def step_response(self, h0, us, ds, tf, incs=[0.10, 0.25, 0.50], normalized=False, read_off=False, plot_title="", measurements=False, filename=None):
        m0 = FourTank.height_to_mass(h0)
        ms = self.get_steady_state(m0, us, ds)
        hs = FourTank.mass_to_height(ms)
        
        in_labels = [r"$u_1$", r"$u_2$"]
        out_labels = [r"$h_1$", r"$h_2$"]

        fig,axs = plt.subplots(nrows=2,ncols=2,figsize=(14,9))

        for c in range(2):
            for i, inc in enumerate(incs):
                us_inc = us.copy()
                us_inc[c] = (1 + inc) * us_inc[c]
                self.ubar = us_inc
                if measurements:
                    self.reset_noise()
                self.simulate(0, tf, hs, ctrl_type="")
                
                ms_new = self.get_steady_state(ms, us_inc, ds)
                hs_new = FourTank.mass_to_height(ms_new)
                for r in range(2): #height index
                    ax = axs[r,c]
                    tt = self.hist['t']
                    hh = self.hist['h'][:,r]
                    yy = self.hist['y'][:,r]

                    if measurements:
                        deviation = hh[:,None] + np.array([[-1,1]])*self.sig_v
                        if normalized:
                            deviation = (deviation - hs[r]) / (us_inc[c] - us[c])
                    
                    if normalized:
                        yy = (yy - hs[r]) / (us_inc[c] - us[c])
                        hh = (hh - hs[r]) / (us_inc[c] - us[c])
                    
                    if measurements:
                        ax.plot(tt, yy, linestyle='-', alpha=0.7, lw=0.5)
                        ax.plot(
                            tt, deviation,
                            linestyle=':', color=f"C{i}"
                        )

                        # ax.plot(
                        #     tt, hh[:,None] + np.array([[-1,1]])*self.sig_v,
                        #     linestyle=':', color=f"C{i}"
                        # )
                    ax.plot(
                        tt, hh,
                        color=f"C{i}",
                        linestyle='--' if measurements else '-'
                    )

                    if r == 0:
                        ax.set_title(f'Input {c+1} ({in_labels[c]})')
                    if c == 0:
                        ylabel = f'Output {r+1}'
                        if normalized:
                            ylabel += " (normalized)"
                        ax.set_ylabel(ylabel)
                    if r == 1:
                        ax.set_xlabel("Time")
                        
                    if read_off:
                        gain = (hs_new[r] - hs[r]) / (us_inc[c] - us[c])
                        ax.axhline(gain, ls='--', color='black', alpha=0.5, label='gain')
                        if r == c:
                            # first order case
                            ix = np.argmax(hh > 0.632*gain)
                            ax.plot(2*[tt[ix]], [0,hh[ix]], ls='-', color='red', label=r"63.2\%")
                        else:
                            # second order case
                            ix1 = np.argmax(hh > 0.353*gain)
                            ix2 = np.argmax(hh > 0.853*gain)
                            ax.plot(2*[tt[ix1]], [0,hh[ix1]], ls='-', color='red', label=r"35.3\%")
                            ax.plot(2*[tt[ix2]], [0,hh[ix2]], ls='-', color='red', label=r"85.3\%")
                        ax.legend()
        
                    ax.grid(True)

        ### Legends
        from matplotlib.lines import Line2D

        # --- Legend 1: Color meaning (Inflow increase) ---
        handles_color = [
            Line2D([0], [0], color=f"C{i}", linestyle='-', label=f"{100*inc:.0f}\\%")
            for i, inc in enumerate(incs)
        ]

        # --- Legend 2: Line style meaning (Measurement interpretation) ---
        handles_style = []
        if measurements:
            handles_style = [
                Line2D([0], [0], color='k', linestyle='-', label=r'$y = h + v$'),
                Line2D([0], [0], color='k', linestyle='--', label=r'$h$'),
                Line2D([0], [0], color='k', linestyle=':', label=r'$h \pm \sigma_v$'),
            ]

        # Leave extra space on the right for legends
        adj = 0.8
        d_adj = 0.01
        plt.subplots_adjust(right=adj)

        # --- Legend 1 (top right, slightly above the second legend) ---
        fig.legend(
            handles=handles_color,
            title="Inflow increase",
            loc="upper left",
            bbox_to_anchor=(adj+d_adj, 0.9)
        )

        # --- Legend 2 (below the first legend) ---
        if measurements:
            fig.legend(
                handles=handles_style,
                title="Outputs",
                loc="upper left",
                bbox_to_anchor=(adj+d_adj, 0.75)
            )
        
        fig.suptitle(plot_title, y=0.95, fontsize=20)
        if filename is not None:
            plt.savefig(filename)
        plt.show()
    
    def get_jacobians(self, hs):
        """Get the Jacobian matrices for the continuous linearization"""
        T = c.A / c.a * np.sqrt(2*hs/c.g)
        A = np.diag(-1/T)
        A[0, 2] = 1/T[2]
        A[1, 3] = 1/T[3]

        B = c.rho * np.array([
            [c.gamma[0], 0],
            [0, c.gamma[1]],
            [0, 1 - c.gamma[1]],
            [1 - c.gamma[0], 0]
        ], dtype=float)

        C = np.diag(1 / (c.rho * c.A))
        Cz = C[:2,:]

        Cd = np.array([
            [0, 0],
            [0, 0],
            [c.rho, 0],
            [0, c.rho]
        ], dtype=float)
        return A,B,C,Cz,Cd

    def lin_continuous(self, tf, xs, us, ds):
        """Continuous linearization around steady state"""
        t0 = 0
        hs = FourTank.mass_to_height(xs)
        A,B,C,Cz,Cd = self.get_jacobians(hs)

        def flin(t,X):
            u_delta = (self.ubar - us)
            d_delta = (self.get_disturbance(t) - ds)
            return A @ X + B @ u_delta + Cd @ d_delta
        t_eval = np.linspace(t0,tf,200)
        sollin = sp.integrate.solve_ivp(fun=flin, t_span=(t0,tf), y0=xs-xs, t_eval=t_eval)
        Xlin = sollin.y
        Ylin = C @ Xlin
        ylin = Ylin + hs[:,None]
        return t_eval, Ylin.T

    def c2dzoh():
        """Continuous-to-discrete ZOH (zero-order hold)"""
        pass
    
    def lin_discrete():
        """Discrete linearization around steady state"""
        pass
    
    def markov_params(self, Ts, n, h0, us, ds, filename=None):
        """Deterministic/stochastic piecewise"""
        # The coordinate (i,j) of the matrix corresponds to impulse response from input to output so input j and output i
        def calc_hat_matrices():
            n_eval = 200
            dTs = Ts/n_eval
            dexp = sp.linalg.expm(A*dTs)
            integrand = dexp
            I = np.zeros(A.shape)
            for i in range(n_eval-1):
                integrand = integrand @ dexp
                I += dTs * integrand
            
            Ahat = sp.linalg.expm(A*Ts)
            Bhat = I @ B
            Cd_hat = I @ Cd
            Cz_hat = Cd_hat
            return Ahat,Bhat,Cd_hat,Cz_hat

        def calc_markov_matrix(n, Ahat, Bhat, Cz_hat):
            A = Ahat
            B = Bhat
            C = Cz_hat
            ns = np.arange(n+1)
            M = np.zeros((2,2,n+1))
            for n in ns:
                M[:,:,n] = np.array([
                    [
                        C[0,0] * A[0,0]**n * B[0,0],
                        C[0,0]*A[0,0]*B[0,1] + C[0,0]*A[0,2] / (A[0,0]-A[2,2]) * (A[0,0]**n - A[2,2]**n)*B[2,1]
                    ],
                    [
                        C[1,1]*A[1,1]*B[1,0] + C[1,1]*A[1,3] / (A[0,0]-A[2,2]) * (A[1,1]**n - A[3,3]**n)*B[3,0],
                        C[1,1] * A[1,1]**n * B[1,1]
                    ]
                ])
            return M
        
        m0 = FourTank.height_to_mass(h0)
        ms = self.get_steady_state(m0, us, ds)
        hs = FourTank.mass_to_height(ms)
        A,B,C,Cz,Cd = self.get_jacobians(hs)
        Ahat,Bhat,Cd_hat,Cz_hat = calc_hat_matrices()
        M = calc_markov_matrix(n, Ahat, Bhat, Cz_hat)
        
        t = np.arange(0, M.shape[2]*Ts, Ts)
        fig,ax = plt.subplots(2,2,figsize=(12,8))
        for i in range(2):
            for j in range(2):
                ax[i,j].plot(t, M[i,j])
                ax[i,j].grid(True)
                if i == 0:
                    ax[i,j].set_title(f'Input {j+1}')
                if j == 0:
                    ax[i,j].set_ylabel(f'Output {i+1}', fontsize=18)
        
        plt.tight_layout()
        if filename is not None:
            plt.savefig(filename)
        plt.show()
        
    
    def plot_lin(self, h0, us, ds, tf, plot_title="", filename=None):
        m0 = FourTank.height_to_mass(h0)
        ms = self.get_steady_state(m0, us, ds)
        hs = FourTank.mass_to_height(ms)

        incs = [0.10, 0.25, 0.50]
        labels = [rf"$h_{i} - (h_s)_{i}$" for i in range(1, 5)]

        fig, axes = plt.subplots(3, 4, figsize=(12, 9))
        fig.suptitle(plot_title, fontsize=16)
        for j, inc in enumerate(incs):
            us_inc = (1 + inc) * us
            self.ubar = us_inc
            t_lin, Y_lin = self.lin_continuous(tf, ms, us, ds)

            self.simulate(0, tf, hs, ctrl_type="")
            t_nonlin = self.hist['t']
            y_nonlin = self.hist['h']
            Y_nonlin = y_nonlin - hs

            for i in range(self.n):
                ax = axes[j, i]
                ax.set_title(labels[i] if j == 0 else "")  # titles only on first row
                ax.plot(t_lin, Y_lin[:, i], label='linear')
                ax.plot(t_nonlin, Y_nonlin[:, i], label='non-linear')
                ax.grid(True)
                if i == 0:
                    ax.set_ylabel(f"Inflow increase: {100*inc:.0f}\\%")

                ax.legend()
                if j == len(incs) - 1:
                    ax.set_xlabel(f'$t$')
        plt.tight_layout()
        fig.suptitle(plot_title, fontsize=20)
        if filename is not None:
            plt.savefig(filename)
        plt.show()
    
    def plot_lin2(self, h0, us, ds, tf, plot_title="", filename=None, normalized=True):
        m0 = FourTank.height_to_mass(h0)
        ms = self.get_steady_state(m0, us, ds)
        hs = FourTank.mass_to_height(ms)

        # incs = [0.10, 0.25, 0.50]
        incs = np.linspace(0.1,1,10)
        labels = [rf"$h_{i} - (h_s)_{i}$" for i in range(1, 5)]

        fig, axes = plt.subplots(4,1, figsize=(9, 18))
        fig.suptitle(plot_title, fontsize=16)
        for j, inc in enumerate(incs):
            us_inc = (1 + inc) * us
            self.ubar = us_inc
            t_lin, Y_lin = self.lin_continuous(tf, ms, us, ds)

            self.simulate(0, tf, hs, ctrl_type="")
            t_nonlin = self.hist['t']
            # y_nonlin = self.hist['h']
            # Y_nonlin = y_nonlin - hs

            for i in range(self.n):
                hh_nonlin = self.hist['h'][:,i]
                hh_lin = Y_lin[:,i]

                if normalized:
                    hh_nonlin = (hh_nonlin - hs[i]) / (us_inc[i] - us[i])
                    hh_lin = (hh_lin - hs[i]) / (us_inc[j] - us[j])
                ax = axes[i]
                ax.plot(t_lin, hh_lin, label=f'linear: {100*inc:.0f}\\%')
                ax.plot(t_nonlin, hh_nonlin, label=f'non-linear: {100*inc:.0f}\\%')
                if j == 0:
                    ax.set_title(labels[i] if j == 0 else "")  # titles only on first row
                    ax.grid(True)

                ax.legend()
                if j == len(incs) - 1:
                    ax.set_xlabel(f'$t$')
        plt.tight_layout()
        fig.suptitle(plot_title, fontsize=20)
        if filename is not None:
            plt.savefig(filename)
        plt.show()
    
    def step_response_lin(self, h0, us, ds, tf, incs=[0.10, 0.25, 0.50], normalized=False, plot_title="", filename=None):
        m0 = FourTank.height_to_mass(h0)
        ms = self.get_steady_state(m0, us, ds)
        hs = FourTank.mass_to_height(ms)
        
        in_labels = [r"$u_1$", r"$u_2$"]
        out_labels = [r"$h_1$", r"$h_2$"]

        fig,axs = plt.subplots(nrows=2,ncols=2,figsize=(14,9))

        for c in range(2):
            for i, inc in enumerate(incs):
                us_inc = us.copy()
                us_inc[c] = (1 + inc) * us_inc[c]
                self.ubar = us_inc
                tt, Y_lin = self.lin_continuous(tf, ms, us, ds)
                
                ms_new = self.get_steady_state(ms, us_inc, ds)
                hs_new = FourTank.mass_to_height(ms_new)
                for r in range(2): #height index
                    ax = axs[r,c]
                    hh = Y_lin[:,r]

                    if normalized:
                        hh = hh / (us_inc[c] - us[c])
                        
                    ax.plot(
                        tt, hh,
                        color=f"C{i}",
                        linestyle='-'
                    )

                    if r == 0:
                        ax.set_title(f'Input {c+1} ({in_labels[c]})')
                    if c == 0:
                        ylabel = f'Output {r+1}'
                        if normalized:
                            ylabel += " (normalized)"
                        ax.set_ylabel(ylabel)
                    if r == 1:
                        ax.set_xlabel("Time")
        
                    ax.grid(True)

        ### Legends
        from matplotlib.lines import Line2D

        # --- Legend 1: Color meaning (Inflow increase) ---
        handles_color = [
            Line2D([0], [0], color=f"C{i}", linestyle='-', label=f"{100*inc:.0f}\\%")
            for i, inc in enumerate(incs)
        ]

        # Leave extra space on the right for legends
        adj = 0.8
        d_adj = 0.01
        plt.subplots_adjust(right=adj)

        # --- Legend 1 (top right, slightly above the second legend) ---
        fig.legend(
            handles=handles_color,
            title="Inflow increase",
            loc="upper left",
            bbox_to_anchor=(adj+d_adj, 0.9)
        )
        
        fig.suptitle(plot_title, y=0.95, fontsize=20)
        if filename is not None:
            plt.savefig(filename)
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
        Poisson intervalled disturbances.
        
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
        
        self.reset_noise()
    
    def generate_disturbances(self):
        t = 0.0
        td_hist = [t]
        d_hist = []

        while t < p.tf:
            d_cand = self.mu_d + self.rng.normal(scale=self.sig_d)
            d_hist.append(np.clip(d_cand, p.umin, p.umax))
            t += self.rng.exponential(scale=self.t_d)
            td_hist.append(t)
        
        self.td_hist = np.array(td_hist[:-1] + [p.tf])
        self.d_hist = np.vstack((d_hist, d_hist[-1]))
    
    def get_disturbance(self, t):
        idx = np.searchsorted(self.td_hist, t, side='right') - 1
        return self.d_hist[idx]
    
    def get_disturbance_hist(self):
        return self.td_hist, self.d_hist

    def generate_measurement_noise(self):
        N = int(np.ceil((p.tf - p.t0) / self.dt)) + 1
        self.v_hist = self.rng.normal(scale=self.sig_v, size=(N, self.n))

    def get_measurement_noise(self, t):
        idx = int(np.ceil(t / self.dt))
        idx = np.clip(idx, 0, len(self.v_hist) - 1)
        return self.v_hist[idx]

class SDE(FourTank):
    def __init__(self, dt, zbar, ubar, sig_v, mu_OU, sig_OU, coef_OU, seed=0):
        """
        log of the process is Ornstein-Uhlenbeck (OU) to avoid negative values.
        OU parameters are in log domain and 2-vectors. Assume OU starts at mu_OU.
        """
        super().__init__(dt, zbar, ubar, seed)

        self.sig_v = sig_v
        self.mu_OU = mu_OU
        self.sig_OU = sig_OU
        self.coef_OU = coef_OU

        self.reset_noise()

    def generate_disturbances(self):
        """
        Disturbances are independent of the rest of the system.
        """
        # dt = np.diff(np.concatenate((np.arange(p.t0, p.tf, self.dt), p.tf)))
        dt = p.dt
        N = p.nt
        dB = self.rng.normal(scale=np.sqrt(dt), size=(N,2))
        OU = np.zeros((N + 1,2))
        OU[0] = self.mu_OU
        for i in range(N-1):
            OU[i+1] = OU[i] + self.coef_OU * (self.mu_OU - OU[i]) * dt + self.sig_OU * dB[i]
        
        td_hist = np.arange(p.t0, p.tf + dt, dt)
        d_hist = np.exp(OU)

        self.td_hist = td_hist
        if p.tf % p.dt == 0:
            self.d_hist = d_hist[:-1]
        else:
            self.d_hist = d_hist

    def get_disturbance(self, t):
        idx = np.ceil(t / self.dt).astype(int)
        return self.d_hist[idx]
    
    def get_disturbance_hist(self):
        return self.td_hist, self.d_hist

    def generate_measurement_noise(self):
        N = int(np.ceil((p.tf - p.t0) / self.dt)) + 1
        self.v_hist = self.rng.normal(scale=self.sig_v, size=(N, self.n))

    def get_measurement_noise(self, t):
        idx = int(np.ceil(t / self.dt))
        idx = np.clip(idx, 0, len(self.v_hist) - 1)
        return self.v_hist[idx]

    def solve_step(self, t, m, d, dt):
        f = self.get_rhs(t, m, d) + np.concatenate((np.zeros(2), self.get_disturbance(t)))
        sol = m + f * dt
        return sol
    
    def get_jacobians(self, hs):
        super().get_jacobians(hs)
        # how to linearize the SDE?
