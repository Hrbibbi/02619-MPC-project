import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import casadi as ca

import src.constants as c
import src.parameters as p
import src.plotting

from tqdm import tqdm

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

    def get_zbar(self, t):
        if isinstance(self.zbar, np.ndarray):
            return self.zbar
        else:
            raise ValueError # implement something similar to ubar
    
    @staticmethod
    def mass_to_height(m):
        """Convert mass to height. For broadcasting to work, m must be N x 4"""
        h = m / (c.rho * c.A)
        return h
    
    @staticmethod
    def height_to_mass(h):
        """Convert height to mass.  For broadcasting to work, h must be N x 4"""
        m = c.rho * c.A * h
        return m
    
    def get_output(m):
        # Get outputs h1 and h2
        out = m[:2] / (c.rho * c.A[:2])
        return out

    def get_flows(self, h, u):
        # Get flows q and q_in
        q = np.zeros(self.n)
        q_in = np.array(
            [c.gamma[0]*u[0], c.gamma[1]*u[1], (1-c.gamma[1])*u[1], (1-c.gamma[0])*u[0]]
        )
        q = c.a * np.sqrt(2 * c.g * h)
        return q_in, q
    
    def get_rhs(self, t, m, u, d):
        # ODE RHS dm/dt = f
        h = FourTank.mass_to_height(m)
        q_in, q = self.get_flows(h, u)
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

    def solve_step(self, t, m, u, d, dt):
        sol = sp.integrate.solve_ivp(
            self.get_rhs, (t, t + dt), m, t_eval=[t + dt],
            args=(u,d), rtol=1e-6, atol=1e-8
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
            u = self.controller(ubar, y[:2], y_prev[:2], dt, ctrl_type)
            
            d = self.get_disturbance(t)
            
            m_prev = m
            m = self.solve_step(t, m, u, d, dt)
            if np.any(m < 0):
                print('hello')
            t = t + dt

            k += 1
            self.hist['t'][k] = t
            self.hist['m'][k] = m
            self.hist['u'][k] = np.atleast_1d(u).astype(float)
        
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
        u = np.clip(u_cand, p.umin, p.umax)

        if "I" in ctrl_type:
            self.I += (self.KI @ e) * free * dt
        
        return u
    
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
        rhs_wrap = lambda m,u,d: self.get_rhs(0, m, u, d)
        ms = sp.optimize.fsolve(rhs_wrap, m0, args=(us,ds))
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
            # print(u_delta,d_delta)
            return A @ X + B @ u_delta + Cd @ d_delta
        # t_eval = np.linspace(t0,tf,200)
        t_eval = np.arange(t0,tf+self.dt,self.dt)
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
                ax[i,j].plot(t, M[i,j], '.')
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
    
    def lin_valid_region(self, h0, us, ds, tf, span=0.30, N=25,
                             axes_in_percent=True, log_scale=False, filename=None):
        """
        Compute and (optionally) plot the region in (u1, u2) where the linear model
        deviates from the nonlinear model by at most `thresh` (e.g. 0.10 = 10%).

        Parameters
        ----------
        h0 : array-like
            Initial heights (used to build m0, then steady-state at us).
        us : array-like, shape (nu,)
            Nominal input vector. Region is evaluated around us[0], us[1].
        ds : array-like
            Disturbance vector passed into both simulations.
        tf : float
            Final time for simulations.
        span : float
            +/- span around each coordinate (0.30 means 70%..130% of nominal).
        N : int
            Grid resolution per axis (N x N simulations).
        log_scale : bool
            If True, color scale is logarithmic.
        axes_in_percent : bool
            If True, plot x/y axes as percent deviation from steady-state us.
            i.e. 100*(u/us - 1). Simulations still run in absolute units.

        Returns
        -------
        U1, U2 : 2D arrays, shape (N, N)
            Meshgrid for u1 and u2.
        err : 2D float array, shape (N, N)
            The scalar error value at each grid point.
        """
        m0 = FourTank.height_to_mass(h0)
        ms = self.get_steady_state(m0, us, ds)
        hs = FourTank.mass_to_height(ms)

        u1_nom, u2_nom = us[0], us[1]
        u1_vals = np.linspace((1 - span) * u1_nom, (1 + span) * u1_nom, N)
        u2_vals = np.linspace((1 - span) * u2_nom, (1 + span) * u2_nom, N)
        U1, U2 = np.meshgrid(u1_vals, u2_vals, indexing="xy")

        err = np.full((N, N), np.nan, dtype=float)

        for j in range(N):
            for i in range(N):
                u = np.array([U1[j, i],U2[j, i]])

                self.ubar = u
                t_lin, Y_lin = self.lin_continuous(tf, ms, us, ds)
                self.simulate(0, tf, hs, ctrl_type="")

                t_nonlin = self.hist["t"]
                y_nonlin = self.hist["h"]
                Y_nonlin = y_nonlin - hs

                e = np.max(np.abs((Y_lin-Y_nonlin)/hs))

                err[j, i] = e

                # if anomaly_count == 0 and e > 50:
                    # anomaly_count += 1
                if i == N//2 and j==N//2:
                    # print('u: ',u)
                    # print('e: ',e)
                    fig,ax=plt.subplots(1,2)
                    for i in range(2):
                        ax[i].plot(t_lin,Y_lin[:,i],label='lin')
                        ax[i].plot(t_nonlin,Y_nonlin[:,i],label='nonlin')
                        plt.legend()
                    plt.show()

        from matplotlib.colors import ListedColormap

        # ---- 4) Plot validity region in (u1, u2) ----
        fig, ax = plt.subplots(1, 1, figsize=(7, 6))

        from matplotlib.colors import LogNorm, Normalize

        # ---- Robust limits ----
        err_finite = err[np.isfinite(err)]

        if log_scale:
            # Log scale requires strictly positive values
            err_pos = err_finite[err_finite > 0]

            if err_pos.size == 0:
                raise ValueError("Log scale requested but no positive error values found.")

            vmin = err_pos.min()
            vmax = err_pos.max()

            if vmax <= vmin:
                vmax = 10 * vmin

            norm = LogNorm(vmin=vmin, vmax=vmax)

        else:
            # Linear scale
            vmin = err_finite.min()
            vmax = err_finite.max()

            if vmax <= vmin:
                vmax = vmin + 1e-12

            norm = Normalize(vmin=vmin, vmax=vmax)

        if axes_in_percent:
            if u1_nom == 0 or u2_nom == 0:
                raise ValueError("axes_in_percent=True requires us[0] and us[1] to be non-zero.")

            U1_plot = 100.0 * (U1 / u1_nom - 1.0)
            U2_plot = 100.0 * (U2 / u2_nom - 1.0)

            u1_mark, u2_mark = 0.0, 0.0  # steady state is 0% deviation
            xlabel = r'\% step response in $u_1$'
            ylabel = r'\% step response in $u_2$'
        else:
            U1_plot, U2_plot = U1, U2
            u1_mark, u2_mark = u1_nom, u2_nom
            xlabel = r'$u_1$'
            ylabel = r'$u_2$'

        pcm = ax.pcolormesh(
            U1_plot, U2_plot, err,
            shading="nearest",
            norm=norm,
            cmap="viridis"
        )

        cbar = fig.colorbar(pcm, ax=ax)
        cbar.set_label(r"max relative error")

        ax.plot(u1_mark, u2_mark, "o", markersize=6, label=r'$u_s$', color='red')

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True)
        plt.legend()

        plt.tight_layout()
        if filename:
            plt.savefig(filename)
        plt.show()

        return U1, U2, err

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
    def __init__(self, dt, zbar, ubar, sig_v, mu_log_OU, sig_OU, coef_OU, seed=0):
        """
        log of the process, F, is Ornstein-Uhlenbeck (OU) to avoid negative values.
        OU parameters are in log domain, Y, and 2-vectors. Assume OU starts at mu_OU.
        We have Y = log F
        """
        super().__init__(dt, zbar, ubar, seed)

        self.sig_v = sig_v
        self.sig_OU = sig_OU
        self.coef_OU = coef_OU

        # ---- mu_log_OU: constant or piecewise schedule ----
        if isinstance(mu_log_OU, list) and len(mu_log_OU) > 0 and isinstance(mu_log_OU[0], tuple):
            self.mu_OU_type = "piecewise"
            self.set_mu_OU_piecewise(mu_log_OU)   # converts + stores internal mu_OU pieces
        else:
            self.mu_OU_type = "constant"
            self.mu_OU = self.mu_log_to_mu_OU(mu_log_OU)

        self.reset_noise()
    
    def mu_log_to_mu_OU(self, mu_log):
        mu_log = np.asarray(mu_log, float)
        if np.any(mu_log <= 0):
            raise ValueError("mu_log_OU must be strictly positive (since log(mu_log_OU) is used).")
        return np.log(mu_log) - (self.sig_OU**2) / (4.0 * self.coef_OU)

    def set_mu_OU_piecewise(self, segments):
        """
        segments: [(t0, mu_log0), (t1, mu_log1), ...]
        mu_logk is desired mean in original domain (positive), length-2.

        We store converted mu_OU values (log OU mean) aligned to sorted times.
        """
        # sort by time to make searchsorted valid
        seg = sorted(((float(t), np.asarray(mu_log, float)) for (t, mu_log) in segments),
                     key=lambda x: x[0])

        self.mu_OU_times = np.array([t for (t, _) in seg], dtype=float)
        self.mu_OU_vals = np.vstack([self.mu_log_to_mu_OU(mu_log) for (_, mu_log) in seg]).astype(float)

        # optional: enforce strictly increasing times
        if np.any(np.diff(self.mu_OU_times) <= 0):
            raise ValueError("mu_log_OU piecewise times must be strictly increasing.")
    
    def get_mu_OU(self, t):
        """
        Return mu_OU(t) where mu_OU is the OU mean in log-state domain.

        Piecewise convention:
        - segments are left-closed/right-open: [t_k, t_{k+1})
        - for t < t0, use the first segment
        - for t >= last breakpoint, use the last segment
        """
        if self.mu_OU_type == "constant":
            return self.mu_OU

        if self.mu_OU_type == "piecewise":
            # idx = largest k such that mu_OU_times[k] <= t
            idx = np.searchsorted(self.mu_OU_times, t, side="right") - 1
            idx = int(np.clip(idx, 0, len(self.mu_OU_times) - 1))
            return self.mu_OU_vals[idx]

        raise ValueError("Unknown mu_OU_type")


    def generate_disturbances(self):
        """
        Disturbances are independent of the rest of the system.
        """
        # dt = np.diff(np.concatenate((np.arange(p.t0, p.tf, self.dt), p.tf)))
        dt = p.dt
        N = p.nt
        t = p.t0
        self.dB = self.rng.normal(scale=np.sqrt(dt), size=(N,2))
        dB = self.dB
        OU = np.zeros((N + 1,2))
        OU[0] = self.get_mu_OU(t)
        for i in range(N-1):
            OU[i+1] = OU[i] + self.coef_OU * (self.get_mu_OU(t) - OU[i]) * dt + self.sig_OU * dB[i]
            t += dt
        
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

    def solve_step(self, t, m, u, d, dt):
        f = self.get_rhs(t, m, u, d)
        sol = m + f * dt
        return sol
    
    def get_jacobians(self, hs):
        super().get_jacobians(hs)
        # how to linearize the SDE?

    def get_V(self,t,F):
        # Currently, Ybar is constant, so independent of t, but could change.
        V = 0.5*F*self.sig_OU**2 - F*self.coef_OU*np.log(F) + F*self.coef_OU*self.get_mu_OU(t)
        return V

    def get_drift(self, t, X, u):
        """Get drift term f of SDE"""
        m,d = X[:4],X[4:]        
        f_m = self.get_rhs(t, m, u, d)
        f = np.concatenate((f_m, self.get_V(t,d)))
        return f
    
    def get_jacobians(self, t,x):
        """Jacobians for SDE model"""
        h = self.mass_to_height(x[:4])
        T = c.A / c.a * np.sqrt(2*h/c.g)
        A = np.zeros((6,6))
        # dV/dF = \sigma^2/2 - a \ln(F) - a + a\bar{Y}
        # dV/d\bar{Y} = aF
        dVdF = [self.sig_OU[i]**2/2 + self.coef_OU[i]*(-np.log(F)-1+self.get_mu_OU(t)[i]) for i,F in enumerate(x[4:])]
        dVdYbar = [self.coef_OU[i]*F for i,F in enumerate(x[4:])]
        A = np.diag(np.concatenate((-1/T, dVdF)))
        A[0, 2] = 1/T[2]
        A[1, 3] = 1/T[3]
        A[2, 4] = c.rho
        A[3, 5] = c.rho

        B = c.rho * np.array([
            [c.gamma[0], 0],
            [0, c.gamma[1]],
            [0, 1 - c.gamma[1]],
            [1 - c.gamma[0], 0],
            [0, 0],
            [0, 0],
        ], dtype=float)
        
        CD = np.zeros((6,2))
        CD[4,0] = dVdYbar[0]
        CD[5,1] = dVdYbar[1]
        
        D = np.zeros((6,2))
        
        # mu_F = self.get_mu_OU_original(self.get_mu_OU(t))
        # D[4,0] = mu_F[0]*self.sig_OU[0]
        # D[5,1] = mu_F[1]*self.sig_OU[1]
        D[4,0] = x[4]*self.sig_OU[0]
        D[5,1] = x[5]*self.sig_OU[1]
        return A,B,CD,D
    
    def lin_continuous(self, tf, xs, us, ds):
        """Continuous linearization around steady state"""
        def refine_brownian_simple(dB, dt, k):
            """
            Refine coarse Brownian increments to a k-times finer grid, enforcing that
            each block of k fine increments sums exactly to the original coarse increment.

            Parameters
            ----------
            dB : ndarray, shape (N, dim)
                Coarse Brownian increments.
            dt : float
                Coarse time step.
            k : int
                Refinement factor (e.g. 10 => 10 fine steps per coarse step).

            Returns
            -------
            dB_fine : ndarray, shape (N*k, dim)
                Fine increments. For each i: sum(dB_fine[i*k:(i+1)*k]) == dB[i].
            """

            dB = np.asarray(dB)
            if dB.ndim != 2:
                raise ValueError("dB must have shape (N, dim)")

            N, dim = dB.shape
            dt_fine = dt / k

            # 1) Propose independent fine increments at the right scale
            fine = self.rng.normal(scale=np.sqrt(dt_fine), size=(N, k, dim))

            # 2) Compute the sum over each coarse interval
            S = fine.sum(axis=1, keepdims=True)  # shape (N, 1, dim)

            # 3) Add a constant correction to each of the k sub-increments so totals match
            correction = (dB[:, None, :] - S) / k  # shape (N, 1, dim)
            fine = fine + correction               # broadcast over k

            return fine.reshape(N * k, dim)
        t0 = 0
        hs = FourTank.mass_to_height(xs)
        Xs = np.concatenate((xs,ds))
        A,B,Cd,D = SDE.get_jacobians(self,t0,Xs)

        steps_per_sampling_time = 1
        dt = self.dt/steps_per_sampling_time
        t_eval = np.arange(t0,tf+dt,dt)
        N = len(t_eval)
        # dB = refine_brownian_simple(self.dB,self.dt,steps_per_sampling_time)
        dB = self.dB
        X = np.zeros((N,6))
        u_delta = (self.ubar - us)
        for i,t in enumerate(t_eval[:-1]):
            X[i+1,:] = X[i] + A @ X[i] * dt + B @ u_delta * dt + D @ dB[i]

        Ylin = self.mass_to_height(X[::steps_per_sampling_time,:4] + xs) - hs
        t_eval = t_eval[::steps_per_sampling_time]
        # def flin(t,X):
        #     u_delta = (self.ubar - us)
        #     d_delta = (self.get_disturbance(t) - ds)
        #     # print(u_delta, d_delta)
        #     # X = np.concatenate((x,d_delta))
        #     return A @ X + B @ u_delta + Cd @ d_delta + D @ 
        # # t_eval = np.linspace(t0,tf,200)
        # t_eval = np.arange(t0,tf+self.dt,self.dt)
        # sollin = sp.integrate.solve_ivp(fun=flin, t_span=(t0,tf), y0=Xs-Xs, t_eval=t_eval)
        # Xlin = sollin.y
        # Ylin = np.diag(1 / (c.rho * c.A)) @ Xlin[:4,:]
        # ylin = Ylin + hs[:,None]
        return t_eval, Ylin

    def get_mu_OU_original(self,Ybar):
        """Get mean of original process F = exp(Y)"""
        Fbar = np.exp(Ybar+self.sig_OU/(4*self.coef_OU))
        return Fbar

    def extended_kalman_NMPC(self, t0, tf, h0, R=None, plot=False, filename=''):
        """Continuous-Discrete Extended Kalman Filter"""
        self.simulate(t0, tf, h0, ctrl_type="")

        tt = self.hist['t']
        hh = self.hist['h']
        yy = self.hist['y']
        dd = self.hist['d']
        uu = self.hist['u']
        
        m0 = self.height_to_mass(h0)
        X0_hat = np.concatenate((m0,self.get_mu_OU_original(self.get_mu_OU(tt[0]))))
        P0_hat = np.eye(6)
        N = len(tt)
        
        def precompute_var(X):
            """Precomputed matrices for fun_var: dP/dt = FP(t) + P(t)F^T + Q, Q=GG^T"""
            G = np.zeros((6,2))
            G[4,0] = X[4]*self.sig_OU[0]
            G[5,1] = X[5]*self.sig_OU[1]
            F = A
            Q = G @ G.T
            return F,Q

        def fun_mean(t, X, u):
            f = self.get_drift(t,X,u)
            return f

        def fun_var(t, P):
            """P has to be a 1D array for solve_ivp, so reshaping is done inside fun. G and F are assumed constant at the initival values."""
            # dV/dF = \sigma^2/2 - a \ln(F) - a + a\bar{Y}
            P = P.reshape((6,6))
            dPdt = F @ P + P @ F.T + Q
            return dPdt.reshape(-1)
        
        def get_dHdX(t, X):
            dHdX = np.zeros((4,6))
            dHdX[:4,:4] = np.diag(1 / (c.rho * c.A))
            return dHdX

        X_hat = X0_hat
        P_hat = P0_hat
        if R is None:
            R = np.diag(np.full((4,),self.sig_v**2))
        
        X_est = np.zeros((N,6))
        X_est[0,:] = X_hat

        e = np.zeros((N-1, 4)) # innovation
        S = np.zeros((N-1, 4, 4)) # covariance of innovation
        for k in tqdm(range(N-1)):
            t = tt[k]
            t_next = tt[k+1]
            Y = yy[k]
            # X = np.concatenate((self.height_to_mass(Y), dd[k]))
            u = uu[k]
            sol_mean = sp.integrate.solve_ivp(
                fun=fun_mean, t_span=(t, t_next), y0=X_hat, t_eval=[t_next], args=(u,),
                rtol=1e-6, atol=1e-8
            )
            X_tilde = sol_mean.y.flatten()
            
            A,_,_,_ = self.get_jacobians(t,X_hat)
            F,Q = precompute_var(X_hat)
            sol_var = sp.integrate.solve_ivp(
                fun=fun_var, t_span=(t, t_next), y0=P_hat.reshape(-1), t_eval=[t_next],
                # rtol=1e-6, atol=1e-8
            )
            P_tilde = sol_var.y.reshape((6,6))
            H = get_dHdX(t,X_hat)

            # Kalman gain:
            S[k] = H @ P_tilde @ H.T + R
            K = sp.linalg.solve(S[k].T, (P_tilde @ H.T).T).T
            # K = P_tilde @ H.T @ np.linalg.inv(H @ P_tilde @ H.T + R)

            e[k] = Y - self.mass_to_height(X_tilde[:4])
            X_hat = X_tilde + K @ e[k]
            P_hat = (np.eye(6) - K @ H) @ P_tilde

            X_est[k+1,:] = X_hat
        
        if plot:
            h_est = self.mass_to_height(X_est[:,:4])
            fig,ax = plt.subplots(4,1, figsize=(10,10))
            for i in range(4):
                ax[i].plot(tt, yy[:,i], label=f'$y_{i+1}$', linestyle='-', marker='.', alpha=0.7)
                ax[i].plot(tt, hh[:,i], label=f'$h_{i+1}$', linestyle='-', alpha=0.7)
                ax[i].plot(tt, h_est[:,i], linestyle='-', label='EKF estimate')
                ax[i].legend()
                ax[i].grid(True)
            
            if filename:
                plt.savefig(filename)
            plt.show()

        return X_est, e, S
    
    def neg_log_likelihood(self, e, S):
        """
        Negative log-likelihood for Gaussian innovations.

        Parameters
        ----------
        e : (N-1, ny) array
            Innovations
        S : (N-1, ny, ny) array
            Innovation covariances

        Returns
        -------
        V : float
            Negative log-likelihood (up to additive constant)
        """
        V = 0.0
        for k in range(e.shape[0]):
            Sk = S[k]
            ek = e[k]

            # Sk = L L^T
            L = np.linalg.cholesky(Sk)

            # log det Sk
            logdet = 2.0 * np.sum(np.log(np.diag(L)))

            # y = L^{-1} e_k
            y = sp.linalg.solve_triangular(L, ek, lower=True)

            # e_k^T S_k^{-1} e_k = y^T y
            quad = y @ y

            V += 0.5 * (logdet + quad)

        return V

    def pem_sweep_R_scalar(
        self,
        t0, tf, h0,
        sigma_grid,
        plot=True, filename=None
    ):
        """
        PEM sweep assuming R = sigma^2 * I.

        Parameters
        ----------
        sigma_grid : array-like
            Candidate values for measurement noise std dev sigma_v

        Returns
        -------
        sigma_grid : ndarray
        V : ndarray
            Negative log-likelihood values
        sigma_hat : float
            MLE estimate of sigma_v
        """
        V = np.zeros(len(sigma_grid))

        for i, sigma in enumerate(sigma_grid):
            R = (sigma**2) * np.eye(4)

            _, e, S = self.extended_kalman_NMPC(
                t0, tf, h0,
                R=R,
                plot=False
            )

            V[i] = self.neg_log_likelihood(e, S)

        idx = np.argmin(V)
        sigma_hat = sigma_grid[idx]
        V_hat = V[idx]

        if plot:
            plt.figure(figsize=(7, 4))
            plt.plot(sigma_grid, V, '-')
            plt.axvline(p.sig_v, linestyle='--', color='k', label=fr'True $\sigma_v = {p.sig_v}$')
            plt.plot(sigma_hat, V_hat, '.', color='red', markersize=14, label=fr'MLE $\sigma_v^\ast = {sigma_hat:.3f}$')
            plt.grid(True)
            plt.xlabel(r'$\sigma_v$ (measurement noise std)')
            plt.ylabel('negative log-likelihood')
            plt.title(r'PEM sweep for $R=\sigma_v^2 I$')
            plt.legend()
            plt.tight_layout()
            if filename:
                plt.savefig(filename)
            plt.show()

        return sigma_grid, V, sigma_hat


    
    def bound_constrained_NMPC(self, t0, tf, h0, Nh, r, Wz, Wu, Wdu, tol=1e-4, maxiter=500, filename=None):
        """
        Results depend highly on tol and maxiter!
        Plotting assumes the weight matrices can be written as $c\cdot I$.
        
        r: reference track, given as piecewise function [(t1,[r11,r12], (t2,[r21,r22])), ...]
        Nh: time horizon T = Nh*dt, dt is sampling time, e.g. time between measurements

        Notation:
          X: states [m1,m2,m3,m4,F_3 or d_1, F_4 or d_2]
          Y: height [h1,h2,h3,h4] obtained from self.mass_to_height([m1,m2,m3,m4])
          Z: [h1,h2]
          EKF: extended Kalman filter
        """
        def construct_r_fun(r):
            """Build a time-callable function to retrieve the reference track point"""
            times, values = zip(*r)
            times = np.asarray(times)
            values = np.asarray(values)
            
            def r_at(t):
                idx = np.searchsorted(times, t, side="right")
                idx = np.clip(idx, 0, len(values) - 1)
                return values[idx]
            return r_at
            
        r_fun = construct_r_fun(r)
        dt = float(self.dt)
        T = Nh*dt
        umin = np.asarray(p.umin, float).reshape((2,))
        umax = np.asarray(p.umax, float).reshape((2,))
        N_steps = int(np.ceil((tf - t0) / dt))

        m0 = self.height_to_mass(h0)
        X0_hat = np.concatenate((m0,self.get_mu_OU_original(self.get_mu_OU(t0))))
        P0_hat = np.eye(6)

        X_hat = X0_hat
        P_hat = P0_hat
        R = np.diag(np.full((4,),self.sig_v**2))
        
        def unpack_var(v):
            x = v[:(4*Nh)].reshape(Nh, 4)
            u = v[(4*Nh):].reshape(Nh, 2)
            return x, u
        def pack_var(x, u):
            return np.concatenate([x.ravel(), u.ravel()])
        
        def horizon_con_fun(X0,v,t):
            out = np.zeros((Nh,4))
            x,u = unpack_var(v)
            x0 = X0[:4]
            d = X0[4:]
            
            f = self.get_rhs(t,x0,u[0],d)
            out[0] = x[0] - x0 - T/Nh * f
            for i in range(Nh-1):
                f = self.get_rhs(t,x[i],u[i+1],d)
                out[i+1] = x[i+1] - x[i] - T/Nh * f
            return out.ravel()
            
        def horizon_obj_fun(v,r,u0):
            x,u = unpack_var(v)
            h = self.mass_to_height(x)
            z = h[:,:2]
            term1 = np.sum((Wz @ (z-r).T)**2)
            term2 = np.sum((Wu @ u.T)**2)
            du = np.vstack((u[0,:]-u0, np.diff(u,axis=0)))
            term3 = np.sum((Wdu @ du.T)**2)
            return term1 + term2 + term3
        
        def optimize_horizon(x0, u0, t):
            """
            In the optimization, the variables are (as required by scipy) the 1D vector [x.ravel, u.ravel()]
            """
            lb = pack_var(
                np.full((Nh,4), 0),
                np.tile(p.umin, (Nh,1))
            )
            ub = pack_var(
                np.full((Nh,4), np.inf),
                np.tile(p.umax, (Nh,1))
            )
            bound_con = sp.optimize.Bounds(lb=lb, ub=ub)
            
            horizon_con = sp.optimize.NonlinearConstraint(lambda v: horizon_con_fun(x0,v,t), 0, 0)
            v0 = pack_var(
                x=np.tile(x0[:4], (Nh,1)),
                u=np.tile(u0, (Nh,1))
            )
            future_times = np.linspace(t, t+T, Nh+1)[1:]
            r = r_fun(future_times)
            res = sp.optimize.minimize(
                fun=lambda v: horizon_obj_fun(v,r,u0),
                x0=v0,
                bounds=bound_con,
                constraints=horizon_con,
                method='SLSQP',
                tol=tol,
                options={'maxiter': maxiter}
            )
            if not res.success:
                raise ValueError(res.message)
            
            x,u = unpack_var(res.x)
            return u[0]
        
        def simulate_one_step(t, m_true, u):
            d_true = self.get_disturbance(t)
            m_next = self.solve_step(t, m_true, u, d_true, dt=dt)
            v_next = self.get_measurement_noise(t + dt)
            y_next = np.clip(self.mass_to_height(m_next) + v_next, 0, None)
            return m_next, y_next
        
        def EKF_predict_one_step(t, X_hat, P_hat, u):
            def precompute_var(X):
                """Precomputed matrices for fun_var: dP/dt = FP(t) + P(t)F^T + Q, Q=GG^T"""
                G = np.zeros((6,2))
                G[4,0] = X[4]*self.sig_OU[0]
                G[5,1] = X[5]*self.sig_OU[1]
                F = A
                Q = G @ G.T
                return F,Q

            def fun_mean(t, X, u):
                f = self.get_drift(t,X,u)
                return f

            def fun_var(t, P):
                """P has to be a 1D array for solve_ivp, so reshaping is done inside fun. G and F are assumed constant at the initival values."""
                # dV/dF = \sigma^2/2 - a \ln(F) - a + a\bar{Y}
                P = P.reshape((6,6))
                dPdt = F @ P + P @ F.T + Q
                return dPdt.reshape(-1)
            
            t_next = t + dt
            sol_mean = sp.integrate.solve_ivp(
                fun=fun_mean, t_span=(t, t_next), y0=X_hat, t_eval=[t_next], args=(u,),
                rtol=1e-6, atol=1e-8
            )
            X_tilde = sol_mean.y.flatten()
            A,_,_,_ = self.get_jacobians(t,X_hat)
            F,Q = precompute_var(X_hat)
            sol_var = sp.integrate.solve_ivp(
                fun=fun_var, t_span=(t, t_next), y0=P_hat.reshape(-1), t_eval=[t_next],
                # rtol=1e-6, atol=1e-8
            )
            P_tilde = sol_var.y.reshape((6,6))
            return X_tilde, P_tilde
        
        def EKF_measurement_update(Y, X_tilde, P_tilde, X_hat):
            def get_dHdX(t, X):
                dHdX = np.zeros((4,6))
                dHdX[:4,:4] = np.diag(1 / (c.rho * c.A))
                return dHdX
            
            H = get_dHdX(t,X_hat)
            K = sp.linalg.solve((H @ P_tilde @ H.T + R).T, (P_tilde @ H.T).T).T
            X_hat = X_tilde + K @ (Y - self.mass_to_height(X_tilde[:4]))
            P_hat = (np.eye(6) - K @ H) @ P_tilde
            return X_hat, P_hat
        
        t = 0
        u_init = (p.umin + p.umax) / 2
        u = np.zeros((N_steps,2))
        y = np.zeros((N_steps+1,4))
        m = np.zeros((N_steps+1,4))
        m[0] = m0
        X_est = np.zeros((N_steps+1,6))
        X_est[0] = X0_hat
        for i in range(N_steps):
            # compute the matrix $f(\hat{X}_i, U_i)$. \hat{X}_i is fixed, so this is a function of U
            # where \hat{X} = [m_1,m_2,m_3,m_4,F_3,F_4] from the EKF estimate.
            
            # now do the horizon optimization problem over the next N_horizon steps.
            # that is, determine u_0,...,u_{N_h-1} minimizing
            # \sum_{j=0}^N ||z_j-r_j||^2_{W_z} + ||u_j||^2_{W_u} + ||\delta u_j||^2_{W_{\delta u}}
            # s.t. x_0 = \hat{X}_i    --- from EKF
            #      x_{j+1} = x_j + T/N_h f(x_j, u_i), j=0,...,N_h-1
            #      u_min[i] <= u_j[i] <= u_max[i], i=1,2
            u[i] = optimize_horizon(X_hat, u_init if i == 0 else u[i-1], t)

            # simulate the true process for the time dt using the optimized u_0
            # this gives the measurement Y_i, which is used in the EKF to estimate \hat{X}_i
            m[i+1], y[i+1] = simulate_one_step(t, m[i], u[i])


            # EKF predict from t to t+dt using applied control
            X_tilde, P_tilde = EKF_predict_one_step(t, X_hat, P_hat, u[i])
            # EKF update at t+dt with new measurement
            X_hat, P_hat = EKF_measurement_update(y[i+1], X_tilde, P_tilde, X_hat)
            
            X_est[i+1] = X_hat
            t += dt
            
        tt = np.arange(t0, tf+dt, dt)
        hh = self.mass_to_height(m)
        h_est = self.mass_to_height(X_est[:,:4])
        rr = r_fun(tt)
        
        # fig,ax = plt.subplots(6,1,figsize=(10,13),constrained_layout=True)
        # plt.suptitle(f'Bound-constrained NMPC with $N={Nh}$, $T={T}$, $W_z={Wz[0,0]}I$, $W_u={Wu[0,0]}I$, $W_{{\Delta u}}={Wdu[0,0]}I$')
        # ax_idx_u = 4
        # for i in range(4):
        #     if i in [0,1]:
        #         ax[i].step(tt, rr[:,i], where='post', label=f'$r_{i+1}$', color='red', linestyle='--')
        #     ax[i].plot(tt, y[:,i], label=f'$y_{i+1}$', linestyle='-', marker='.', alpha=0.7)
        #     ax[i].plot(tt, hh[:,i], label=f'$h_{i+1}$', linestyle='-', alpha=0.7)
        #     ax[i].plot(tt, h_est[:,i], linestyle='-', label='EKF estimate')
        #     ax[i].grid(True)
        #     ax[i].legend(loc='upper left', bbox_to_anchor=(1.02, 1))
        #     # ax[i].set_xlabel('$t$')
            
        #     # ax[ax_idx_u+i].axhline(umin[i], linestyle="--", color='red')
        #     # ax[ax_idx_u+i].axhline(umax[i], linestyle="--", color='red')
        #     if i in [0,1]:
        #         ax[ax_idx_u+i].plot([t0, tf], [umin[i], umin[i]], linestyle="--", color="red", label=f'Bounds on $u_{i+1}$')
        #         ax[ax_idx_u+i].plot([t0, tf], [umax[i], umax[i]], linestyle="--", color="red")
        #         ax[ax_idx_u+i].step(tt, np.r_[u[:,i],u[-1,i]], where='post', label=f'$u_{i+1}$')
        #         ax[ax_idx_u+i].grid(True)
        #         ax[ax_idx_u+i].legend(loc='upper left', bbox_to_anchor=(1.02, 1))
        #         if i == 1:
        #             ax[ax_idx_u+i].set_xlabel('$t$')

        ###########
        # --- disturbance history (piecewise) ---
        # Disturbance history (piecewise)
        td_hist, d_hist = self.get_disturbance_hist()   # td_hist: (K,), d_hist: (K,2)

        fig, ax = plt.subplots(3, 1, figsize=(10, 11), constrained_layout=True)
        plt.suptitle(
            f'Bound-constrained NMPC with $N={Nh}$, $T={T}$, '
            f'$W_z={Wz[0,0]}I$, $W_u={Wu[0,0]}I$, $W_{{\\Delta u}}={Wdu[0,0]}I$'
        )

        # Base colors for channel 1 and 2
        c1, c2 = 'C0', 'C1'

        # Make r a "lighter shade" of the corresponding channel color using alpha
        # (same hue, more see-through) while keeping h more solid.
        alpha_r = 0.8
        alpha_y = 0.45

        # ---------- Subplot 1: outputs (h1,h2) + measurements (y1,y2) + references (r1,r2) ----------
        ax_y = ax[0]

        # h (solid, strong)
        ax_y.plot(tt, hh[:, 0], color=c1, linestyle='-', linewidth=2.2, label=r'$h_1$')
        ax_y.plot(tt, hh[:, 1], color=c2, linestyle='-', linewidth=2.2, label=r'$h_2$')

        # y (dots only, transparent)
        ax_y.plot(tt, y[:, 0], color=c1, linestyle='None', marker='.', alpha=alpha_y, label=r'$y_1$')
        ax_y.plot(tt, y[:, 1], color=c2, linestyle='None', marker='.', alpha=alpha_y, label=r'$y_2$')

        # r (dashed, lighter via alpha)
        ax_y.step(tt, rr[:, 0], where='post', color=c1, linestyle='--', linewidth=2.0, alpha=alpha_r, label=r'$r_1$')
        ax_y.step(tt, rr[:, 1], where='post', color=c2, linestyle='--', linewidth=2.0, alpha=alpha_r, label=r'$r_2$')

        ax_y.grid(True)
        ax_y.set_ylabel('outputs')
        ax_y.legend(loc='upper left', bbox_to_anchor=(1.02, 1))

        # ---------- Subplot 2: inputs u1/u2 with bounds ----------
        ax_u = ax[1]

        # bounds (one label each to avoid duplicate legend entries)
        for i in [0, 1]:
            ax_u.plot([t0, tf], [umin[i], umin[i]], linestyle="--", color="red",
                    alpha=0.8)
            ax_u.plot([t0, tf], [umax[i], umax[i]], linestyle="--", color="red",
                    alpha=0.8)

        # inputs (step)
        ax_u.step(tt, np.r_[u[:, 0], u[-1, 0]], where='post', color=c1, label=r'$u_1$')
        ax_u.step(tt, np.r_[u[:, 1], u[-1, 1]], where='post', color=c2, label=r'$u_2$')

        ax_u.grid(True)
        ax_u.set_ylabel('$u$')
        ax_u.legend(loc='upper left', bbox_to_anchor=(1.02, 1))

        # ---------- Subplot 3: disturbances d1/d2 ----------
        ax_d = ax[2]

        ax_d.step(td_hist, d_hist[:, 0], where='post', color=c1, label=r'$d_1$')
        ax_d.step(td_hist, d_hist[:, 1], where='post', color=c2, label=r'$d_2$')

        ax_d.grid(True)
        ax_d.set_xlabel('$t$')
        ax_d.set_ylabel('$d$')
        ax_d.legend(loc='upper left', bbox_to_anchor=(1.02, 1))

        # Optional: force same x-limits everywhere
        for a in ax:
            a.set_xlim(t0, tf)

        if filename:
            plt.savefig(filename)

        plt.show()
        ###########
