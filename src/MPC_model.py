import numpy as np
import numpy.linalg as npL
import scipy as sp
import src.constants as c
import src.parameters as p
from src.models import FourTank, StochasticBrownian, StochasticPiecewise
from qpsolvers import Problem, solve_problem
#FT here should probably be a model like StochasticBrownian or Determinisc??
class FourTank_MPC:
    def __init__(self, FT: FourTank, m_ss, u_ss,d_ss,R,Wz,Wu,Wdu,Ws,Wt,I_C,O_C,static_kf=False,MPC_type=None,use_kf=True,NL_sim=False):
        #SOU is a vector of soft output constraints vectors
        self.FT = FT
        self.dt = FT.dt
        self.Wz = Wz
        self.Wu = Wu    
        self.Wdu = Wdu
        self.Ws = Ws
        self.Wt = Wt
        self.I_C = I_C
        self.O_C = O_C
        self.m_ss = np.reshape(m_ss,(len(m_ss),-1))
        self.u_ss = np.reshape(u_ss,(-1,1))
        self.SDE = False
        self.Q = None
        self.SDE = True if isinstance(FT,StochasticBrownian) else False
        self.pois = True if isinstance(FT, StochasticPiecewise) else False
        self.stoch = self.SDE or self.pois
        self.use_kf = use_kf
        self.R = R
        self.Rr = np.linalg.cholesky(self.R)
        self.L = None
        self.NL_sim = NL_sim
        
        if self.I_C:
            SIC = np.broadcast_to(
                np.stack([p.umin, p.umax, p.dumin, p.dumax])[:, None, :],
                (4, p.nt, 2)
            )
            self.umin = SIC[0]
            self.umax = SIC[1]
            self.dumin = SIC[2]
            self.dumax = SIC[3]
        if self.O_C:
            self.ocmax = p.ocmax
            self.ocmin = p.ocmin
        
        #Linearize fourtank system to get continious time matrices
        Ac,Bc,Cc,Gdc,Gbc,Dc = self.FT.linearize_fourtank(m_ss,u_ss,self.SDE)
        
        
        #Compute discrete time matrices
        self.ds = Discrete_system(Ac,Bc,Cc,Gdc,Dc,Gbc,self.dt)
        if Gbc is not None:
            self.Q = self.discretize_process_noise(Ac, Gbc, self.dt)
            self.L = np.linalg.cholesky(self.Q)
        else:
            self.Q = np.zeros((self.FT.n,self.FT.n))
            self.L = self.Q
        
        #Initialize MPC system
        self.MPC_type = MPC_type
        self.controller = None
        
        #Log history
        self.hist = self.FT.hist
        
        #Initial kf
        #x0 = np.zeros((Ac.shape[0], 1))-self.m_ss
        x0 = np.zeros((Ac.shape[0], 1))
        P0 = np.eye(Ac.shape[0]) * 1.0
        if self.use_kf:
            if static_kf:
                self.kf = StaticKF(self.ds, x0, P0, Q=self.Q, R=R,rng=self.FT.rng)
            else:
                self.kf = DynamicKF(self.ds, x0, P0, Q=self.Q, R=R,rng=self.FT.rng)
        else:
            self.kf = None
                
        self.y_ss = np.reshape(self.ds.get_y(self.m_ss,self.u_ss,np.zeros((self.FT.n,1))),(self.FT.n,-1))
        self.z_ss = self.ds.get_z(self.m_ss)
        self.d_ss = np.reshape(d_ss, (1, -1))
        #np.zeros((1, self.ds.Gdd.shape[1])) 
    
    @staticmethod
    def discretize_process_noise(A, Gw, dt):
        n = A.shape[0]
        M = np.zeros((2*n, 2*n))
        M[:n, :n]     =  -A
        #M[:n, n:]     =  Gw @ Qc @ Gw.T
        M[:n, n:]     =  Gw @ Gw.T
        M[n:, n:]     = A.T
        Md = sp.linalg.expm(M * dt)
        #Ad = Md[:n, :n]
        #Qx = Ad @ Md[:n, n:]    
        Phi22 = Md[n:, n:]
        Phi12 = Md[:n, n:]    
        Q = Phi22.T @ Phi12
        return np.round(Q,decimals=7)
    
    def get_measurement_noise(self):
        #return self.rng.normal(scale=self.R)
        #return np.reshape(self.rng.multivariate_normal(mean=np.zeros(self.R.shape[0]),cov=self.R),(self.R.shape[0],-1))
        e = self.FT.rng.normal(size=self.R.shape[0])
        v = self.Rr @ e
        v = np.reshape(v, (self.R.shape[0],-1))
        return v
    
    def get_process_noise(self):
        e = self.FT.rng.normal(size=self.Q.shape[0])
        w = self.L @ e
        w = np.reshape(w, (self.Q.shape[0],-1))
        #w = #np.reshape(self.rng.multivariate_normal(mean=np.zeros(self.Q.shape[0]),cov=self.Q),(self.Q.shape[0],-1))
        w[:4] = 0
        return w
    
    @staticmethod
    def qp_solve(H,g,l,u,A,bl,bu,xinit=None):
        H = (H + H.T) * 0.5 # Numerical stabilty?
        if A is not None:
            G = np.concatenate([A, -A], axis=0)
            h = np.concatenate([bu, -bl]).ravel()
        else:
            G, h = None, None
        if xinit is not None:
            problem = Problem(P=H, q=g.ravel(), G=G, h=h, A=None, b=None,lb=None if l is None else l.ravel(),
                                                                        ub=None if u is None else u.ravel(), xinit=xinit)
        else:
            problem = Problem(P=H, q=g.ravel(), G=G, h=h, A=None, b=None,lb=None if l is None else l.ravel(),
                                                                        ub=None if u is None else u.ravel())
        sol = solve_problem(problem, solver="gurobi", verbose=False)
        x = sol.x                 # primal solution
        info_like = {
            "status": sol.extras.get("status"),
            "message": sol.extras.get("message"),
            "iterations": sol.extras.get("iterations") or sol.extras.get("itercount"),
            "runtime_sec": sol.extras.get("runtime") or sol.extras.get("time"),
            "is_optimal": sol.is_optimal(1e-8),
        }
        return x, info_like
    
    def step_NLmodel(self,t,m,d):
        self.FT.u = self.u.ravel()
        #d = self.FT.get_disturbance(t)
        #d = self.FT.F_t(d)
        m = self.FT.solve_step(t, m, d, self.dt)
        return m
    
                
    #When we do computations its always in mass units
    def kf_step(self,y_meas,u,d,w=None):
        x_pred, P_pred = self.kf.predict(u, d,w)     
        x_filt, P_filt = self.kf.update(y_meas, u)
        z_hat = self.ds.get_z(x_filt)       
        return x_pred, P_pred, x_filt, P_filt, z_hat  
    
    def simulate(self, t0, tf, h0, T):
        """
        Closed Loop Simulation of the four tank system in the time-interval [t0, tf] w/
        MPC controller
        """        
        self.t0 = self.FT.t0 = t0
        self.tf = self.FT.tf = tf
        self.N = int(np.ceil((tf-t0) / self.dt))
        t = float(t0)

        h0[:4] = self.FT.height_to_mass(h0[:4])
        m0 = np.reshape(h0,(len(h0),-1))
        m = np.asarray(m0, dtype=float).copy()
        if self.kf is not None:
            self.kf.x = np.reshape(m0-self.m_ss,(len(m0),-1))
        x_plant_dev = np.reshape(m0-self.m_ss,(len(m0),-1))
        x_plant = m0
        #u0 = np.atleast_1d(self.FT.get_ubar(t0)).astype(float)
            
        
        self.u = u0 = np.reshape(self.FT.u,(-1,1))
        u_dev = self.u-self.u_ss
        y = self.ds.get_y(x_plant, self.u, None)
        y_dev = y - self.y_ss
        nx, nu = m.size, u0.size
        if self.SDE:
            nx -= 2
        nz = self.ds.Cdz.shape[0]
        md = self.ds.Gdd.shape[1]
        #For the innovation form model this term is important? I think maybe this term is only applied when we have 
        #Innovation form since the covariance between S = cov(w,v) > 0
        #However since we have a disturbance term this what should include both the zeros and the new disturbance term so we just set what to be 
        #The disturbance
        if self.MPC_type == "linear":
            self.controller = LMPC_controller(self.ds, self.Wz, self.Wu, self.Wdu,self.Ws,self.Wt,T,nx,nz)
        elif self.MPC_type == "Economic":
            self.controller = EMPC_controller(self.ds,self.Wz,self.Wu,self.Wdu,self.Ws,self.Wt)
        elif self.MPC_type == "nonlinear":
            self.controller = NMPC_controller(self.FT, self.Wz, self.Wu, self.Wdu,self.Ws,self.Wt, self.dt, nlp_solver=...)
        elif self.MPC_type == "":
            pass
        else:
            raise ValueError("Unknown MPC_type")
        
        
        what = np.zeros((self.FT.n-2, 1))
        td_hist, d_hist = self.FT.construct_disturbance_disc(self.N)
        
        self.hist.setdefault('qp_status', np.empty(self.N+1, dtype=object))
        
        self.hist['zbar'] = np.empty((self.N + 1, 2), dtype=float)
        self.hist['t'] = np.empty(self.N + 1, dtype=float)
        self.hist['m'] = np.empty((self.N + 1, nx), dtype=float)
        self.hist['y'] = np.empty((self.N + 1, nx), dtype=float) # y is h + noise
        self.hist['u'] = np.empty((self.N + 1, nu), dtype=float)
        self.hist['xhat'] = np.zeros_like(self.hist['m'],dtype=float)
        self.hist['zhat'] = np.zeros((self.N + 1, self.ds.Cdz.shape[0]),dtype=float)
        self.hist['Pdiag'] = np.zeros((self.N + 1, self.ds.Ad.shape[0]),dtype=float)
        self.hist['d'] = np.empty((self.N + 1,2),dtype=float)
        self.hist['d'][0] = m[4:].ravel() if self.SDE else d_hist[0]
        self.hist['t'][0] = t
        self.hist['m'][0] = m[:4].ravel()
        self.hist['u'][0] = u0.ravel()
        self.hist['zbar'][0] = np.atleast_1d(self.FT.get_zbar(t))[:2]
        #if self.log_NL:
        m_nl = m0[:4, 0].copy() 
        self.hist['m_nl'] = np.empty((self.N + 1, nx), dtype=float)    # nonlinear masses
        self.hist['h_nl'] = np.empty((self.N + 1, nx), dtype=float)
        self.hist['y_nl'] = np.empty((self.N + 1, nx), dtype=float)
        self.hist['m_nl'][0] = m_nl.ravel()
        self.hist['h_nl'][0] = FourTank.mass_to_height(m_nl).ravel()
        
            
        #Construct matrices for controller
        self.k = 0
        v = None

        while (self.k < self.N) and (t < tf):
            dt = min(self.dt, tf - t)
            
            #Noises
            if T+self.k > self.N:
                T = self.N-self.k
            
            
            d = d_hist[self.k]
            d_dev = np.reshape(d-self.d_ss,(self.FT.n-2,-1))
            
            
            #Maybe we use np.vstack here instead of concatenate
            zbar_stack = np.reshape(np.concatenate([self.FT.get_zbar(t+(i+1)*self.dt) for i in range(T)],axis=0), (-1,2))
            #print(zbar_stack)
            Ubark = np.reshape(np.concatenate([self.FT.get_ubar(t+(i+1)*self.dt) for i in range(T)],axis=0),(-1,2))
            self.hist['zbar'][self.k] = np.atleast_1d(zbar_stack[0][:2])
            zbar_stack_dev = np.reshape(zbar_stack-self.z_ss[:,0],(-1,1))
            Ubark_dev = np.reshape(Ubark-self.u_ss[:,0],(-1,1))
            #Just set what to be disturbance since S=0
            what = d_dev
            

            Umin_dev = np.reshape(self.umin[self.k:self.k+T]-self.u_ss[:,0],(-1,1)) if self.I_C else None
            Umax_dev = np.reshape(self.umax[self.k:self.k+T]-self.u_ss[:,0],(-1,1)) if self.I_C else None
            duMin_dev = np.reshape(self.dumin[self.k:self.k+T]-self.u_ss[:,0],(-1,1)) if self.I_C else None
            duMax_dev = np.reshape(self.dumax[self.k:self.k+T]-self.u_ss[:,0],(-1,1)) if self.I_C else None
            Zmin_dev = np.reshape((zbar_stack - self.ocmin)-self.z_ss[:,0],(-1,1)) if self.O_C else None
            Zmax_dev = np.reshape((zbar_stack + self.ocmax)-self.z_ss[:,0],(-1,1)) if self.O_C else None
            
            if self.controller is not None:
                u_seq_dev, info = self.controller.control(self.kf.x[:4], what,T, zbar_stack_dev,u_dev,Ubark_dev,Umin_dev,Umax_dev,duMin_dev,duMax_dev,Zmin_dev,Zmax_dev)
                self.u = np.reshape(u_seq_dev[:nu],(-1,1)) + self.u_ss
                
            u_dev = self.u-self.u_ss
            
            #w = self.kf.get_process_noise() if self.SDE else None
            w_plant = self.get_process_noise() if self.SDE else None
            w_filter = self.get_process_noise() if self.SDE and self.kf is not None else None
            v = self.get_measurement_noise() if self.stoch else None
        
            #Get measurement noise,noisy measurement and plant disturbance
            x_plant_dev = self.ds.step(x_plant_dev,u_dev,d_dev,w_plant)
            x_plant = x_plant_dev+np.reshape(self.m_ss,(len(self.m_ss),-1))
            
            y_l = self.ds.get_y(x_plant,self.u,v)
            self.hist['y'][self.k] = np.clip(y_l[:,0], 0, None)
            #m = x_plant
            m_l = x_plant
            #self.ds.get_y(x_plant_dev,u_dev,v)      
            y_dev_lin = y_l - self.y_ss
            
            #if self.log_NL:
            #    self.FT.k = self.k
            #    if self.SDE:
            #        d_phy = m[4:]
            #    else:
            #        d_phy = d
            self.FT.k = self.k
            d_phy = m_l[4:]
            m_nl = self.step_NLmodel(t,m_nl,d_phy.ravel())
            self.hist['m_nl'][self.k] = m_nl.ravel()
            y_nl = FourTank.mass_to_height(m_nl)[None,:].T + self.get_measurement_noise() if self.stoch else FourTank.mass_to_height(m_nl)
            self.hist['y_nl'][self.k] = np.clip(y_nl,0,None).ravel()
            
            if self.NL_sim:
                m = m_nl
                y = y_nl  
                y_dev = y - self.y_ss
            else:
                m = x_plant
                y = y_l
                y_dev = y_dev_lin
            
            if self.kf is not None:
                x_pred, P_pred, x_filt, P_filt, z_hat = self.kf_step(y_dev,u_dev,d_dev,w_filter)
                self.kf.x = x_filt
                self.kf.P = P_filt
                self.hist['xhat'][self.k] = x_filt[:4].ravel()
                self.hist['zhat'][self.k] = z_hat.ravel()
                self.hist['Pdiag'][self.k] = np.diag(P_filt)
            else:
                self.hist['xhat'][self.k] = x_plant_dev[:4].ravel()
                
            if np.any(m < 0):
                print('negative mass, yo?')
            t = t + dt

            self.k += 1
            self.hist['t'][self.k] = t
            self.hist['m'][self.k] = m_l[:4].ravel()
            self.hist['u'][self.k] = np.atleast_1d(self.u.ravel()).astype(float)
            self.hist['d'][self.k] = m_l[4:].ravel() if self.SDE else d
            if self.controller is not None:
                self.hist['qp_status'][self.k] = info['status']
        
        self.hist['h'] = FourTank.mass_to_height(self.hist['m'])
        self.hist['y'][-1] = np.clip(y + self.get_measurement_noise() if self.stoch else y, 0, None).ravel()
        self.hist['zbar'][-1] = np.atleast_1d(self.FT.get_zbar(self.tf))[:2]
        self.hist['td'] = td_hist
        #if self.log_NL:
        self.hist['m_nl'][-1] = m_nl.ravel()
        self.hist['h_nl'] = FourTank.mass_to_height(self.hist['m_nl'])  
        self.hist['y_nl'][-1] = np.clip(FourTank.mass_to_height(m_nl)[None, :].T + self.get_measurement_noise() if self.stoch else FourTank.mass_to_height(m_nl), 0, None).ravel()




    
    
        
class Discrete_system():
    def __init__(self, Ac,Bc,Cc,Gc,Dc,Gb,dt):
        self.Ad, self.Bd, self.Cd, self.Gdd,self.Dd = Discrete_system.get_discrete(Ac,Bc,Cc,Gc,dt,Dc)
        self.Cdz  = self.Cd[0:2,:]
        self.Ddz = np.zeros((self.Cdz.shape[0], self.Dd.shape[1]))
        self.Gb = Gb
    @staticmethod
    def get_discrete(A,B,C,G,dt,D=None):
        mu = B.shape[1]
        md = G.shape[1]
        py = C.shape[0]
        B_aug = np.hstack([B,G])
        D_aug = np.zeros((py, mu+md))
        Ad, Bd_aug, Cd, Dd_aug, _ = sp.signal.cont2discrete((A, B_aug,C, D_aug), dt, method='zoh')
        Bd = Bd_aug[:, :mu]  
        Gd = Bd_aug[:, mu:mu+md]           
        Dd   = Dd_aug[:, :mu]
        #Cdz = self.C_d[0:2,:]
        return Ad, Bd, Cd, Gd, Dd
    def step(self,x,u,d,w):
        if w is None:
            return self.Ad @ x + self.Bd @ u +self.Gdd @ d
        return self.Ad @ x + self.Bd @ u +self.Gdd @ d +  w
    def get_y(self, x,u,v):
        if v is None:
            return self.Cd @ x + self.Dd @ u
        return self.Cd @ x + self.Dd @ u + v
    def get_z(self, x):
        return self.Cdz @ x
             

class MPC_controller():
    def __init__(self,sys, Wz,Wu,Wdu,Ws,Wt):
        self.sys = sys 
        self.Wz = Wz
        self.Wu = Wu
        self.Wdu = Wdu
        self.Ws1 = Ws[0]
        self.Ws2 = Ws[1]
        self.Wt1 = Wt[0]
        self.Wt2 = Wt[1]
    def control(self, *args, **kwargs):
        raise NotImplementedError

    def construct_MPC_obj(self, *args, **kwargs):
        raise NotImplementedError

    def solve_QP(self, *args, **kwargs):
        raise NotImplementedError
    
        
class LMPC_controller(MPC_controller):
    def __init__(self, ds, Wz, Wu, Wdu,Ws,Wt,T_max,nx,nz):
        super().__init__(ds, Wz, Wu, Wdu,Ws,Wt)
        self.T_max = T_max
        self.nx = nx
        self.nz = nz
        self.Phix, self.Phiw, self.Gamma = self.get_system_matrices(T_max)
        #Construct weight matrices for MPC
        self.I_T = np.eye(T_max, dtype=float)
        
    def build_blocks(self,T):
        #ny, nx = self.sys.Cdz.shape
        md = self.sys.Gdd.shape[1]
        nu = self.sys.Bd.shape[1]
        Phix = np.empty((T * self.nz, self.nx), dtype=self.sys.Ad.dtype)
        Phiw = np.empty((T * self.nz, md), dtype=self.sys.Ad.dtype)
        Hs   = np.empty((T, self.nz, nu), dtype=self.sys.Ad.dtype)

        Q = np.eye(self.nx, dtype=self.sys.Ad.dtype)   # Q = A^i, start with i=0
        for i in range(T):
            rows = slice(i*self.nz, (i+1)*self.nz)

            CzQ = self.sys.Cdz[:self.nz, :self.nx] @ Q           # C * A^i
            Phix[rows, :] = CzQ @ self.sys.Ad[:self.nx,:self.nx]  # C * A^(i+1)
            Phiw[rows, :] = CzQ @ self.sys.Gdd[:self.nx,:md]  # C * A^i * G
            Hs[i, :, :]    = CzQ @ self.sys.Bd[:self.nx,:nu]  # C * A^i * B

            Q = Q @ self.sys.Ad[:self.nx,:self.nx]  # A^i -> A^(i+1)

        return Phix, Phiw, Hs
    
    def get_system_matrices(self,T):        
        #Phix = np.concatenate([self.ds.Cdz@ (npL.matrix_power(self.ds.Ad,i)) for i in range(1,T+1)],axis=0)
        #hiw = np.concatenate([self.ds.Cdz@(npL.matrix_power(self.ds.Ad,i))@self.ds.Gd for i in range(T)],axis=0)
        #Hs = [self.ds.Cdz @ npL.matrix_power(self.ds.Ad,i-1) @ self.ds.Bd for i in range(1,T+1)]
        Phix, Phiw, Hs = self.build_blocks(T)
        L = np.eye(T, k=-1, dtype=int)
        Gamma = sum(np.kron(npL.matrix_power(L, k), Hs[k]) for k in range(T))
        return Phix,Phiw,Gamma
    
    def construct_MPC_obj(self,Rk,bk,Gamma,Lambda,u_prev,I0,Ubark):
        #Classic input constraint MPC
        Wbarz = np.kron(self.I_T, self.Wz)
        ck = Rk-bk
        Hz = (Wbarz @ Gamma).T @ (Wbarz @ Gamma)
        gz = -(Wbarz @ Gamma).T @ Wbarz @ ck
        rhoz = 0.5 * ck.T @ Wbarz.T @ Wbarz @ ck
        Wbaru = np.kron(self.I_T, self.Wu)
        Hu = Wbaru.T @ Wbaru
        gu = - Wbaru.T @ Wbaru @ Ubark
        rhou = 0.5 *( Ubark.T @ Wbaru.T @ Wbaru @ Ubark)
        Wbardu = np.kron(self.I_T, self.Wdu)
        Hdu = (Wbardu @ Lambda).T @ (Wbardu @ Lambda)
        gdu = -(Wbardu @ Lambda).T @ Wbardu @ I0 @ u_prev
        rhodu = 0.5 * (Wbardu @ I0 @ u_prev).T @ (Wbardu @ I0 @ u_prev)
        return Hz+Hu+Hdu, gz+gu+gdu, rhoz+rhou+rhodu

    def solve_QP(self, H,g,rho, Umin, Umax, duMin, duMax,zmin,zmax,Gamma,bk,Lambda,I0,u_prev,T, nu, nz, xinit):
        H = H
        g = g 
        nU = T * nu
        nZ = T * nz
        if zmin is not None and zmax is not None:
            # x = [U; S; T]
            nX = nU + 2 * nZ
        else:
            # x = U only
            nX = nU
        l = None
        u  = None
        A = None
        bl = None
        bu = None
        l = -np.inf * np.ones((nX, 1))
        u =  np.inf * np.ones((nX, 1))
        
        if Umin is not None and Umax is not None:
            #l = Umin
            #u = Umax
            l[:nU, :] = Umin
            u[:nU, :] = Umax
        
        if duMin is not None and duMax is not None:
            A_rate = Lambda
            #A_rate = np.hstack([Lambda,
            #                np.zeros((T*nu, 2*nZ))])
            bl_rate = duMin + I0 @ u_prev
            bu_rate = duMax + I0 @ u_prev
            A  = A_rate
            bl = bl_rate
            bu = bu_rate
        
        if zmin is not None and zmax is not None:
            ZS = np.zeros((T*nz, nZ))
            #A_z  = Gamma
            bl_z = zmin - bk
            bu_z = zmax - bk
            l[nU:nU+nZ,   :] = 0.0   # S
            l[nU+nZ:nX,   :] = 0.0   # T        
            #l_s = np.zeros_like(zmin)
            #u_s = np.ones_like(l_s)*np.inf
            #l_t = np.zeros_like(zmax)
            #u_t = np.ones_like(l_t)*np.inf
            bl_low = np.full_like(bl_z,-np.inf)
            bl_up = np.full_like(bu_z,np.inf,)
            A_low = np.hstack([Gamma,np.eye(nZ),ZS])
            A_up = np.hstack([Gamma, ZS, -np.eye(nZ)])
            
            #if l is None:
            #    l = np.vstack((l_s,l_t))
            #    u = np.vstack((u_s,u_t))
            #else:
            #    l = np.vstack((l,l_s,l_t))
            #    u = np.vstack((u,u_s,u_t))
                
            if A is None:
                A  = np.vstack((A_low,A_up))
                bl = np.vstack((bl_z,bl_low))
                bu = np.vstack((bl_up,bu_z))
            else:
                A_u = np.hstack([A,np.zeros((T*nu, 2*nZ))])
                A = np.vstack((A_u,A_low,A_up))
                bl = np.vstack((bl,bl_z,bl_low))
                bu = np.vstack((bu,bl_up,bu_z))
                #A  = np.vstack((A,  A_z))
                #bl = np.vstack((bl, bl_z))
                #bu = np.vstack((bu, bu_z))
            
            
        x, info = FourTank_MPC.qp_solve(H, g, l, u, A, bl, bu, xinit)
        return x,info   
    
    def extend_output_constraints(self,H,g,nU,nZ):
        #S vars
        Wbars1 = np.kron(self.I_T, self.Ws1)
        Wbars2 = np.kron(self.I_T, self.Ws2)
        Hs = Wbars2.T @ Wbars2
        gs = Wbars1 @ np.ones(Wbars1.shape[0])
        H[nU:nU+nZ, nU:nU+nZ] = Hs
        g[nU:nU+nZ] = gs[None,:].T
        #t vars
        Wbart1 = np.kron(self.I_T, self.Wt1)
        Wbart2 = np.kron(self.I_T, self.Wt2)
        Ht = Wbart2.T @ Wbart2
        gt = Wbart1 @ np.ones(Wbart1.shape[0])
        H[nU+nZ:, nU+nZ:] = Ht
        g[nU+nZ:] = gt[None,:].T
        return H,g
        
    def control(self,xhat, what,T, zbar, uhat,Ubark,Umin,Umax,duMin,duMax,zmin,zmax,xinit=None):
        if T > self.T_max:
            raise ValueError("Requested horizon T larger than precomputed T_max")
        nu = uhat.shape[0]
        nz = self.sys.Cdz.shape[0]
        self.I_T = np.eye(T, dtype=float)
        Phix, Phiw, Gamma = self.Phix[:T*nz], self.Phiw[:T*nz], self.Gamma[:T*nz,:T*nu]
                
        bk = Phix @ xhat + Phiw @ what
        
        I_nu = np.eye(nu)   
        Lambda_s = np.eye(T)
        
        I0 = np.zeros((T*nu, nu))
        I0[:nu,:nu] = np.eye(nu)
        Lambda_s[1:, :-1] += -np.eye(T-1)
        Lambda = np.kron(Lambda_s, I_nu)

        H_u, g_u, rho = self.construct_MPC_obj(zbar, bk, Gamma, Lambda,uhat,I0, Ubark)
        
        if zmin is not None and zmax is not None:
            nU = T * nu
            nZ = T * nz
            nX = nU + 2 * nZ

            H = np.zeros((nX, nX), dtype=H_u.dtype)
            g = np.zeros((nX, 1),    dtype=g_u.dtype)
            H[:nU, :nU] = H_u
            g[:nU, 0]   = g_u.ravel()
            H, g = self.extend_output_constraints(H,g,nU,nZ)
        else:
            H = H_u
            g = g_u
        
        x,info = self.solve_QP(H,g,rho,Umin,Umax,duMin,duMax,zmin,zmax,Gamma,bk,Lambda,I0,uhat,T,nu,nz,xinit)
        return x,info

class NMPC_controller(MPC_controller):
    def __init__(self, sys, Wz, Wu, Wdu,Ws,Wt,T_max,nx,nz):
        super().__init__(sys, Wz, Wu, Wdu,Ws,Wt)
        self.T_max = T_max
        self.nx = nx
        self.nz = nz
        

class EMPC_controller(MPC_controller):
    def __init__(self, sys, Wz, Wu, Wdu,Ws,Wt):
        super().__init__(sys, Wz, Wu, Wdu,Ws,Wt)

      
                        
class KalmanFilter:
    def __init__(self, system: Discrete_system, x0, P0, Q, R,rng):
        self.s = system 
        self.x = x0.copy() 
        self.P = P0.copy()
        self.Q = Q
        #self.L=np.linalg.cholesky(Q)
        self.R = R
        #self.Rr = np.linalg.cholesky(R)
        #self.S = np.cov(Q,R)
        self.S = None
        self.rng = rng

    def predict(self, u,w=None):
        raise(NotImplementedError)

    def update(self, y, u):
        raise(NotImplementedError)


class DynamicKF(KalmanFilter):
    def __init__(self, system, x0, P0, Q, R,rng=None):
        super().__init__(system, x0, P0, Q, R,rng)
            
    def predict(self, u,d,w=None):
        if w is None:
            w = np.zeros((self.s.Ad.shape[0], 1))
        self.x = self.s.step(self.x,u,d,w)
        self.P = self.s.Ad @ self.P @ self.s.Ad.T + self.Q
        return self.x, self.P

    def update(self, y, u):
        yhat = self.s.Cd @ self.x + self.s.Dd @ u
        e = y-yhat
        Re = self.s.Cd @ self.P @ self.s.Cd.T + self.R
        Kfx = self.P @ self.s.Cd.T @ np.linalg.inv(Re)
        if (self.S is not None) and np.any(self.S):
            Kfw = self.S @ np.linalg.inv(Re)
        else:   
            Kfw = None
        self.x = self.x + Kfx @ e
        self.P = (np.eye(self.P.shape[0]) - Kfx @ self.s.Cd) @ self.P
        return self.x, self.P
    
class StaticKF(KalmanFilter):
    def __init__(self, system, x0, P0, Q, R,rng=None):
        super().__init__(system, x0, P0, Q, R,rng)
        #Solve Ricatti (DARE) equation for a global P 
        self.P = sp.linalg.solve_discrete_are(self.s.Ad.T, self.s.Cd.T, Q, self.R)
        self.Re = self.s.Cd @ self.P @ self.s.Cd.T + self.R
        self.Kfx = self.P @ self.s.Cd.T @ np.linalg.inv(self.Re)
    def predict(self, u,d,w=None):
        if w is None:
            w = np.zeros((self.s.Ad.shape[0], 1))
        self.x = self.s.step(self.x,u,d,w)
        return self.x, self.P

    def update(self, y, u):
        yhat = self.s.Cd @ self.x + self.s.Dd @ u
        e = y-yhat
        self.x = self.x + self.Kfx @ e
        return self.x, self.P 
    

