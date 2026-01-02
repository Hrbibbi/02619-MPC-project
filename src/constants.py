import numpy as np

a = np.array([1.2272, 1.2272, 1.2272, 1.2272])
A = np.array([380.1327, 380.1327, 380.1327, 380.1327])
gamma = np.array([0.58, 0.72])
#gamma = np.array([0.45,0.40])
rho = 1.0
g = 981




def construct_cost_poisson(nt, dt, mu_c, sig_c, t_c, seed=0):
    rng = np.random.default_rng(seed)

    mu_c  = np.asarray(mu_c, float)     
    sig_c = np.asarray(sig_c, float)     
    n_u   = mu_c.shape[0]               

    T = (nt - 1) * dt
    event_times = [0.0]
    t = 0.0
    while t <= T:
        t += rng.exponential(scale=t_c)  
        event_times.append(t)
    event_times = np.array(event_times)

    eps = rng.normal(size=(n_u, event_times.size))
    c_vals = mu_c[:, None] + sig_c[:, None] * eps

    c_vals = np.clip(c_vals, 0.0, None)

    t_grid = np.arange(nt) * dt 
    idx = np.searchsorted(event_times, t_grid, side='right') - 1
    series = c_vals[:, idx]
    return series.T
