import numpy as np
from src.models import Deterministic, StochasticPiecewise, StochasticBrownian
from src.plotting import plot_hist
import src.parameters as p
import src.constants as c
import scipy as sp
import matplotlib.pyplot as plt

def compute_ss(model,x0):
    #Get u by simulating the system with no controltype
    #model.simulate(p.t0, p.tf, p.h0, ctrl_type="")
    def wrap_rhs(m):
        d = model.get_disturbance(0) 
        #d = np.array([0,0])
        return model.get_rhs(0,m,d)
    sol = sp.optimize.root(wrap_rhs,x0)
    h_ss = sol.x/(c.rho*c.A)
    return sol.x, h_ss


def make_step_response(model,x_ss,percentage,varidx):
    base_u = np.asarray(model.get_ubar(p.t0), float)
    step_u = base_u.copy()
    step_u[varidx] = base_u[varidx] * (1.0 + float(percentage))
    model.ubar_type = 'piecewise'
    model.set_ubar_piecewise([
        (p.t0,       base_u),
        (p.tf//10,  step_u),
    ])
    model.simulate(p.t0, p.tf, x_ss, ctrl_type="")
    return model.hist
    




#noise_levels = np.array([20,100,250])
#percentage_levels = np.array([0.10,0.25,0.50])
#for sig_lvl in noise_levels:
#    d_determ = np.full((2,), sig_lvl)
#    deterministic_step = Deterministic(p.dt, p.zbar, p.ubar, d_determ)
##    model = deterministic_step
#    x0 = 5000*np.ones(4)
#    m_ss, h_ss = compute_ss(model,x0)
#    make_step_response(model,h_ss,0.10,0)
#    plot_hist(model, f"10% step response", plot_disturbance=True, filename=None)
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.pyplot as plt
import numpy as np

def plot_deterministic_steps(percentage_levels, varidx=0, disturbance_level=100,
                                   x0=None, normalized=False, cell_size=(8.5, 4.5), dpi=200):
    if x0 is None:
        x0 = np.zeros(4)

    d_determ = np.full((2,), disturbance_level)
    model_ss = Deterministic(p.dt, p.zbar, p.ubar, d_determ)
    _, h_ss = compute_ss(model_ss, x0)
    base_u = np.asarray(model_ss.get_ubar(p.t0), float)

    fig, (ax_y, ax_u) = plt.subplots(1, 2, figsize=cell_size, dpi=dpi)
    
    ls_map = {
        0.10: '-',             
        0.25: (0, (10, 4)),            
        0.50: '--',    
    }
    color_map = {0.10: '#1f77b4', 0.25: '#2ca02c', 0.50: '#9467bd'}
    legends_done = False
    du_for_label = {}
    

    for pct in percentage_levels:

        model = Deterministic(p.dt, p.zbar, p.ubar, d_determ)


        step_u = base_u.copy()
        step_u[varidx] = base_u[varidx] * (1.0 + float(pct))
        model.ubar_type = 'piecewise'
        model.set_ubar_piecewise([
            (p.t0, base_u),
            (p.tf//10, step_u),
        ])
        model.simulate(p.t0, p.tf, h_ss, ctrl_type="")

        t = model.hist['t']
        y = model.hist['y'] if 'y' in model.hist else model.hist['h']
        u = model.hist['u']
        
        if normalized:
            du = base_u[varidx] * float(pct)
            if np.isclose(du, 0.0):
                raise ValueError("Baseline u is zero; a percentage step yields Δu=0. Set a nonzero baseline.")
            y_plot = (y - np.tile(h_ss, (y.shape[0], 1))) / du
            ylab = r"$\Delta y / \Delta u$"
            u_plot = (u-base_u[varidx]) / du
        else:
            du = None
            y_plot = y
            ylab = "Output (height)"

        du_for_label[pct] = du

        ls = ls_map.get(float(pct), '-')  
        #, label=f"{int(pct*100)}% – h_1"
        ax_y.plot(t, y_plot[:, 0],color="orange", linestyle=ls)
        ax_y.plot(t, y_plot[:, 1],color="blue", linestyle=ls)
        #, label=f"{int(pct*100)}% – h_2"
        #if normalized:
        #    pass
        #else:
        #    ax_u.plot(t, u[:, varidx],color="orange", linestyle=ls, label=f"{int(pct*100)}% on u1")

    if normalized:
        #ax_u.plot(t, u_plot[:,varidx],color="orange", label="Step on u1")
        ax_u.plot(t, (u[:, 0]-base_u[0])/du,color="orange", label="Step on u1")
        ax_u.plot(t, (u[:, 1]-base_u[1])/du ,color="blue", label="Step on u2")
    else:    
        ax_u.plot(t, u[:, 1], color="blue", label=f"{0} % on u2")
        ax_u.plot(t, u[:, 0], color="orange", label=f"{0} % on u1")

    ax_y.set_title(f"Deterministic normalized steps on u{varidx+1}" if normalized
                   else f"Deterministic steps on u{varidx+1}")
    ax_y.set_xlabel("Time [s]")
    ax_y.set_ylabel(ylab)
    ax_y.grid(True, alpha=0.3)
    #ax_y.legend(ncol=2)
    var_handles = [
    plt.Line2D([], [], color="orange", lw=2, label="h1"),
    plt.Line2D([], [], color="blue", lw=2, label="h2")
    ]

    step_handles = [
        plt.Line2D([], [], color="k", linestyle=ls_map[p], lw=1, label=f"{int(p*100)}%")
        for p in percentage_levels
    ]
    all_handles = var_handles + step_handles

    ax_y.legend(
        handles=all_handles,
        title="Vars/Step sizes",
        loc="upper left",
    )

    ax_u.set_title(f"Inputs steps u_0")
    ax_u.set_xlabel("Time [s]")
    ax_u.set_ylabel("u")
    ax_u.grid(True, alpha=0.3)
    ax_u.legend(loc="upper left")
    

    fig.tight_layout(pad=0.25)
    return fig

percentage_levels = np.array([0.10, 0.25, 0.50])
fig = plot_deterministic_steps(
    percentage_levels=percentage_levels,
    varidx=1,                  
    disturbance_level=100,     
    x0=np.zeros(4),        
    normalized=True,            
    cell_size=(8.2, 3.6),
    dpi=100
)
plt.show()
