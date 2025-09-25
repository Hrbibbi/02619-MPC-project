
import numpy as np
import matplotlib.pyplot as plt
from D_NL_MODEL import four_tank_D
from S_NL_MODEL import four_tank_S
from SDE_NL_MODEL import four_tank_SDE
from params import *
def run_piecewise_open_loop(model, segments, x0, dt, flag="P"):
    if hasattr(model, "KP"): model.KP[...] = 0.0
    if hasattr(model, "KI"): model.KI[...] = 0.0
    if hasattr(model, "KD"): model.KD[...] = 0.0
    if hasattr(model, "I"):  model.I[...]  = 0.0

    t_all = []
    m_all = []
    u_all = []

    x_init = x0.copy()
    first = True
    for (ts, te, ubar_seg) in segments:
        model.ubar = np.asarray(ubar_seg, dtype=float)

        t_hist, m_hist, u_hist = model.simulate(ts, te, x_init, flag)

        if first:
            t_all.append(t_hist)
            m_all.append(m_hist)
            u_all.append(u_hist)
            first = False
        else:
            t_all.append(t_hist[1:])
            m_all.append(m_hist[1:])
            u_all.append(u_hist[1:])

        x_init = m_hist[-1]

    t_all = np.concatenate(t_all, axis=0)
    m_all = np.vstack(m_all)
    u_all = np.vstack(u_all)
    return t_all, m_all, u_all

def step_plot_inputs_outputs(T, U, H, title):
    fig, axs = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

    axs[0].step(T, U[:,0], where="post", label="u1")
    axs[0].step(T, U[:,1], where="post", label="u2")
    axs[0].set_ylabel("Inputs")
    axs[0].legend()
    axs[0].grid(True, alpha=0.3)

    axs[1].plot(T, H[:,0], label="h1")
    axs[1].plot(T, H[:,1], label="h2")
    axs[1].set_xlabel("time [s]")
    axs[1].set_ylabel("Heights")
    axs[1].legend()
    axs[1].grid(True, alpha=0.3)

    fig.suptitle(title)
    plt.tight_layout()
    plt.show()

segments = [
    (t0,   400.0, np.array([100.0, 300.0])),
    (400.0,800.0, np.array([400.0, 600.0])),
    (800.0,te,    np.array([250.0, 400.0])),
]


sde = four_tank_SDE(gamma, F0, a, A, rho, ubar, KP.copy(), KI.copy(), I.copy(),
                    KD.copy(), umin, umax, hbar, Q, Rvv, dt)


stoch = four_tank_S(gamma, F0, a, A, rho, ubar, KP.copy(), KI.copy(), I.copy(),
                    KD.copy(), umin, umax, hbar, d_p, Rvv, dt)

det = four_tank_D(gamma, F0, a, A, rho, ubar, KP.copy(), KI.copy(), I.copy(),
                  KD.copy(), umin, umax, hbar, d_c, dt)

# ----- Simulation ---------------
T_D,   M_D,   U_D   = run_piecewise_open_loop(det,   segments, x0, dt, flag="P")
T_S,   M_S,   U_S   = run_piecewise_open_loop(stoch, segments, x0, dt, flag="P")
T_SDE, M_SDE, U_SDE = run_piecewise_open_loop(sde,   segments, x0, dt, flag="P")

H_D   = det.get_heights(M_D)
H_S   = stoch.get_heights(M_S)
H_SDE = sde.get_heights(M_SDE)

step_plot_inputs_outputs(T_D,   U_D,   H_D,   "Open-loop — Deterministic with piecewise-constant disturbances")
step_plot_inputs_outputs(T_S,   U_S,   H_S,   "Open-loop — Stochastic")
step_plot_inputs_outputs(T_SDE, U_SDE, H_SDE, "Open-loop — SDE")
