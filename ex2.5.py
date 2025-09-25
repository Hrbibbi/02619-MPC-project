from D_NL_MODEL import four_tank_D
from S_NL_MODEL import four_tank_S
from SDE_NL_MODEL import four_tank_SDE
from params import *
import numpy as np
import matplotlib.pyplot as plt

def simulate_closed_loop(model, t0, te, x0, flag):
    return model.simulate(t0, te, x0, flag)

def plot_cl(title, T, H, U):
    fig, axs = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

    axs[0].plot(T, H[:,0], label="h1")
    axs[0].plot(T, H[:,1], label="h2")
    axs[0].set_ylabel("Heights [cm]")
    axs[0].legend(); axs[0].grid(True, alpha=0.3)

    axs[1].step(T, U[:,0],where="post", label="u1")
    axs[1].step(T, U[:,1],where="post", label="u2")
    axs[1].set_xlabel("time [s]")
    axs[1].set_ylabel("Inputs")
    axs[1].legend(); axs[1].grid(True, alpha=0.3)

    fig.suptitle(title)
    plt.tight_layout()
    plt.show()


plant_D = four_tank_D(gamma, F0, a, A, rho, ubar, KP.copy(), KI.copy(), I.copy(),
                      KD.copy(), umin, umax, hbar, d_c, dt)

# ----------------- Closed-loop: P / PI / PID -----------------
# P
if False:
    plant_P = four_tank_D(gamma, F0, a, A, rho, ubar, KP.copy(), KI*0.0, I*0.0,
                        KD*0.0, umin, umax, hbar, d_c, dt)
    T_P, M_P, U_P = simulate_closed_loop(plant_P, t0, te, x0, flag="P")
    H_P = plant_P.get_heights(M_P)
    plot_cl("Closed-loop — P control (Deterministic)", T_P, H_P, U_P)

    # PI
    plant_PI = four_tank_D(gamma, F0, a, A, rho, ubar, KP.copy(), KI.copy(), I*0.0,
                        KD*0.0, umin, umax, hbar, d_c, dt)
    T_PI, M_PI, U_PI = simulate_closed_loop(plant_PI, t0, te, x0, flag="PI")
    H_PI = plant_PI.get_heights(M_PI)
    plot_cl("Closed-loop — PI control (Deterministic)", T_PI, H_PI, U_PI)

    # PID
    plant_PID = four_tank_D(gamma, F0, a, A, rho, ubar, KP.copy(), KI.copy(), I*0.0,
                            KD.copy(), umin, umax, hbar, d_c, dt)
    T_PID, M_PID, U_PID = simulate_closed_loop(plant_PID, t0, te, x0, flag="PID")
    H_PID = plant_PID.get_heights(M_PID)
    plot_cl("Closed-loop — PID control (Deterministic)", T_PID, H_PID, U_PID)


if False:
    plant_P = four_tank_S(gamma, F0, a, A, rho, ubar, KP.copy(), KI*0.0, np.zeros_like(I),
    KD*0.0, umin, umax, hbar,d_p, Rvv,dt)
    T_P, M_P, U_P = simulate_closed_loop(plant_P, t0, te, x0, flag="P")
    H_P = plant_P.get_heights(M_P)
    plot_cl("Closed-loop — P control (Stochastic)", T_P, H_P, U_P)

    # PI
    plant_PI = four_tank_S(gamma, F0, a, A, rho, ubar, KP.copy(), KI*0.0, np.zeros_like(I),
    KD*0.0, umin, umax, hbar,d_p, Rvv,dt)
    T_PI, M_PI, U_PI = simulate_closed_loop(plant_PI, t0, te, x0, flag="PI")
    H_PI = plant_PI.get_heights(M_PI)
    plot_cl("Closed-loop — PI control (Stochastic)", T_PI, H_PI, U_PI)

    # PID
    plant_PID = four_tank_S(gamma, F0, a, A, rho, ubar, KP.copy(), KI*0.0, np.zeros_like(I),
    KD*0.0, umin, umax, hbar,d_p, Rvv,dt)
    T_PID, M_PID, U_PID = simulate_closed_loop(plant_PID, t0, te, x0, flag="PID")
    H_PID = plant_PID.get_heights(M_PID)
    plot_cl("Closed-loop — PID control (Stochastic)", T_PID, H_PID, U_PID)


plant_P = four_tank_SDE(
    gamma, F0, a, A, rho, ubar, KP.copy(), KI*0.0, np.zeros_like(I),
    KD*0.0, umin, umax, hbar, Q, Rvv, dt
)

plant_PI = four_tank_SDE(
    gamma, F0, a, A, rho, ubar, KP.copy(), KI.copy(), np.zeros_like(I),
    KD*0.0, umin, umax, hbar, Q, Rvv, dt
)

plant_PID = four_tank_SDE(
    gamma, F0, a, A, rho, ubar, KP.copy(), KI.copy(), np.zeros_like(I),
    KD.copy(), umin, umax, hbar, Q, Rvv, dt
)


T_P, M_P, U_P = simulate_closed_loop(plant_P, t0, te, x0, flag="P")
H_P = plant_P.get_heights(M_P)
plot_cl("Closed-loop — P control (SDE)", T_P, H_P, U_P)

T_PI, M_PI, U_PI = simulate_closed_loop(plant_PI, t0, te, x0, flag="PI")
H_PI = plant_PI.get_heights(M_PI)
plot_cl("Closed-loop — PI control (SDE)", T_PI, H_PI, U_PI)


T_PID, M_PID, U_PID = simulate_closed_loop(plant_PID, t0, te, x0, flag="PID")
H_PID = plant_PID.get_heights(M_PID)
plot_cl("Closed-loop — PID control (SDE)", T_PID, H_PID, U_PID)
