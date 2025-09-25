from D_NL_MODEL import four_tank_D
from S_NL_MODEL import four_tank_S
from SDE_NL_MODEL import four_tank_SDE
from params import *
import numpy as np
import matplotlib.pyplot as plt




plant_S = four_tank_S(gamma, F0, a, A, rho, ubar, KP.copy(), KI.copy(), I.copy(),
                      KD.copy(), umin, umax, hbar, d_p, Rvv,dt)




samps = np.linspace(0,te,10_000)
d_samples = np.array([plant_S.get_disturbance_constant(t) for t in samps]) 

fig, ax = plt.subplots(figsize=(8, 3.5))
for i in range(d_samples.shape[1]):
    ax.step(samps, d_samples[:, i], where='post', label=f"d[{i}]")

ax.set_xlabel("time [s]")
ax.set_ylabel("disturbance")
ax.set_title("Piecewise-constant disturbance plot")
ax.grid(True, alpha=0.3)
ax.legend()
plt.tight_layout()
plt.show()

    