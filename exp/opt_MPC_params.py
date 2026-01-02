import numpy as np
import src.parameters as p
from src.models import StochasticBrownian

from optimize_MPC_params import optimize_MPC

delta = 0.3
def loss(e, delta=0.3):
    ae = np.abs(e)
    huber = np.where(ae <= delta, 0.5*ae**2, delta*(ae - 0.5*delta))
    return float(huber.sum())



Y_transform = lambda F: np.log(np.maximum(F, 1e-12))
F_transform = lambda Y: np.exp(Y)

def mpc_env(seed: int):
    return StochasticBrownian(
        p.dt, p.zbar, p.ubar, p.sig_v,
        p.D_state, p.S_state, p.F0,
        Y_transform, F_transform,
        seed=seed
    )
    

Wz, Wu, Wdu, LOSS, res = optimize_MPC(
    model_factory=mpc_env,
    loss_fn=loss,
    maxiter=1,
    n_realizations=1,
    T_horizon=10,
    MPC_type="linear",  
    NL_sim=True
)

print("Best Wz:\n", Wz)
print("Best Wu:\n", Wu)
print("Best Wdu:\n", Wdu)
#print("Best Ws:\n", Ws)
#print("Best Wt:\n", Wt)
print("Expected loss:", LOSS)