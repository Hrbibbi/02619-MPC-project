import numpy as np
from typing import Callable, Dict, Any
from scipy.optimize import dual_annealing, minimize, differential_evolution
import src.parameters as p
from numpy.linalg import norm

CTRL_TYPE = "PID"
BOUNDS = [(0.0, 15.0), (0.0, 15.0),   # KP11, KP22
          (0.0, 5.0),  (0.0, 5.0),    # KI11, KI22
          (0.0, 15.0), (0.0, 15.0)]   # KD11, KD22

def to_gains(theta):
    kp1,kp2, ki1,ki2, kd1,kd2 = theta
    KP = np.zeros((2,2)); KP[0,0]=kp1; KP[1,1]=kp2
    KI = np.zeros((2,2)); KI[0,0]=ki1; KI[1,1]=ki2
    KD = np.zeros((2,2)); KD[0,0]=kd1; KD[1,1]=kd2
    return KP, KI, KD


def optimize_pid(model_factory: Callable[[], Any],loss_fn,bounds=BOUNDS,seed=0,maxiter=200):  
    model_seed = np.random.SeedSequence(seed).generate_state(1)[0]
    def run_sim(theta):
        # Run the model simulation--
        KP, KI, KD = to_gains(theta)
        model = model_factory(model_seed) 
        model.KP, model.KI, model.KD = KP, KI, KD
        model.simulate(t0=p.t0, tf=p.tf, h0=p.h0, ctrl_type=CTRL_TYPE)
        y_hist = model.hist['h'][:,:2]
        z_hist  = model.hist['zbar']
        e = (y_hist - z_hist)
        loss = loss_fn(e)
        return loss

    def wrapped(theta):
            return float(run_sim(theta))

    res = dual_annealing(wrapped, bounds=bounds, maxiter=maxiter, seed=seed)
    KP, KI, KD = to_gains(res.x)
    return KP,KI,KD,res.fun


def run_expected(model,loss,maxiter=10, runs=10,base_seed = 0):
    KPs,KIs,KDs,LOSSES = [],[],[],[]
    seeds = np.random.SeedSequence(base_seed).generate_state(runs)
    for s in seeds:
        KP,KI,KD,loss_e = optimize_pid(model, loss, maxiter=maxiter,seed=int(s))
        KPs.append(KP)
        KIs.append(KI)
        KDs.append(KD)
        LOSSES.append(loss_e)
    KP_mean = np.mean(np.stack(KPs, axis=0), axis=0)
    KI_mean = np.mean(np.stack(KIs, axis=0), axis=0)
    KD_mean = np.mean(np.stack(KDs, axis=0), axis=0)
    LOSS_mean = float(np.mean(LOSSES))
    return KP_mean, KI_mean, KD_mean, LOSS_mean

def optimize_pid_expected(model_factory: Callable[[int], Any],loss_fn,
    bounds=BOUNDS,
    seed=0,
    maxiter=200,
    n_realizations=8
):
    ss = np.random.SeedSequence(seed)
    sub_seeds = ss.generate_state(n_realizations).astype(np.uint32)

    def single_run_loss(theta, s):
        KP, KI, KD = to_gains(theta)
        model = model_factory(int(s))
        model.KP, model.KI, model.KD = KP, KI, KD
        model.simulate(t0=p.t0, tf=p.tf, h0=p.h0, ctrl_type=CTRL_TYPE)
        y_hist = model.hist['h'][:, :2]         
        z_hist = model.hist['zbar']             
        e = y_hist - z_hist                      
        return loss_fn(e) * p.opt_dt


    def objective(theta):
        vals = [single_run_loss(theta, s) for s in sub_seeds]
        val = np.mean(vals)
        return val

    res = dual_annealing(objective, bounds=bounds, maxiter=maxiter, seed=seed)
    KP, KI, KD = to_gains(res.x)
    return KP, KI, KD, float(res.fun)
