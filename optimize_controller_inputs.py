import numpy as np
from typing import Callable, Dict, Any
from scipy.optimize import dual_annealing, minimize, differential_evolution
import src.parameters as p
from numpy.linalg import norm

#Todo: Vi skal lade variable zbar være en ting og så skal vi køre n iterationer af optimeringsalgoritmen hvor hver setpoint so svare
#Til en realisering som optimeres på og vi tager så expectation af alle losses.


CTRL_TYPE = "PID"
BOUNDS = [(0.0, 10.0), (0.0, 10.0),   # KP11, KP22
          (0.0, 5.0),  (0.0, 5.0),    # KI11, KI22
          (0.0, 10.0), (0.0, 10.0)]   # KD11, KD22

def to_gains(theta):
    kp1,kp2, ki1,ki2, kd1,kd2 = theta
    KP = np.zeros((2,2)); KP[0,0]=kp1; KP[1,1]=kp2
    KI = np.zeros((2,2)); KI[0,0]=ki1; KI[1,1]=ki2
    KD = np.zeros((2,2)); KD[0,0]=kd1; KD[1,1]=kd2
    return KP, KI, KD


def optimize_pid(model_factory: Callable[[], Any],bounds=BOUNDS,seed=0,maxiter=200):  
    def run_sim(theta):
        # Run the model simulation--
        KP, KI, KD = to_gains(theta)
        model = model_factory() 
        model.KP, model.KI, model.KD = KP, KI, KD
        model.simulate(t0=p.t0, tf=p.tf, h0=p.h0, ctrl_type=CTRL_TYPE)
        y_hist = model.hist['h'][:,:2]
        e = (y_hist - p.zbar)
        #we can change loss function maybe
        #lam = 1e-4
        #.ravel()
        #loss_f = norm(e.ravel(),1)
        delta = 0.3
        ae = np.abs(e)
        huber = np.where(ae <= delta, 0.5*ae**2, delta*(ae - 0.5*delta))
        loss_f = huber.sum()
        #loss_f = norm(e,'fro')
        #reg  = lam * np.sum(np.square(theta))
        reg = 0
        
        #loss_f = np.max(np.abs(e))
        return loss_f+reg

    def wrapped(theta):
            return float(run_sim(theta))

    res = dual_annealing(wrapped, bounds=bounds, maxiter=maxiter, seed=seed)
    KP, KI, KD = to_gains(res.x)
    return KP,KI,KD,res.fun


