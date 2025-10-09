import numpy as np
#import sys
#import os
#sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import src.parameters as p
from src.models import StochasticBrownian
from src.plotting import plot_hist
from optimize_controller_inputs import optimize_pid

def opt_env():
    return StochasticBrownian(p.opt_dt, p.zbar, p.ubar, p.sig_v,
                              p.D_state_opt, p.S_state, p.F0, seed=0)
    
KP,KI,KD,LOSS = optimize_pid(model_factory=opt_env, maxiter=10)
print(f"Best found KP: {KP}, Best found KI: {KI}, Best found KD: {KD}")
print(f"Best found controller param loss: {LOSS}" )
for ctrl_type in ['P', 'PI', 'PID']:
    stoch_brown = StochasticBrownian(p.dt, p.zbar, p.ubar, p.sig_v,p.D_state,p.S_state,p.F0)
    stoch_brown.KP, stoch_brown.KI, stoch_brown.KD = KP,KI,KD  
    stoch_brown.simulate(p.t0, p.tf, p.h0, ctrl_type=ctrl_type)
    plot_hist(stoch_brown, f"Closed-loop {ctrl_type}-controller - Stochastic brownian", plot_disturbance=True, filename=f'p3_stoch_brown_{ctrl_type}',step=False)   
    
    #stoch_brown = StochasticBrownian(p.dt, p.zbar, p.ubar, p.sig_v,p.D_state,p.S_state,p.F0)
    #stoch_brown.simulate(p.t0, p.tf, p.h0, ctrl_type=ctrl_type)
    #plot_hist(stoch_brown, f"Closed-loop {ctrl_type}-controller - Stochastic brownian", plot_disturbance=True, filename=f'p3_stoch_brown_{ctrl_type}',step=False)   
    #stoch_piece = StochasticPiecewise(p.dt, p.zbar, p.ubar, p.sig_v, p.mu_d, p.sig_d, p.t_d)
    #stoch_piece.simulate(p.t0, p.tf, p.h0, ctrl_type=ctrl_type)
    #plot_hist(stoch_piece, f"Closed-loop {ctrl_type}-controller - Stochastic piecewise", filename=f'p3_stoch_piece_{ctrl_type}')
    
    
