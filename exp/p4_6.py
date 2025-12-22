import numpy as np
import matplotlib.pyplot as plt

from src.models import Deterministic, StochasticPiecewise, SDE
import src.parameters as p

determ = Deterministic(p.dt, p.zbar, p.ubar, p.d_determ)

tf = 1000
Ts = 10
determ.markov_params(Ts=Ts, n=np.ceil(tf/Ts).astype(int), h0=p.h0, us=p.ubar, ds=p.d_determ, filename='figs/p4/markov_params.pdf')