import numpy as np

from src.models import Deterministic, StochasticPiecewise, StochasticBrownian
import src.parameters as p

determ = Deterministic(p.dt, p.zbar, p.ubar, p.d_determ)
determ.step_response(p.h0, p.ubar, p.mu_d, p.tf)