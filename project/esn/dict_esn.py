from project.esn import trainer as t
from project.esn import updater as u
from project.esn import matrix as g
import numpy as np

example_dict = {
    "trainer" : t.ridge_reg,
    "updater" : u.leaking_update,
    "non_linearity": np.tanh,
    "res_gen": g.generate_rresvoir,
    "in_gen": g.generate_rmatrix,
    "output": u.default_output,
    "output_f":lambda x : x,

    "leaking_rate": 0.3,

    "reg_coef" : 1e-8,

    "density":0.25,

    "bound":0.5,

    "spectral_radius": 0.0,
}
