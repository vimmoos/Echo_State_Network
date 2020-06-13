
import numpy as np
from scipy import sparse,stats,linalg

def apply_leak(state,update,
                     leaking_rate=0.3):
    return (1-leaking_rate) * state  +  leaking_rate * update

def default_update(W_in,W_res,
                        input,state,
                        non_linearity=np.tanh):

     return non_linearity(W_in.dot(input)
                           + W_res.dot(state))


def feedback_update(W_in,W_res,W_feb,
                         input,state,output,
                         non_linearity=np.tanh):

     return non_linearity(W_in.dot(input)
                           + W_res.dot(state)
                           + W_feb.dot(output))
