
import numpy as np
from scipy import sparse,stats,linalg
from project.esn import utils as u

def apply_leak(state,update,
               leaking_rate=0.3,**kwargs):
    return (1-leaking_rate) * state  +  leaking_rate * update

def leaking_update(W_in,W_res,
                   input,state,
                   non_linearity=np.tanh,
                   leaking_rate=0.3,**kwargs):
    return apply_leak(state,
                      default_update(W_in,W_res,
                                     input,state,non_linearity),leaking_rate)

def default_update(W_in,W_res,
                   input,state,
                   non_linearity=np.tanh,**kwargs):

     return non_linearity(W_in.dot(input)
                          + W_res.dot(state))


def feedback_update(W_in,W_res,W_feb,
                    input,state,output,
                    non_linearity=np.tanh,**kwargs):

     return non_linearity(W_in.dot(input)
                          + W_res.dot(state)
                          + W_feb.dot(output))

def default_output(W_out,input,state,output_f = lambda x : x,**kwargs):
    return output_f(W_out.dot(
        u.build_extended_states(np.matrix(input),
                                np.matrix(state)).T)).reshape(-1)
