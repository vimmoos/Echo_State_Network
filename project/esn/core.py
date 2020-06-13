
import numpy as np
from scipy import sparse,stats,linalg
from project.esn import updater as up


def run_states(W_in,W_res,inputs,init_state):
    states = np.zeros((len(inputs)+1,W_res.shape[0]))
    states[0,:] = init_state
    for (u,i) in zip(inputs,range(len(inputs))):
        states[i+1,:] = up.apply_leak(states[i,:],
                                 up.default_update(W_in,W_res,u,states[i,:]))
    return  states[1:,:]



def build_extended_states(inputs,states,start_index=0):
    return np.vstack((inputs.T[:,start_index:],
                      states.T[:,start_index:])).T


def run_gen_mode(W_in,W_res,W_out,init_input,init_state,run_length,
                      fun = lambda x :x ):
    state = init_state
    input  = init_input

    outputs = np.zeros((run_length,W_out.shape[0]))
    for t in range(run_length-1):
        state = up.apply_leak(state,
                                 up.default_update(W_in,W_res,input,state))
        outputs[t,:] = fun(W_out.dot(np.vstack((input,np.matrix(state).T))))
        input = outputs[t,:]
    return outputs
