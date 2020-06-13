import numpy as np

def build_extended_states(m_inputs,states,init_len=0,**kwargs):
    return np.vstack((m_inputs.T[:,init_len:],
                      states.T[:,init_len:])).T
