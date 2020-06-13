
import numpy as np
from scipy import sparse,stats,linalg
from project.esn import updater as up
from project.esn import utils as ut



class ESN():
    def __init__(self,in_out_size,res_size,
                 res_gen,in_gen,
                 **kwargs):
        super().__init__()
        np.random.seed(42)
        self.params = {**kwargs,"W_in":in_gen(res_size,in_out_size,**kwargs),
                       "W_res":res_gen(res_size,**kwargs)}

    def _runnner(self,desired,**kwargs):
        X1,last_state = run_extended(**kwargs,**self.params)
        return self.params["trainer"](X1,desired,**self.params,**kwargs),last_state

    def __enter__(self):
        return (self._runnner,
                lambda **kwargs:run_gen_mode(**self.params,**kwargs))


    def __exit__(self,err_t,err_v,traceback):
        return

def run_states(W_in,W_res,inputs,init_state,
               updater=up.leaking_update,**kwargs):
    states = np.zeros((len(inputs)+1,W_res.shape[0]))
    states[0,:] = init_state
    for (u,i) in zip(inputs,range(len(inputs))):
        states[i+1,:] = updater(W_in,W_res,u,states[i,:],**kwargs)
    return  states[1:,:]




def run_gen_mode(W_in,W_res,W_out,init_input,init_state,run_length,
                 fun = lambda x :x ,
                 updater = up.leaking_update,
                 output = up.default_output,
                 transformer = lambda x : x,
                 **kwargs):
    state = init_state
    input  = init_input

    outputs = np.zeros((run_length,W_out.shape[0]))
    for t in range(run_length-1):
        state = updater(W_in,W_res,input,state,**kwargs)
        outputs[t,:] = transformer(output(W_out,input,state,**kwargs))
        input = outputs[t,:]
    return outputs

def run_extended(W_in,W_res,inputs,init_state,**kwargs):
    states = run_states(W_in,W_res,inputs,init_state,**kwargs)
    return ut.build_extended_states(np.matrix(inputs),
                                    states,
                                    **kwargs), states[-1,:]
