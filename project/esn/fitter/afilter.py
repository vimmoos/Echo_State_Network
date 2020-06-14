from project.esn.fitter  import ifitter
from project.esn.fitter.logger import logger
import typing
from  dataclasses  import dataclass


@dataclass
class DemoFitter(ifitter.IFitter):
    state: dict
    limit: int = 10
    i: int = 0
    step: float = 0.01
    trace: typing.Any = None

    def current_state(self):
        return self.state

    def swap_state(self,new_state):
        self.state = new_state
        logger.debug(f"New_state = {self.state}")
        return self


    def get_candidates(self,_state):
        '''take a setep in both directions for each float in state, then
        perform a cartesian product

        '''
        return [{"leaking_rate":x,"seed":_state["seed"]} for x in
                [_state["leaking_rate"] + (1 + x )/100
                 for x in range(8)]]

    def pick_canditates(self,cands):
        '''simply returns all candidates
        '''
        return cands

    def evaluate_results(self,res_iter,dry=False):
        '''Greedy pick the best among results
        looking at the 'z' key in the result dict.
        If dry is not truthy it will increase the counter
        for iteration limit of the fitter
        '''
        rsort =  sorted(res_iter,key=lambda x: x['mse'])
        # greedy
        best = rsort[0]
        logger.debug(f"Best: {best}")

        if not dry:
            self.i += 1
            if not self.trace is None:
                self.trace.append([best['args']["leaking_rate"],best['mse']])

        return best['args']

    def stop_criterion(self):
        '''Stops if the iteration limit was reached
        '''
        return self.i >= self.limit

    def state_to_args(_state):
        '''In this simple example the state of the fitter
        is direcltly the signature for the task
        '''
        return [_state]
