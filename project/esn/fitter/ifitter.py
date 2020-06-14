from  abc  import abstractmethod,abstractclassmethod
from celery import group
import celery as cel
from project.esn.fitter.logger import logger
from dataclasses import dataclass
import typing

class TaskRunningError(Exception):
    pass

class TaskNotRunningError(Exception):
    pass

class IFitter():
    """Pseudo interface for auto fitters.

    """

    @abstractmethod
    def current_state(self):
        """Should return the current state of the fitter.
        This should be suitable input for `get_candidates`

        """
        pass

    @abstractmethod
    def swap_state(self,new_state):
        pass

    @abstractmethod
    def get_candidates(self,current_state):
        """Should implement the neighborghood
        expansion procedure given the current state.
        """
        pass

    @abstractmethod
    def pick_canditates(self,candidates):
        """Should implement the selection procedure
        that filters the products of `self.get_candidates`

        Its product should should be suitable input for mapping with
        type(self).state_to_args static method to produce task
        signature iterables.

        """
        pass

    @abstractmethod
    def evaluate_results(self,results_iter):
        """Should determine what is the new current state given an iterable of
        results, presumably from task batch completion.

        Its result should be suitable equivalent to self.current_state()
        as it will be fet to get_candidates unless the stop criterion is met

        """
        pass


    @abstractmethod
    def stop_criterion(self):
        """Should return True when the fitter judges its job is done.

        """
        pass


    @abstractclassmethod
    def state_to_args(state):
        """Should statically implement the mapping from candidate state to
        task signature for this fitter.  A task signature is any
        iterable that is valid as arguments to task callables when
        spliced. So if the task is f(x,y) this function should for
        example map {'x':0,'y':1} -> [0,1].

        Some times this can simply be the idetity function

        """
        pass



@dataclass(init=True, repr=True)
class ADispatcher():
    algortihm_task : cel.Task
    current_task : typing.Any = None

    def _build_task(self,fitter):
        """Build the celery task callable for the given fitter
        """
        logger.debug(f"Building new task for {repr(fitter)}")
        curr = fitter.current_state()
        cands = list(fitter.get_candidates(curr))
        logger.debug(f"Candidates: {cands}")

        picked = fitter.pick_canditates(cands)
        logger.debug(f"Picked: {picked}")

        as_args = list(map(type(fitter).state_to_args, picked))
        logger.debug(f"Args: {as_args} for {self.algortihm_task}")

        task_group =  group(self.algortihm_task.s(*a) for a in as_args)()
        logger.debug(f"group: {task_group}")

        return  task_group

    def dispatch_task(self,fitter):
        if not self.current_task:
            self.current_task =  self._build_task(fitter)
        else:
            raise TaskRunningError(self)
        return self.current_task

    def wait(self):
        if self.current_task:
            task = self.current_task
            result = task.get()
            self.current_task = None
            return result, task
        else:
            raise TaskNotRunningError(self)

@dataclass
class Combiner():
    fitter: IFitter
    dispatcher: ADispatcher

    def __call__(self):
        logger.debug(f"Fit-combiner step")
        r = self.dispatcher.dispatch_task(self.fitter)
        logger.debug(f"Fit-combiner dispatched jobs, waiting for results")
        results,_ = self.dispatcher.wait()
        logger.debug(f"Fit-combiner evaulating results")
        next_state = self.fitter.evaluate_results(results)
        self.fitter.swap_state(next_state)
        logger.debug(f"Fit-combiner transition:\n{repr(self.fitter)}")
        return self

    def __invert__(self):
        try:
            while True:
                self()
                if self.fitter.stop_criterion():
                    return self
        except KeyboardInterrupt:
            logger.debug("Interrupted combiner loop")
