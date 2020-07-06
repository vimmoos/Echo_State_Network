# default PSO params from
# https://pdfs.semanticscholar.org/a4ad/7500b64d70a2ec84bf57cfc2fedfdf770433.pdf

# TODO - redundancy (each position should be tried multiple times for each
#        particle, then take mean)
#      - reproducibility (save seeds)

import random as r
from dataclasses import dataclass
from pprint import pprint

import numpy as np
from aenum import Enum

import project.esn.core as core
import project.esn.transformer as transf
import project.stats.metrics as met
import project.test.music_test as tmusic

transf.fill_squeezer()

esn_gen = {
    "data":
    core.Data(np.array(list(~(tmusic.test * 100))), (tmusic.test * 100).tempo,
              960, 5500, 5500),
    "in_out":
    9,
    "reg":
    1e-8,
    "noise":
    0,
    "t_squeeze":
    np.tanh,
    "t_param": ((1 * 2) / 10) + 0.2,
    "transformer":
    transf.Transformers.sig_prob,
    "squeeze_o":
    transf._identity,
}


def add_net_params(**kwargs) -> dict:
    for k, v in kwargs.items():
        esn_gen[k] = v
    return esn_gen


def bind_enum_idx(idx: float, e: Enum) -> int:
    return int(idx) if int(idx) < len(list(e)) else int(idx) - 1


def map_params(**PSO_kwargs) -> dict:
    for k, v in PSO_kwargs.items():
        if k in ["t_squeeze", "squeeze_o"]:
            PSO_kwargs[k] = list(transf.Squeezers)[bind_enum_idx(
                v, transf.Squeezers)].value
        if k == "transformer":
            PSO_kwargs[k] = list(transf.Transformers)[bind_enum_idx(
                v, transf.Transformers)]
    return PSO_kwargs


def run_netwok(out_transf: transf.Transformer = transf.Transformers.sig_prob,
               **PSO_kwargs) -> tuple:
    # pprint(PSO_kwargs)
    PSO_kwargs = map_params(**PSO_kwargs)
    # pprint(PSO_kwargs)
    # pprint(add_net_params(**PSO_kwargs))
    run_dict = core.Run(**add_net_params(**PSO_kwargs)).__enter__()()
    return ((raw_out := run_dict["output"]),
            out_transf.value(0.8,
                             transf._identity)(raw_out), run_dict["desired"])


def eval_config(**PSO_kwargs) -> float:
    met = PSO_kwargs.pop("metric")
    raw_out, out, target = run_netwok(**PSO_kwargs)
    return met.value((out[:500] if met.name != "teacher_loss_nd" else raw_out),
                     target[:500])()


# would like to specify call=True by default, but the caller needs to still
# give it a value or it will be moved to **kwargs
check_default = lambda obj, default, call, *args, **kwargs: (
    obj if obj is not None else
    (default(*args, **kwargs)
     if hasattr(default, "__call__") and call else default))

# test_dimensions = {"leaking_rate": (0.05, 1), "spectral_radius": (0.05, 1.5)}
# test_dimensions = {"transformer": transf.Transformers}
test_dimensions = {
    "transformer": transf.Transformers,
    "squeeze_o": transf.Squeezers,
    "leaking_rate": (0.05, 1)
}


def enum_rand_idx(e: Enum, velocity: bool) -> float:
    return np.random.uniform(0 if not velocity else -len(list(e)),
                             len(list(e)))


def check_transformer(dims: dict):
    if "transformer" in (keys := dims.keys()):
        if "t_param" not in keys:
            dims["t_param"] = (0.05, 1)
        if "t_squeeze" not in keys:
            dims["t_squeeze"] = transf.Squeezers


def distribute_dimensions(dims: dict, velocity: bool = False) -> float:
    check_transformer(dims)
    ret = []
    for k, v in dims.items():
        if k in ["t_squeeze", "squeeze_o"] and v.__name__ == "Squeezers":
            ret.append(enum_rand_idx(transf.Squeezers, velocity))
        elif k == "transformer" and v.__name__ == "Transformers":
            ret.append(enum_rand_idx(transf.Transformers, velocity))
        elif isinstance(v, tuple):
            up, low = v
            ret.append(
                np.random.uniform(up, low) if not velocity else np.random.
                uniform(-np.abs(up - low), np.abs(up - low)))
    return np.array(ret)


# NOTE need checking when default args are passed during init to enforce
# correctness of values given (they must be inside provided bounds to make sense)
@dataclass
class Particle:
    _dims: dict
    position: np.ndarray = None
    pbest_position: np.ndarray = None
    pbest_value: float = -np.inf
    velocity: np.ndarray = None

    def __post_init__(self):
        self.position = distribute_dimensions(self._dims)
        self.pbest_position = self.position
        self.velocity = distribute_dimensions(self._dims, velocity=True)

    @property
    def ndim(self):
        return len(self._dims.keys())

    @property
    def dims(self):
        return list(self._dims.keys())

    @property
    def dim_bounds(self):
        return list(
            map(lambda x: x if isinstance(x, tuple) else (0, len(list(x)) - 1),
                self._dims.values()))

    def adjust_position(self):
        for idx, bound in enumerate(self.dim_bounds):
            low_b, high_b = bound
            if self.position[idx] > high_b:
                self.position[idx] = high_b
            if self.position[idx] < low_b:
                self.position[idx] = low_b

    def move(self, new_velocity: np.ndarray):
        self.velocity = new_velocity
        self.position += self.velocity
        self.adjust_position()


@dataclass
class Landscape:
    _dims: dict
    cost_func: met.Metric = met.Metrics.np_cor
    termination_func: callable = None
    n_particles: int = 20
    max_iterations: int = 1
    particles: np.ndarray = None
    gbest_position: np.ndarray = None
    gbest_value: float = -np.inf
    _pso_params: dict = None

    def __post_init__(self):
        self._pso_params = check_default(self._pso_params, {
            "W": 0.6571,
            "phi_cog": 1.6319,
            "phi_soc": 0.6239
        }, False)
        self.particles = np.array(
            [Particle(self._dims) for _ in range(self.n_particles)])
        self.gbest_position = distribute_dimensions(self._dims)
        self.termination_func = check_default(
            self.termination_func, lambda it: it >= self.max_iterations, False)

    def update_vel_component(self,
                             part: Particle,
                             comp: str,
                             global_=False) -> float:
        return (self._pso_params[comp] * np.random.uniform() * (
            (part.pbest_position if not global_ else self.gbest_position) -
            part.position))

    def update_part_velocity(self, part: Particle):
        return (self.W * part.velocity +
                self.update_vel_component(part, "phi_cog") +
                self.update_vel_component(part, "phi_soc", global_=True))

    def part_fitness(self, part: Particle) -> float:
        args = {k: v for k, v in zip(self._dims.keys(), part.position)}
        args["metric"] = self.cost_func
        return eval_config(**args)

    def update_pbest_candidate(self, part: Particle) -> bool:
        fitness_value = self.part_fitness(part)
        if fitness_value <= part.pbest_value:
            return False
        part.pbest_value = fitness_value
        part.pbest_position = part.position
        return True

    def update_gbest_candidate(self, part: Particle):
        if part.pbest_value <= self.gbest_value:
            return
        self.gbest_value = part.pbest_value
        # print(f"old gbest pos -> {self.gbest_position}")
        self.gbest_position = part.pbest_position
        # print(f"Updated gbest position -> {self.gbest_position}")

    def update_best_candidate(self, part: Particle):
        if self.update_pbest_candidate(part):
            self.update_gbest_candidate(part)

    def run(self, *args, **kwargs):
        it = 0
        while not self.termination_func(it, *args, **kwargs):
            print(f"iter {it}")
            for part in self.particles:
                part.move(self.update_part_velocity(part))
                self.update_best_candidate(part)
            it += 1
        return self.print_results(it)

    # NOTE args and kwargs are given only for the custom termination function,
    # in reality they should already be saved at initialization (when the term
    # func is provided) and not passed as parameters here. However, idk how to
    # to do that... (I believe the custom termination func should have 'self'
    # as a parameter)
    def __call__(self, *args, **kwargs):
        return self.run(*args, **kwargs)

    def print_results(self, it: int):
        # res = (f"# Iterations: {it}\nFinal {self.cost_func.name}: "
        #        f"{self.gbest_value}\nBest config: "
        #        f"{}")
        dio_porco = lambda k, v: (v.name if k == "transformer" else v
                                  if not callable(v) else v.__name__)
        return {
            "iteration": it,
            "metric": self.cost_func.name,
            "best": self.gbest_value,
            **{
                k: dio_porco(k, v)
                for k, v in map_params(**{
                    k: v
                    for k, v in zip(self._dims.keys(), self.gbest_position)
                }).items()
            }
        }

    # def print_results(self, it: int):
    #     res = self._print_results
    #     return {i for k,v in res}

    @property
    def W(self):
        return self._pso_params.get("W", None)

    @property
    def phi_cog(self):
        return self._pso_params.get("phi_cog", None)

    @property
    def phi_soc(self):
        return self._pso_params.get("phi_soc", None)
