# default pso params from
# https://pdfs.semanticscholar.org/a4ad/7500b64d70a2ec84bf57cfc2fedfdf770433.pdf

from dataclasses import dataclass
from pprint import pprint

import numpy as np

import project.esn.core as core
import project.esn.transformer as transf
import project.stats.metrics as met
import project.test.music_test as tmusic

make_Data = lambda pattern, init_len, train_len, test_len: (core.Data(
    np.array(list(~pattern)), pattern.tempo, init_len, train_len, test_len))


def add_net_params(**kwargs) -> dict:
    return {
        "data": make_Data((tmusic.test_patterns[0] * 300), 200, 1200, 1200),
        "in_out": 9,
        **kwargs
    }


def run_netwok(transformer: transf.Transformer = transf.Transformers.sig_prob,
               **PSO_kwargs) -> tuple:
    run_dict = core.Run(**add_net_params(**PSO_kwargs)).__enter__()()
    return ((raw_out := run_dict["output"]),
            transformer.value(0.8,
                              transf._identity)(raw_out), run_dict["desired"])


def eval_config(**PSO_kwargs) -> float:
    met = PSO_kwargs.pop("metric")
    raw_out, out, target = run_netwok(**PSO_kwargs)
    # print(f"raw -> {raw_out}, transformed -> {out}")
    return met.value((out[:500] if met.name != "teacher_loss_nd" else raw_out),
                     target[:500])()


test_dimensions = {
    "leaking_rate": (0.05, 1)
    # "density": (0.05, 0.95),
    # "spectral_radius": (0.05, 1.5)
}

make_uniform_dist = lambda bounds: np.array(
    [np.random.uniform(low_b, high_b) for low_b, high_b in bounds])

# would like to specify call=True by default, but the caller needs to still
# give it a value or it will be moved to **kwargs
check_default = lambda obj, default, call, *args, **kwargs: (
    obj if obj is not None else
    (default(*args, **kwargs)
     if hasattr(default, "__call__") and call else default))


# NOTE need checking when default args are passed during init to enforce
# correctness of values given (they must be inside provided bounds to make sense)
@dataclass
class Particle:
    _dims: dict
    position: np.ndarray = None
    pbest_position: np.ndarray = None
    pbest_value: float = np.inf
    velocity: np.ndarray = None

    def __post_init__(self):
        limits = self._dims.values()
        self.position = check_default(self.position, make_uniform_dist, True,
                                      limits)
        self.pbest_position = self.position
        self.velocity = check_default(self.velocity, make_uniform_dist, True,
                                      [(-(np.abs(up - low)), np.abs(up - low))
                                       for low, up in limits])

    @property
    def ndim(self):
        return len(self._dims.keys())

    @property
    def dims(self):
        return list(self._dims.keys())

    @property
    def dim_bounds(self):
        return list(self._dims.values())

    def move(self, new_velocity: np.ndarray):
        self.velocity = new_velocity
        self.position += self.velocity


@dataclass
class Landscape:
    _dims: dict
    cost_func: met.Metric = met.Metrics.nmse
    termination_func: callable = None
    n_particles: int = 20
    max_iterations: int = 50
    particles: np.ndarray = None
    gbest_position: np.ndarray = None
    gbest_value: float = np.inf
    _pso_params: dict = None

    def __post_init__(self):
        self._pso_params = check_default(self._pso_params, {
            "W": 0.6571,
            "phi_cog": 1.6319,
            "phi_soc": 0.6239
        }, False)
        self.particles = np.array(
            [Particle(self._dims) for _ in range(self.n_particles)])
        self.gbest_position = check_default(self.gbest_position,
                                            make_uniform_dist, True,
                                            self._dims.values())
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
        pprint(fitness_value)
        if fitness_value >= part.pbest_value:
            return False
        part.pbest_value = fitness_value
        part.pbest_position = part.position
        return True

    def update_gbest_position(self, part: Particle):
        if part.pbest_value >= self.gbest_value:
            return
        self.gbest_value = part.pbest_value
        self.gbest_position = part.pbest_position

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
        self.print_results(it)
        # return self

    # NOTE args and kwargs are given only for the custom termination function,
    # in reality they should already be saved at initialization (when the term
    # func is provided) and not passed as parameters here. However, idk how to
    # to do that... (I believe the custom termination func should have 'self'
    # as a parameter)
    def __call__(self, *args, **kwargs):
        self.run(*args, **kwargs)
        # return self.run(*args, **kwargs)

    def print_results(self, it: int):
        res = (
            f"# Iterations: {it}\nBest fitness value: {self.gbest_value}\n"
            f"Best config: "
            f"{ {k: v for k, v in zip(self._dims.keys(), self.gbest_position)} }"
        )
        print(res)

    @property
    def W(self):
        return self._pso_params.get("W", None)

    @property
    def phi_cog(self):
        return self._pso_params.get("phi_cog", None)

    @property
    def phi_soc(self):
        return self._pso_params.get("phi_soc", None)
