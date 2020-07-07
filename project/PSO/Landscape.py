# default PSO params from
# https://pdfs.semanticscholar.org/a4ad/7500b64d70a2ec84bf57cfc2fedfdf770433.pdf

# TODO - redundancy (each position should be tried multiple times for each
#        particle, then take mean)
#      - reproducibility (save seeds)

from dataclasses import dataclass

import numpy as np

import project.PSO.utils as u
import project.stats.metrics as met
from project.PSO.Particle import Particle


def eval_config(slice_len: int, **PSO_kwargs) -> float:
    met = PSO_kwargs.pop("metric")
    raw_out, out, target = u.run_network(**PSO_kwargs)
    return met.value((out[:slice_len] if met.name != "teacher_loss_nd" else
                      raw_out)[:slice_len], target[:slice_len])()


@dataclass
class Landscape:
    _dims: dict
    n_particles: int = 2
    max_iter: int = 100
    restart_limit: int = 3
    it: int = 0
    particles: np.ndarray = None
    gbest_position: np.ndarray = None
    gbest_value: list = None
    _pso_params: dict = None
    cost_func: met.Metric = met.Metrics.mse_cor
    termination_func: callable = None
    eval_network_len: int = 500

    def __post_init__(self):
        self._pso_params = u.check_default(self._pso_params, {
            "W": 0.6571,
            "phi_cog": 1.6319,
            "phi_soc": 0.6239
        }, False)
        self.particles = np.array(
            [Particle(self._dims) for _ in range(self.n_particles)])
        self.gbest_position = u.distribute_dimensions(self._dims)
        self.gbest_value = [-np.inf, 0]
        self.termination_func = u.check_default(
            self.termination_func, lambda: self.it >= self.max_iter, False)

    def __repr__(self) -> str:
        return str(self.state)

    def update_vel_component(self,
                             part: Particle,
                             comp: str,
                             global_=False) -> np.ndarray:
        return (self._pso_params[comp] * np.random.uniform() * (
            (part.pbest_position if not global_ else self.gbest_position) -
            part.position))

    def update_part_velocity(self, part: Particle) -> np.ndarray:
        return (self.W * part.velocity +
                self.update_vel_component(part, "phi_cog") +
                self.update_vel_component(part, "phi_soc", global_=True))

    def part_fitness(self, part: Particle) -> float:
        args = {k: v for k, v in zip(self._dims.keys(), part.position)}
        args["metric"] = self.cost_func
        return eval_config(self.eval_network_len, **args)

    def update_pbest_candidate(self, part: Particle) -> bool:
        fitness_value = self.part_fitness(part)
        if fitness_value <= part.pbest_value:
            return False
        part.pbest_value = fitness_value
        part.pbest_position = part.position
        return True

    def update_gbest_candidate(self, part: Particle):
        if part.pbest_value <= self.gbest_value[0]:
            return
        self.gbest_value = [part.pbest_value, 0]
        self.gbest_position = part.pbest_position

    def update_best_candidate(self, part: Particle):
        if self.update_pbest_candidate(part):
            self.update_gbest_candidate(part)

    def run(self, *args, **kwargs):
        while not self.termination_func(*args, **kwargs):
            print(self.it)
            for part in self.particles:
                part.move(self.update_part_velocity(part))
                self.update_best_candidate(part)
            self.random_restart()
        return self

    def random_restart(self):
        self.it += 1
        self.gbest_value[1] += 1
        if self.gbest_value[1] >= self.restart_limit:
            last_gbest_val = self.gbest_value[0]
            last_gbest_pos = self.gbest_position
            self.__post_init__()
            self.gbest_value[0] = last_gbest_val
            self.gbest_position = last_gbest_pos

    # NOTE args and kwargs are given only for the custom termination function,
    # in reality they should already be saved at initialization (when the term
    # func is provided) and not passed as parameters here. However, idk how to
    # to do that... (I believe the custom termination func should have 'self'
    # as a parameter)
    def __call__(self, *args, **kwargs):
        return self.run(*args, **kwargs)

    @property
    def state(self) -> dict:
        replace_name = lambda k, v: (v.name if k == "transformer" else v
                                     if not callable(v) else v.__name__)
        return {
            "iteration": self.it,
            "max_iteration": self.max_iter,
            "metric": self.cost_func.name,
            "slice_transform_out": self.eval_network_len,
            "best": self.gbest_value,
            **{
                k: replace_name(k, v)
                for k, v in u.map_params(**{
                    k: v
                    for k, v in zip(self._dims.keys(), self.gbest_position)
                }).items()
            }
        }

    @property
    def W(self) -> float:
        return self._pso_params.get("W", None)

    @property
    def phi_cog(self) -> float:
        return self._pso_params.get("phi_cog", None)

    @property
    def phi_soc(self) -> float:
        return self._pso_params.get("phi_soc", None)
