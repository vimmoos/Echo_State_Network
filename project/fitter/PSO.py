from dataclasses import dataclass

import numpy as np

import project.expander.expander as expand
import project.stats.metrics as met

# from pprint import pprint

leak_rate_b = (0.05, 1)
density_b = (0.05, 0.95)
spec_radius_b = (0.05, 1.5)

test_dimensions = {
    "leaking_rate": leak_rate_b,
    "density": density_b,
    "spectral_radius": spec_radius_b
}

# def make_uniform_dist(bounds):
#     # print(bounds)
#     return np.array(
#         [np.random.uniform(low_b, high_b) for low_b, high_b in bounds])

# def check_default(obj, default, call, *args, **kwargs):
#     # print(obj)
#     # print(default)
#     # print(f"call {call}")
#     # print(args)
#     # print(kwargs)
#     return (obj if obj is not None else
#             (default(*args, **kwargs)
#              if hasattr(default, "__call__") and call else default))

make_uniform_dist = lambda bounds: np.array(
    [np.random.uniform(low_b, high_b) for low_b, high_b in bounds])

# would like to specify call=True by default, but the caller needs to still
# give it a value or it will be moved to **kwargs
check_default = lambda obj, default, call, *args, **kwargs: (
    obj if obj is not None else
    (default(*args, **kwargs)
     if hasattr(default, "__call__") and call else default))

# c.Run(**kwargs).__enter__()()


# NOTE need checking when default args are passed during init to enforce
# correctness of values given (they must be inside provided bounds)
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
    cost_func: callable = met.nmse
    termination_func: callable = None
    n_particles: int = 50
    max_iterations: int = 50
    particles: np.ndarray = None
    gbest_position: np.ndarray = None
    gbest_value: float = np.inf
    _pso_params: dict = None

    def __post_init__(self):
        # https://pdfs.semanticscholar.org/a4ad/7500b64d70a2ec84bf57cfc2fedfdf770433.pdf
        self._pso_params = check_default(self._pso_params, {
            "W": 0.6571,
            "phi_cog": 1.6319,
            "phi_soc": 0.6239
        }, False)
        self.particles = np.array(
            [Particle(self._dims) for _ in range(len(self.n_particles))])
        self.gbest_position = check_default(self.gbest_position,
                                            make_uniform_dist, True,
                                            self._dims.values())
        self.termination_func = check_default(
            self.termination_func, lambda it: it <= self.max_iterations, False)

    def update_vel_component(self,
                             part: Particle,
                             comp: str,
                             global_=False) -> float:
        return (self._pso_params[comp] * np.random.uniform() *
                ((part.pbest_postion if not global_ else self.gbest_position) -
                 part.position))

    def update_part_velocity(self, part: Particle):
        return (self.W * part.velocity +
                self.update_vel_component(part, "phi_cog") +
                self.update_vel_component(part, "phi_soc", global_=True))

    def part_fitness(self, part: Particle):
        pass

    def update_pbest_candidate(self, part: Particle) -> bool:
        fitness_value = self.part_fitness(part)
        if fitness_value >= part.pbest_value:
            return False
        part.pbest_value = fitness_value
        part.pbest_position = part.position
        return True

    def update_gbest_position(self, part: Particle):
        if part.pbest_position >= self.gbest_value:
            return
        self.gbest_value = part.pbest_value
        self.gbest_position = part.pbest_position

    def update_best_candidate(self, part: Particle):
        if self.update_pbest_candidate(part):
            self.update_gbest_candidate(part)

    def run(self, *args, **kwargs):
        it = 0
        while not self.termination_func(it, *args, **kwargs):
            for part in self.particles:
                part.move(self.update_part_velocity(part))
                self.update_best_candidate(part)
            it += 1
        self.print_results(it)
        return self

    # NOTE args and kwargs are given only for the custom termination function,
    # in reality they should already be saved at initialization (when the term
    # func is provided) and not passed as parameters here. However, idk how to
    # to do that... (I believe the custom termination func should have 'self'
    # as a parameter)
    def __call__(self, *args, **kwargs):
        return self.run(*args, **kwargs)

    def print_results(self, it: int):
        res = (
            f"# Iterations: {it}\nBest fitness value: {self.gbest_value}\n"
            f"Best config: "
            f"{{k: v for k, v in zip(self._dims.keys(), self.gbest_position)}}"
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
