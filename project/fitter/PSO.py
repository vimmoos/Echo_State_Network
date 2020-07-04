import pickle
from dataclasses import dataclass
from pprint import pprint

import numpy as np

import project.stats.metrics as met
import project.stats.pre_post_proc as processor


def write_processed_dump(fname: str, out):
    with open(fname, "wb") as f:
        pickle.dump(processor.process_data(out), f)


def read_processed_dump(fname: str):
    with open(fname, "rb") as f:
        return pickle.load(f)


def process_dump(path: str, fname: str):
    write_processed_dump(fname, processor.get_data(path))
    return read_processed_dump(fname)


leak_rate_b = (0.05, 1)
density_b = (0.05, 0.95)
spec_radius_b = (0.05, 1.5)

test_dimensions = {
    "leaking_rate": leak_rate_b,
    "density": density_b,
    "spectral_radius": spec_radius_b
}

make_uniform_dist = lambda bounds: np.array(
    [np.random.uniform(low_b, high_b) for low_b, high_b in bounds])


@dataclass
class Particle:
    _dims: dict
    position: np.ndarray = None
    pbest_position: np.ndarray = None
    velocity: np.ndarray = None
    pbest_value: float = np.inf

    def __post_init__(self):
        limits = self._dims.values()
        self.position = make_uniform_dist(limits)
        self.pbest_position = self.position
        self.velocity = make_uniform_dist([
            (-(np.abs(up - low)), np.abs(up - low)) for low, up in limits
        ])

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
    cost_func: callable = met.nmse
    n_particles: int = 50
    _particles: np.ndarray = None
    W: float = .5  # inertia
    phi_cog: float = 1.  # individual, cognitive drive
    phi_soc: float = 2.  # social drive

    def __post_init__(self):
        pass

    @np.vectorize
    def update_velocities(self):
        pass

    def update_positions(self):
        pass

    def update_particles(self):
        pass
