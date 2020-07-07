from dataclasses import dataclass

import numpy as np

import utils as u


# NOTE need checking when default args are passed during init to enforce
# correctness of values given (they must be inside provided bounds to make
# sense)
@dataclass
class Particle:
    """Representation of multidimensional set of solutions for an
    optimization problem. The style of this representation is that
    described in the Particle Swarm Optimization heuristic search:
    the dimensions of a Particle are encoded as a position in the
    search space, which is changed stochastically with the addition
    of velocity in the exploration of the search space."""

    _dims: dict = None
    """The dimensions of the search space explored in the optimization
    problem (they have continuous domain are continuous)"""
    position: np.ndarray = None
    """A configuration of values for each of the particle's dimensions
    (a vector)"""
    pbest_position: np.ndarray = None
    """The particle's fittest configuration of values for the dimensions
    of the search space"""
    pbest_value: float = -np.inf
    """The particle's fittest configuration of values according to some
    evaluation function"""
    velocity: np.ndarray = None
    """The particle's velocity (a multidimensional vector), which
    stochastically determines the particle's rate of exploration of the
    search space. The update formula depends on the particle's inertia, a
    cognitive component and a social component
    """
    """Initialize a particle's position and velocity in the search space
    randomly, and save best position as current one"""
    def __post_init__(self):
        self.position = u.distribute_dimensions(self._dims)
        self.pbest_position = self.position
        self.velocity = u.distribute_dimensions(self._dims, velocity=True)

    def __repr__(self) -> str:
        return str(self.state)

    """The number of dimensions of this particle"""

    @property
    def ndim(self) -> int:
        return len(self._dims.keys())

    """The identifiers for the dimensions of the search space, as strings"""

    @property
    def dims(self) -> list:
        return list(self._dims.keys())

    """The bounds of the particle's dimensions"""

    @property
    def dim_bounds(self) -> list:
        return list(
            map(lambda x: x if isinstance(x, tuple) else (0, len(list(x)) - 1),
                self._dims.values()))

    """The representation of all the parameters of the particle as a
    dictionary"""

    @property
    def state(self) -> dict:
        return {
            "dimensions": {k: v
                           for k, v in zip(self.dims, self.dim_bounds)},
            "position:": self.position,
            "pbest_position": self.pbest_position,
            "velocity": self.velocity
        }

    """Bound the value of a dimension of the particle inside its domain
    (the stochastic update may bring it out of bounds)"""

    def adjust_position(self):
        for idx, bound in enumerate(self.dim_bounds):
            low_b, high_b = bound
            if self.position[idx] > high_b:
                self.position[idx] = high_b
            if self.position[idx] < low_b:
                self.position[idx] = low_b

    """Update the position of the particle in the search space by
    updating its velocity and position"""

    def move(self, new_velocity: np.ndarray):
        self.velocity = new_velocity
        self.position += self.velocity
        self.adjust_position()
