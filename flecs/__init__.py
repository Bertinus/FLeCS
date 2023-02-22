from importlib.metadata import version
from flecs import cell_population, data, decay, intervention, mutation, production, sets, trajectory, utils


__all__ = [
    cell_population,
    data,
    decay,
    intervention,
    mutation,
    production,
    sets,
    trajectory,
    utils,
]

__version__ = version("flecs")