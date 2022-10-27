"""Functions for modelling decay processes in cells."""


def alpha_decay(obj, node_type):
    return obj[node_type]["alpha"] * obj[node_type].state
