"""Functions for modelling decay processes in cells."""


def exponential_decay(obj, node_type, alpha=None):
    return alpha * obj[node_type].state
