"""Functions for modelling decay processes in cells."""


def exponential_decay(obj, node_type, lambda_c=None):
    return lambda_c * obj[node_type].state
