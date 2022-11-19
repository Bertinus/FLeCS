"""Functions for modelling decay processes in cells."""


def exponential_decay(obj, node_type, alpha=None):
    """

    Args:
        obj:
        node_type:
        alpha:

    Returns:

    """
    return alpha * obj[node_type].state
