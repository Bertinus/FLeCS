"""Used to initialize model parameters."""
from torch.distributions.normal import Normal


def init_normal(obj, group_name, param_name, mu, sigma):
    obj[group_name][param_name] = Normal(mu, sigma).sample((len(obj[group_name]),))