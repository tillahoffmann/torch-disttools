import torch as th
from torch import distributions
import typing
from .util import _setattr


def _get_dict_event_dims(distribution: th.distributions.Bernoulli):
    return {name: constraint.event_dim for name, constraint in distribution.arg_constraints.items()
            if name in distribution.__dict__}


def _get_lkj_cholesky_event_dims(distribution: th.distributions.LKJCholesky):
    return {"dim": None, "concentration": distribution.arg_constraints["concentration"].event_dim}


def _get_logistic_normal_event_dims(distribution: th.distributions.LogisticNormal):
    return {name: constraint.event_dim + 1 for name, constraint
            in distribution.arg_constraints.items()}


EVENT_DIM_LOOKUP = {
    th.distributions.Bernoulli: _get_dict_event_dims,
    th.distributions.Binomial: _get_dict_event_dims,
    th.distributions.Categorical: _get_dict_event_dims,
    th.distributions.ContinuousBernoulli: _get_dict_event_dims,
    th.distributions.Geometric: _get_dict_event_dims,
    th.distributions.LKJCholesky: _get_lkj_cholesky_event_dims,
    th.distributions.LogisticNormal: _get_logistic_normal_event_dims,
    th.distributions.NegativeBinomial: _get_dict_event_dims,
    th.distributions.MultivariateNormal: _get_dict_event_dims,
}


def reshape(
        distribution: distributions.Distribution, batch_shape: th.Size,
        event_dims: typing.Optional[typing.Mapping[str, int]] = None) -> distributions.Distribution:
    """
    Reshape the batch dimensions of a distribution.

    Args:
        distribution: Distribution whose batch dimensions to reshape.
        batch_shape: Target shape.
        event_dims: Mapping of distribution argument names to the number of event dimensions of the
            argument or a callable that returns such a mapping given a distribution. If not given,
            the function will attempt to look up the event dimensions from `EVENT_DIM_LOOKUP`. If
            not possible, the function will try to infer the event dimensions.
    """
    # Identify the event dimensions of each parameter.
    if not event_dims:
        if func := EVENT_DIM_LOOKUP.get(type(distribution)):
            event_dims = func(distribution)
        else:
            event_dims = {name: constraint.event_dim for name, constraint
                          in distribution.arg_constraints.items()}
    # Get the parameters from the distribution and reshape them.
    args = {}
    for name, event_dim in event_dims.items():
        value: th.Tensor = getattr(distribution, name)
        if event_dim is None:  # Leave the value as is.
            args[name] = value
        else:  # Reshape the value.
            event_shape = value.shape[-event_dim:] if event_dim else ()
            shape = batch_shape + event_shape
            args[name] = value.reshape(shape)
    # Create a new distribution of the same class.
    return type(distribution)(**args)


_setattr(distributions.Distribution, "reshape", reshape)
