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


def _get_multinomial_event_dims(distribution: th.distributions.Multinomial):
    event_dims = {"total_count": None}
    event_dims.update(_get_dict_event_dims(distribution._categorical))
    # Remove "probs" if "logits" exists.
    if "logits" in event_dims:
        event_dims.pop("probs", None)
    return event_dims


def _get_one_hot_categorical_event_dims(distribution: th.distributions.Multinomial):
    event_dims = _get_dict_event_dims(distribution._categorical)
    # Remove "probs" if "logits" exists.
    if "logits" in event_dims:
        event_dims.pop("probs", None)
    return event_dims


def _get_relaxed_bernoulli_event_dims(distribution: th.distributions.RelaxedBernoulli):
    event_dims = {"temperature": None}
    event_dims.update(_get_dict_event_dims(distribution.base_dist))
    # Remove "probs" if "logits" exists.
    if "logits" in event_dims:
        event_dims.pop("probs", None)
    return event_dims


def _get_relaxed_one_hot_categorical_event_dims(
        distribution: th.distributions.RelaxedOneHotCategorical):
    event_dims = {"temperature": None}
    event_dims.update(_get_dict_event_dims(distribution.base_dist._categorical))
    # Remove "probs" if "logits" exists.
    if "logits" in event_dims:
        event_dims.pop("probs", None)
    return event_dims


EVENT_DIM_LOOKUP = {
    th.distributions.Bernoulli: _get_dict_event_dims,
    th.distributions.Binomial: _get_dict_event_dims,
    th.distributions.Categorical: _get_dict_event_dims,
    th.distributions.ContinuousBernoulli: _get_dict_event_dims,
    th.distributions.Geometric: _get_dict_event_dims,
    th.distributions.LKJCholesky: _get_lkj_cholesky_event_dims,
    th.distributions.LogisticNormal: _get_logistic_normal_event_dims,
    th.distributions.Multinomial: _get_multinomial_event_dims,
    th.distributions.MultivariateNormal: _get_dict_event_dims,
    th.distributions.NegativeBinomial: _get_dict_event_dims,
    th.distributions.OneHotCategorical: _get_one_hot_categorical_event_dims,
    th.distributions.OneHotCategoricalStraightThrough: _get_one_hot_categorical_event_dims,
    th.distributions.RelaxedBernoulli: _get_relaxed_bernoulli_event_dims,
    th.distributions.RelaxedOneHotCategorical: _get_relaxed_one_hot_categorical_event_dims,
    th.distributions.Wishart: _get_dict_event_dims,
}


def _reshape_transformed_distribution(
        distribution: th.distributions.TransformedDistribution, batch_shape: th.Size,
        event_dims: typing.Optional[typing.Mapping[str, int]] = None) \
        -> th.distributions.TransformedDistribution:
    reshaped_base_dist = reshape(distribution.base_dist, batch_shape, event_dims)
    return type(distribution)(reshaped_base_dist, distribution.transforms)


def _reshape_independent_distribution(
        distribution: th.distributions.Independent, batch_shape: th.Size,
        event_dims: typing.Optional[typing.Mapping[str, int]] = None) \
        -> th.distributions.Independent:
    # The base distribution will have more dimensions than the base distribution. So we need to
    # handle the extra dimensions explicitly.
    base_dist = distribution.base_dist
    reinterpreted_shape = base_dist.batch_shape[-distribution.reinterpreted_batch_ndims:]
    reshaped_base_dist = reshape(distribution.base_dist, batch_shape + reinterpreted_shape,
                                 event_dims)
    return type(distribution)(reshaped_base_dist, distribution.reinterpreted_batch_ndims)


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
    # Dispatch to if this *is* a TransformedDistribution distribution, not just an instance.
    if type(distribution) is th.distributions.TransformedDistribution:
        return _reshape_transformed_distribution(distribution, batch_shape, event_dims)
    if type(distribution) is th.distributions.Independent:
        return _reshape_independent_distribution(distribution, batch_shape, event_dims)
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
