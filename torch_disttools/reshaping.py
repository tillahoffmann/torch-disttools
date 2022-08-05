import torch as th
from torch import distributions
import typing
from .util import _setattr, get_event_dims


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
    # The base distribution will have more dimensions than the independent distribution. So we need
    # to handle the extra dimensions explicitly.
    base_dist = distribution.base_dist
    reinterpreted_shape = base_dist.batch_shape[-distribution.reinterpreted_batch_ndims:]
    reshaped_base_dist = reshape(distribution.base_dist, batch_shape + reinterpreted_shape,
                                 event_dims)
    return type(distribution)(reshaped_base_dist, distribution.reinterpreted_batch_ndims)


def _reshape_mixture_distribution(
        distribution: th.distributions.MixtureSameFamily, batch_shape: th.Size,
        event_dims: typing.Optional[typing.Mapping[str, int]] = None) \
        -> th.distributions.MixtureSameFamily:
    # We need to reshape the mixture and component distributions separately, then reconstruct the
    # mixture.
    mixture_distribution: th.distributions.Categorical = \
        reshape(distribution.mixture_distribution, batch_shape)
    component_shape = batch_shape + (mixture_distribution._num_events,)
    component_distribution = reshape(distribution.component_distribution, component_shape)
    return type(distribution)(mixture_distribution, component_distribution)


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

    Returns:
        reshaped: Distribution with reshaped batch dimension.
    """
    # Dispatch to if this *is* a TransformedDistribution distribution, not just an instance.
    if type(distribution) is th.distributions.TransformedDistribution:
        return _reshape_transformed_distribution(distribution, batch_shape, event_dims)
    if type(distribution) is th.distributions.Independent:
        return _reshape_independent_distribution(distribution, batch_shape, event_dims)
    if type(distribution) is th.distributions.MixtureSameFamily:
        return _reshape_mixture_distribution(distribution, batch_shape)

    # Get the parameters from the distribution and reshape them.
    args = {}
    event_dims = event_dims or get_event_dims(distribution)
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
