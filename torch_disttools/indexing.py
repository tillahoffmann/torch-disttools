import torch as th
from torch import distributions
import typing
from .util import _setattr, get_event_dims


def index(
        distribution: th.distributions.Distribution, key,
        event_dims: typing.Optional[typing.Mapping[str, int]] = None
        ) -> distributions.Distribution:
    """
    Index the batch dimensions of a distribution.

    Args:
        distribution: Distribution whose batch dimensions to reshape.
        batch_shape: Target shape.
        event_dims: Mapping of distribution argument names to the number of event dimensions of the
            argument or a callable that returns such a mapping given a distribution. If not given,
            the function will attempt to look up the event dimensions from `EVENT_DIM_LOOKUP`. If
            not possible, the function will try to infer the event dimensions.

    Returns:
        indexed: Distribution with indexed batch dimension.
    """
    # If the key is not a tuple, turn it into one.
    if not isinstance(key, tuple):
        key = (key,)
    # Get the parameters from the distribution and index them.
    args = {}
    event_dims = event_dims or get_event_dims(distribution)
    for name, event_dim in event_dims.items():
        value: th.Tensor = getattr(distribution, name)
        if event_dim is None:  # Leave the value as is.
            args[name] = value
        else:  # Index the value.
            # If there are ellipsis, we need to add the trailing event dimensions explicitly.
            if Ellipsis in key:
                arg_key = key + (slice(None),) * event_dim
            else:
                arg_key = key
            args[name] = value[arg_key]
    # Create a new distribution of the same class.
    return type(distribution)(**args)


_setattr(distributions.Distribution, "__getitem__", index)
