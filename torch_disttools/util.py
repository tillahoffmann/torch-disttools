import torch as th
from torch.distributions import constraints
import typing
import warnings


def _setattr(obj: typing.Type, name: str, value: typing.Callable, override: bool = False) -> None:
    """
    Assign `value` to the `name`d attribute of `obj`, making sure it does not already exist.
    """
    if hasattr(obj, name) and not override:
        warnings.warn(f"type {obj} already has attribute {name}; rerun with `override = True` to "
                      "override the attribute", UserWarning)
        return
    setattr(obj, name, value)


def sample_value(shape_or_value: typing.Union[th.Size, typing.Any], batch_shape: typing.Tuple[int],
                 constraint: constraints.Constraint) -> th.Tensor:
    """
    Sample a value with the given batch shape satisfying a constraint. If the first argument is not
    shape-like, it is returned as-is and other arguments are ignored.

    Args:
        shape_or_value: Shape-like object corresponding to the batch shape of the value to be
            sampled or a value to be returned as-is.
        batch_shape: Batch shape of the sampled value.
        constraint: Constraint that the (sampled) value must satisfy.

    Returns:
        value: (Sampled) value that satisfies the constraint.
    """
    if isinstance(shape_or_value, (tuple, th.Size)):
        unconstrained = th.randn(batch_shape + shape_or_value)
        # Manually apply a positive definite constraint (see
        # https://github.com/pytorch/pytorch/pull/76777 for details).
        if constraint is constraints.positive_definite:
            value: th.Tensor = th.distributions.LowerCholeskyTransform()(unconstrained)
            return value @ value.mT + 1e-3 * th.eye(value.shape[-1])
        else:
            transform: th.distributions.Transform = th.distributions.transform_to(constraint)
            return transform(unconstrained)
    elif constraint and not constraint.check(shape_or_value).all():
        raise ValueError(f"the value {shape_or_value} does not satisfy the constraint {constraint}")
    return shape_or_value
