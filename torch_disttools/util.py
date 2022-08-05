import torch as th
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


def get_event_dims(distribution: th.distributions.Distribution) -> typing.Mapping[str, int]:
    """
    Get the event dimensions of the parameters of a distribution.

    Args:
        distribution: Distribution for which to infer parameter event dimensions.

    Returns:
        event_dims: Mapping from parameter names to the number of event dimensions.
    """
    if func := EVENT_DIM_LOOKUP.get(type(distribution)):
        return func(distribution)
    return {name: constraint.event_dim for name, constraint in distribution.arg_constraints.items()}
