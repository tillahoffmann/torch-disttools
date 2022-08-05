import pytest
import torch as th
from torch.distributions import constraints
import typing


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
        # We use a down-scaled normal distribution in the unconstrained space to avoid running into
        # numerical errors by accident.
        unconstrained = th.randn(batch_shape + shape_or_value) / 10
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


@pytest.fixture(params=[
    (th.distributions.Bernoulli, {"probs": ()}),
    (th.distributions.Bernoulli, {"logits": ()}),
    (th.distributions.Beta, {"concentration0": (), "concentration1": ()}),
    (th.distributions.Binomial, {"total_count": th.randint(low=1, high=100, size=()), "probs": ()}),
    (th.distributions.Categorical, {"probs": (7,)}),
    (th.distributions.Cauchy, {"loc": (), "scale": ()}),
    (th.distributions.Chi2, {"df": ()}),
    (th.distributions.ContinuousBernoulli, {"probs": ()}),
    (th.distributions.ContinuousBernoulli, {"logits": ()}),
    (th.distributions.Dirichlet, {"concentration": (9,)}),
    (th.distributions.Exponential, {"rate": ()}),
    (th.distributions.FisherSnedecor, {"df1": (), "df2": ()}),
    (th.distributions.Gamma, {"concentration": (), "rate": ()}),
    (th.distributions.Geometric, {"probs": ()}),
    (th.distributions.Geometric, {"logits": ()}),
    (th.distributions.Gumbel, {"loc": (), "scale": ()}),
    (th.distributions.HalfCauchy, {"scale": ()}),
    (th.distributions.HalfNormal, {"scale": ()}),
    (th.distributions.Kumaraswamy, {"concentration0": (), "concentration1": ()}),
    (th.distributions.Laplace, {"loc": (), "scale": ()}),
    (th.distributions.LKJCholesky, {"dim": 4, "concentration": ()}),
    (th.distributions.LogisticNormal, {"loc": (7,), "scale": (7,)}),
    (th.distributions.LogNormal, {"loc": (), "scale": ()}),
    (th.distributions.LowRankMultivariateNormal, {"loc": (7,), "cov_factor": (7, 7),
                                                  "cov_diag": (7,)}),
    (th.distributions.Multinomial, {"total_count": 23, "probs": (5,)}),
    (th.distributions.Multinomial, {"total_count": 31, "logits": (5,)}),
    (th.distributions.MultivariateNormal, {"loc": (13,), "covariance_matrix": (13, 13)}),
    (th.distributions.MultivariateNormal, {"loc": (13,), "precision_matrix": (13, 13)}),
    (th.distributions.MultivariateNormal, {"loc": (13,), "scale_tril": (13, 13)}),
    (th.distributions.NegativeBinomial, {"total_count": (), "probs": ()}),
    (th.distributions.Normal, {"loc": (), "scale": ()}),
    (th.distributions.OneHotCategorical, {"probs": (17,)}),
    (th.distributions.OneHotCategorical, {"logits": (17,)}),
    (th.distributions.OneHotCategoricalStraightThrough, {"probs": (17,)}),
    (th.distributions.OneHotCategoricalStraightThrough, {"logits": (17,)}),
    (th.distributions.Pareto, {"scale": (), "alpha": ()}),
    (th.distributions.Poisson, {"rate": ()}),
    (th.distributions.RelaxedBernoulli, {"probs": (), "temperature": th.scalar_tensor(2)}),
    (th.distributions.RelaxedBernoulli, {"logits": (), "temperature": th.scalar_tensor(1.5)}),
    (th.distributions.RelaxedOneHotCategorical, {"probs": (4,),
                                                 "temperature": th.scalar_tensor(1.1)}),
    (th.distributions.RelaxedOneHotCategorical, {"logits": (4,),
                                                 "temperature": th.scalar_tensor(1.3)}),
    (th.distributions.StudentT, {"df": (), "loc": (), "scale": ()}),
    (th.distributions.VonMises, {"loc": (), "concentration": ()}),
    (th.distributions.Weibull, {"scale": (), "concentration": ()}),
])
def distribution_specification(request: pytest.FixtureRequest):
    """
    Return the distribution class and parameter specifications. Each parameter specification is
    either a tuple indicating the event shape or a literal value to be passed to the distribution.
    """
    return request.param


def distribution_from_spec(
        batch_shape: th.Size, cls: typing.Type[th.distributions.Distribution],
        params: typing.Mapping[str, typing.Any]) -> th.distributions.Distribution:
    args = {name: sample_value(shape, batch_shape, cls.arg_constraints.get(name))
            for name, shape in params.items()}
    dist = cls(**args)
    assert dist.batch_shape == batch_shape
    return dist
