import pytest
import torch as th
from torch.distributions import constraints
import torch_disttools as td
import typing
import warnings


def sample_value(shape_or_value, batch_shape: typing.Tuple[int],
                 constraint: constraints.Constraint) -> th.Tensor:
    """
    Sample a value with the given batch shape satisfying a constraint. If the first argument is not
    shape-like, it is returned as is and other arguments are ignored.
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
    else:
        return shape_or_value


def check_reshaped_dist(dist: th.distributions.Distribution, shape_to: typing.Tuple[int]) -> None:
    # Reshape and make sure the dimensions behave as expected.
    reshaped = td.reshape(dist, shape_to)
    assert reshaped.batch_shape == shape_to

    # Ensure the means are the same after reshaping.
    try:
        reshaped_mean: th.Tensor = dist.mean.reshape(shape_to + dist.event_shape)
        assert th.allclose(reshaped_mean.nan_to_num(th.pi), reshaped.mean.nan_to_num(th.pi))
    except NotImplementedError:
        pass

    # Ensure that samples have the correct shape.
    assert reshaped.sample().shape == shape_to + dist.event_shape

    # Ensure that samples are the same after reshaping by setting a seed.
    seed = th.randint(1024, ())
    th.manual_seed(seed)
    reshaped_sample = dist.sample().reshape(shape_to + dist.event_shape)
    th.manual_seed(seed)
    assert th.allclose(reshaped_sample.nan_to_num(th.pi), reshaped.sample().nan_to_num(th.pi))


@pytest.fixture(params=[
    ((5 * 2,), (5, 2)),
    ((2, 3, 5), (2 * 3, 5)),
    ((2, 3, 5), (3, 2 * 5)),
])
def batch_shapes(request: pytest.FixtureRequest):
    """
    Combinations of source and target batch shapes.
    """
    return request.param


# Distribution classes and the shape of samples in unconstrained space.
@pytest.mark.parametrize("spec", [
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
    (th.distributions.RelaxedBernoulli, {"probs": (), "temperature": 2}),
    (th.distributions.RelaxedBernoulli, {"logits": (), "temperature": 1.5}),
    (th.distributions.RelaxedOneHotCategorical, {"probs": (4,), "temperature": 1.1}),
    (th.distributions.RelaxedOneHotCategorical, {"logits": (4,), "temperature": 1.3}),
    (th.distributions.StudentT, {"df": (), "loc": (), "scale": ()}),
    (th.distributions.VonMises, {"loc": (), "concentration": ()}),
    (th.distributions.Weibull, {"scale": (), "concentration": ()}),
])
def test_reshape(
    batch_shapes: typing.Tuple[th.Size, th.Size],
    spec: typing.Tuple[
        typing.Type[th.distributions.Distribution],
        typing.Iterable[typing.Tuple[str, th.Size]]
        ]):
    shape_from, shape_to = batch_shapes
    cls, arg_shapes = spec

    # Generate the parameters we need and create a distribution.
    args = {name: sample_value(shape, shape_from, cls.arg_constraints.get(name))
            for name, shape in arg_shapes.items()}

    dist = cls(**args)
    assert dist.batch_shape == shape_from
    check_reshaped_dist(dist, shape_to)


@pytest.mark.parametrize("p", [2, 3])
@pytest.mark.parametrize("arg", ["covariance_matrix", "precision_matrix", "scale_tril"])
def test_reshape_wishart(batch_shapes: typing.Tuple[int], p: int, arg: str):
    """
    Separate test required because df depends on the dimensionality of the matrix.
    """
    shape_from, shape_to = batch_shapes
    value = sample_value((p, p), shape_from, th.distributions.Wishart.arg_constraints[arg])
    # Sample a large-ish df to make most samples well-behaved, e.g., not singular.
    df = sample_value((), shape_from, constraints.greater_than(10 + p))
    dist = th.distributions.Wishart(df, **{arg: value})
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "Singular sample detected.")
        check_reshaped_dist(dist, shape_to)


def test_reshape_uniform(batch_shapes: typing.Tuple[int]):
    """
    Separate test required because `low` and `high` have mutual constraints.
    """
    shape_from, shape_to = batch_shapes
    low = sample_value((), shape_from, constraints.real)
    width = sample_value((), shape_from, constraints.positive)
    high = low + width
    dist = th.distributions.Uniform(low, high)
    check_reshaped_dist(dist, shape_to)


@pytest.mark.parametrize("reinterpreted_batch_ndims", [1, 2, 3])
def test_reshape_independent(batch_shapes: typing.Tuple[int], reinterpreted_batch_ndims: int):
    shape_from, shape_to = batch_shapes
    base_shape = shape_from + (7, 9, 11)[:reinterpreted_batch_ndims]
    base_dist = th.distributions.Normal(th.randn(base_shape), th.randn(base_shape).exp())
    dist = th.distributions.Independent(base_dist, reinterpreted_batch_ndims)
    check_reshaped_dist(dist, shape_to)


@pytest.mark.skip("not yet implemented")
def test_reshape_mixture_same_family(batch_shapes: typing.Tuple[int]):
    raise NotImplementedError


@pytest.mark.parametrize("transform", [
    th.distributions.AbsTransform(),
    th.distributions.AffineTransform(3, 2),
    pytest.param(th.distributions.CatTransform, marks=pytest.mark.skip),
    th.distributions.ExpTransform(),
    th.distributions.SigmoidTransform(),
    pytest.param(th.distributions.ComposeTransform, marks=pytest.mark.skip),
    th.distributions.CorrCholeskyTransform(),
    th.distributions.CumulativeDistributionTransform(th.distributions.Normal(3, 2)),
    th.distributions.IndependentTransform(th.distributions.ExpTransform(), 2),
    th.distributions.LowerCholeskyTransform(),
    th.distributions.PowerTransform(2.5),
    pytest.param(th.distributions.ReshapeTransform, marks=pytest.mark.skip),
    th.distributions.SoftmaxTransform(),
    th.distributions.SoftplusTransform(),
    pytest.param(th.distributions.StackTransform, marks=pytest.mark.skip),
    th.distributions.StickBreakingTransform(),
    th.distributions.TanhTransform(),
])
def test_reshape_transformed_distribution(batch_shapes: typing.Tuple[int],
                                          transform: th.distributions.Transform):
    shape_from, shape_to = batch_shapes
    shape = shape_from + (10,) * transform.domain.event_dim
    base_dist = th.distributions.Normal(th.randn(shape), th.randn(shape).exp())
    dist = th.distributions.TransformedDistribution(base_dist, transform)
    check_reshaped_dist(dist, shape_to)
