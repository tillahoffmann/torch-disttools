import pytest
import torch as th
from torch.distributions import constraints
import torch_disttools as td
import typing
import warnings
from .conftest import distribution_from_spec, sample_value


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


def test_reshape(batch_shapes: typing.Tuple[th.Size, th.Size], distribution_specification):
    shape_from, shape_to = batch_shapes
    dist = distribution_from_spec(shape_from, *distribution_specification)
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
    """
    Separate test required because `reinterpreted_batch_ndims` doesn't fit into the standard
    parameter scheme.
    """
    shape_from, shape_to = batch_shapes
    base_shape = shape_from + (7, 9, 11)[:reinterpreted_batch_ndims]
    base_dist = th.distributions.Normal(th.randn(base_shape), th.randn(base_shape).exp())
    dist = th.distributions.Independent(base_dist, reinterpreted_batch_ndims)
    check_reshaped_dist(dist, shape_to)


@pytest.mark.parametrize("num_components", [1, 2, 3])
@pytest.mark.parametrize("scalar", [False, True])
def test_reshape_mixture_same_family(batch_shapes: typing.Tuple[int], num_components: int,
                                     scalar: bool):
    shape_from, shape_to = batch_shapes
    mixture_shape = *shape_from, num_components
    mixture_distribution = th.distributions.Categorical(logits=th.randn(mixture_shape))
    if scalar:
        component_distribution = th.distributions.Normal(
            th.randn(mixture_shape), th.randn(mixture_shape).exp())
    else:
        num_dims = 5
        loc = th.randn(*mixture_shape, num_dims)
        cov = th.randn(*mixture_shape, num_dims).exp().diag_embed()
        component_distribution = th.distributions.MultivariateNormal(loc, cov)

    assert mixture_distribution.batch_shape + (num_components,) \
        == component_distribution.batch_shape, "batch shapes do not match"
    dist = th.distributions.MixtureSameFamily(mixture_distribution, component_distribution)
    check_reshaped_dist(dist, shape_to)


@pytest.mark.parametrize("transform", [
    th.distributions.AbsTransform(),
    th.distributions.AffineTransform(3, 2),
    pytest.param(th.distributions.CatTransform, marks=pytest.mark.skip),
    th.distributions.ExpTransform(),
    th.distributions.SigmoidTransform(),
    th.distributions.ComposeTransform([
        th.distributions.LowerCholeskyTransform(),
        th.distributions.ExpTransform(),
    ]),
    th.distributions.CorrCholeskyTransform(),
    th.distributions.CumulativeDistributionTransform(th.distributions.Normal(3, 2)),
    th.distributions.IndependentTransform(th.distributions.ExpTransform(), 2),
    th.distributions.LowerCholeskyTransform(),
    th.distributions.PowerTransform(2.5),
    th.distributions.ReshapeTransform((10, 10), (2, 5, 5, 2)),
    th.distributions.SoftmaxTransform(),
    th.distributions.SoftplusTransform(),
    pytest.param(th.distributions.StackTransform, marks=pytest.mark.skip),
    th.distributions.StickBreakingTransform(),
    th.distributions.TanhTransform(),
])
def test_reshape_transformed_distribution(batch_shapes: typing.Tuple[int],
                                          transform: th.distributions.Transform):
    """
    Separate test required because we need to create transforms in addition to the base
    distribution.
    """
    shape_from, shape_to = batch_shapes
    shape = shape_from + (10,) * transform.domain.event_dim
    base_dist = th.distributions.Normal(th.randn(shape), th.randn(shape).exp())
    dist = th.distributions.TransformedDistribution(base_dist, transform)
    check_reshaped_dist(dist, shape_to)
