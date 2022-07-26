import pytest
import torch as th
from torch.distributions.constraints import Constraint
import torch_disttools as td
import typing


# Combiations of source and target batch shapes.
@pytest.mark.parametrize("batch_shapes", [
    ((5 * 2,), (5, 2)),
    ((2, 3, 5), (2 * 3, 5)),
    ((2, 3, 5), (3, 2 * 5)),
])
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
    pytest.param(th.distributions.LowRankMultivariateNormal, marks=pytest.mark.skip),
    pytest.param(th.distributions.MixtureSameFamily, marks=pytest.mark.skip),
    pytest.param(th.distributions.Multinomial, marks=pytest.mark.skip),
    pytest.param((th.distributions.MultivariateNormal, {"loc": (13,), "covariance_matrix": None}),
                 marks=pytest.mark.skip(reason="https://github.com/pytorch/pytorch/pull/76777")),
    pytest.param((th.distributions.MultivariateNormal, {"loc": (13,), "precision_matrix": None}),
                 marks=pytest.mark.skip(reason="https://github.com/pytorch/pytorch/pull/76777")),
    (th.distributions.MultivariateNormal, {"loc": (13,), "scale_tril": (13, 13)}),
    (th.distributions.NegativeBinomial, {"total_count": (), "probs": ()}),
    (th.distributions.Normal, {"loc": (), "scale": ()}),
    pytest.param(th.distributions.OneHotCategorical, marks=pytest.mark.skip),
    pytest.param(th.distributions.OneHotCategoricalStraightThrough, marks=pytest.mark.skip),
    (th.distributions.Pareto, {"scale": (), "alpha": ()}),
    (th.distributions.Poisson, {"rate": ()}),
    pytest.param(th.distributions.RelaxedBernoulli, marks=pytest.mark.skip),
    pytest.param(th.distributions.RelaxedOneHotCategorical, marks=pytest.mark.skip),
    (th.distributions.StudentT, {"df": (), "loc": (), "scale": ()}),
    pytest.param((th.distributions.Uniform, {"low": (), "high": ()}),
                 marks=pytest.mark.skip("cannot transform to dependent constraints")),
    (th.distributions.VonMises, {"loc": (), "concentration": ()}),
    (th.distributions.Weibull, {"scale": (), "concentration": ()}),
    pytest.param((th.distributions.Wishart, {"df": (), "covariance_matrix": None}),
                 marks=pytest.mark.skip(reason="https://github.com/pytorch/pytorch/pull/76777")),
    pytest.param((th.distributions.Wishart, {"df": (), "precision_matrix": None}),
                 marks=pytest.mark.skip(reason="https://github.com/pytorch/pytorch/pull/76777")),
    pytest.param((th.distributions.Wishart, {"df": (), "scale_tril": (13, 13)}),
                 marks=pytest.mark.skip(reason="df should have dependent constraint > dim - 1")),
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
    args = {}
    for name, shape in arg_shapes.items():
        if isinstance(shape, (tuple, th.Size)):
            unconstrained = th.randn(shape_from + shape)
            constraint: Constraint = cls.arg_constraints[name]
            transform: th.distributions.Transform = th.distributions.transform_to(constraint)
            args[name] = transform(unconstrained)
        else:
            args[name] = shape  # Just copy the literal value.

    dist = cls(**args)
    assert dist.batch_shape == shape_from

    # Reshape and make sure the dimensions behave as expected.
    reshaped = td.reshape(dist, shape_to)
    assert reshaped.batch_shape == shape_to

    # Ensure the means are the same after reshaping.
    try:
        reshaped_mean = dist.mean.reshape(shape_to + dist.event_shape)
        assert (reshaped_mean.nan_to_num(th.pi) == reshaped.mean.nan_to_num(th.pi)).all()
    except NotImplementedError:
        pass

    # Ensure that samples have the correct shape.
    assert reshaped.sample().shape == shape_to + dist.event_shape
