import pytest
import torch as th
import torch_disttools as td
import typing
from .conftest import assert_allclose, distribution_from_spec


class KeyGetter:
    """
    Helper class for getting the key equivalent to indexing with brackets.
    """
    def __getitem__(self, key):
        return key


key_getter = KeyGetter()


def check_indexed_dist(dist: th.distributions.Distribution, key, expected_shape):
    indexed = td.index(dist, key)
    assert indexed.batch_shape == th.zeros(dist.batch_shape)[key].shape
    assert indexed.batch_shape == expected_shape, (dist, key, expected_shape)

    # We need to explicitly account for the trailing dimension of samples if there are ellipsis.
    if isinstance(key, tuple) and Ellipsis in key:
        sample_key = key + (slice(None),) * len(dist.event_shape)
    else:
        sample_key = key

    try:
        assert_allclose(dist.mean[sample_key], indexed.mean, dist)
    except NotImplementedError:
        pass

    x = dist.sample()
    assert_allclose(dist.log_prob(x)[key], indexed.log_prob(x[sample_key]))


@pytest.fixture(params=[
    ((3, 4, 5), key_getter[0, 1:, ::2], (3, 3)),
    ((3, 4, 5), key_getter[0, 1:, [2, 4]], (3, 2)),
    ((3, 4, 5), 2, (4, 5)),
    ((2,), key_getter[..., None], (2, 1)),
])
def batch_shape_and_key_and_expected_shape(request: pytest.FixtureRequest):
    """
    Tuple of batch shapes and keys for indexing.
    """
    return request.param


def test_reshape(
        batch_shape_and_key_and_expected_shape: typing.Tuple[th.Size, typing.Tuple, typing.Tuple],
        distribution_specification):
    batch_shape, key, expected_shape = batch_shape_and_key_and_expected_shape
    dist = distribution_from_spec(batch_shape, *distribution_specification)
    check_indexed_dist(dist, key, expected_shape)
