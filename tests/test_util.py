import pytest
import torch as th
from torch.distributions import constraints
from torch_disttools import util


def test_setattr():
    class Foo:
        pass
    util._setattr(Foo, "foo", "bar")
    assert Foo.foo == "bar"


@pytest.mark.parametrize("override", [False, True])
def test_setattr_exists(override: bool):
    class Foo:
        foo = "baz"
    if override:
        util._setattr(Foo, "foo", "bar", override=override)
        assert Foo.foo == "bar"
    else:
        with pytest.warns(UserWarning):
            util._setattr(Foo, "foo", "bar", override=override)
        assert Foo.foo == "baz"
