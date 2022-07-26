import pytest
from torch_disttools import util


def test_setattr():
    class Foo:
        pass
    util._setattr(Foo, "foo", "bar")
    assert Foo.foo == "bar"


def test_setattr_exists():
    class Foo:
        foo = "baz"
    with pytest.warns(UserWarning):
        util._setattr(Foo, "foo", "bar")
    assert Foo.foo == "baz"
