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
