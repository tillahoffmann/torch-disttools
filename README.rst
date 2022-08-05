torch_disttools
===============

.. image:: https://github.com/tillahoffmann/torch-disttools/actions/workflows/main.yml/badge.svg
  :target: https://github.com/tillahoffmann/torch-disttools/actions/workflows/main.yml


:code:`torch_disttools` is a simple library that extends functionality in the :code:`torch.distributions` module.

Reshaping distributions
-----------------------

The batch dimensions of a distribution can be reshaped like a tensor.

.. doctest:: group

  >>> import torch as th
  >>> import torch_disttools as td  # Importing adds member functions to distributions.
  >>>
  >>> # Create a simple distribution.
  >>> batch_shape = (3, 4, 5)
  >>> dist = th.distributions.Normal(th.randn(batch_shape), th.ones(batch_shape))
  >>> dist
  Normal(loc: torch.Size([3, 4, 5]), scale: torch.Size([3, 4, 5]))
  >>> # Reshape it.
  >>> batch_reshaped = (10, 6)
  >>> reshaped = dist.reshape((10, 6))  # Or use td.reshape((10, 6)).
  >>> reshaped
  Normal(loc: torch.Size([10, 6]), scale: torch.Size([10, 6]))
  >>> # Check that the reshaped mean and the mean of the reshaped distribution are the same.
  >>> th.allclose(dist.mean.reshape(batch_reshaped), reshaped.mean)
  True


  >>> # A simple doctest.
  >>> 2 + 2
  4
