Exporting models
================

.. py:currentmodule:: metatomic.torch

Exporting models to work with any metatomic-compatible simulation engine is
done with the :py:class:`AtomisticModel` class. This class takes in an
arbitrary :py:class:`torch.nn.Module`, with a forward function that follows the
:py:class:`ModelInterface`. In addition to the actual model, you also need to
define some information about the model, using :py:class:`ModelMetadata` and
:py:class:`ModelCapabilities`.

For simple energy models, you can use :py:class:`ModelAdapter` as a base class
to avoid writing the ``forward()`` boilerplate yourself, and
:py:func:`export_model` to export in a single call.

.. autoclass:: metatomic.torch.ModelInterface
    :members:
    :show-inheritance:

.. autoclass:: metatomic.torch.AtomisticModel
    :members:

.. autoclass:: metatomic.torch.ModelAdapter
    :members:
    :show-inheritance:

.. autofunction:: metatomic.torch.export_model
