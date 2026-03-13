import warnings
from typing import List, Optional, Union

import numpy as np
import torch

from . import System


# Try to import featomic's implementation (preferred)
try:
    from featomic.torch import systems_to_torch as _featomic_systems_to_torch

    HAS_FEAUTOMIC = True
except ImportError:
    HAS_FEAUTOMIC = False


try:
    import ase

    HAS_ASE = True
except ImportError:
    HAS_ASE = False


class IntoSystem:
    """
    A type that can be converted into a :py:class:`metatomic.torch.System`.

    This is an abstract class that is used to indicate a class whose objects can be
    converted into a :py:class:`System`. For the moment, the only supported type is
    :py:class:`ase.Atoms`.
    """


def systems_to_torch(
    systems: Union[IntoSystem, List[IntoSystem]],
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
    positions_requires_grad: Optional[bool] = None,
    cell_requires_grad: Optional[bool] = None,
) -> Union[System, List[System]]:
    """
    Convert a system or a list of systems into a ``metatomic.torch.System`` or a list
    of such objects.

    If ``featomic`` is installed, this function uses featomic's implementation which
    supports additional system types (chemfiles, pyscf, etc.) and provides better
    handling of existing torch Systems.

    :param systems: The system or list of systems to convert.
    :param dtype: The dtype of the output tensors. If ``None``, the default
        dtype is used.
    :param device: The device of the output tensors. If ``None``, the default
        device is used.
    :param positions_requires_grad: The value of ``requires_grad`` on the output
        ``positions``. If ``None`` and the positions of the input is already a
        :py:class:`torch.Tensor`, ``requires_grad`` is kept the same. Otherwise it is
        initialized to ``False``.
    :param cell_requires_grad: The value of ``requires_grad`` on the output ``cell``. If
        ``None`` and the cell of the input is already a :py:class:`torch.Tensor`,
        ``requires_grad`` is kept the same. Otherwise it is initialized to ``False``.

    :return: The converted system or list of systems.
    """

    # Use featomic's implementation if available
    if HAS_FEAUTOMIC:
        # featomic now handles dtype/device natively
        return _featomic_systems_to_torch(
            systems,
            positions_requires_grad=positions_requires_grad,
            cell_requires_grad=cell_requires_grad,
            dtype=dtype,
            device=device,
        )

    # Fallback to ASE-only implementation
    if isinstance(systems, list):
        return [
            _system_to_torch(
                system, dtype, device, positions_requires_grad, cell_requires_grad
            )
            for system in systems
        ]
    else:
        return _system_to_torch(
            systems, dtype, device, positions_requires_grad, cell_requires_grad
        )



def _system_to_torch(
    system: IntoSystem,
    dtype: Optional[torch.dtype],
    device: Optional[torch.device],
    positions_requires_grad: Optional[bool],
    cell_requires_grad: Optional[bool],
) -> System:
    """
    Fallback implementation: Converts an ASE Atoms object into a ``metatomic.torch.System``.

    :param system: The system to convert.
    :param dtype: The dtype of the output tensors. If ``None``, the default
        dtype is used.
    :param device: The device of the output tensors. If ``None``, the default
        device is used.
    :param positions_requires_grad: Whether the positions tensors of
        the outputs should require gradients.
    :param cell_requires_grad: Whether the cell tensors of the outputs
        should require gradients.

    :return: The converted system.
    """
    if not HAS_ASE:
        raise RuntimeError("The `ase` package is required to convert systems to torch.")

    if not isinstance(system, ase.Atoms):
        raise ValueError(
            "Only `ase.Atoms` objects can be converted to `System`s "
            f"for now; got {type(system)}."
        )

    if dtype is None:
        # this is necessary because creating torch tensors from numpy arrays
        # takes the dtype from the numpy array, which is not always the default
        # dtype
        dtype = torch.get_default_dtype()

    positions = torch.tensor(
        system.positions,
        requires_grad=positions_requires_grad or False,
        dtype=dtype,
        device=device,
    )

    cell_vectors_are_not_zero = np.any(system.cell != 0, axis=1)
    if not np.all(cell_vectors_are_not_zero == system.pbc):
        warnings.warn(
            "A conversion to `System` was requested for an `ase.Atoms` object "
            "with one or more non-zero cell vectors but where the corresponding "
            "boundary conditions are set to `False`. "
            "The corresponding cell vectors will be set to zero.",
            stacklevel=3,
        )

    cell = torch.zeros((3, 3), dtype=dtype, device=device)

    pbc = torch.tensor(system.pbc, dtype=torch.bool, device=device)

    cell[pbc] = torch.tensor(system.cell[system.pbc], dtype=dtype, device=device)

    types = torch.tensor(system.numbers, device=device, dtype=torch.int32)

    return System(positions=positions, cell=cell, types=types, pbc=pbc)
