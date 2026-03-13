import warnings
from typing import List, Optional, Union

import numpy as np
import torch

from . import System


try:
    import ase

    HAS_ASE = True
except ImportError:
    HAS_ASE = False


try:
    import pyscf

    HAS_PYSCF = True
except ImportError:
    HAS_PYSCF = False


try:
    import chemfiles

    HAS_CHEMFILES = True
except ImportError:
    HAS_CHEMFILES = False


class IntoSystem:
    """
    A type that can be converted into a :py:class:`metatomic.torch.System`.

    This is an abstract class that is used to indicate a class whose objects can be
    converted into a :py:class:`System`. Supported types are:

    - :py:class:`ase.Atoms`: the Atomistic Simulation Environment Atoms class
    - :py:class:`pyscf.gto.mole.Mole` or :py:class:`pyscf.pbc.gto.cell.Cell`: pyscf system types
    - :py:class:`chemfiles.Frame`: chemfiles frame type
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

    This function supports multiple input types through automatic type detection:

    - ASE Atoms objects (requires ``ase`` package)
    - PySCF Mole or Cell objects (requires ``pyscf`` package)
    - Chemfiles Frame objects (requires ``chemfiles`` package)

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
    Converts a system into a ``metatomic.torch.System``.

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
    if dtype is None:
        dtype = torch.get_default_dtype()

    # Handle requires_grad: if None, default to False
    if positions_requires_grad is None:
        positions_requires_grad = False
    if cell_requires_grad is None:
        cell_requires_grad = False

    # Detect system type and extract data
    if HAS_ASE and isinstance(system, ase.Atoms):
        types, positions, cell, pbc = _extract_from_ase(system)
    elif HAS_PYSCF:
        result = _try_extract_from_pyscf(system)
        if result is not None:
            types, positions, cell, pbc = result
        else:
            raise TypeError(f"unknown system type: {type(system)}")
    elif HAS_CHEMFILES:
        result = _try_extract_from_chemfiles(system)
        if result is not None:
            types, positions, cell, pbc = result
        else:
            raise TypeError(f"unknown system type: {type(system)}")
    else:
        raise TypeError(f"unknown system type: {type(system)}")

    # Create torch tensors with proper dtype/device
    types_tensor = torch.tensor(types, dtype=torch.int32, device=device)
    positions_tensor = torch.tensor(positions, dtype=dtype, device=device)
    cell_tensor = torch.tensor(cell, dtype=dtype, device=device)
    pbc_tensor = torch.tensor(pbc, dtype=torch.bool, device=device)

    # Apply requires_grad if requested
    if positions_requires_grad:
        positions_tensor.requires_grad_(True)
    if cell_requires_grad:
        cell_tensor.requires_grad_(True)

    return System(
        positions=positions_tensor,
        cell=cell_tensor,
        types=types_tensor,
        pbc=pbc_tensor,
    )


def _extract_from_ase(atoms):
    """Extract system data from ASE Atoms object."""
    # Careful PBC handling: check for mismatched cell vectors and PBC flags
    cell_vectors_are_not_zero = np.any(atoms.cell != 0, axis=1)
    if not np.all(cell_vectors_are_not_zero == atoms.pbc):
        warnings.warn(
            "A conversion to `System` was requested for an `ase.Atoms` object "
            "with one or more non-zero cell vectors but where the corresponding "
            "boundary conditions are set to `False`. "
            "The corresponding cell vectors will be set to zero.",
            stacklevel=3,
        )

    # Build cell tensor, zeroing out non-periodic directions
    cell = np.zeros((3, 3), dtype=np.float64)
    pbc = np.array(atoms.pbc, dtype=bool)
    cell[pbc] = atoms.cell[pbc]

    return atoms.numbers, atoms.positions, cell, pbc


def _try_extract_from_pyscf(mol):
    """Try to extract system data from PySCF Mole or Cell object."""
    if not HAS_PYSCF:
        return None

    # Check for Mole or Cell
    if isinstance(mol, pyscf.gto.mole.Mole):
        # Extract from Mole
        positions = np.array(mol.atom_coords())
        numbers = np.array([mol.atom_charge(i) for i in range(mol.natm)])

        # Check if it's a periodic calculation
        if hasattr(mol, 'a') and mol.a is not None:
            cell = np.array(mol.a)
            pbc = np.array([True, True, True])
        else:
            cell = np.zeros((3, 3), dtype=np.float64)
            pbc = np.array([False, False, False])

        return numbers, positions, cell, pbc

    return None


def _try_extract_from_chemfiles(frame):
    """Try to extract system data from Chemfiles Frame object."""
    if not HAS_CHEMFILES:
        return None

    if isinstance(frame, chemfiles.Frame):
        # Extract from Frame
        topology = frame.topology
        positions = frame.positions
        cell = frame.unit_cell

        # Get atomic numbers from atom types
        numbers = []
        for atom in topology.atoms:
            # Try to get atomic number, fallback to mass-based guess
            atomic_num = atom.atomic_number
            if atomic_num == 0:
                # Guess from mass (approximate)
                mass = atom.mass
                if mass < 2:
                    atomic_num = 1  # H
                elif mass < 5:
                    atomic_num = 2  # He
                else:
                    atomic_num = int(round(mass / 2))  # Rough guess
            numbers.append(atomic_num)

        numbers = np.array(numbers, dtype=np.int32)

        # Convert cell to matrix form
        cell_matrix = np.zeros((3, 3), dtype=np.float64)
        if cell is not None:
            # Chemfiles stores cell as [a, b, c, alpha, beta, gamma]
            # or as a matrix - check which format
            if hasattr(cell, 'shape') and cell.shape == (3, 3):
                cell_matrix = np.array(cell)
            else:
                # Assume orthorhombic if we get lengths
                cell_matrix[0, 0] = cell[0] if len(cell) > 0 else 0.0
                cell_matrix[1, 1] = cell[1] if len(cell) > 1 else 0.0
                cell_matrix[2, 2] = cell[2] if len(cell) > 2 else 0.0

        pbc = np.array([True, True, True]) if np.any(cell_matrix != 0) else np.array([False, False, False])

        return numbers, positions, cell_matrix, pbc

    return None
