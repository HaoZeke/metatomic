from typing import Dict, List, Optional

import torch
from metatensor.torch import Labels, TensorBlock, TensorMap

from . import ModelOutput, NeighborListOptions, System


class ModelAdapter(torch.nn.Module):
    """Adapts a simple energy function to the full ModelInterface protocol.

    Subclass this and implement :py:meth:`compute_energy`. The adapter handles
    System unpacking, neighbor list management, and TensorMap construction.

    The ``compute_energy`` method receives raw tensors extracted from each
    :py:class:`System` and should return either a scalar tensor (total energy)
    or a 1D ``(n_atoms,)`` tensor (per-atom energies). The adapter then
    constructs the correct :py:class:`TensorMap` output with proper Labels,
    handles ``per_atom`` vs total energy requests, and applies
    ``selected_atoms`` filtering.

    This class is TorchScript-compatible: ``compute_energy`` is a concrete
    method (not abstract) that raises ``NotImplementedError`` at runtime.

    Example usage::

        class MyModel(ModelAdapter):
            def __init__(self):
                super().__init__(cutoff=5.0)
                self.linear = torch.nn.Linear(1, 1)

            def compute_energy(self, positions, types, cell, pbc, neighbors):
                distances = neighbors.values.reshape(-1, 3)
                r = torch.linalg.vector_norm(distances, dim=1)
                return torch.sum(1.0 / r**2)

    :param cutoff: neighbor list cutoff distance
    :param full_neighbor_list: whether to request a full (symmetric) neighbor
        list. Default ``True``.
    :param strict_neighbor_list: whether to request a strict neighbor list
        (excluding pairs beyond cutoff). Default ``True``.
    """

    def __init__(
        self,
        cutoff: float,
        full_neighbor_list: bool = True,
        strict_neighbor_list: bool = True,
    ):
        super().__init__()
        self._nl_options = NeighborListOptions(
            cutoff=cutoff,
            full_list=full_neighbor_list,
            strict=strict_neighbor_list,
        )

    def compute_energy(
        self,
        positions: torch.Tensor,
        types: torch.Tensor,
        cell: torch.Tensor,
        pbc: torch.Tensor,
        neighbors: TensorBlock,
    ) -> torch.Tensor:
        """Compute energy from raw system tensors.

        Override this method in your subclass. Return either:
        - a scalar tensor for total energy, or
        - a 1D ``(n_atoms,)`` tensor for per-atom energies.

        :param positions: atomic positions, shape ``(n_atoms, 3)``
        :param types: atomic types, shape ``(n_atoms,)``, int32
        :param cell: unit cell matrix, shape ``(3, 3)``
        :param pbc: periodic boundary conditions, shape ``(3,)``, bool
        :param neighbors: neighbor list as a :py:class:`TensorBlock`
        :returns: energy tensor (scalar or per-atom)
        """
        raise NotImplementedError("subclass must implement compute_energy")

    def requested_neighbor_lists(self) -> List[NeighborListOptions]:
        return [self._nl_options]

    def forward(
        self,
        systems: List[System],
        outputs: Dict[str, ModelOutput],
        selected_atoms: Optional[Labels],
    ) -> Dict[str, TensorMap]:
        results: Dict[str, TensorMap] = {}

        if "energy" not in outputs:
            return results

        per_atom_requested = outputs["energy"].per_atom

        all_block_values: List[torch.Tensor] = []
        all_block_samples: List[torch.Tensor] = []

        for system_i, system in enumerate(systems):
            positions = system.positions
            types = system.types
            cell = system.cell
            pbc = system.pbc
            neighbors = system.get_neighbor_list(self._nl_options)

            energy = self.compute_energy(positions, types, cell, pbc, neighbors)

            n_atoms = positions.shape[0]
            device = positions.device

            if energy.dim() == 0:
                # scalar: total energy
                if per_atom_requested:
                    # distribute evenly across atoms for per-atom output
                    per_atom_energy = (
                        energy
                        / n_atoms
                        * torch.ones(n_atoms, dtype=energy.dtype, device=device)
                    )
                else:
                    per_atom_energy = energy.reshape(1)
            else:
                # 1D tensor: per-atom energies
                per_atom_energy = energy

            if per_atom_requested:
                if selected_atoms is not None:
                    current_system_mask = selected_atoms.column("system") == system_i
                    current_atoms = selected_atoms.column("atom")
                    current_atoms = current_atoms[current_system_mask].to(torch.long)

                    per_atom_energy = per_atom_energy[current_atoms]

                    sample_values = torch.stack(
                        [
                            torch.full(
                                (len(current_atoms),),
                                system_i,
                                dtype=torch.int32,
                                device=device,
                            ),
                            current_atoms.to(torch.int32),
                        ],
                        dim=1,
                    )
                else:
                    sample_values = torch.stack(
                        [
                            torch.full(
                                (n_atoms,),
                                system_i,
                                dtype=torch.int32,
                                device=device,
                            ),
                            torch.arange(n_atoms, dtype=torch.int32, device=device),
                        ],
                        dim=1,
                    )

                all_block_values.append(per_atom_energy.reshape(-1, 1))
                all_block_samples.append(sample_values)
            else:
                # total energy per system
                if energy.dim() > 0 and energy.numel() > 1:
                    # per-atom energies, sum them
                    total = energy.sum().reshape(1, 1)
                else:
                    total = energy.reshape(1, 1)

                sample_values = torch.tensor(
                    [[system_i]], dtype=torch.int32, device=device
                )
                all_block_values.append(total)
                all_block_samples.append(sample_values)

        if len(all_block_values) > 0:
            block_values = torch.cat(all_block_values, dim=0)
            block_samples_values = torch.cat(all_block_samples, dim=0)
        else:
            # no systems: empty output
            if per_atom_requested:
                block_values = torch.zeros((0, 1), dtype=torch.float64)
                block_samples_values = torch.zeros((0, 2), dtype=torch.int32)
            else:
                block_values = torch.zeros((0, 1), dtype=torch.float64)
                block_samples_values = torch.zeros((0, 1), dtype=torch.int32)

        if per_atom_requested:
            sample_names = ["system", "atom"]
        else:
            sample_names = ["system"]

        energy_block = TensorBlock(
            values=block_values,
            samples=Labels(sample_names, block_samples_values),
            components=torch.jit.annotate(List[Labels], []),
            properties=Labels(
                ["energy"],
                torch.tensor([[0]], device=block_values.device),
            ),
        )

        energy_map = TensorMap(
            keys=Labels(["_"], torch.tensor([[0]], device=block_values.device)),
            blocks=[energy_block],
        )

        results["energy"] = energy_map
        return results
