import os
from typing import Dict

import pytest
import torch
from metatensor.torch import Labels, TensorBlock

from metatomic.torch import (
    ModelAdapter,
    ModelEvaluationOptions,
    ModelMetadata,
    ModelOutput,
    NeighborListOptions,
    System,
    export_model,
    load_atomistic_model,
)


# ---------------------------------------------------------------------------
# Inline test model: simple sum-of-inverse-r^2 pair potential via ModelAdapter
# ---------------------------------------------------------------------------


class SimplePairAdapter(ModelAdapter):
    """Minimal pair potential: sum of 1/r^2 over neighbor pairs.

    Returns per-atom energies (each atom gets half of each pair interaction).
    """

    def __init__(self, cutoff: float = 5.0):
        super().__init__(cutoff=cutoff, full_neighbor_list=False)

    def compute_energy(
        self,
        positions: torch.Tensor,
        types: torch.Tensor,
        cell: torch.Tensor,
        pbc: torch.Tensor,
        neighbors: TensorBlock,
    ) -> torch.Tensor:
        n_atoms = positions.shape[0]
        dtype = positions.dtype
        device = positions.device

        distances = neighbors.values.reshape(-1, 3)
        r2 = (distances * distances).sum(dim=1)
        pair_energy = 1.0 / r2

        all_i = neighbors.samples.column("first_atom").to(torch.long)
        all_j = neighbors.samples.column("second_atom").to(torch.long)

        energy = torch.zeros(n_atoms, dtype=dtype, device=device)
        energy = energy.index_add(0, all_i, pair_energy * 0.5)
        energy = energy.index_add(0, all_j, pair_energy * 0.5)

        return energy


class ScalarEnergyAdapter(ModelAdapter):
    """Returns total energy as a scalar (tests the scalar code path)."""

    def __init__(self, cutoff: float = 5.0):
        super().__init__(cutoff=cutoff, full_neighbor_list=False)

    def compute_energy(
        self,
        positions: torch.Tensor,
        types: torch.Tensor,
        cell: torch.Tensor,
        pbc: torch.Tensor,
        neighbors: TensorBlock,
    ) -> torch.Tensor:
        distances = neighbors.values.reshape(-1, 3)
        r2 = (distances * distances).sum(dim=1)
        return torch.sum(1.0 / r2)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_system_with_neighbors(
    nl_options: NeighborListOptions,
    dtype: torch.dtype = torch.float64,
) -> System:
    """Create a 2-atom system with a single neighbor pair."""
    positions = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=dtype)
    positions.requires_grad_(True)
    types = torch.tensor([1, 1])
    cell = torch.zeros((3, 3), dtype=dtype)
    pbc = torch.tensor([False, False, False])

    system = System(positions=positions, types=types, cell=cell, pbc=pbc)

    # Build neighbor list: atom 0 and atom 1 are neighbors at distance (1,0,0)
    nl_values = torch.tensor([[[1.0], [0.0], [0.0]]], dtype=dtype)
    nl_samples = Labels(
        ["first_atom", "second_atom", "cell_shift_a", "cell_shift_b", "cell_shift_c"],
        torch.tensor([[0, 1, 0, 0, 0]], dtype=torch.int32),
    )
    nl_components = [Labels.range("xyz", 3)]
    nl_properties = Labels.range("distance", 1)

    nl_block = TensorBlock(
        values=nl_values,
        samples=nl_samples,
        components=nl_components,
        properties=nl_properties,
    )
    system.add_neighbor_list(nl_options, nl_block)

    return system


# ---------------------------------------------------------------------------
# Tests for export_model
# ---------------------------------------------------------------------------


class TestExportModel:
    def test_export_basic(self, tmp_path):
        """export_model produces a loadable .pt file."""
        model = SimplePairAdapter(cutoff=5.0).eval()
        path = str(tmp_path / "model.pt")

        export_model(
            model,
            path,
            atomic_types=[1],
            interaction_range=5.0,
        )

        assert os.path.exists(path)
        loaded = load_atomistic_model(path)
        assert loaded is not None

    def test_export_custom_outputs(self, tmp_path):
        """Non-default outputs dict survives export."""
        model = SimplePairAdapter(cutoff=5.0).eval()
        path = str(tmp_path / "model.pt")

        custom_outputs = {
            "energy": ModelOutput(
                quantity="energy",
                unit="eV",
                per_atom=True,
            ),
        }

        export_model(
            model,
            path,
            atomic_types=[1, 6, 8],
            interaction_range=5.0,
            length_unit="angstrom",
            outputs=custom_outputs,
        )

        loaded = load_atomistic_model(path)
        caps = loaded.capabilities()
        assert "energy" in caps.outputs
        assert caps.outputs["energy"].unit == "eV"
        assert caps.outputs["energy"].per_atom

    def test_export_metadata(self, tmp_path):
        """ModelMetadata survives round-trip export/load."""
        model = SimplePairAdapter(cutoff=5.0).eval()
        path = str(tmp_path / "model.pt")

        metadata = ModelMetadata(
            name="test-model",
            description="a test",
            authors=["Test Author"],
        )

        export_model(
            model,
            path,
            atomic_types=[1],
            interaction_range=5.0,
            metadata=metadata,
        )

        loaded = load_atomistic_model(path)
        loaded_meta = loaded.metadata()
        assert loaded_meta.name == "test-model"
        assert loaded_meta.description == "a test"
        assert loaded_meta.authors == ["Test Author"]

    def test_export_defaults(self, tmp_path):
        """Default arguments produce sensible capabilities."""
        model = SimplePairAdapter(cutoff=5.0).eval()
        path = str(tmp_path / "model.pt")

        export_model(
            model,
            path,
            atomic_types=[1, 8],
            interaction_range=5.0,
        )

        loaded = load_atomistic_model(path)
        caps = loaded.capabilities()
        assert caps.length_unit == "angstrom"
        assert caps.dtype == "float64"
        assert "cpu" in caps.supported_devices
        assert caps.atomic_types == [1, 8]
        assert caps.interaction_range == 5.0


# ---------------------------------------------------------------------------
# Tests for ModelAdapter
# ---------------------------------------------------------------------------


class TestModelAdapter:
    def test_adapter_per_atom_energy(self):
        """SimplePairAdapter returns correct per-atom energy structure."""
        model = SimplePairAdapter(cutoff=5.0)
        model.eval()

        nl_options = model.requested_neighbor_lists()[0]
        system = _make_system_with_neighbors(nl_options)

        outputs = {"energy": ModelOutput(per_atom=True)}
        result = model.forward([system], outputs, None)

        assert "energy" in result
        energy_map = result["energy"]
        block = energy_map.block(0)

        # 2 atoms, per-atom output
        assert block.values.shape == (2, 1)
        assert block.samples.names == ["system", "atom"]

        # energy should be 0.5 / r^2 for each atom (half-list, split equally)
        # r = 1.0, so each atom gets 0.5
        expected = 0.5
        assert torch.allclose(
            block.values,
            torch.tensor([[expected], [expected]], dtype=torch.float64),
        )

    def test_adapter_total_energy(self):
        """ScalarEnergyAdapter returns correct total energy structure."""
        model = ScalarEnergyAdapter(cutoff=5.0)
        model.eval()

        nl_options = model.requested_neighbor_lists()[0]
        system = _make_system_with_neighbors(nl_options)

        outputs = {"energy": ModelOutput(per_atom=False)}
        result = model.forward([system], outputs, None)

        assert "energy" in result
        energy_map = result["energy"]
        block = energy_map.block(0)

        # total energy for 1 system
        assert block.values.shape == (1, 1)
        assert block.samples.names == ["system"]

        # 1/r^2 = 1.0 (r=1)
        assert torch.allclose(
            block.values,
            torch.tensor([[1.0]], dtype=torch.float64),
        )

    def test_adapter_multiple_systems(self):
        """Adapter handles multiple systems correctly."""
        model = SimplePairAdapter(cutoff=5.0)
        model.eval()

        nl_options = model.requested_neighbor_lists()[0]
        s1 = _make_system_with_neighbors(nl_options)
        s2 = _make_system_with_neighbors(nl_options)

        outputs = {"energy": ModelOutput(per_atom=False)}
        result = model.forward([s1, s2], outputs, None)

        block = result["energy"].block(0)
        assert block.values.shape == (2, 1)
        assert block.samples.names == ["system"]

        # Both systems should have the same energy
        assert torch.allclose(block.values[0], block.values[1])

    def test_adapter_selected_atoms(self):
        """Adapter respects selected_atoms filtering."""
        model = SimplePairAdapter(cutoff=5.0)
        model.eval()

        nl_options = model.requested_neighbor_lists()[0]
        system = _make_system_with_neighbors(nl_options)

        # Select only atom 0
        selected = Labels(
            ["system", "atom"],
            torch.tensor([[0, 0]], dtype=torch.int32),
        )

        outputs = {"energy": ModelOutput(per_atom=True)}
        result = model.forward([system], outputs, selected)

        block = result["energy"].block(0)
        # Only 1 atom selected
        assert block.values.shape == (1, 1)
        assert block.samples.names == ["system", "atom"]

    def test_adapter_empty_outputs(self):
        """Adapter returns empty dict when energy not requested."""
        model = SimplePairAdapter(cutoff=5.0)
        model.eval()

        nl_options = model.requested_neighbor_lists()[0]
        system = _make_system_with_neighbors(nl_options)

        outputs: Dict[str, ModelOutput] = {}
        result = model.forward([system], outputs, None)

        assert result == {}

    def test_adapter_no_systems(self):
        """Adapter handles zero systems."""
        model = SimplePairAdapter(cutoff=5.0)
        model.eval()

        outputs = {"energy": ModelOutput(per_atom=False)}
        result = model.forward([], outputs, None)

        assert "energy" in result
        block = result["energy"].block(0)
        assert block.values.shape[0] == 0

    def test_adapter_export_roundtrip(self, tmp_path):
        """Full pipeline: ModelAdapter -> export_model -> load -> forward."""
        model = SimplePairAdapter(cutoff=5.0)
        path = str(tmp_path / "adapter_model.pt")

        export_model(
            model,
            path,
            atomic_types=[1],
            interaction_range=5.0,
            outputs={
                "energy": ModelOutput(
                    quantity="energy",
                    unit="",
                    per_atom=True,
                ),
            },
        )

        loaded = load_atomistic_model(path)

        # Build a system with neighbor list matching the loaded model's requests
        nl_requests = loaded.requested_neighbor_lists()
        assert len(nl_requests) == 1

        system = _make_system_with_neighbors(nl_requests[0])

        eval_outputs = {
            "energy": ModelOutput(per_atom=True),
        }
        eval_options = ModelEvaluationOptions(
            length_unit="angstrom", outputs=eval_outputs
        )

        result = loaded([system], eval_options, check_consistency=True)

        assert "energy" in result
        energy_map = result["energy"]
        block = energy_map.block(0)
        assert block.values.shape == (2, 1)

        # Non-zero energy
        assert torch.any(block.values != 0.0)

    def test_adapter_forces_through_autograd(self):
        """Autograd forces work through the adapter."""
        model = SimplePairAdapter(cutoff=5.0)
        model.eval()

        nl_options = model.requested_neighbor_lists()[0]
        system = _make_system_with_neighbors(nl_options)

        outputs = {"energy": ModelOutput(per_atom=False)}
        result = model.forward([system], outputs, None)

        energy = result["energy"].block(0).values.sum()
        energy.backward()

        forces = -system.positions.grad
        assert forces is not None
        assert forces.shape == (2, 3)
        # For 1/r^2 potential at r=1, forces should be non-zero along x
        assert torch.any(forces.abs() > 0)


# ---------------------------------------------------------------------------
# Tests using lj-test (skipped if not installed)
# ---------------------------------------------------------------------------


class TestLJModel:
    @pytest.fixture
    def lj_model(self):
        metatomic_lj_test = pytest.importorskip("metatomic_lj_test")
        return metatomic_lj_test.lennard_jones_model(
            atomic_type=1,
            cutoff=5.0,
            epsilon=0.5,
            sigma=1.5,
            length_unit="angstrom",
            energy_unit="eV",
            with_extension=False,
        )

    @pytest.fixture
    def lj_system(self, lj_model):
        """Create a system compatible with the LJ model."""
        positions = torch.tensor(
            [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]],
            dtype=torch.float64,
        )
        positions.requires_grad_(True)
        types = torch.tensor([1, 1, 1])
        cell = torch.zeros((3, 3), dtype=torch.float64)
        pbc = torch.tensor([False, False, False])

        system = System(positions=positions, types=types, cell=cell, pbc=pbc)

        # Add neighbor lists for all requests
        for nl_options in lj_model.requested_neighbor_lists():
            # Build a simple neighbor list for the 3-atom cluster
            pairs = []
            pair_distances = []
            for i in range(3):
                for j in range(i + 1, 3):
                    dist = positions[j].detach() - positions[i].detach()
                    r = torch.linalg.vector_norm(dist)
                    if r.item() <= nl_options.cutoff:
                        pairs.append([i, j, 0, 0, 0])
                        pair_distances.append(dist.tolist())

                        if nl_options.full_list:
                            pairs.append([j, i, 0, 0, 0])
                            pair_distances.append((-dist).tolist())

            if len(pairs) > 0:
                nl_samples = Labels(
                    [
                        "first_atom",
                        "second_atom",
                        "cell_shift_a",
                        "cell_shift_b",
                        "cell_shift_c",
                    ],
                    torch.tensor(pairs, dtype=torch.int32),
                )
                nl_values = torch.tensor(pair_distances, dtype=torch.float64).reshape(
                    -1, 3, 1
                )
            else:
                nl_samples = Labels(
                    [
                        "first_atom",
                        "second_atom",
                        "cell_shift_a",
                        "cell_shift_b",
                        "cell_shift_c",
                    ],
                    torch.zeros((0, 5), dtype=torch.int32),
                )
                nl_values = torch.zeros((0, 3, 1), dtype=torch.float64)

            nl_block = TensorBlock(
                values=nl_values,
                samples=nl_samples,
                components=[Labels.range("xyz", 3)],
                properties=Labels.range("distance", 1),
            )
            system.add_neighbor_list(nl_options, nl_block)

        return system

    def test_export_lj_model(self, lj_model, tmp_path):
        """LJ model exports and reloads via export_model (re-wrapping)."""
        # lj_model is already an AtomisticModel, so we extract the inner module
        inner = lj_model.module
        caps = lj_model.capabilities()

        path = str(tmp_path / "lj.pt")
        export_model(
            inner,
            path,
            atomic_types=caps.atomic_types,
            interaction_range=caps.interaction_range,
            length_unit=caps.length_unit,
            dtype=caps.dtype,
            supported_devices=caps.supported_devices,
            outputs=caps.outputs,
        )

        loaded = load_atomistic_model(path)
        assert loaded is not None

    def test_export_lj_forward(self, lj_model, lj_system, tmp_path):
        """Exported LJ model produces correct output structure."""
        inner = lj_model.module
        caps = lj_model.capabilities()

        path = str(tmp_path / "lj.pt")
        export_model(
            inner,
            path,
            atomic_types=caps.atomic_types,
            interaction_range=caps.interaction_range,
            length_unit=caps.length_unit,
            dtype=caps.dtype,
            supported_devices=caps.supported_devices,
            outputs=caps.outputs,
        )

        loaded = load_atomistic_model(path)

        eval_outputs = {
            "energy": ModelOutput(per_atom=False),
        }
        eval_options = ModelEvaluationOptions(
            length_unit="angstrom", outputs=eval_outputs
        )

        result = loaded([lj_system], eval_options, check_consistency=True)

        assert "energy" in result
        block = result["energy"].block(0)
        assert block.values.shape == (1, 1)

    def test_export_lj_forces(self, lj_model, lj_system, tmp_path):
        """Autograd forces work through exported LJ model."""
        inner = lj_model.module
        caps = lj_model.capabilities()

        path = str(tmp_path / "lj.pt")
        export_model(
            inner,
            path,
            atomic_types=caps.atomic_types,
            interaction_range=caps.interaction_range,
            length_unit=caps.length_unit,
            dtype=caps.dtype,
            supported_devices=caps.supported_devices,
            outputs=caps.outputs,
        )

        loaded = load_atomistic_model(path)

        eval_outputs = {
            "energy": ModelOutput(per_atom=False),
        }
        eval_options = ModelEvaluationOptions(
            length_unit="angstrom", outputs=eval_outputs
        )

        result = loaded([lj_system], eval_options, check_consistency=True)

        energy = result["energy"].block(0).values.sum()
        energy.backward()

        forces = -lj_system.positions.grad
        assert forces is not None
        assert forces.shape == (3, 3)
        assert torch.any(forces.abs() > 0)
