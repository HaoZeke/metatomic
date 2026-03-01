"""Tests for MetatomicCalculator model loading paths."""

import os

import metatomic_lj_test
import pytest
import torch
from ase.calculators.calculator import InputError

from metatomic.ase import MetatomicCalculator


CUTOFF = 5.0
SIGMA = 1.5808
EPSILON = 0.1729


def _make_lj_model():
    return metatomic_lj_test.lennard_jones_model(
        atomic_type=28,
        cutoff=CUTOFF,
        sigma=SIGMA,
        epsilon=EPSILON,
        length_unit="Angstrom",
        energy_unit="eV",
        with_extension=False,
    )


@pytest.fixture
def lj_model():
    return _make_lj_model()


def test_load_from_pt_file(lj_model, tmp_path):
    """Model loads from a saved .pt file."""
    pt_path = os.path.join(str(tmp_path), "test_model.pt")
    lj_model.save(pt_path)
    calc = MetatomicCalculator(pt_path)
    assert calc._device is not None


def test_load_from_atomistic_model(lj_model):
    """Model loads from an AtomisticModel instance."""
    calc = MetatomicCalculator(lj_model, uncertainty_threshold=None)
    assert calc._device is not None


def test_load_from_scripted_model(lj_model):
    """Model loads from a torch.jit.script-ed AtomisticModel."""
    scripted = torch.jit.script(lj_model)
    calc = MetatomicCalculator(scripted, uncertainty_threshold=None)
    assert calc._device is not None


def test_nonexistent_path_raises(tmp_path):
    """InputError raised for a path that does not exist."""
    with pytest.raises(InputError, match="does not exist"):
        MetatomicCalculator(os.path.join(str(tmp_path), "nonexistent.pt"))


def test_wrong_model_type_raises():
    """TypeError raised when passing an unsupported type."""
    with pytest.raises(TypeError, match="unknown type for model"):
        MetatomicCalculator(42)


def test_non_atomisticmodel_scriptmodule_raises():
    """InputError raised for a ScriptModule that is not AtomisticModel."""

    class Dummy(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x

    dummy_scripted = torch.jit.script(Dummy())
    with pytest.raises(InputError, match="must be 'AtomisticModel'"):
        MetatomicCalculator(dummy_scripted)
