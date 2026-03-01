from typing import Dict, List, Optional

import torch

from . import ModelCapabilities, ModelMetadata, ModelOutput
from .model import AtomisticModel


def export_model(
    model: torch.nn.Module,
    path: str,
    *,
    atomic_types: List[int],
    interaction_range: float,
    length_unit: str = "angstrom",
    dtype: str = "float64",
    supported_devices: Optional[List[str]] = None,
    outputs: Optional[Dict[str, ModelOutput]] = None,
    metadata: Optional[ModelMetadata] = None,
    collect_extensions: Optional[str] = None,
):
    """Export a model implementing ``ModelInterface`` to a ``.pt`` file.

    This is a convenience function that constructs
    :py:class:`ModelCapabilities`, wraps the model in an
    :py:class:`AtomisticModel`, and saves it via TorchScript. The resulting
    file can be loaded by any simulation engine that supports metatomic models.

    The ``model`` must implement the ``ModelInterface`` protocol (i.e. its
    ``forward()`` must accept ``systems``, ``outputs``, ``selected_atoms``
    and return ``Dict[str, TensorMap]``). Models built with
    :py:class:`ModelAdapter` satisfy this automatically.

    :param model: a ``torch.nn.Module`` implementing ``ModelInterface``.
        Must be in eval mode (``model.eval()``).
    :param path: file path for the exported ``.pt`` model
    :param atomic_types: list of atomic types (integers) the model supports
    :param interaction_range: interaction range (cutoff) in ``length_unit``
    :param length_unit: unit for lengths/positions. Default ``"angstrom"``.
    :param dtype: model dtype, ``"float32"`` or ``"float64"``.
        Default ``"float64"``.
    :param supported_devices: devices the model can run on.
        Default ``["cpu"]``.
    :param outputs: dictionary of named :py:class:`ModelOutput` the model
        produces. Default: ``{"energy": ModelOutput(quantity="energy",
        unit="", per_atom=False)}``.
    :param metadata: optional :py:class:`ModelMetadata` (name, authors, etc.)
    :param collect_extensions: if not None, collect TorchScript extensions
        into this directory.
    """
    if supported_devices is None:
        supported_devices = ["cpu"]

    if outputs is None:
        outputs = {
            "energy": ModelOutput(
                quantity="energy",
                unit="",
                per_atom=False,
            ),
        }

    if metadata is None:
        metadata = ModelMetadata()

    capabilities = ModelCapabilities(
        outputs=outputs,
        atomic_types=atomic_types,
        interaction_range=interaction_range,
        length_unit=length_unit,
        supported_devices=supported_devices,
        dtype=dtype,
    )

    model = model.eval()
    atomistic = AtomisticModel(model, metadata, capabilities)
    atomistic.save(path, collect_extensions=collect_extensions)
