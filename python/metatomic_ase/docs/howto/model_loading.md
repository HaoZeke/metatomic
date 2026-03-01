# Loading models

`MetatomicCalculator` accepts several input formats. Each section below shows
one loading pattern.

## From a saved `.pt` file

The most common case. Pass the path to a TorchScript-exported metatomic model:

```python
from metatomic.ase import MetatomicCalculator

calc = MetatomicCalculator("path/to/model.pt")
```

The file must exist and contain a valid `AtomisticModel`. An `InputError` is
raised if the path does not exist.

## From a metatrain checkpoint

Pass a `.ckpt` path to load a metatrain checkpoint directly. This requires the
`metatrain` package:

```python
calc = MetatomicCalculator("path/to/checkpoint.ckpt")
```

The checkpoint is exported to an `AtomisticModel` internally.

## PET-MAD shortcut

The string `"pet-mad"` downloads and loads the PET-MAD universal model:

```python
calc = MetatomicCalculator("pet-mad")
```

This also requires `metatrain` to be installed. The model weights are fetched
from HuggingFace on first use.

## From a Python AtomisticModel

If you already have an `AtomisticModel` instance (for example, built
programmatically):

```python
from metatomic.torch import AtomisticModel

atomistic_model = build_my_model()  # returns AtomisticModel
calc = MetatomicCalculator(atomistic_model, device="cuda")
```

## From a TorchScript RecursiveScriptModule

If you have a scripted model loaded via `torch.jit.load`:

```python
import torch

scripted = torch.jit.load("model.pt")
calc = MetatomicCalculator(scripted)
```

The script module must have `original_name == "AtomisticModel"`. Otherwise an
`InputError` is raised.

## Selecting a device

By default, `MetatomicCalculator` picks the best device from the model's
`supported_devices`. Override with the `device` parameter:

```python
calc = MetatomicCalculator("model.pt", device="cuda:0")
```

## Extensions directory

Some models require compiled TorchScript extensions. Point to their location
with `extensions_directory`:

```python
calc = MetatomicCalculator(
    "model.pt",
    extensions_directory="path/to/extensions/",
)
```
