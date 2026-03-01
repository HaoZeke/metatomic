# metatomic-ase

ASE calculator for metatomic atomistic models.

Wraps metatomic models as ASE `Calculator` instances, enabling their use in
ASE molecular dynamics, geometry optimization, and other simulation workflows.

## Installation

```bash
pip install metatomic-ase
```

For rotational symmetrization (SymmetrizedCalculator):

```bash
pip install metatomic-ase[symmetrized]
```

To use metatrain checkpoints (`.ckpt` files) or the `pet-mad` shortcut:

```bash
pip install metatomic-ase[metatrain]
```

## Usage

```python
from metatomic.ase import MetatomicCalculator

# From a saved .pt model
calc = MetatomicCalculator("model.pt")

# Attach to ASE Atoms
atoms.calc = calc
energy = atoms.get_potential_energy()
forces = atoms.get_forces()
stress = atoms.get_stress()
```
