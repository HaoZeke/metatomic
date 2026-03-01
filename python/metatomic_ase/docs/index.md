# metatomic-ase

```{toctree}
:hidden:

tutorials/getting_started
```

```{toctree}
:hidden:
:caption: How-to Guides

howto/model_loading
howto/symmetrized
```

```{toctree}
:hidden:
:caption: Understanding

explanation/architecture
```

```{toctree}
:hidden:
:caption: Reference

autoapi/metatomic/ase/index
changelog
```

**metatomic-ase** provides an [ASE](https://wiki.fysik.dtu.dk/ase/) calculator
for [metatomic](https://docs.metatensor.org/metatomic/latest/) atomistic models.

## Features

- Run any metatomic-compatible model (PET-MAD, MACE, etc.) as an ASE calculator
- Compute energies, forces, and stresses
- Rotational averaging via `SymmetrizedCalculator` (O(3) and space group)
- GPU-accelerated neighbor lists via nvalchemiops when available

## Quick install

```bash
pip install metatomic-ase
```

## Minimal example

```python
from metatomic.ase import MetatomicCalculator
import ase.build

atoms = ase.build.bulk("Si", "diamond", a=5.43)
atoms.calc = MetatomicCalculator("model.pt")

energy = atoms.get_potential_energy()
forces = atoms.get_forces()
stress = atoms.get_stress()
```
