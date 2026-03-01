# Getting started

This tutorial walks through computing energies and forces with a metatomic model
using the ASE calculator interface, then runs a short geometry optimization.

## Prerequisites

Install the package:

```bash
pip install metatomic-ase
```

You need a saved metatomic model file (`.pt`). If you have a metatrain
checkpoint (`.ckpt`), install metatrain as well:

```bash
pip install metatrain
```

## Load the model

```python
from metatomic.ase import MetatomicCalculator

calc = MetatomicCalculator("path/to/model.pt")
```

The calculator detects the model's dtype and supported devices automatically.
Pass `device="cuda"` to run on GPU.

## Attach to ASE Atoms

```python
import ase.build

atoms = ase.build.bulk("Si", "diamond", a=5.43, cubic=True)
atoms.calc = calc
```

## Compute properties

```python
energy = atoms.get_potential_energy()   # scalar, eV
forces = atoms.get_forces()            # (n_atoms, 3), eV/A
stress = atoms.get_stress()            # Voigt 6-vector, eV/A^3
```

## Run a geometry optimization

```python
from ase.optimize import BFGS

atoms.rattle(stdev=0.05)  # perturb positions
opt = BFGS(atoms)
opt.run(fmax=0.01)
```

The optimizer calls `get_forces()` internally. Results are cached by ASE, so
redundant model evaluations are avoided.

## Next steps

- {doc}`/howto/model_loading` covers all supported input formats
- {doc}`/howto/symmetrized` explains rotational averaging with `SymmetrizedCalculator`
- {doc}`/explanation/architecture` describes the internals
