# Symmetrized calculator

The `SymmetrizedCalculator` wraps a `MetatomicCalculator` and averages its
predictions over rotations to enforce equivariance. This is useful for models
that are not inherently equivariant or that show residual orientation dependence.

## Basic usage

```python
from metatomic.ase import MetatomicCalculator, SymmetrizedCalculator

base_calc = MetatomicCalculator("model.pt")
calc = SymmetrizedCalculator(base_calc, l_max=3)

atoms.calc = calc
energy = atoms.get_potential_energy()
forces = atoms.get_forces()
```

## How it works

The calculator applies a quadrature over the orthogonal group O(3):

1. **Lebedev quadrature** on the unit sphere S^2 for polar angles
2. **Equispaced sampling** of the unit circle S^1 for in-plane rotations
3. **Inversion** (optional) to cover both proper and improper rotations

For each quadrature point, the input structure is rotated, the base calculator
is evaluated, and the results are back-rotated and averaged with the
quadrature weights.

## Choosing `l_max`

The `l_max` parameter controls the quadrature order. It should match the
highest spherical harmonic degree that your model can represent. Higher values
give more accurate averaging but cost more evaluations.

```python
# Low-order model
calc = SymmetrizedCalculator(base_calc, l_max=3)

# Higher-order model
calc = SymmetrizedCalculator(base_calc, l_max=9)
```

## Batch size

By default, all rotated systems are evaluated at once. For large systems or
high `l_max`, this can exceed GPU memory. Use `batch_size` to evaluate in
chunks:

```python
calc = SymmetrizedCalculator(base_calc, l_max=9, batch_size=10)
```

## Space group averaging

For periodic systems, you can also average over the discrete space group
operations. This is applied after the O(3) averaging:

```python
calc = SymmetrizedCalculator(
    base_calc,
    l_max=3,
    apply_space_group_symmetry=True,
)
```

This requires `spglib` to be installed (`pip install metatomic-ase[symmetrized]`).

## Rotational standard deviation

To monitor how much the predictions vary with orientation, enable
`store_rotational_std`:

```python
calc = SymmetrizedCalculator(base_calc, l_max=3, store_rotational_std=True)
atoms.calc = calc
atoms.get_forces()

# Access standard deviations from the results dict
print(calc.results.get("energy_rot_std"))
print(calc.results.get("forces_rot_std"))
```
