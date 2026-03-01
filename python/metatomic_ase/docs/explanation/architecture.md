# Architecture

This page explains how `MetatomicCalculator` bridges ASE and metatomic.

## ASE Calculator protocol

ASE calculators implement a `calculate(atoms, properties, system_changes)`
method. When a user calls `atoms.get_potential_energy()`, ASE checks whether
results are cached, and if not, calls `calculate()` with the appropriate
properties list.

`MetatomicCalculator` maps ASE property names (`energy`, `forces`, `stress`)
to metatomic `ModelOutput` requests, runs the model, and stores the results in
`self.results` for ASE to consume.

## Atoms to System conversion

ASE represents structures as `ase.Atoms` objects with numpy arrays. Metatomic
models expect `metatomic.torch.System` with torch tensors.

The conversion (`_ase_to_torch_data`) handles:

- Atomic numbers to `types` (int32 tensor)
- Positions to the model's dtype and device
- Cell vectors, respecting mixed PBC
- PBC flags as a boolean tensor

## Forces via autograd

When conservative forces are requested (the default), forces are computed as
the negative gradient of the energy with respect to positions:

```
F_i = -dE/dr_i
```

Positions are set to `requires_grad_(True)` before the forward pass, and
`backward()` is called on the energy output.

## Stress via the strain trick

Stress uses the Knuth strain trick. An identity strain tensor
(3x3, `requires_grad=True`) is applied to both positions and cell:

```
r' = r @ strain
h' = h @ strain
```

The stress per system is:

```
sigma = (1/V) * dE/d(strain)
```

This gives the full 3x3 stress tensor, which is then converted to Voigt
6-vector format for ASE.

## Non-conservative mode

Setting `non_conservative=True` uses model-predicted forces and stresses
directly, skipping the backward pass. This can be faster but may not conserve
energy in MD simulations.

## Neighbor lists

Models specify required neighbor lists via `requested_neighbor_lists()`. The
calculator computes them using:

- **vesin**: Default backend for CPU and GPU. Handles half and full neighbor
  lists. Systems on non-CPU/CUDA devices are temporarily moved to CPU.
- **nvalchemiops**: Used automatically on CUDA for full neighbor lists when
  installed. Keeps everything on GPU, avoiding host-device transfers.

## SymmetrizedCalculator

The `SymmetrizedCalculator` wraps any `MetatomicCalculator` and applies
rotational averaging via O(3) quadrature. It rotates the input structure,
evaluates the base calculator for each rotation, back-rotates the results, and
computes weighted averages. Optionally, discrete space group operations are
applied as a post-processing step using `spglib`.

## Why a separate package

metatomic-ase has its own versioning, release schedule, and dependency set
(`ase`). Keeping it separate from metatomic-torch avoids forcing an ASE
dependency on users who only need the core torch functionality or other
integrations like TorchSim.

The package is pure Python with no compiled extensions, making it lightweight
to install.
