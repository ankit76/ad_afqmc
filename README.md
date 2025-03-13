# Automatically differentiable AFQMC

An end-to-end differentiable Auxiliary Field Quantum Monte Carlo (AFQMC) code based on Jax.

## Usage

The code can be installed as a package using pip:

```
  pip install .
```

For use on GPUs with CUDA, install as:

```
  pip install .[gpu]
```

Currently MPI is only used for CPU calculations. To install with MPI support, use:

```
  pip install .[mpi]
```

This code is interfaced with pyscf for molecular integral evaluation. The examples therefore require pyscf to be installed and can be run as:

```
  python hchain.py > hchain.out
  python nh3.py > nh3.out
```
