# Automatic differentiable AFQMC

A code to calculate derivatives of energy in auxiliary field quantum Monte Carlo calculations using automatic differentiation.

## Usage

Requires:

* python (>= 3.7)
* mpi4py
* numpy, scipy
* jax, jaxlib 

The only finicky dependency is mpi4py, which requires an openmpi setup. Once all the requirements are installed this package can be used by adding it to the PYTHONPATH. 

This code is interfaced with pyscf for molecular integral evaluation. The examples therefore require pyscf to be installed and can be run as:

```
  python hchain.py > hchain.out
  python nh3.py > nh3.out
```

MPI calls are handled internally (rather crudely), which may lead to issues in some MPI setups. In this case, pyscf calculations need to be performed separately before the MPI calls to ad_afqmc.

