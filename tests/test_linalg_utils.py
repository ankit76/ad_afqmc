import os
import pytest
import numpy as np
os.environ['JAX_ENABLE_X64'] = 'True'
os.environ['JAX_PLATFORM_NAME'] = 'cpu'
from jax import numpy as jnp
from ad_afqmc import linalg_utils

seed = 102
np.random.seed(seed)

def test_modified_cholesky():
  norb = 5
  size = norb*(norb+1)//2
  mat = np.random.rand(size,size)
  mat = mat @ mat.T
  mat = jnp.array(mat)
  chol = linalg_utils.modified_cholesky(mat, norb)
  mat_chol = 0. * mat
  for i in range(chol.shape[0]):
    chol_vec = chol[i].reshape(norb,norb)[np.tril_indices(norb)]
    mat_chol += np.einsum('i,j->ij', chol_vec, chol_vec) 
  assert np.allclose(mat, mat_chol)

if __name__ == '__main__':
  test_modified_cholesky()
