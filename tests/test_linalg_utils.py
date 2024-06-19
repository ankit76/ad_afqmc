import os

import numpy as np
import pytest

os.environ["JAX_ENABLE_X64"] = "True"
os.environ["JAX_PLATFORM_NAME"] = "cpu"
from jax import numpy as jnp

from ad_afqmc import linalg_utils

seed = 102
np.random.seed(seed)


def test_modified_cholesky():
    norb = 5
    size = norb**2
    mat = np.random.randn(size, size)
    mat = mat @ mat.T
    mat = jnp.array(mat)
    chol = linalg_utils.modified_cholesky(mat, norb, size)
    mat_chol = 0.0 * mat
    for i in range(chol.shape[0]):
        chol_vec = chol[i]
        mat_chol += np.einsum("i,j->ij", chol_vec, chol_vec)
    assert np.allclose(mat, mat_chol)

def test_sherman_morrison():
    tol = 1e-15
    norb = 5
    size = norb**2
    A = np.random.randn(size, size)
    A = A @ A.T
    
    while abs(jnp.linalg.det(A)) < tol:
        A = np.random.randn(size, size)
        A = A @ A.T

    Ainv = jnp.linalg.inv(A)
    u = np.random.randn(size)
    v = np.random.randn(size)

    while abs(1 + v @ Ainv @ u) < tol:
        u = np.random.randn(size)
        v = np.random.randn(size)

    mat = A + jnp.outer(u, v)
    matinv = jnp.linalg.inv(mat)
    test = linalg_utils.sherman_morrison(Ainv, u, v)
    assert np.allclose(matinv, test)

def test_mat_det_lemma():
    tol = 1e-15
    norb = 5
    size = norb**2
    A = np.random.randn(size, size)
    A = A @ A.T
    
    while abs(jnp.linalg.det(A)) < tol:
        A = np.random.randn(size, size)
        A = A @ A.T

    Ainv = jnp.linalg.inv(A)
    detA = jnp.linalg.det(A)
    u = np.random.randn(size)
    v = np.random.randn(size)

    mat = A + jnp.outer(u, v)
    detmat = jnp.linalg.det(mat)
    test = linalg_utils.mat_det_lemma(detA, Ainv, u, v)
    assert np.allclose(detmat, test)


if __name__ == "__main__":
    test_modified_cholesky()
    test_sherman_morrison()
    test_mat_det_lemma()
