import numpy as np
from jax import numpy as jnp
import unittest

from ad_afqmc import config
config.setup_jax()
from ad_afqmc import linalg_utils

seed = 102
np.random.seed(seed)

class TestLinalgUtils(unittest.TestCase):

    # Trick to avoid it running with pytest
    __test__ = False

    def test_modified_cholesky(self):
        norb = 5
        size = norb * norb
        mat = np.random.randn(size, size)
        mat = mat @ mat.T
        mat = jnp.array(mat)
        chol = linalg_utils.modified_cholesky(mat, norb, size)
        mat_chol = 0.0 * mat
        for i in range(chol.shape[0]):
            chol_vec = chol[i]
            mat_chol += np.einsum("i,j->ij", chol_vec, chol_vec)
        self.assertTrue(np.allclose(mat, mat_chol))

if __name__ == "__main__":
    unittest.main()
