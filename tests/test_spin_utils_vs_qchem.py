import numpy as np

from ad_afqmc import config

config.setup_jax()

from ad_afqmc.spin_utils import get_spin_covariance, spin_collinearity_test

seed = 102
np.random.seed(seed)

def test_spin_collinearity_test():
    filetag = 'spin_cov_test'
    spin_ref = np.loadtxt(f'{filetag}.spin.csv', dtype=float, delimiter=',', skiprows=1)
    spin_cov_ref = np.loadtxt(f'{filetag}.spin_cov.csv', dtype=np.complex128, delimiter=',', skiprows=1)
    A_ref = np.loadtxt(f'{filetag}.A.csv', dtype=float, delimiter=',', skiprows=1)
    Aevals_ref = np.loadtxt(f'{filetag}.Aevals.csv', dtype=float, delimiter=',', skiprows=1)
    Aevecs_ref = np.loadtxt(f'{filetag}.Aevecs.csv', dtype=float, delimiter=',', skiprows=1)
    psi0_ref = np.loadtxt(f'{filetag}.psi0.csv', dtype=float, delimiter=',', skiprows=1)

    nbsf, nocc = psi0_ref.shape
    nbsf = nbsf // 2
    ao_ovlp = np.eye(nbsf)

    epsilon0, mu, evals, evecs, A = spin_collinearity_test(psi0_ref, ao_ovlp, debug=True)

    np.testing.assert_allclose(A, A_ref, atol=1e-14)
    np.testing.assert_allclose(mu, Aevals_ref[0], atol=1e-14)
    np.testing.assert_allclose(evals, Aevals_ref, atol=1e-14)

    for i in range(3):
        try: np.testing.assert_allclose(evecs[:, i], Aevecs_ref[:, i], atol=1e-14)
        except Exception as e: np.testing.assert_allclose(evecs[:, i], -Aevecs_ref[:, i], atol=1e-14)

if __name__ == "__main__":
    test_spin_collinearity_test()






