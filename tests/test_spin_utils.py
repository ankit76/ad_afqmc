import numpy as np
import sys
sys.path.append('/burg/home/su2254/libs/ad_afqmc')

from ad_afqmc import config

config.setup_jax()

from ad_afqmc.spin_utils import get_spin_covariance, spin_collinearity_test

seed = 102
np.random.seed(seed)

def test_spin_collinearity_test():
    nbsf = 4
    ao_ovlp = np.eye(nbsf)

    # Random dm and coeffs.
    Paa = np.random.rand(nbsf, nbsf)
    Pbb = np.random.rand(nbsf, nbsf)
    Paa += Paa.T.conj()
    Pbb += Pbb.T.conj()
    _, Ca = np.linalg.eigh(Paa)
    _, Cb = np.linalg.eigh(Pbb)

    # Test 1: UHF polarized.
    print('\n# ------')
    print('# Test 1')
    print('# ------')
    nelec = (3, 0)
    nocc = np.sum(nelec)
    psi0 = np.zeros((2*nbsf, nocc))
    psi0[:nbsf, :nelec[0]] = Ca[:, :nelec[0]]
    psi0[nbsf:, nelec[0]:] = Cb[:, :nelec[1]]
    psi0a = psi0[:nbsf] # (nbsf, nocc)
    psi0b = psi0[nbsf:]

    Aref, _ = get_spin_covariance(psi0a, psi0b, ao_ovlp)
    epsilon0, mu, evals, evecs, A = spin_collinearity_test(psi0, ao_ovlp, debug=True)

    np.testing.assert_allclose(A, Aref, atol=1e-14)
    np.testing.assert_allclose(epsilon0, 0.5*(nelec[0] - nelec[1]), atol=1e-14)
    np.testing.assert_allclose(mu, 0., atol=1e-14)

    # Check equality up to a sign.
    try: np.testing.assert_allclose(evecs[:, 0], np.array([0., 0., 1.]), atol=1e-14)
    except Exception as e: np.testing.assert_allclose(evecs[:, 0], -np.array([0., 0., 1.]), atol=1e-14)


    # Test 2: UHF unpolarized.
    print('\n# ------')
    print('# Test 2')
    print('# ------')
    nelec = (4, 4)
    nocc = np.sum(nelec)
    psi0 = np.zeros((2*nbsf, nocc))
    psi0[:nbsf, :nelec[0]] = Ca[:, :nelec[0]]
    psi0[nbsf:, nelec[0]:] = Cb[:, :nelec[1]]
    psi0a = psi0[:nbsf] # (nbsf, nocc)
    psi0b = psi0[nbsf:]

    Aref, _ = get_spin_covariance(psi0a, psi0b, ao_ovlp)
    epsilon0, mu, evals, evecs, A = spin_collinearity_test(psi0, ao_ovlp, debug=True)

    np.testing.assert_allclose(A, Aref, atol=1e-14)
    np.testing.assert_allclose(epsilon0, 0.5*(nelec[0] - nelec[1]), atol=1e-14)
    np.testing.assert_allclose(mu, 0., atol=1e-14)

    # Check equality up to a sign.
    #try: np.testing.assert_allclose(evecs[:, 0], np.array([0., 0., 1.]), atol=1e-14)
    #except Exception as e: np.testing.assert_allclose(evecs[:, 0], -np.array([0., 0., 1.]), atol=1e-14)

    # Test 3: rotated UHF polarized.
    print('\n# ------')
    print('# Test 3')
    print('# ------')
    nelec = (3, 0)
    nocc = np.sum(nelec)
    psi0 = np.zeros((2*nbsf, nocc))
    psi0[:nbsf, :nelec[0]] = Ca[:, :nelec[0]]
    psi0[nbsf:, nelec[0]:] = Cb[:, :nelec[1]]
    psi0a = psi0[:nbsf] # (nbsf, nocc)
    psi0b = psi0[nbsf:]

    # Rotate spin axis to x-axis.
    theta = np.pi / 2.
    Uspin = np.array(
                [[np.cos(theta / 2.0), -np.sin(theta / 2.)],
                 [np.sin(theta / 2.0), np.cos(theta / 2.)]])
    U = np.kron(Uspin, np.eye(nbsf))
    psi0 = U.dot(psi0)

    Aref, _ = get_spin_covariance(psi0[:nbsf], psi0[nbsf:], ao_ovlp)
    epsilon0, mu, evals, evecs, A = spin_collinearity_test(psi0, ao_ovlp, debug=True)

    np.testing.assert_allclose(A, Aref, atol=1e-14)
    np.testing.assert_allclose(epsilon0, 0.5*(nelec[0] - nelec[1]), atol=1e-14)
    np.testing.assert_allclose(mu, 0., atol=1e-14)

    # Check equality up to a sign.
    try: np.testing.assert_allclose(evecs[:, 0], np.array([1., 0., 0.]), atol=1e-14)
    except Exception as e: np.testing.assert_allclose(evecs[:, 0], -np.array([1., 0., 0.]), atol=1e-14)

    # Test 4: UHF generic spin.
    print('\n# ------')
    print('# Test 4')
    print('# ------')
    nelec = (4, 2)
    nocc = np.sum(nelec)
    psi0 = np.zeros((2*nbsf, nocc))
    psi0[:nbsf, :nelec[0]] = Ca[:, :nelec[0]]
    psi0[nbsf:, nelec[0]:] = Cb[:, :nelec[1]]
    psi0a = psi0[:nbsf] # (nbsf, nocc)
    psi0b = psi0[nbsf:]

    Aref, _ = get_spin_covariance(psi0[:nbsf], psi0[nbsf:], ao_ovlp)
    epsilon0, mu, evals, evecs, A = spin_collinearity_test(psi0, ao_ovlp, debug=True)

    np.testing.assert_allclose(A, Aref, atol=1e-14)
    np.testing.assert_allclose(epsilon0, 0.5*(nelec[0] - nelec[1]), atol=1e-14)
    np.testing.assert_allclose(mu, 0., atol=1e-14)

    # Check equality up to a sign.
    try: np.testing.assert_allclose(evecs[:, 0], np.array([0., 0., 1.]), atol=1e-14)
    except Exception as e: np.testing.assert_allclose(evecs[:, 0], -np.array([0., 0., 1.]), atol=1e-14)
    
    # Test 5: GHF.
    print('\n# ------')
    print('# Test 5')
    print('# ------')
    nocc = 5
    P = np.random.rand(2*nbsf, 2*nbsf)
    P += P.T.conj()
    _, C = np.linalg.eigh(P)
    psi0 = C[:, :nocc]

    Aref, _ = get_spin_covariance(psi0[:nbsf], psi0[nbsf:], ao_ovlp)
    epsilon0, mu, evals, evecs, A = spin_collinearity_test(psi0, ao_ovlp, debug=True)
    epsilon0_int = np.rint(epsilon0)

    np.testing.assert_allclose(A, Aref, atol=1e-14)
    np.testing.assert_array_less(0., np.absolute(epsilon0 - epsilon0_int))
    np.testing.assert_array_less(0., np.absolute(mu))

if __name__ == "__main__":
    test_spin_collinearity_test()






