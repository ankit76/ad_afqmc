import numpy as np

from ad_afqmc import config

config.setup_jax()

from ad_afqmc.spin_utils import (
        get_spin_covariance, 
        spin_collinearity_test,
        get_spin_rotation_matrix,
        align_spin_axis
        )

seed = 102
np.random.seed(seed)

def test_spin_collinearity_test():
    nbsf = 10
    ao_ovlp = np.eye(nbsf)

    # Random dm and coeffs.
    Paa = np.random.rand(nbsf, nbsf)
    Pbb = np.random.rand(nbsf, nbsf)
    Paa += Paa.T.conj()
    Pbb += Pbb.T.conj()
    _, Ca = np.linalg.eigh(Paa)
    _, Cb = np.linalg.eigh(Pbb)

    # -------------------------------------------------------------------------
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

    # -------------------------------------------------------------------------
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

    # -------------------------------------------------------------------------
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
                [[np.cos(theta / 2.0), np.sin(theta / 2.)],
                 [np.sin(theta / 2.0), -np.cos(theta / 2.)]])
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

def test_get_spin_rotation_matrix():
    nbsf = 10

    # -------------------------------------------------------------------------
    # Test 1.
    print('\n# ------')
    print('# Test 1')
    print('# ------')
    v = np.array([0., 0., 1.])
    U = get_spin_rotation_matrix(nbsf, v)
    np.testing.assert_allclose(U, np.eye(2*nbsf), atol=1e-14)

    # -------------------------------------------------------------------------
    # Test 2.
    print('\n# ------')
    print('# Test 2')
    print('# ------')
    v = np.array([0., 1., 0.])
    U = get_spin_rotation_matrix(nbsf, v)
    
    theta, phi = np.pi/2., np.pi/2.
    evecs = np.array([[np.cos(theta/2.),                   np.sin(theta/2.)],
                      [np.exp(phi*1.j) * np.sin(theta/2.), -np.exp(phi*1.j) * np.cos(theta/2.)]])
    U_ref = np.kron(evecs, np.eye(nbsf))
    np.testing.assert_allclose(U, U_ref, atol=1e-14)

    # Test 3.
    print('\n# ------')
    print('# Test 3')
    print('# ------')
    theta = np.random.rand() * np.pi
    phi = np.random.rand() * 2*np.pi
    v = np.array([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)])
    U = get_spin_rotation_matrix(nbsf, v)
    
    evecs = np.array([[np.cos(theta/2.),                    np.sin(theta/2.)],
                      [np.exp(phi*1.j) * np.sin(theta/2.), -np.exp(phi*1.j) * np.cos(theta/2.)]])
    U_ref = np.kron(evecs, np.eye(nbsf))
    np.testing.assert_allclose(U, U_ref, atol=1e-14)

def test_align_spin_axis():
    nbsf = 10
    ao_ovlp = np.eye(nbsf)

    # Random dm and coeffs.
    Paa = np.random.rand(nbsf, nbsf)
    Pbb = np.random.rand(nbsf, nbsf)
    Paa += Paa.T.conj()
    Pbb += Pbb.T.conj()
    _, Ca = np.linalg.eigh(Paa)
    _, Cb = np.linalg.eigh(Pbb)

    # -------------------------------------------------------------------------
    # Test 1: UHF polarized.
    print('\n# ------')
    print('# Test 1')
    print('# ------')
    nelec = (3, 0)
    nocc = np.sum(nelec)
    psi0 = np.zeros((2*nbsf, nocc))
    psi0[:nbsf, :nelec[0]] = Ca[:, :nelec[0]]
    psi0[nbsf:, nelec[0]:] = Cb[:, :nelec[1]]

    epsilon0, mu, spin_axis = spin_collinearity_test(psi0, ao_ovlp)
    rot_psi0 = align_spin_axis(psi0, spin_axis)

    print(f'\n# After rotation from axis \n{spin_axis}')
    rot_epsilon0, rot_mu, rot_spin_axis = spin_collinearity_test(rot_psi0, ao_ovlp)
    np.testing.assert_allclose(np.absolute(rot_spin_axis), np.array([0., 0., 1.]), atol=1e-14)
    np.testing.assert_allclose(rot_epsilon0, epsilon0, atol=1e-14)
    np.testing.assert_allclose(rot_mu, mu, atol=1e-14)

    # -------------------------------------------------------------------------
    # Test 2: Rotated UHF polarized.
    print('\n# ------')
    print('# Test 2')
    print('# ------')
    nelec = (4, 0)
    nocc = np.sum(nelec)
    psi0 = np.zeros((2*nbsf, nocc))
    psi0[:nbsf, :nelec[0]] = Ca[:, :nelec[0]]
    psi0[nbsf:, nelec[0]:] = Cb[:, :nelec[1]]

    # Rotate spin axis to x-axis.
    theta = np.pi / 2.
    Uspin = np.array(
                [[np.cos(theta / 2.0), -np.sin(theta / 2.)],
                 [np.sin(theta / 2.0), np.cos(theta / 2.)]])
    U = np.kron(Uspin, np.eye(nbsf))
    psi0 = U.dot(psi0)

    epsilon0, mu, spin_axis = spin_collinearity_test(psi0, ao_ovlp)
    rot_psi0 = align_spin_axis(psi0, spin_axis)

    print(f'\n# After rotation from axis \n{spin_axis}')
    rot_epsilon0, rot_mu, rot_spin_axis = spin_collinearity_test(rot_psi0, ao_ovlp)
    np.testing.assert_allclose(np.absolute(rot_spin_axis), np.array([0., 0., 1.]), atol=1e-14)
    np.testing.assert_allclose(rot_epsilon0, epsilon0, atol=1e-14)
    np.testing.assert_allclose(rot_mu, mu, atol=1e-14)

    # Test 3: UHF generic spin.
    print('\n# ------')
    print('# Test 3')
    print('# ------')
    nelec = (4, 2)
    nocc = np.sum(nelec)
    psi0 = np.zeros((2*nbsf, nocc))
    psi0[:nbsf, :nelec[0]] = Ca[:, :nelec[0]]
    psi0[nbsf:, nelec[0]:] = Cb[:, :nelec[1]]

    epsilon0, mu, spin_axis = spin_collinearity_test(psi0, ao_ovlp)
    rot_psi0 = align_spin_axis(psi0, spin_axis)

    print(f'\n# After rotation from axis \n{spin_axis}')
    rot_epsilon0, rot_mu, rot_spin_axis = spin_collinearity_test(rot_psi0, ao_ovlp)
    np.testing.assert_allclose(np.absolute(rot_spin_axis), np.array([0., 0., 1.]), atol=1e-14)
    np.testing.assert_allclose(rot_epsilon0, epsilon0, atol=1e-14)
    np.testing.assert_allclose(rot_mu, mu, atol=1e-14)
    
    # -------------------------------------------------------------------------
    # Test 4: Rotated UHF generic spin.
    print('\n# ------')
    print('# Test 5')
    print('# ------')
    nelec = (5, 1)
    nocc = np.sum(nelec)
    psi0 = np.zeros((2*nbsf, nocc))
    psi0[:nbsf, :nelec[0]] = Ca[:, :nelec[0]]
    psi0[nbsf:, nelec[0]:] = Cb[:, :nelec[1]]

    # Rotate spin axis.
    theta = np.random.rand() * np.pi
    phi = np.random.rand() * 2*np.pi
    Uspin = np.array(
                [[np.cos(theta / 2.0), np.sin(theta / 2.)],
                 [np.exp(phi*1.j)*np.sin(theta / 2.0), -np.exp(phi*1.j)*np.cos(theta / 2.)]])
    U = np.kron(Uspin, np.eye(nbsf))
    psi0 = U.dot(psi0)

    epsilon0, mu, spin_axis = spin_collinearity_test(psi0, ao_ovlp)
    rot_psi0 = align_spin_axis(psi0, spin_axis)

    print(f'\n# After rotation from axis \n{spin_axis}')
    rot_epsilon0, rot_mu, rot_spin_axis = spin_collinearity_test(rot_psi0, ao_ovlp)
    np.testing.assert_allclose(np.absolute(rot_spin_axis), np.array([0., 0., 1.]), atol=1e-14)
    np.testing.assert_allclose(rot_epsilon0, epsilon0, atol=1e-14)
    np.testing.assert_allclose(rot_mu, mu, atol=1e-14)
    

if __name__ == "__main__":
    test_spin_collinearity_test()
    test_get_spin_rotation_matrix()
    test_align_spin_axis()
