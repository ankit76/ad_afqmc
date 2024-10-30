import numpy as np

# Pauli matrices.
PAULI_X = np.array([[0., 1.], [1., 0.]])
PAULI_Y = np.array([[0., -1.j], [1.j, 0.]])
PAULI_Z = np.array([[1., 0.], [0., -1.]])


def get_spin_average(psi0a, psi0b, ao_ovlp):
    """
    Computes the expectation value <S> = [ <Sx>, <Sy>, <Sz> ].

    Args
        psi0a : np.ndarray
            alpha spin coefficient matrix with shape (nbsf, nocca).
        psi0b : np.ndarray
            beta spin coefficient matrix with shape (nbsf, noccb).
        ao_ovlp : np.ndarray
            AO overlap matrix with shape (nbsf, nsbf).

    Returns
        The expectation value of the spin vector operator.
    """
    nocc = psi0a.shape[1]
    mo_ovlp_aa = psi0a.T.conj() @ ao_ovlp @ psi0a
    mo_ovlp_ab = psi0a.T.conj() @ ao_ovlp @ psi0b
    mo_ovlp_bb = psi0b.T.conj() @ ao_ovlp @ psi0b

    Sz, Sp, Sm = 0., 0., 0. # Sz, S+, S-.

    for i in range(nocc):
        Sz += 0.5 * (mo_ovlp_aa[i, i] - mo_ovlp_bb[i, i])
        Sp += mo_ovlp_ab[i, i]
        Sm += mo_ovlp_ab[i, i].conj()

    Sx = 0.5 * np.real(Sp + Sm)
    Sy = 0.5 * np.imag(Sp - Sm)
    return np.array([Sx, Sy, Sz])

def get_spin_covariance(psi0a, psi0b, ao_ovlp):
    """
    Computes the matrix A as defined in Eq (11) of 10.197.16.115 and the
    spin covariance matrix:

        [[ <Sx2>,  <SxSy>, <SxSz> ],
        [[ <SySx>, <Sy2>,  <SySz> ],
        [[ <SzSx>, <SzSy>, <Sz2>  ]]

    Args
        psi0a : np.ndarray
            alpha spin coefficient matrix with shape (nbsf, nocca).
        psi0b : np.ndarray
            beta spin coefficient matrix with shape (nbsf, noccb).
        ao_ovlp : np.ndarray
            AO overlap matrix with shape (nbsf, nsbf).

    Returns
        The matrix A and the spin covariance matrix.
    """
    nbsf, nocc = psi0a.shape
    mo_ovlp_aa = psi0a.T.conj() @ ao_ovlp @ psi0a
    mo_ovlp_ab = psi0a.T.conj() @ ao_ovlp @ psi0b
    mo_ovlp_ba = mo_ovlp_ab.T.conj()
    mo_ovlp_bb = psi0b.T.conj() @ ao_ovlp @ psi0b

    SpSp = 0.
    SmSm = 0.
    SpSm = 0.
    SmSp = 0.
    SpSz = 0.
    SmSz = 0.
    Sz2 = 0.

    for i in range(nocc):
        Sz2 += 0.25 * (mo_ovlp_aa[i, i] + mo_ovlp_bb[i, i])
        SpSm += mo_ovlp_aa[i, i]
        SmSp += mo_ovlp_bb[i, i]
        SpSz += -0.5 * mo_ovlp_ab[i, i]
        SmSz += 0.5 * mo_ovlp_ba[i, i]

        for j in range(nocc):
            Sz2 += 0.25 * (mo_ovlp_aa[i, i]*mo_ovlp_aa[j, j] - mo_ovlp_aa[i, j]*mo_ovlp_aa[j, i])
            Sz2 += 0.25 * (mo_ovlp_bb[i, i]*mo_ovlp_bb[j, j] - mo_ovlp_bb[i, j]*mo_ovlp_bb[j, i])
            Sz2 += 0.5 * np.real(mo_ovlp_bb[i, j]*mo_ovlp_aa[j, i] - mo_ovlp_aa[i, i]*mo_ovlp_bb[j, j])

            SpSm += mo_ovlp_ab[i, i]*mo_ovlp_ba[j, j] - mo_ovlp_ba[i, j]*mo_ovlp_ab[j, i]
            SmSp += mo_ovlp_ab[i, i]*mo_ovlp_ba[j, j] - mo_ovlp_ba[i, j]*mo_ovlp_ab[j, i]
            SmSm += mo_ovlp_ba[i, i]*mo_ovlp_ba[j, j] - mo_ovlp_ba[i, j]*mo_ovlp_ba[j, i]

            SpSz += 0.5 * (mo_ovlp_ab[i, i]*mo_ovlp_aa[j, j] - mo_ovlp_aa[i, j]*mo_ovlp_ab[j, i])
            SpSz += -0.5 * (mo_ovlp_ab[i, i]*mo_ovlp_bb[j, j] - mo_ovlp_bb[i, j]*mo_ovlp_ab[j, i])

            SmSz += 0.5 * (mo_ovlp_ba[i, i]*mo_ovlp_aa[j, j] - mo_ovlp_aa[i, j]*mo_ovlp_ba[j, i])
            SmSz += -0.5 * (mo_ovlp_ba[i, i]*mo_ovlp_bb[j, j] - mo_ovlp_bb[i, j]*mo_ovlp_ba[j, i])

    SpSp = SmSm.conj()
    
    Sx2 = 0.25 * (SpSp + SpSm + SmSp + SmSm)
    Sy2 = -0.25 * (SpSp - SpSm - SmSp + SmSm)
    SxSy = -0.25j * (SpSp - SpSm + SmSp - SmSm)
    SySx = np.conj(SxSy)
    SxSz = 0.5 * (SpSz + SmSz)
    SySz = -0.5j * (SpSz - SmSz)
    SzSx = np.conj(SxSz)
    SzSy = np.conj(SySz)

    spin_cov = np.array([[Sx2,  SxSy, SxSz],
                         [SySx, Sy2,  SySz],
                         [SzSx, SzSy, Sz2]])
    return spin_cov

def get_mo_spin_average(psi0a, psi0b, ao_ovlp):
    nocc = psi0a.shape[1]
    mo_ovlp_aa = psi0a.T.conj() @ ao_ovlp @ psi0a
    mo_ovlp_ab = psi0a.T.conj() @ ao_ovlp @ psi0b
    mo_ovlp_bb = psi0b.T.conj() @ ao_ovlp @ psi0b

    Sz = np.zeros(nocc)
    Sp = np.zeros(nocc) # S+
    Sm = np.zeros(nocc) # S-

    for i in range(nocc):
        Sz[i] = 0.5 * (mo_ovlp_aa[i, i] - mo_ovlp_bb[i, i])
        Sp[i] = mo_ovlp_ab[i, i]
        Sm[i] = mo_ovlp_ab[i, i].conj()

    Sx = 0.5 * np.real(Sp + Sm)
    Sy = 0.5 * np.imag(Sp - Sm)
    return np.array([Sx, Sy, Sz])

def get_ao_spin_average(psi0):
    nbsf = psi0.shape[0] // 2
    dm = psi0 @ psi0.T.conj()
    dm_aa = dm[:nbsf, :nbsf]
    dm_ab = dm[:nbsf, nbsf:]
    dm_ba = dm[nbsf:, :nbsf]
    dm_bb = dm[nbsf:, nbsf:]

    Sz = np.zeros(nbsf)
    Sp = np.zeros(nbsf) # S+
    Sm = np.zeros(nbsf) # S-

    for i in range(nbsf):
        Sz[i] = 0.5 * (dm_aa[i, i] - dm_bb[i, i])
        Sp[i] = dm_ab[i, i]
        Sm[i] = dm_ab[i, i].conj()

    Sx = 0.5 * np.real(Sp + Sm)
    Sy = 0.5 * np.imag(Sp - Sm)
    return np.array([Sx, Sy, Sz])
    
def get_A_matrix(psi0a, psi0b, ao_ovlp):
    spin_avg = get_spin_average(psi0a, psi0b, ao_ovlp)
    spin_cov = get_spin_covariance(psi0a, psi0b, ao_ovlp)
    spin_avg2 = np.outer(spin_avg, spin_avg)
    return np.real(spin_cov - spin_avg2)

def spin_collinearity_test(psi0, ao_ovlp, debug=False, verbose=False):
    """
    Spin collinearity test as introduced in 10.197.16.115
    """
    nbsf = psi0.shape[0] // 2
    nocc = psi0.shape[1]
    psi0a = psi0[:nbsf]
    psi0b = psi0[nbsf:]
        
    # -------------------------------------------------------------------------
    # Compute <S> = (Sx, Sy, Sz).
    spin = get_spin_average(psi0a, psi0b, ao_ovlp)
   
    # -------------------------------------------------------------------------
    # Compute the A matrix given in Eq (27) of ref.
    # Spin matrices, S_m = 0.5 * pauli_m, m = x, y, z.
    spin_mats = 0.5 * np.array([PAULI_X, PAULI_Y, PAULI_Z])

    # Eq (28) of ref.
    O_mats = np.array([psi0.T.conj() @ np.kron(spin_mat, ao_ovlp) @ psi0 for spin_mat in spin_mats])
    
    # Generate lower triangular matrix.
    tril = np.zeros((3, 3))

    for i in range(3):
        for j in range(i+1):
            tril[i, j] = -np.trace(O_mats[i] @ O_mats[j])
    
    for i in range(3): tril[i, i] += nocc / 4.
    A = np.where(tril, tril, tril.T)

    # -------------------------------------------------------------------------
    # Collinearity test.
    epsilon0 = np.linalg.norm(spin)
    evals, evecs = np.linalg.eigh(A)
    mu = evals[0]
    spin_axis = evecs[:, 0]
    
    if verbose:
        print(f'\n# ----------------------')
        print(f'# Spin collinearity test')
        print(f'# ----------------------')
        print(f'# epsilon0 = {epsilon0}')
        print(f"# non {'half-' if nocc%2 else ''}integer value indicates non-collinearity")

        print(f'\n# minimum mu = {mu}')
        print(f'# Value is 0 iff wavefunction is collinear')

        print(f'\n# If mu = 0, the collinear spin axis is: \n{spin_axis}')
    
    if debug: return epsilon0, mu, evals, evecs, A
    return epsilon0, mu, spin_axis

def get_spin_rotation_matrix(nbsf, v):
    """
    Returns the spin rotation matrix that rotates the spin axis along vector
    v to the z-axis.
    """
    Sv = v[0]*PAULI_X + v[1]*PAULI_Y + v[2]*PAULI_Z
    Sv *= 0.5
    _, Uspin = np.linalg.eigh(Sv)

    # Fix gauge s.t. the real part of the first element in each eigenvector is positive.
    for icol in range(Uspin.shape[1]):
        evec = Uspin[:, icol]
        if evec[0].real < 0: Uspin[:, icol] *= -1
    
    Uspin = Uspin[:, ::-1]
    U = np.kron(Uspin, np.eye(nbsf))

    # Check if unitary.
    np.testing.assert_allclose(U @ U.T.conj(), np.eye(2*nbsf), atol=1e-14)
    np.testing.assert_allclose(U.T.conj() @ U, np.eye(2*nbsf), atol=1e-14)

    return U

def align_spin_axis(psi, v):
    """
    Rotate the spin axis of state psi from the vector v to the z-axis.
    """
    nbsf = psi.shape[0] // 2
    U = get_spin_rotation_matrix(nbsf, v)
    return U.T.conj() @ psi

