import sys
import struct
import time
from functools import partial
from typing import Any, Optional, Sequence, Tuple, Union

import h5py
import jax.numpy as jnp
import numpy as np
import scipy
from pyscf import __config__, ao2mo, df, dft, lib, mcscf, scf
from pyscf.cc.ccsd import CCSD
from pyscf.cc.uccsd import UCCSD

from ad_afqmc.logger import Logger

print = partial(print, flush=True)


# modified cholesky for a given matrix
def modified_cholesky(mat: np.ndarray, max_error: float = 1e-6) -> np.ndarray:
    """Modified cholesky decomposition for a given matrix.

    Args:
        mat (np.ndarray): Matrix to decompose.
        max_error (float, optional): Maximum error allowed. Defaults to 1e-6.

    Returns:
        np.ndarray: Cholesky vectors.
    """
    diag = mat.diagonal()
    size = mat.shape[0]
    nchol_max = size
    chol_vecs = np.zeros((nchol_max, nchol_max))
    # ndiag = 0
    nu = np.argmax(diag)
    delta_max = diag[nu]
    Mapprox = np.zeros(size)
    chol_vecs[0] = np.copy(mat[nu]) / delta_max**0.5

    nchol = 0
    while abs(delta_max) > max_error and (nchol + 1) < nchol_max:
        Mapprox += chol_vecs[nchol] * chol_vecs[nchol]
        delta = diag - Mapprox
        nu = np.argmax(np.abs(delta))
        delta_max = np.abs(delta[nu])
        R = np.dot(chol_vecs[: nchol + 1, nu], chol_vecs[: nchol + 1, :])
        chol_vecs[nchol + 1] = (mat[nu] - R) / (delta_max + 1e-10) ** 0.5
        nchol += 1

    return chol_vecs[:nchol]


# prepare phaseless afqmc with mf trial
def prep_afqmc(
    mf_or_cc: Union[scf.uhf.UHF, scf.rhf.RHF, CCSD, UCCSD],
    basis_coeff: Optional[np.ndarray] = None,
    norb_frozen: int = 0,
    chol_cut: float = 1e-5,
    integrals: Optional[dict] = None,
    tmpdir: str = "./",
    verbose: int = 0,
):
    """Prepare AFQMC calculation with mean field trial wavefunction. Writes integrals and mo coefficients to disk.

    Args:
        mf (Union[scf.uhf.UHF, scf.rhf.RHF, mcscf.mc1step.CASSCF]): pyscf mean field object. Used for generating integrals (if not provided) and trial.
        basis_coeff (np.ndarray, optional): Orthonormal basis used for afqmc, given in the basis of ao's. If not provided mo_coeff of mf is used as the basis.
        norb_frozen (int, optional): Number of frozen orbitals. Not supported for custom integrals.
        chol_cut (float, optional): Cholesky decomposition cutoff.
        integrals (dict, optional): Dictionary of integrals in an orthonormal basis, {"h0": enuc, "h1": h1e, "h2": eri}.
        tmpdir (str, optional): Directory to write integrals and mo coefficients. Defaults to "./".
    """
    log = Logger(sys.stdout, verbose)

    log.log("#\n# Preparing AFQMC calculation")

    if isinstance(mf_or_cc, (CCSD, UCCSD)):
        mf, cc, norb_frozen = read_pyscf_ccsd(log, mf_or_cc, tmpdir)
    else:
        mf = mf_or_cc

    mol = mf.mol
    # choose the orbital basis
    if basis_coeff is None:
        if isinstance(mf, scf.uhf.UHF):
            basis_coeff = mf.mo_coeff[0]
        else:
            basis_coeff = mf.mo_coeff

    # calculate cholesky integrals
    h1e, chol, nelec, enuc, nbasis, nchol = compute_cholesky_integrals(log, mol, mf, basis_coeff, integrals, norb_frozen, chol_cut)

    log.log("# Finished calculating Cholesky integrals\n#")

    nbasis = h1e.shape[-1]
    log.log("# Size of the correlation space:")
    log.log(f"# Number of electrons: {nelec}")
    log.log(f"# Number of basis functions: {nbasis}")
    log.log(f"# Number of Cholesky vectors: {chol.shape[0]}\n#")
    chol = chol.reshape((-1, nbasis, nbasis))
    v0 = 0.5 * np.einsum("nik,njk->ij", chol, chol, optimize="optimal")
    h1e_mod = h1e - v0
    chol = chol.reshape((chol.shape[0], -1))

    # write trial mo coefficients
    trial_coeffs = write_trial(mol, mf, basis_coeff, nbasis, norb_frozen, tmpdir) 

    write_dqmc(
        h1e,
        h1e_mod,
        chol,
        sum(nelec),
        nbasis,
        enuc,
        ms=mol.spin,
        filename=tmpdir + "/FCIDUMP_chol",
        mo_coeffs=trial_coeffs,
    )

def getCollocationMatrices(
    mol, grid_level=0, thc_eps=1.0e-4, mo1=None, mo2=None, alpha=0.25
):
    """Return the matrices X1 and X2 the equation
    Z(ab,r) = X1(a,P) X2(b,P) Xi(P,r)
    can be satisfied well. Note that in this code we do not evaluate Xi, for that call the least square solver.

    The two matrices X1 and X2 are the same when the space spanned by {a} and {b} is the same, e.g. all AOs.

    Args:
        mol: pysf Mol object
        grid_level (int, optional): The size of the Becke grid used in these calculations. Defaults to 0.
        thc_eps (double, optional): The threshold used in the THC calculation, the lower it is the more accurate the approximation. Defaults to 1.e-4.
        mo1 (array, optional): The orbital coefficient in terms of AOs mo1(ao, mo)
                Defaults to None, so this assumes mo1 is identity and all AOs are fit
                Other options might be all mos, all occupied, all virtual etc.
        mo2 (array, optional): The orbital coefficient in terms of AOs mo2(ao, mo)
                Defaults to None, so this assumes mo2 is identity and all AOs are fit
                Other options might be all mos, all occupied, all virtual etc.
        alpha (double, optional): the exponent on the weight in the collocation matrix
            X(a,P) = phi_a(r_P) w_p^\alpha
    """

    grids = dft.gen_grid.Grids(mol)  # type: ignore
    grids.level = 0
    grids.build()

    coords = grids.coords
    weights = grids.weights
    ao = mol.eval_gto("GTOval_sph", coords)  ##aos on coords
    X1 = np.einsum("ri,r->ri", (ao @ mo1), abs(weights) ** alpha)  ##mos on coords
    X2 = np.einsum("ri,r->ri", (ao @ mo2), abs(weights) ** alpha)  ##mos on coords

    P = doISDF(X1, X2, thc_eps)
    return X1[P], X2[P]


def doISDF(X1: np.ndarray, X2: np.ndarray, thc_eps=1.0e-4):
    """Return the pivot points P such that the equation
    Z(ab,r) = X1(a,P) X2(b,P) Xi(P,r)
    can be satisfied to high accuracy.

    Args:
        X1 (np.ndarray): The first matrix
        X2 (np.ndarray): The second matrix
        thc_eps (_type_, optional): The threshold up to which the pivot points are retained. Defaults to 1.e-4.
    """

    S = np.einsum("ri,si->rs", X1, X1) * np.einsum("ri,si->rs", X2, X2)
    R, P, rankc, info = scipy.linalg.lapack.dpstrf(S, thc_eps**2)
    P = P[:rankc] - 1
    return P


def solveLS(T, X1, X2):
    """T is a three index quantity and we want to obtain Xi such that
    T(a,b,M) = X1(P,a) X2(P,b) Xi(P, M)
    is satisfied approximately using least square minimization.

    Args:
        T (array): The input three index quantity
        X1 (array): The two index collocation array, the size of the second index should be equal to the size of first index in T
        X2 (array): The two index collocation array, the size of the seonc idnex should be equal to the size of the second index in T
             and the size of the first index should be equal to the size of first index in X1

    Returns:
        Xi (array): The matrix that satisfies the equation above
    """

    S = np.einsum("Pa,Qa->PQ", X1, X1) * np.einsum("Pb,Qb->PQ", X2, X2)
    X12 = jnp.einsum("Pa,Pb->abP", X1, X2)
    Stilde = jnp.einsum("abP, abM->PM", X12, T)

    L = scipy.linalg.cholesky(S, lower=True)
    Xi = scipy.linalg.cho_solve((L, True), Stilde, overwrite_b=True)

    return Xi


def solveLS_twoSided(T, X1, X2):
    """T is a four index quantity and we want to obtain V such that
    T(a,b,c,d) = X1(P,a) X2(P,b) V(P, Q) X1(Q,c) X2(Q,d)
    is satisfied approximately using least square minimization.

    Args:
        T (array): The input four index quantity
        X1 (array): The two index collocation array, the size of the second index should be equal to the size of first and third index in T
        X2 (array): The two index collocation array, the size of the second idnex should be equal to the size of the second and fourth index in T
             and the size of the first index should be equal to the size of first index in X1

    Returns:
        V (array): The matrix that satisfies the equation above
    """

    S = np.einsum("Pa,Qa->PQ", X1, X1) * np.einsum("Pb,Qb->PQ", X2, X2)
    L = scipy.linalg.cholesky(S, lower=True)

    X12 = jnp.einsum("Pa,Pb->abP", X1, X2)
    E = jnp.einsum("abP, abcd->Pcd", X12, T).reshape(X1.shape[0], -1)

    E = scipy.linalg.cho_solve((L, True), E).reshape(-1, T.shape[2], T.shape[3])

    E = jnp.einsum("Pcd, cdQ->PQ", E, X12).T
    V = scipy.linalg.cho_solve((L, True), E)

    ##symmetrize it
    V = 0.5 * (V + V.T)

    return V


# cholesky generation functions are from pauxy
def generate_integrals(log, mol, hcore, X, chol_cut=1e-5, DFbas=None):
    # Unpack SCF data.
    # Step 1. Rotate core Hamiltonian to orthogonal basis.
    log.log(" # Transforming hcore and eri to ortho AO basis.")
    if len(X.shape) == 2:
        h1e = np.dot(X.T.conj(), np.dot(hcore, X))
    elif len(X.shape) == 3:
        h1e = np.dot(X[0].T.conj(), np.dot(hcore, X[0]))

    if DFbas is not None:
        chol_vecs = df.incore.cholesky_eri(mol, auxbasis=DFbas)
        chol_vecs = lib.unpack_tril(chol_vecs).reshape(chol_vecs.shape[0], -1)
    else:  # do cholesky
        # nbasis = h1e.shape[-1]
        # Step 2. Genrate Cholesky decomposed ERIs in non-orthogonal AO basis.
        log.log(" # Performing modified Cholesky decomposition on ERI tensor.")
        chol_vecs = chunked_cholesky(log, mol, max_error=chol_cut)

    log.log(" # Orthogonalising Cholesky vectors.")
    start = time.time()

    # Step 2.a Orthogonalise Cholesky vectors.
    if len(X.shape) == 2 and X.shape[0] != X.shape[1]:
        chol_vecs = ao2mo_chol_copy(chol_vecs, X)
    elif len(X.shape) == 2:
        ao2mo_chol(chol_vecs, X)
    elif len(X.shape) == 3:
        ao2mo_chol(chol_vecs, X[0])
    log.log(" # Time to orthogonalise: %f" % (time.time() - start))
    enuc = mol.energy_nuc()
    # Step 3. (Optionally) freeze core / virtuals.
    nelec = mol.nelec
    return h1e, chol_vecs, nelec, enuc


def ao2mo_chol(eri, C):
    nb = C.shape[-1]
    for i, cv in enumerate(eri):
        half = np.dot(cv.reshape(nb, nb), C)
        eri[i] = np.dot(C.conj().T, half).ravel()


def ao2mo_chol_copy(eri, C):
    nb = C.shape[0]
    nmo = C.shape[1]
    eri_copy = np.zeros((eri.shape[0], nmo * nmo))
    for i, cv in enumerate(eri):
        half = np.dot(cv.reshape(nb, nb), C)
        eri_copy[i] = np.dot(C.conj().T, half).ravel()
    return eri_copy


def chunked_cholesky(log, mol, max_error=1e-6, cmax=10):
    """Modified cholesky decomposition from pyscf eris.

    See, e.g. [Motta17]_

    Only works for molecular systems.

    Parameters
    ----------
    mol : :class:`pyscf.mol`
        pyscf mol object.
    orthoAO: :class:`numpy.ndarray`
        Orthogonalising matrix for AOs. (e.g., mo_coeff).
    delta : float
        Accuracy desired.
    verbose : bool
        If true print out convergence progress.
    cmax : int
        nchol = cmax * M, where M is the number of basis functions.
        Controls buffer size for cholesky vectors.

    Returns
    -------
    chol_vecs : :class:`numpy.ndarray`
        Matrix of cholesky vectors in AO basis.
    """
    nao = mol.nao_nr()
    diag = np.zeros(nao * nao)
    nchol_max = cmax * nao
    # This shape is more convenient for pauxy.
    chol_vecs = np.zeros((nchol_max, nao * nao))
    ndiag = 0
    dims = [0]
    nao_per_i = 0
    for i in range(0, mol.nbas):
        l = mol.bas_angular(i)
        nc = mol.bas_nctr(i)
        nao_per_i += (2 * l + 1) * nc
        dims.append(nao_per_i)
    # print (dims)
    for i in range(0, mol.nbas):
        shls = (i, i + 1, 0, mol.nbas, i, i + 1, 0, mol.nbas)
        buf = mol.intor("int2e_sph", shls_slice=shls)
        di, dk, dj, dl = buf.shape
        diag[ndiag : ndiag + di * nao] = buf.reshape(di * nao, di * nao).diagonal()
        ndiag += di * nao
    nu = np.argmax(diag)
    delta_max = diag[nu]
    log.debug("# Generating Cholesky decomposition of ERIs.")
    log.debug("# max number of cholesky vectors = %d" % nchol_max)
    log.debug("# iteration %5d: delta_max = %f" % (0, delta_max))
    j = nu // nao
    l = nu % nao
    sj = np.searchsorted(dims, j)
    sl = np.searchsorted(dims, l)
    if dims[sj] != j and j != 0:
        sj -= 1
    if dims[sl] != l and l != 0:
        sl -= 1
    Mapprox = np.zeros(nao * nao)
    # ERI[:,jl]
    eri_col = mol.intor(
        "int2e_sph", shls_slice=(0, mol.nbas, 0, mol.nbas, sj, sj + 1, sl, sl + 1)
    )
    cj, cl = max(j - dims[sj], 0), max(l - dims[sl], 0)
    chol_vecs[0] = np.copy(eri_col[:, :, cj, cl].reshape(nao * nao)) / delta_max**0.5

    nchol = 0
    while abs(delta_max) > max_error:
        # Update cholesky vector
        start = time.time()
        # M'_ii = L_i^x L_i^x
        Mapprox += chol_vecs[nchol] * chol_vecs[nchol]
        # D_ii = M_ii - M'_ii
        delta = diag - Mapprox
        nu = np.argmax(np.abs(delta))
        delta_max = np.abs(delta[nu])
        # Compute ERI chunk.
        # shls_slice computes shells of integrals as determined by the angular
        # momentum of the basis function and the number of contraction
        # coefficients. Need to search for AO index within this shell indexing
        # scheme.
        # AO index.
        j = nu // nao
        l = nu % nao
        # Associated shell index.
        sj = np.searchsorted(dims, j)
        sl = np.searchsorted(dims, l)
        if dims[sj] != j and j != 0:
            sj -= 1
        if dims[sl] != l and l != 0:
            sl -= 1
        # Compute ERI chunk.
        eri_col = mol.intor(
            "int2e_sph", shls_slice=(0, mol.nbas, 0, mol.nbas, sj, sj + 1, sl, sl + 1)
        )
        # Select correct ERI chunk from shell.
        cj, cl = max(j - dims[sj], 0), max(l - dims[sl], 0)
        Munu0 = eri_col[:, :, cj, cl].reshape(nao * nao)
        # Updated residual = \sum_x L_i^x L_nu^x
        R = np.dot(chol_vecs[: nchol + 1, nu], chol_vecs[: nchol + 1, :])
        chol_vecs[nchol + 1] = (Munu0 - R) / (delta_max) ** 0.5
        nchol += 1
        step_time = time.time() - start
        info = (nchol, delta_max, step_time)
        log.debug("# iteration %5d: delta_max = %13.8e: time = %13.8e" % info)

    return chol_vecs[:nchol]


# write cholesky integrals
def write_dqmc(
    hcore,
    hcore_mod,
    chol,
    nelec,
    nmo,
    enuc,
    ms=0,
    filename="FCIDUMP_chol",
    mo_coeffs=None,
):
    assert len(chol.shape) == 2
    with h5py.File(filename, "w") as fh5:
        fh5["header"] = np.array([nelec, nmo, ms, chol.shape[0]])
        fh5["hcore"] = hcore.flatten()
        fh5["hcore_mod"] = hcore_mod.flatten()
        fh5["chol"] = chol.flatten()
        fh5["energy_core"] = enuc
        if mo_coeffs is not None:
            fh5["mo_coeffs_up"] = mo_coeffs[0]
            fh5["mo_coeffs_dn"] = mo_coeffs[1]


def finite_difference_properties(
    mol,
    observable,
    observable_constant=0.0,
    epsilon=1.0e-5,
    norb_frozen=0,
    hf_type="rhf",
    relaxed=True,
    dm=None,
):
    from pyscf import cc

    print(
        f'#\n# Orbital {"" if relaxed else "un"}relaxed finite difference properties using {hf_type} reference'
    )
    print(f"# epsilon: {epsilon}")
    mf_coeff, mo_occ = None, None
    if hf_type == "rhf":
        mf = scf.RHF(mol)
    elif hf_type == "uhf":
        mf = scf.UHF(mol)
    if not relaxed:
        mf.verbose = 1
        if dm is None:
            mf.kernel()
        else:
            mf.kernel(dm)
        if hf_type == "uhf":
            mo1 = mf.stability(external=True)[0]
            mf = mf.newton().run(mo1, mf.mo_occ)  # type: ignore
            mo1 = mf.stability(external=True)[0]  # type: ignore
            mf = mf.newton().run(mo1, mf.mo_occ)  # type: ignore
        mf_coeff = mf.mo_coeff.copy()  # type: ignore
        mf_occ = mf.mo_occ.copy()  # type: ignore

    h1e = mf.get_hcore() - epsilon * observable
    mf.get_hcore = lambda *args: h1e  # type: ignore
    mf.verbose = 1
    if relaxed:
        if dm is None:
            mf.kernel()
        else:
            mf.kernel(dm)
        if hf_type == "uhf":
            mo1 = mf.stability(external=True)[0]  # type: ignore
            mf = mf.newton().run(mo1, mf.mo_occ)  # type: ignore
            mo1 = mf.stability(external=True)[0]  # type: ignore
            mf = mf.newton().run(mo1, mf.mo_occ)  # type: ignore
    emf_m = mf.e_tot - epsilon * observable_constant
    mycc = cc.CCSD(mf)
    mycc.frozen = norb_frozen
    mycc.kernel()
    emp2_m = mycc.e_hf + mycc.emp2 - epsilon * observable_constant  # type: ignore
    eccsd_m = mycc.e_tot - epsilon * observable_constant
    et = mycc.ccsd_t()
    eccsdpt_m = mycc.e_tot + et - epsilon * observable_constant

    if hf_type == "rhf":
        mf = scf.RHF(mol)
    elif hf_type == "uhf":
        mf = scf.UHF(mol)
    if not relaxed:
        mf.verbose = 1
        mf.mo_coeff = mf_coeff  # type: ignore
        mf.mo_occ = mf_occ
    h1e = mf.get_hcore() + epsilon * observable
    mf.get_hcore = lambda *args: h1e  # type: ignore
    mf.verbose = 1
    if relaxed:
        if dm is None:
            mf.kernel()
        else:
            mf.kernel(dm)
        if hf_type == "uhf":
            mo1 = mf.stability(external=True)[0]  # type: ignore
            mf = mf.newton().run(mo1, mf.mo_occ)  # type: ignore
            mo1 = mf.stability(external=True)[0]  # type: ignore
            mf = mf.newton().run(mo1, mf.mo_occ)  # type: ignore
    emf_p = mf.e_tot + epsilon * observable_constant
    mycc = cc.CCSD(mf)
    mycc.frozen = norb_frozen
    mycc.kernel()
    emp2_p = mycc.e_hf + mycc.emp2 + epsilon * observable_constant  # type: ignore
    eccsd_p = mycc.e_tot + epsilon * observable_constant
    et = mycc.ccsd_t()
    eccsdpt_p = mycc.e_tot + et + epsilon * observable_constant

    print("# FD single point energies:")
    print(f"# emf_m: {emf_m}, emf_p: {emf_p}")
    print(f"# emp2_m: {emp2_m}, emp2_p: {emp2_p}")
    print(f"# eccsd_m: {eccsd_m}, eccsd_p: {eccsd_p}")
    print(f"# eccsdpt_m: {eccsdpt_m}, eccsd_p: {eccsdpt_p}")

    obs_mf = (emf_p - emf_m) / 2 / epsilon
    obs_mp2 = (emp2_p - emp2_m) / 2 / epsilon
    obs_ccsd = (eccsd_p - eccsd_m) / 2 / epsilon
    obs_ccsdpt = (eccsdpt_p - eccsdpt_m) / 2 / epsilon

    print("# FD Observables:")
    if relaxed:
        print(f"HF observable: {obs_mf}")
    print(f"MP2 observable: {obs_mp2}")
    print(f"CCSD observable: {obs_ccsd}")
    print(f"CCSD(T) observable: {obs_ccsdpt}")
    return obs_mf, obs_mp2, obs_ccsd, obs_ccsdpt


def get_fci_state(
    fci: Any, ndets: Optional[int] = None, tol: float = 1.0e-4, root=0
) -> dict:
    """Get FCI state from a pyscf FCI object.

    Args:
        fci: FCI object.
        ndets: Number of determinants to include (sorted by coeffs).
        tol: Tolerance for including determinants.

    Returns:
        Dictionary with determinants as keys and coefficients as values.
    """
    ci_coeffs = fci.ci
    if isinstance(ci_coeffs, list):
        ci_coeffs = ci_coeffs[root]
    norb = fci.norb
    nelec = fci.nelec
    if ndets is None:
        ndets = int(ci_coeffs.size)
    coeffs, occ_a, occ_b = zip(
        *fci.large_ci(ci_coeffs, norb, nelec, tol=tol, return_strs=False)
    )
    coeffs, occ_a, occ_b = zip(
        *sorted(zip(coeffs, occ_a, occ_b), key=lambda x: -abs(x[0]))
    )
    state = {}
    for i in range(min(ndets, len(coeffs))):
        det = [[0 for _ in range(norb)], [0 for _ in range(norb)]]
        for j in range(nelec[0]):
            det[0][occ_a[i][j]] = 1
        for j in range(nelec[1]):
            det[1][occ_b[i][j]] = 1
        state[tuple(map(tuple, det))] = coeffs[i]
    return state


def get_excitations(
    state: Optional[dict] = None,
    num_core: int = 0,
    fname: str = "dets.bin",
    ndets: Optional[int] = None,
    max_excitation: int = 10,
) -> Tuple[dict, dict, dict, dict, dict, np.ndarray]:
    """Use given a state or read determinants from a binary file and return excitations.

    | psi_t > = sum_i coeff_i E_i | d_0 > (note that coeff_i differ from those in sum_i coeff_i_1 | d_i > by parity factors).

    Args:
        state: Dictionary with determinants as keys and coefficients as values.
        num_core: Number of core orbitals.
        fname: Binary file containing determinants.
        ndets: Number of determinants to read.
        max_excitation: Maximum excitation level (alpha + beta) to consider.

    Returns:
        Acre: Alpha creation indices.
        Ades: Alpha destruction indices.
        Bcre: Beta creation indices.
        Bdes: Beta destruction indices.
        coeff: Coefficients of the determinants.
        ref_det: Reference determinant.
    """
    if state is None:
        _, state, ndets_all = read_dets(fname, ndets)
        if ndets is None:
            ndets = ndets_all
    else:
        if ndets is None:
            ndets = len(state)

    dets = list(state.keys())[:ndets]
    Acre, Ades, Bcre, Bdes, coeff = {}, {}, {}, {}, {}
    d0 = dets[0]
    d0a, d0b = np.asarray(d0[0]), np.asarray(d0[1])
    for d in dets:
        dia, dib = np.asarray(d[0]), np.asarray(d[1])
        nex = (np.sum(abs(dia - d0a)) // 2, np.sum(abs(dib - d0b)) // 2)
        if nex[0] + nex[1] > max_excitation:
            continue
        coeff[nex] = coeff.get(nex, []) + [state[d]]
        if nex[0] > 0 and nex[1] > 0:
            occ_idx_a_rel = (
                np.arange(sum(d0a))[
                    (np.cumsum(d0a) - 1)[np.nonzero((d0a - dia) > 0)[0]]
                ],
            )
            occ_idx_b_rel = (
                np.arange(sum(d0b))[
                    (np.cumsum(d0b) - 1)[np.nonzero((d0b - dib) > 0)[0]]
                ],
            )
            Acre[nex], Ades[nex], Bcre[nex], Bdes[nex] = (
                Acre.get(nex, []) + [occ_idx_a_rel],
                Ades.get(nex, []) + [np.nonzero((d0a - dia) < 0)],
                Bcre.get(nex, []) + [occ_idx_b_rel],
                Bdes.get(nex, []) + [np.nonzero((d0b - dib) < 0)],
            )
            coeff[nex][-1] *= parity(
                d0a, np.nonzero((d0a - dia) > 0), Ades[nex][-1]
            ) * parity(d0b, np.nonzero((d0b - dib) > 0), Bdes[nex][-1])

        elif nex[0] > 0 and nex[1] == 0:
            occ_idx_a_rel = (
                np.arange(sum(d0a))[
                    (np.cumsum(d0a) - 1)[np.nonzero((d0a - dia) > 0)[0]]
                ],
            )
            Acre[nex], Ades[nex] = Acre.get(nex, []) + [occ_idx_a_rel], Ades.get(
                nex, []
            ) + [np.nonzero((d0a - dia) < 0)]
            coeff[nex][-1] *= parity(d0a, np.nonzero((d0a - dia) > 0), Ades[nex][-1])

        elif nex[0] == 0 and nex[1] > 0:
            occ_idx_b_rel = (
                np.arange(sum(d0b))[
                    (np.cumsum(d0b) - 1)[np.nonzero((d0b - dib) > 0)[0]]
                ],
            )
            Bcre[nex], Bdes[nex] = Bcre.get(nex, []) + [occ_idx_b_rel], Bdes.get(
                nex, []
            ) + [np.nonzero((d0b - dib) < 0)]
            coeff[nex][-1] *= parity(d0b, np.nonzero((d0b - dib) > 0), Bdes[nex][-1])

    coeff[(0, 0)] = np.asarray(coeff[(0, 0)]).reshape(
        -1,
    )

    # fill up the arrays up to max_excitation
    for i in range(1, max_excitation + 1):
        # singe alpha excitation
        if (i, 0) in Ades:
            Ades[(i, 0)] = np.asarray(Ades[(i, 0)]).reshape(-1, i) + num_core
            Acre[(i, 0)] = np.asarray(Acre[(i, 0)]).reshape(-1, i) + num_core
            coeff[(i, 0)] = np.asarray(coeff[(i, 0)]).reshape(
                -1,
            )
        else:
            Ades[(i, 0)] = np.zeros((1, i), dtype=int)
            Acre[(i, 0)] = np.zeros((1, i), dtype=int)
            coeff[(i, 0)] = np.zeros((1,))

        # singe beta excitation
        if (0, i) in Bdes:
            Bdes[(0, i)] = np.asarray(Bdes[(0, i)]).reshape(-1, i) + num_core
            Bcre[(0, i)] = np.asarray(Bcre[(0, i)]).reshape(-1, i) + num_core
            coeff[(0, i)] = np.asarray(coeff[(0, i)]).reshape(
                -1,
            )
        else:
            Bdes[(0, i)] = np.zeros((1, i), dtype=int)
            Bcre[(0, i)] = np.zeros((1, i), dtype=int)
            coeff[(0, i)] = np.zeros((1,))

        # alpha-beta
        if i != 0:
            for j in range(1, max_excitation + 1 - i):
                if (i, j) in Ades:
                    Ades[(i, j)] = np.asarray(Ades[(i, j)]).reshape(-1, i) + num_core
                    Acre[(i, j)] = np.asarray(Acre[(i, j)]).reshape(-1, i) + num_core
                    Bdes[(i, j)] = np.asarray(Bdes[(i, j)]).reshape(-1, j) + num_core
                    Bcre[(i, j)] = np.asarray(Bcre[(i, j)]).reshape(-1, j) + num_core
                    coeff[(i, j)] = np.asarray(coeff[(i, j)]).reshape(
                        -1,
                    )
                else:
                    Ades[(i, j)] = np.zeros((1, j), dtype=int)
                    Acre[(i, j)] = np.zeros((1, j), dtype=int)
                    Bdes[(i, j)] = np.zeros((1, j), dtype=int)
                    Bcre[(i, j)] = np.zeros((1, j), dtype=int)
                    coeff[(i, j)] = np.zeros((1,))
    ref_det = np.array([d0a, d0b])
    return Acre, Ades, Bcre, Bdes, coeff, ref_det


# reading dets from dice
def read_dets(
    fname: str = "dets.bin", ndets: Optional[int] = None
) -> Tuple[int, dict, int]:
    """Read determinants from a binary file generated by Dice.

    Args:
        fname: Binary file containing determinants.
        ndets: Number of determinants to read.

    Returns:
        norbs: Number of orbitals.
        state: Dictionary with determinant (tuple of up and down occupation number tuples) keys and coefficient values.
        ndets_all: Total number of determinants in the file.
    """
    state = {}
    norbs = 0
    with open(fname, "rb") as f:
        ndets_all = struct.unpack("i", f.read(4))[0]
        norbs = struct.unpack("i", f.read(4))[0]
        if ndets is None:
            ndets = int(ndets_all)
        for _ in range(ndets):
            coeff = struct.unpack("d", f.read(8))[0]
            det = [[0 for _ in range(norbs)], [0 for _ in range(norbs)]]
            for j in range(norbs):
                occ = struct.unpack("c", f.read(1))[0]
                if occ == b"a":
                    det[0][j] = 1
                elif occ == b"b":
                    det[1][j] = 1
                elif occ == b"2":
                    det[0][j] = 1
                    det[1][j] = 1
            state[tuple(map(tuple, det))] = coeff

    return norbs, state, ndets_all


def write_dets(state: dict, norbs: int, fname: str = "dets.bin") -> None:
    """Write determinants to a binary file in Dice format.

    Args:
        state: Dictionary with determinant (tuple of up and down occupation number tuples) keys
              and coefficient values.
        norbs: Number of orbitals.
        fname: Output binary filename.

    The binary format is:
    - Number of determinants (4 bytes, integer)
    - Number of orbitals (4 bytes, integer)
    For each determinant:
    - Coefficient (8 bytes, double)
    - Orbital occupations (1 byte per orbital, character):
      '0' for empty, 'a' for alpha, 'b' for beta, '2' for double
    """
    import struct

    ndets = len(state)

    with open(fname, "wb") as f:
        # Write number of determinants and orbitals
        f.write(struct.pack("i", ndets))
        f.write(struct.pack("i", norbs))

        # Write each determinant
        for det, coeff in state.items():
            # Write coefficient
            f.write(struct.pack("d", coeff))

            # Convert occupation numbers to Dice format and write
            alpha, beta = det
            for i in range(norbs):
                if alpha[i] == 0 and beta[i] == 0:
                    f.write(struct.pack("c", b"0"))
                elif alpha[i] == 1 and beta[i] == 0:
                    f.write(struct.pack("c", b"a"))
                elif alpha[i] == 0 and beta[i] == 1:
                    f.write(struct.pack("c", b"b"))
                elif alpha[i] == 1 and beta[i] == 1:
                    f.write(struct.pack("c", b"2"))
                else:
                    raise ValueError(
                        f"Invalid occupation numbers at orbital {i}: "
                        f"alpha={alpha[i]}, beta={beta[i]}"
                    )


def parity(d0: np.ndarray, cre: Sequence, des: Sequence) -> float:
    """Compute the parity for an excitation.

    Args:
        d0: Reference determinant.
        cre: Creation indices.
        des: Destruction indices.

    Returns:
        Parity of the excitation.
    """
    d = 1.0 * d0
    parity = 1.0

    C = np.asarray(cre).flatten()
    D = np.asarray(des).flatten()
    for i in range(C.shape[0]):
        I, A = min(D[i], C[i]), max(D[i], C[i])
        parity *= 1.0 - 2.0 * ((np.sum(d[I + 1 : A])) % 2)
        d[C[i]] = 0
        d[D[i]] = 1
    return float(parity)

def read_pyscf_ccsd(log, mf_or_cc, tmpdir):
    # raise warning about mpi
    log.warn(
        "# Note that PySCF CC module uses MPI. This could potentially lead to MPI initialization issues in some cases."
    )
    mf = mf_or_cc._scf
    cc = mf_or_cc
    if cc.frozen is not None:
        norb_frozen = cc.frozen
    else:
        norb_frozen = 0
    if isinstance(cc, UCCSD):
        ci2aa = cc.t2[0] + 2 * np.einsum("ia,jb->ijab", cc.t1[0], cc.t1[0])
        ci2aa = (ci2aa - ci2aa.transpose(0, 1, 3, 2)) / 2
        ci2aa = ci2aa.transpose(0, 2, 1, 3)
        ci2bb = cc.t2[2] + 2 * np.einsum("ia,jb->ijab", cc.t1[1], cc.t1[1])
        ci2bb = (ci2bb - ci2bb.transpose(0, 1, 3, 2)) / 2 
        ci2bb = ci2bb.transpose(0, 2, 1, 3)
        ci2ab = cc.t2[1] + np.einsum("ia,jb->ijab", cc.t1[0], cc.t1[1])
        ci2ab = ci2ab.transpose(0, 2, 1, 3)
        ci1a = np.array(cc.t1[0])
        ci1b = np.array(cc.t1[1])
        np.savez(
            tmpdir + "/amplitudes.npz",
            ci1a=ci1a,
            ci1b=ci1b,
            ci2aa=ci2aa,
            ci2ab=ci2ab,
            ci2bb=ci2bb,
        )
    else:
        ci2 = cc.t2 + np.einsum("ia,jb->ijab", np.array(cc.t1), np.array(cc.t1))
        ci2 = ci2.transpose(0, 2, 1, 3)
        ci1 = np.array(cc.t1)
        np.savez(tmpdir + "/amplitudes.npz", ci1=ci1, ci2=ci2)

    return mf, cc, norb_frozen

def compute_cholesky_integrals(log, mol, mf, basis_coeff, integrals, norb_frozen, chol_cut):
    log.log("# Calculating Cholesky integrals")
    assert basis_coeff.dtype == "float", "Only implemented for real-valued MOs"
    h1e, chol, nelec, enuc, nbasis, nchol = [None] * 6
    if integrals is not None:
        assert norb_frozen == 0, "Frozen orbitals not supported for custom integrals"
        enuc = integrals["h0"]
        h1e = integrals["h1"]
        eri = integrals["h2"]
        nelec = mol.nelec
        nbasis = h1e.shape[-1]
        norb = nbasis
        eri = ao2mo.restore(4, eri, norb)
        chol0 = modified_cholesky(eri, chol_cut)
        nchol = chol0.shape[0]
        chol = np.zeros((nchol, norb, norb))
        for i in range(nchol):
            for m in range(norb):
                for n in range(m + 1):
                    triind = m * (m + 1) // 2 + n
                    chol[i, m, n] = chol0[i, triind]
                    chol[i, n, m] = chol0[i, triind]

        # basis transformation
        h1e = basis_coeff.T @ h1e @ basis_coeff
        for gamma in range(nchol):
            chol[gamma] = basis_coeff.T @ chol[gamma] @ basis_coeff

    else:
        DFbas = None
        if getattr(mf, "with_df", None) is not None:
            DFbas = mf.with_df.auxmol.basis  # type: ignore
        h1e, chol, nelec, enuc = generate_integrals(
            log, mol, mf.get_hcore(), basis_coeff, chol_cut, DFbas=DFbas,
        )
        nbasis = h1e.shape[-1]
        nelec = mol.nelec

        if norb_frozen > 0:
            assert norb_frozen * 2 < sum(
                nelec
            ), "Frozen orbitals exceed number of electrons"
            mc = mcscf.CASSCF(
                mf, mol.nao - norb_frozen, mol.nelectron - 2 * norb_frozen
            )
            nelec = mc.nelecas  # type: ignore
            mc.mo_coeff = basis_coeff  # type: ignore
            h1e, enuc = mc.get_h1eff()  # type: ignore
            chol = chol.reshape((-1, nbasis, nbasis))
            chol = chol[:, mc.ncore : mc.ncore + mc.ncas, mc.ncore : mc.ncore + mc.ncas]  # type: ignore
    return h1e, chol, nelec, enuc, nbasis, nchol

def write_trial(mol, mf, basis_coeff, nbasis, norb_frozen, tmpdir):
    assert basis_coeff.dtype == "float", "Only implemented for real-valued MOs"
    trial_coeffs = np.empty((2, nbasis, nbasis))
    overlap = mf.get_ovlp(mol)
    if isinstance(mf, (scf.uhf.UHF, scf.rohf.ROHF)):
        uhfCoeffs = np.empty((nbasis, 2 * nbasis))
        if isinstance(mf, scf.uhf.UHF):
            q, r = np.linalg.qr(
                basis_coeff[:, norb_frozen:]
                .T.dot(overlap)
                .dot(mf.mo_coeff[0][:, norb_frozen:])
            )
            sgn = np.sign(r.diagonal())
            q = np.einsum("ij,j->ij", q, sgn)
            # q2 = basis_coeff[:, norb_frozen:].T.dot(overlap).dot(mf.mo_coeff[0][:, norb_frozen:])
            # print("max err a", np.max(abs(q-q2)))
            # q, _ = np.linalg.qr(
            #    basis_coeff[:, norb_frozen:]
            #    .T.dot(overlap)
            #    .dot(mf.mo_coeff[0][:, norb_frozen:])
            # )
            uhfCoeffs[:, :nbasis] = q
            q, r = np.linalg.qr(
                basis_coeff[:, norb_frozen:]
                .T.dot(overlap)
                .dot(mf.mo_coeff[1][:, norb_frozen:])
            )
            sgn = np.sign(r.diagonal())
            q = np.einsum("ij,j->ij", q, sgn)
            # q2 = basis_coeff[:, norb_frozen:].T.dot(overlap).dot(mf.mo_coeff[1][:, norb_frozen:])
            # print("max err b", np.max(abs(q-q2)))
            # import pdb
            # pdb.set_trace()
            # q, _ = np.linalg.qr(
            #     basis_coeff[:, norb_frozen:]
            #     .T.dot(overlap)
            #     .dot(mf.mo_coeff[1][:, norb_frozen:])
            # )
            uhfCoeffs[:, nbasis:] = q
        else:
            q, r = np.linalg.qr(
                basis_coeff[:, norb_frozen:]
                .T.dot(overlap)
                .dot(mf.mo_coeff[:, norb_frozen:])
            )
            sgn = np.sign(r.diagonal())
            q = np.einsum("ij,j->ij", q, sgn)
            uhfCoeffs[:, :nbasis] = q
            uhfCoeffs[:, nbasis:] = q

        trial_coeffs[0] = uhfCoeffs[:, :nbasis]
        trial_coeffs[1] = uhfCoeffs[:, nbasis:]
        # np.savetxt("uhf.txt", uhfCoeffs)
        np.savez(tmpdir + "/mo_coeff.npz", mo_coeff=trial_coeffs)

    elif isinstance(mf, scf.rhf.RHF):
        q, r = np.linalg.qr(
            basis_coeff[:, norb_frozen:]
            .T.dot(overlap)
            .dot(mf.mo_coeff[:, norb_frozen:])
        )
        sgn = np.sign(r.diagonal())
        q = np.einsum("ij,j->ij", q, sgn)
        trial_coeffs[0] = q
        trial_coeffs[1] = q
        np.savez(tmpdir + "/mo_coeff.npz", mo_coeff=trial_coeffs)

    return trial_coeffs

def prep_afqmc_ghf_complex(mol, gmf: scf.ghf.GHF, tmpdir, chol_cut=1e-5, verbose: int=0):
    import scipy.linalg as la
    log = Logger(sys.stdout, verbose)

    norb = np.shape(gmf.mo_coeff)[-1]//2
    mo_coeff = gmf.mo_coeff

    # Chol ao to mo
    chol_vecs = chunked_cholesky(log, mol, max_error=chol_cut)
    nchol = chol_vecs.shape[0]
    chol = np.zeros((nchol, 2*norb, 2*norb), dtype=complex)
    for i in range(nchol):
        chol_i = chol_vecs[i].reshape(norb, norb)
        chol_i = la.block_diag(chol_i, chol_i)
        chol[i] = mo_coeff.T.conj() @ chol_i @ mo_coeff

    # h ao to mo
    h = mo_coeff.T.conj() @ gmf.get_hcore() @ mo_coeff

    enuc = mol.energy_nuc()
    nbasis = h.shape[-1]
    log.log(f'nelec: {mol.nelec}')
    log.log(f'nbasis: {nbasis}')
    log.log(f'chol.shape: {chol.shape}')

    # Modified one-electron integrals
    chol = chol.reshape((-1, nbasis, nbasis))
    v0 = 0.5 * np.einsum('gik,gkj->ij', chol, chol, optimize='optimal')
    h_mod = h - v0
    chol = chol.reshape((chol.shape[0], -1))

    # Save
    write_dqmc(h, h_mod, chol, sum(mol.nelec), nbasis, enuc, ms=mol.spin, filename=tmpdir+'/FCIDUMP_chol')

    ovlp = gmf.get_ovlp(mol)
    q, r = np.linalg.qr(
        mo_coeff.T.conj() @ ovlp @ mo_coeff
    )
    sgn = np.sign(r.diagonal())
    q = np.einsum("ij,j->ij", q, sgn)
    np.savez(tmpdir + "/mo_coeff.npz", mo_coeff=[q,q])

    return h, h_mod, chol

def prep_afqmc_spinor(mol, mo_coeff, h_ao, n_ao, tmpdir, chol_cut=1e-5, verbose: int=0):
    import scipy.linalg as la
    from socutils.scf import spinor_hf

    log = Logger(sys.stdout, verbose)

    norb = n_ao

    # Chol ao to mo
    chol_vecs = chunked_cholesky(log, mol, max_error=chol_cut)
    nchol = chol_vecs.shape[0]
    chol = np.zeros((nchol, norb, norb), dtype=complex)
    for i in range(nchol):
        chol_i = chol_vecs[i].reshape(norb//2, norb//2)
        chol_i = spinor_hf.sph2spinor(mol, la.block_diag(chol_i, chol_i))
        chol[i] = mo_coeff.T.conj() @ chol_i @ mo_coeff

    # h ao to mo
    h = mo_coeff.T.conj() @ h_ao @ mo_coeff

    enuc = mol.energy_nuc()
    nbasis = h.shape[-1]
    log.log(f'nelec: {mol.nelec}')
    log.log(f'nbasis: {nbasis}')
    log.log(f'chol.shape: {chol.shape}')

    # Modified one-electron integrals
    chol = chol.reshape((-1, nbasis, nbasis))
    v0 = 0.5 * np.einsum('gik,gkj->ij', chol, chol, optimize='optimal')
    h_mod = h - v0
    chol = chol.reshape((chol.shape[0], -1))

    # Save
    write_dqmc(h, h_mod, chol, sum(mol.nelec), nbasis, enuc, ms=mol.spin, filename=tmpdir+'/FCIDUMP_chol')

    ovlp = mol.intor("int1e_ovlp_spinor")
    q, r = np.linalg.qr(
        mo_coeff.T.conj() @ ovlp @ mo_coeff
    )
    sgn = np.sign(r.diagonal())
    q = np.einsum("ij,j->ij", q, sgn)
    np.savez(tmpdir + "/mo_coeff.npz", mo_coeff=[q,q])

    return h, h_mod, chol
