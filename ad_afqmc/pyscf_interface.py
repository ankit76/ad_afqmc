import pickle
import struct
import time
from functools import partial

import h5py
import numpy as np
from pyscf import __config__, ao2mo, fci, gto, mcscf, scf

print = partial(print, flush=True)


# modified cholesky for a given matrix
def modified_cholesky(mat, max_error=1e-6):
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
def prep_afqmc(mf, mo_coeff=None, norb_frozen=0, chol_cut=1e-5, integrals=None, 
               filename="FCIDUMP_chol"):
    print("#\n# Preparing AFQMC calculation")

    mol = mf.mol
    # choose the orbital basis
    if mo_coeff is None:
        if isinstance(mf, scf.uhf.UHF):
            mo_coeff = mf.mo_coeff[0]
        elif isinstance(mf, scf.rhf.RHF):
            mo_coeff = mf.mo_coeff
        else:
            raise Exception("# Invalid mean field object!")

    # calculate cholesky integrals
    print("# Calculating Cholesky integrals")
    h1e, chol, nelec, enuc, nbasis, nchol = [None] * 6
    if integrals is not None:
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

    else:
        h1e, chol, nelec, enuc = generate_integrals(
            mol, mf.get_hcore(), mo_coeff, chol_cut
        )
        nbasis = h1e.shape[-1]
        nelec = mol.nelec

        if norb_frozen > 0:
            assert norb_frozen * 2 < sum(nelec)
            mc = mcscf.CASSCF(
                mf, mol.nao - norb_frozen, mol.nelectron - 2 * norb_frozen
            )
            nelec = mc.nelecas
            mc.mo_coeff = mo_coeff
            h1e, enuc = mc.get_h1eff()
            chol = chol.reshape((-1, nbasis, nbasis))
            chol = chol[:, mc.ncore : mc.ncore + mc.ncas, mc.ncore : mc.ncore + mc.ncas]

    print("# Finished calculating Cholesky integrals\n")

    nbasis = h1e.shape[-1]
    print("# Size of the correlation space:")
    print(f"# Number of electrons: {nelec}")
    print(f"# Number of basis functions: {nbasis}")
    print(f"# Number of Cholesky vectors: {chol.shape[0]}\n")
    chol = chol.reshape((-1, nbasis, nbasis))
    v0 = 0.5 * np.einsum("nik,njk->ij", chol, chol, optimize="optimal")
    h1e_mod = h1e - v0
    chol = chol.reshape((chol.shape[0], -1))

    # write mo coefficients
    trial_coeffs = np.empty((2, nbasis, nbasis))
    overlap = mf.get_ovlp(mol)
    if isinstance(mf, (scf.uhf.UHF, scf.rohf.ROHF)):
        hf_type = "uhf"
        uhfCoeffs = np.empty((nbasis, 2 * nbasis))
        if isinstance(mf, scf.uhf.UHF):
            q, r = np.linalg.qr(
                mo_coeff[:, norb_frozen:]
                .T.dot(overlap)
                .dot(mf.mo_coeff[0][:, norb_frozen:])
            )
            uhfCoeffs[:, :nbasis] = q
            q, r = np.linalg.qr(
                mo_coeff[:, norb_frozen:]
                .T.dot(overlap)
                .dot(mf.mo_coeff[1][:, norb_frozen:])
            )
            uhfCoeffs[:, nbasis:] = q
        else:
            q, r = np.linalg.qr(
                mo_coeff[:, norb_frozen:]
                .T.dot(overlap)
                .dot(mf.mo_coeff[:, norb_frozen:])
            )
            uhfCoeffs[:, :nbasis] = q
            uhfCoeffs[:, nbasis:] = q

        trial_coeffs[0] = uhfCoeffs[:, :nbasis]
        trial_coeffs[1] = uhfCoeffs[:, nbasis:]
        # np.savetxt("uhf.txt", uhfCoeffs)
        np.savez("uhf.npz", mo_coeff=trial_coeffs)

    elif isinstance(mf, scf.rhf.RHF):
        hf_type = "rhf"
        q, r = np.linalg.qr(
            mo_coeff[:, norb_frozen:].T.dot(overlap).dot(mf.mo_coeff[:, norb_frozen:])
        )
        trial_coeffs[0] = q
        trial_coeffs[1] = q
        np.savetxt("rhf.txt", q)

    write_dqmc(
        h1e,
        h1e_mod,
        chol,
        sum(nelec),
        nbasis,
        enuc,
        ms=mol.spin,
        filename=filename,
        mo_coeffs=trial_coeffs,
    )


# cholesky generation functions are from pauxy
def generate_integrals(mol, hcore, X, chol_cut=1e-5, verbose=False):
    # Unpack SCF data.
    # Step 1. Rotate core Hamiltonian to orthogonal basis.
    if verbose:
        print(" # Transforming hcore and eri to ortho AO basis.")
    if len(X.shape) == 2:
        h1e = np.dot(X.T, np.dot(hcore, X))
    elif len(X.shape) == 3:
        h1e = np.dot(X[0].T, np.dot(hcore, X[0]))

    # nbasis = h1e.shape[-1]
    # Step 2. Genrate Cholesky decomposed ERIs in non-orthogonal AO basis.
    if verbose:
        print(" # Performing modified Cholesky decomposition on ERI tensor.")
    chol_vecs = chunked_cholesky(mol, max_error=chol_cut, verbose=verbose)
    if verbose:
        print(" # Orthogonalising Cholesky vectors.")
    start = time.time()
    # Step 2.a Orthogonalise Cholesky vectors.
    if len(X.shape) == 2 and X.shape[0] != X.shape[1]:
        chol_vecs = ao2mo_chol_copy(chol_vecs, X)
    elif len(X.shape) == 2:
        ao2mo_chol(chol_vecs, X)
    elif len(X.shape) == 3:
        ao2mo_chol(chol_vecs, X[0])
    if verbose:
        print(" # Time to orthogonalise: %f" % (time.time() - start))
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


def chunked_cholesky(mol, max_error=1e-6, verbose=False, cmax=10):
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
    if verbose:
        print("# Generating Cholesky decomposition of ERIs." % nchol_max)
        print("# max number of cholesky vectors = %d" % nchol_max)
        print("# iteration %5d: delta_max = %f" % (0, delta_max))
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
        if verbose:
            step_time = time.time() - start
            info = (nchol, delta_max, step_time)
            print("# iteration %5d: delta_max = %13.8e: time = %13.8e" % info)

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
            mf = mf.newton().run(mo1, mf.mo_occ)
            mo1 = mf.stability(external=True)[0]
            mf = mf.newton().run(mo1, mf.mo_occ)
        mf_coeff = mf.mo_coeff.copy()
        mf_occ = mf.mo_occ.copy()

    h1e = mf.get_hcore() - epsilon * observable
    mf.get_hcore = lambda *args: h1e
    mf.verbose = 1
    if relaxed:
        if dm is None:
            mf.kernel()
        else:
            mf.kernel(dm)
        if hf_type == "uhf":
            mo1 = mf.stability(external=True)[0]
            mf = mf.newton().run(mo1, mf.mo_occ)
            mo1 = mf.stability(external=True)[0]
            mf = mf.newton().run(mo1, mf.mo_occ)
    emf_m = mf.e_tot - epsilon * observable_constant
    mycc = cc.CCSD(mf)
    mycc.frozen = norb_frozen
    mycc.kernel()
    emp2_m = mycc.e_hf + mycc.emp2 - epsilon * observable_constant
    eccsd_m = mycc.e_tot - epsilon * observable_constant
    et = mycc.ccsd_t()
    eccsdpt_m = mycc.e_tot + et - epsilon * observable_constant

    if hf_type == "rhf":
        mf = scf.RHF(mol)
    elif hf_type == "uhf":
        mf = scf.UHF(mol)
    if not relaxed:
        mf.verbose = 1
        mf.mo_coeff = mf_coeff
        mf.mo_occ = mf_occ
    h1e = mf.get_hcore() + epsilon * observable
    mf.get_hcore = lambda *args: h1e
    mf.verbose = 1
    if relaxed:
        if dm is None:
            mf.kernel()
        else:
            mf.kernel(dm)
        if hf_type == "uhf":
            mo1 = mf.stability(external=True)[0]
            mf = mf.newton().run(mo1, mf.mo_occ)
            mo1 = mf.stability(external=True)[0]
            mf = mf.newton().run(mo1, mf.mo_occ)
    emf_p = mf.e_tot + epsilon * observable_constant
    mycc = cc.CCSD(mf)
    mycc.frozen = norb_frozen
    mycc.kernel()
    emp2_p = mycc.e_hf + mycc.emp2 + epsilon * observable_constant
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


# dice to noci
# not tested
def hci_to_noci(nelec_sp, fname="dets.bin", ndets=None):
    state = {}
    norbs = 0
    with open(fname, "rb") as f:
        ndetsAll = struct.unpack("i", f.read(4))[0]
        norbs = struct.unpack("i", f.read(4))[0]
        if ndets is None:
            ndets = ndetsAll
        dets_up = np.zeros((ndets, norbs, nelec_sp[0]))
        dets_dn = np.zeros((ndets, norbs, nelec_sp[1]))
        for i in range(ndets):
            coeff = struct.unpack("d", f.read(8))[0]
            det = [[0 for i in range(norbs)], [0 for i in range(norbs)]]
            for j in range(norbs):
                nelec_up_counter = 0
                nelec_dn_counter = 0
                occ = struct.unpack("c", f.read(1))[0]
                if occ == b"a":
                    det[0][j] = 1
                    dets_up[i][j, nelec_up_counter] = 1.0
                    nelec_up_counter += 1
                elif occ == b"b":
                    det[1][j] = 1
                    dets_dn[i][j, nelec_dn_counter] = 1.0
                    nelec_dn_counter += 1
                elif occ == b"2":
                    det[0][j] = 1
                    det[1][j] = 1
                    dets_up[i][j, nelec_up_counter] = 1.0
                    dets_dn[i][j, nelec_dn_counter] = 1.0
                    nelec_up_counter += 1
                    nelec_dn_counter += 1
            state[tuple(map(tuple, det))] = coeff

    return norbs, state, ndetsAll


def fci_to_noci(fci, ndets=None, tol=1.0e-4):
    ci_coeffs = fci.ci
    norb = fci.norb
    nelec = fci.nelec
    if ndets is None:
        ndets = ci_coeffs.shape[0]
    coeffs, occ_a, occ_b = zip(
        *fci.large_ci(ci_coeffs, norb, nelec, tol=tol, return_strs=False)
    )
    dets_up = np.zeros((ndets, norb, nelec[0]))
    dets_dn = np.zeros((ndets, norb, nelec[1]))
    for i in range(ndets):
        for j in range(nelec[0]):
            dets_up[i][occ_a[i][j], j] = 1.0
        for j in range(nelec[1]):
            dets_dn[i][occ_b[i][j], j] = 1.0
    with open("dets.pkl", "wb") as f:
        pickle.dump([np.array(coeffs[:ndets]), [dets_up, dets_dn]], f)
    return np.array(coeffs[:ndets]), [dets_up, dets_dn]


if __name__ == "__main__":
    from pyscf import fci, gto, scf

    from ad_afqmc import pyscf_interface

    atomstring = f"""
                     O 0.00000000 -0.13209669 0.00000000
                     H 0.00000000 0.97970006  1.43152878
                     H 0.00000000  0.97970006 -1.43152878
                  """
    mol = gto.M(atom=atomstring, basis="sto-6g", verbose=3, unit="bohr", symmetry=1)
    mf = scf.RHF(mol)
    mf.kernel()

    ci = fci.FCI(mf)
    ci.kernel()

    ci_coeffs, dets = pyscf_interface.fci_to_noci(ci, ndets=5)
