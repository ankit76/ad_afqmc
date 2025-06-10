import os
import numpy as np
from pyscf import gto, scf, df, lib, grad
from ad_afqmc import pyscf_interface
from scipy.linalg import fractional_matrix_power
import copy


def get_transformation_matrix(S, lin_dep_thresh=1e-10):
    # Diagonalize the overlap matrix S to obtain the transformation matrix X
    Seigval, Sorbs = np.linalg.eigh(S)
    emin = Seigval[-1] * lin_dep_thresh
    X = Sorbs[:, Seigval > emin]
    w = Seigval[Seigval > emin]

    # To obtain the orthonormal set of basis functions use "canonical orthogonalization"
    np.sqrt(w, out=w)
    w[:] = 1.0 / w
    np.multiply(X, w, X)
    return X


def _ao2mo(chol, C):  # Convert the 2e integrals from AO to MO basis
    chol2 = np.zeros((chol.shape[0], C.shape[0], C.shape[0]))
    for i in range(chol.shape[0]):
        chol2[i] = np.dot(C.T, np.dot(chol[i], C))
    return chol2


def FD_integrals(mf, dR=0.00001):
    basis = mf.mol.basis
    mol = mf.mol

    coords = mol.atom_coords()  # This returns coords in Bohr
    atom_symbols = [mol.atom_pure_symbol(i) for i in range(len(coords))]
    unit = "Bohr"

    coords = np.array(coords)
    df0 = df.incore.cholesky_eri(mol, auxbasis=mf.auxbasis)

    basis0 = np.array(fractional_matrix_power(mf.get_ovlp(), -0.5), dtype="float64")
    h1_der_array = np.zeros((mol.natm, 3, mol.nao, mol.nao))
    h2_der_array = np.zeros((mol.natm, 3, df0.shape[0], mol.nao, mol.nao))

    for i in range(len(atom_symbols)):
        coords_m = coords.copy()
        coords_p = coords.copy()
        for j in range(3):
            coords_p = copy.copy(coords)
            coords_p[i][j] += dR
            atom_p = list(zip(atom_symbols, coords_p))
            mol_p = gto.M(atom=atom_p, basis=basis, verbose=1, unit=unit)
            mf_p = df.density_fit(scf.RHF(mol_p), auxbasis=mf.auxbasis)
            basis_p = np.array(
                fractional_matrix_power(mf_p.get_ovlp(), -0.5), dtype="float64"
            )
            h1_p = np.array(basis_p.T @ mf_p.get_hcore() @ basis_p, dtype="float64")

            df_p = df.incore.cholesky_eri(mol_p, auxbasis=mf.auxbasis)
            df_p = lib.unpack_tril(df_p)
            chol_p = _ao2mo(df_p, basis_p)

            coords_m = copy.copy(coords)
            coords_m[i][j] -= dR
            atom_m = list(zip(atom_symbols, coords_m))
            mol_m = gto.M(atom=atom_m, basis=basis, verbose=3, unit=unit)
            mf_m = df.density_fit(scf.RHF(mol_m), auxbasis=mf.auxbasis)
            basis_m = np.array(
                fractional_matrix_power(mf_m.get_ovlp(), -0.5), dtype="float64"
            )
            h1_m = np.array(basis_m.T @ mf_m.get_hcore() @ basis_m, dtype="float64")

            df_m = df.incore.cholesky_eri(mol_m, auxbasis=mf.auxbasis)
            df_m = lib.unpack_tril(df_m)
            chol_m = _ao2mo(df_m, basis_m)

            h1_der = (h1_p - h1_m) / (2.0 * dR)
            h2_der = (chol_p - chol_m) / (2.0 * dR)

            h1_der_array[i, j] = h1_der
            h2_der_array[i, j] = h2_der
    h0_der = grad.rhf.grad_nuc(mol)

    if isinstance(mf, scf.uhf.UHF):
        dm0 = [
            basis0.T @ mf.get_ovlp() @ mf.make_rdm1()[0] @ mf.get_ovlp() @ basis0,
            basis0.T @ mf.get_ovlp() @ mf.make_rdm1()[1] @ mf.get_ovlp() @ basis0,
        ]
    elif isinstance(mf, scf.rhf.RHF):
        dm0 = basis0.T @ mf.get_ovlp() @ mf.make_rdm1() @ mf.get_ovlp() @ basis0
    np.savez(
        "Integral_der.npz",
        array1=h1_der_array,
        array2=h2_der_array,
        array3=h0_der,
        dm=dm0,
    )


def write_integrals_lowdins(mf):
    mol = mf.mol
    basis0 = np.array(fractional_matrix_power(mf.get_ovlp(), -0.5), dtype="float64")
    X = basis0.copy()
    X_inv = np.linalg.inv(X)
    c_ao = X_inv @ mf.mo_coeff
    if isinstance(mf, scf.uhf.UHF):
        c_ao = [X_inv @ mf.mo_coeff[0], X_inv @ mf.mo_coeff[1]]
    elif isinstance(mf, scf.rhf.RHF):
        c_ao = X_inv @ mf.mo_coeff
    h1 = np.array(basis0.T @ mf.get_hcore() @ basis0, dtype="float64")

    df0 = df.incore.cholesky_eri(mol, auxbasis=mf.auxbasis)  # ,aosym='s1')
    df0 = lib.unpack_tril(df0)
    chol1 = _ao2mo(df0, basis0)

    h1e = h1.copy()
    chol = chol1.copy()
    nbasis = h1e.shape[-1]
    nelec = mol.nelec
    enuc = mol.energy_nuc()
    chol = chol.reshape((-1, nbasis, nbasis))
    v0 = 0.5 * np.einsum("nik,njk->ij", chol, chol, optimize="optimal")
    h1e_mod = h1e - v0
    chol = chol.reshape((chol.shape[0], -1))
    q = c_ao

    if isinstance(mf, scf.uhf.UHF):
        np.savez(
            "mo_coeff.npz",
            mo_coeff=np.array(q),
            X=X,
            X_inv=X_inv,
            can_mo=mf.mo_coeff,
        )
    elif isinstance(mf, scf.rhf.RHF):
        np.savez("mo_coeff.npz", mo_coeff=[q, q], X=X, X_inv=X_inv, can_mo=mf.mo_coeff)

    pyscf_interface.write_dqmc(
        h1e,
        h1e_mod,
        chol,
        sum(nelec),
        nbasis,
        enuc,
        ms=mol.spin,
        filename="FCIDUMP_chol",
    )


def prep_afqmc_nuc_grad(mf, dR=1e-5):
    print("Removing old files")
    os.system("rm -f en_der_afqmc_*.npz")
    FD_integrals(mf, dR=dR)
    write_integrals_lowdins(mf)


def reject_outliers(data, m=10.0):
    # Filter out NaN values
    not_nan_mask = ~np.isnan(data)
    data = data[not_nan_mask]

    # Calculate deviations from median and identify outliers
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d / mdev if mdev else 0.0

    # Filter out the outliers
    non_outliers_mask = s < m
    final_mask = not_nan_mask.copy()
    final_mask[not_nan_mask] = non_outliers_mask

    return data[non_outliers_mask], final_mask


def find_nproc():
    import glob
    import re

    files = glob.glob("en_der_afqmc_*.npz")
    pattern = re.compile(r"en_der_afqmc_(\d+)\.npz")

    indices = []
    for file in files:
        match = pattern.search(file)
        if match:
            indices.append(int(match.group(1)))
    print(max(indices) + 1)
    return max(indices) + 1


def get_rdmsDer(norb, nchol, filename="en_der_afqmc.npz", nproc=4):
    rdm1 = []
    rdm2 = []
    weight = []
    for i in range(nproc):
        weighti = np.load(f"en_der_afqmc_{i}.npz")["weight"]
        rdm1i = np.load(f"en_der_afqmc_{i}.npz")["rdm1"].reshape(
            2, weighti.shape[0], norb, norb
        )
        rdm2i = np.load(f"en_der_afqmc_{i}.npz")["rdm2"].reshape(
            weighti.shape[0], nchol, norb, norb
        )
        for j in range(rdm1i.shape[1]):
            rdm1.append(rdm1i[:, j, :, :])
            rdm2.append(rdm2i[j])  # ,:,:])
            weight.append(weighti[j])

    return np.array(rdm1), np.array(rdm2), np.array(weight)


def reject_nan(data):
    not_nan_mask = ~np.isnan(data)
    return data[not_nan_mask], not_nan_mask


def weighted_std(data, weights):
    # Calculate the weighted mean
    weighted_mean = np.sum(weights * data) / np.sum(weights)
    # Calculate the weighted variance
    weighted_variance = np.sum(weights * (data - weighted_mean) ** 2) / np.sum(weights)
    # Standard deviation is the square root of variance
    weighted_std_dev = np.sqrt(weighted_variance) / np.sqrt(len(weights))  # SEM
    return weighted_mean, weighted_std_dev


def calculate_nuc_gradients(
    integral_der="Integral_der.npz",
    energy_der="en_der_afqmc.npz",
    printG=True,
    reject_outliers_enabled=True,
):
    h1_der = np.load(integral_der)["array1"]  # (natm,3,norb,norb)
    h2_der = np.load(integral_der)["array2"]  # (natm,3,nchol,norb,norb)
    h0_der = np.load(integral_der)["array3"]  # (natm,3)

    rdm1, rdm2, weights = get_rdmsDer(
        h1_der.shape[2], h2_der.shape[2], nproc=find_nproc(), filename=energy_der
    )
    obs1 = np.einsum("rxpq,npq->nrx", h1_der, rdm1[:, 0, :, :]) + np.einsum(
        "rxpq,npq->nrx", h1_der, rdm1[:, 1, :, :]
    )
    obs2 = np.einsum("rxgpq,ngpq->nrx", h2_der, rdm2)
    obs0 = h0_der
    gradients = obs1 + obs2 + obs0

    natm = gradients.shape[1]
    error = np.zeros((natm, 3))
    avg = np.zeros((natm, 3))
    for i in range(natm):
        for j in range(3):
            if reject_outliers_enabled:
                data, mask = reject_outliers(gradients[:, i, j], m=10)
                masked_weight = weights[mask]
                if np.sum(mask == False):
                    print(f"({i},{j}) Outliers removed: {np.sum(mask == False)}")
            else:
                data, mask = reject_nan(gradients[:, i, j])
                masked_weight = weights[mask]
                if np.sum(mask == False):
                    print(f"({i},{j}) NaN removed: {np.sum(mask == False)}")
            grad_avg, grad_err = weighted_std(data, masked_weight)
            avg[i, j] = grad_avg
            error[i, j] = grad_err
    if printG:
        print("Blocking analysis done", flush=True)
        print("--------------- AFQMC Gradient ---------------\n", flush=True)
        print(avg, flush=True)
        print("--------------- Error ---------------\n", flush=True)
        print(error, flush=True)
        print("----------------------------------------------", flush=True)
    return avg, error


def append_to_array(filename, array_data1, array_data2, array_data3):
    # Load the existing arrays from the .npz file
    if os.path.exists(filename):
        with np.load(filename) as data:
            array1 = data["rdm1"]
            array2 = data["rdm2"]
            array3 = data["weight"]
    else:
        array1 = np.array([])
        array2 = np.array([])
        array3 = np.array([])
    # Append new data to array1
    array1 = np.append(array1, array_data1)
    array2 = np.append(array2, array_data2)
    array3 = np.append(array3, array_data3)

    # Save all arrays back to the .npz file
    np.savez(filename, rdm1=array1, rdm2=array2, weight=array3)
    del array1, array2, array3  # Free up memory
