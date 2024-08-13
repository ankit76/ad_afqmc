import pickle
import struct
import time
from functools import partial

import h5py
import numpy as np
from pyscf import __config__, ao2mo, fci, gto, mcscf, scf
from ad_afqmc import pyscf_interface, stat_utils
from scipy.linalg import fractional_matrix_power
import numpy as np
from pyscf import df, gto, grad, scf, lib, ao2mo
import pyscf
import copy
import math


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


def FD_integrals_chol(mf, dR=1e-5, chol_cut=1e-5):
    basis = mf.mol.basis
    mol = mf.mol
    # atoms = mf.mol.atom
    atom_symbols, coords = [list(t) for t in zip(*mol.atom)]
    unit = mf.mol.unit
    # print("unit=",unit)
    coords = np.array(coords)
    # df0 = df.incore.cholesky_eri(mol,auxbasis=mf.auxbasis)
    eri0 = mol.intor("int2e")
    df0 = pyscf_interface.chunked_cholesky(mol, max_error=chol_cut)

    basis0 = np.array(fractional_matrix_power(mf.get_ovlp(), -0.5), dtype="float64")
    h1_der_array = np.zeros((mol.natm, 3, mol.nao, mol.nao))
    h2_der_array = np.zeros((mol.natm, 3, df0.shape[0], mol.nao, mol.nao))

    for i in range(len(atom_symbols)):
        coords_m = coords.copy()
        coords_p = coords.copy()
        for j in [2]:  # range(3):
            coords_p = copy.copy(coords)  # coords.copy()
            coords_p[i][j] += dR
            atom_p = list(zip(atom_symbols, coords_p))
            mol_p = gto.M(atom=atom_p, basis=basis, verbose=1, unit=unit)
            mf_p = df.density_fit(scf.RHF(mol_p), auxbasis=mf.auxbasis)
            basis_p = np.array(
                fractional_matrix_power(mf_p.get_ovlp(), -0.5), dtype="float64"
            )
            # basis_p = get_transformation_matrix(mf_p.get_ovlp())#np.array(fractional_matrix_power(mf_p.get_ovlp(), -0.5),dtype="float64")
            h1_p = np.array(basis_p.T @ mf_p.get_hcore() @ basis_p, dtype="float64")

            # df_p = df.incore.cholesky_eri(mol_p,auxbasis=mf.auxbasis)
            # df_p = lib.unpack_tril(df_p)
            # chol_p = _ao2mo(df_p,basis_p)

            coords_m = copy.copy(coords)
            coords_m[i][j] -= dR
            atom_m = list(zip(atom_symbols, coords_m))
            # print(i,j,atom_m)
            mol_m = gto.M(atom=atom_m, basis=basis, verbose=3, unit=unit)
            mf_m = df.density_fit(scf.RHF(mol_m), auxbasis=mf.auxbasis)
            # mf_m.kernel()
            basis_m = np.array(
                fractional_matrix_power(mf_m.get_ovlp(), -0.5), dtype="float64"
            )
            # basis_m = get_transformation_matrix(mf_m.get_ovlp()) #np.array(fractional_matrix_power(mf_m.get_ovlp(), -0.5),dtype="float64")
            h1_m = np.array(basis_m.T @ mf_m.get_hcore() @ basis_m, dtype="float64")

            # df_m = df.incore.cholesky_eri(mol_m,auxbasis=mf.auxbasis)
            # df_m = lib.unpack_tril(df_m)
            # chol_m = _ao2mo(df_m,basis_m)

            eri_p = mol_p.intor("int2e")
            eri_p = ao2mo.kernel(eri_p, basis_p, compact=False)
            eri_m = mol_m.intor("int2e")
            eri_m = ao2mo.kernel(eri_m, basis_m, compact=False)
            deri = (eri_p - eri_m) / (2.0 * dR)
            deri = (
                deri
                + np.transpose(deri, (2, 3, 0, 1))
                + np.transpose(deri, (1, 0, 3, 2))
                + np.transpose(deri, (3, 2, 1, 0))
            ) / 4.0
            h2_der = pyscf_interface.modified_cholesky(
                deri.reshape(mol.nao**2, mol.nao**2), max_error=chol_cut
            )
            import pdb

            pdb.set_trace()
            h1_der = (h1_p - h1_m) / (2.0 * dR)
            # h2_der = (chol_p - chol_m)/(2.*dR)

            h1_der_array[i, j] = h1_der
            h2_der_array[i, j] = h2_der
    h0_der = pyscf.grad.rhf.grad_nuc(mol)
    dm0 = basis0.T @ mf.get_ovlp() @ mf.make_rdm1() @ mf.get_ovlp() @ basis0
    np.savez(
        "Integral_der.npz",
        array1=h1_der_array,
        array2=h2_der_array,
        array3=h0_der,
        dm=dm0,
    )


def FD_integrals(mf, dR=0.00001):
    basis = mf.mol.basis
    mol = mf.mol
    
    coords = mol.atom_coords()  # This returns coords in Bohr
    atom_symbols = [mol.atom_pure_symbol(i) for i in range(len(coords))]
    unit = "Bohr"  # mf.mol.unit

    coords = np.array(coords)
    df0 = df.incore.cholesky_eri(mol, auxbasis=mf.auxbasis)

    basis0 = np.array(fractional_matrix_power(mf.get_ovlp(), -0.5), dtype="float64")
    h1_der_array = np.zeros((mol.natm, 3, mol.nao, mol.nao))
    h2_der_array = np.zeros((mol.natm, 3, df0.shape[0], mol.nao, mol.nao))

    for i in range(len(atom_symbols)):
        coords_m = coords.copy()
        coords_p = coords.copy()
        for j in range(3):
            coords_p = copy.copy(coords)  # coords.copy()
            # print(coords_p)
            coords_p[i][j] += dR
            atom_p = list(zip(atom_symbols, coords_p))
            mol_p = gto.M(atom=atom_p, basis=basis, verbose=1, unit=unit)
            mf_p = df.density_fit(scf.RHF(mol_p), auxbasis=mf.auxbasis)
            basis_p = np.array(
                fractional_matrix_power(mf_p.get_ovlp(), -0.5), dtype="float64"
            )
            # basis_p = get_transformation_matrix(mf_p.get_ovlp())#np.array(fractional_matrix_power(mf_p.get_ovlp(), -0.5),dtype="float64")
            h1_p = np.array(basis_p.T @ mf_p.get_hcore() @ basis_p, dtype="float64")

            df_p = df.incore.cholesky_eri(mol_p, auxbasis=mf.auxbasis)
            df_p = lib.unpack_tril(df_p)
            chol_p = _ao2mo(df_p, basis_p)

            coords_m = copy.copy(coords)
            # print(coords_m)
            coords_m[i][j] -= dR
            atom_m = list(zip(atom_symbols, coords_m))
            # print(i,j,atom_m)
            mol_m = gto.M(atom=atom_m, basis=basis, verbose=3, unit=unit)
            mf_m = df.density_fit(scf.RHF(mol_m), auxbasis=mf.auxbasis)
            # mf_m.kernel()
            basis_m = np.array(
                fractional_matrix_power(mf_m.get_ovlp(), -0.5), dtype="float64"
            )
            # basis_m = get_transformation_matrix(mf_m.get_ovlp()) #np.array(fractional_matrix_power(mf_m.get_ovlp(), -0.5),dtype="float64")
            h1_m = np.array(basis_m.T @ mf_m.get_hcore() @ basis_m, dtype="float64")

            df_m = df.incore.cholesky_eri(mol_m, auxbasis=mf.auxbasis)
            df_m = lib.unpack_tril(df_m)
            chol_m = _ao2mo(df_m, basis_m)

            h1_der = (h1_p - h1_m) / (2.0 * dR)
            h2_der = (chol_p - chol_m) / (2.0 * dR)

            h1_der_array[i, j] = h1_der
            h2_der_array[i, j] = h2_der
    h0_der = pyscf.grad.rhf.grad_nuc(mol)

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
    q = c_ao  # np.linalg.qr(dm0)[0]

    if isinstance(mf, scf.uhf.UHF):
        np.savez("uhf.npz", mo_coeff=q)
    elif isinstance(mf, scf.rhf.RHF):
        np.savez(
            "rhf.npz", mo_coeff=q
        )  # np.savez("rhf.npz", wave_data=q,mo_coeff=[basis0,basis0])

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
    FD_integrals(mf, dR=dR)
    write_integrals_lowdins(mf)


def round_to_1_sig_fig(num):
    if num == 0:
        return 0, 0
    else:
        decimal_places = -int(math.floor(math.log10(abs(num))))
        rounded_num = round(num, decimal_places)
        if decimal_places <= 0:
            decimal_points = 0
        else:
            str_rounded_num = f"{rounded_num:.{decimal_places}f}"
            decimal_points = len(str_rounded_num.split(".")[1])

        return rounded_num, decimal_points


def remove_nan_elements(array1, array2):
    # Check for NaNs in array1
    nan_mask_array1 = np.isnan(array1).any(axis=(1, 2))

    # Create a combined mask for elements to keep (non-NaN elements)
    combined_mask = ~nan_mask_array1

    # Filter out the elements in both arrays
    filtered_array1 = array1[combined_mask]
    filtered_array2 = array2[combined_mask]

    return filtered_array1, filtered_array2


def blocking_nuc_grad(
    npz_filename="grad_afqmc.npz",
    neql=0,
    printQ=False,
    writeBlockedQ=False,
    printG=True,
):
    data = np.load(npz_filename)
    gradients = data["grad"]  # [:,0,2]
    weights = data["weight"]
    # print(gradients)
    gradients, weights = remove_nan_elements(gradients, weights)
    natm = gradients.shape[1]
    error = np.zeros((natm, 3))
    avg = np.zeros((natm, 3))
    for i in range(natm):
        for j in range(3):
            data, mask = reject_outliers(gradients[:, i, j], m=6)
            # import pdb;pdb.set_trace()
            if np.sum(mask == False):
                print(f"({i},{j}) Outliers removed: {np.sum(mask == False)}")
                # print(gradients[:,i,j][~mask])
            grad_avg, grad_err = stat_utils.blocking_analysis(
                weights[mask],
                data,
                neql=neql,
                printQ=printQ,
                writeBlockedQ=writeBlockedQ,
            )
            if grad_avg is None:
                print(f"({i},{j}) Couldnt find average estimate")
                grad_avg = 0.0
            if grad_err is None:
                if grad_avg == 0.0:
                    grad_err = 0.0
                else:
                    print(f"({i},{j}) Couldnt find error estimate")
                    grad_err = 0.0
            avg[i, j] = grad_avg
            error[i, j] = grad_err
            if error[i, j] is not None:
                error[i, j], sig_dec = round_to_1_sig_fig(error[i, j])
                avg[i, j] = round(avg[i, j], sig_dec)
            else:
                avg[i, j] = f"{avg[i,j]:.8f}"
                error[i, j] = ""
    if printG:
        print("Blocking analysis done")
        print("--------------- AFQMC Gradient ---------------\n")
        print(avg)
        print("--------------- Error ---------------\n")
        print(error)
        print("----------------------------------------------")
    return avg, error


def print_xyz(mol):
    num_atoms = mol.natm  # number of atoms
    atom_coords = mol.atom_coords()  # coordinates of atoms
    atom_symbols = [mol.atom_pure_symbol(i) for i in range(num_atoms)]  # atomic symbols
    # print(num_atoms)
    for symbol, coord in zip(atom_symbols, atom_coords):
        print(f"{symbol} {coord[0]:.6f} {coord[1]:.6f} {coord[2]:.6f}")


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

    # Step 1: Get the list of files
    files = glob.glob("en_der_afqmc_*.npz")
    pattern = re.compile(r"en_der_afqmc_(\d+)\.npz")

    indices = []
    for file in files:
        match = pattern.search(file)
        if match:
            indices.append(int(match.group(1)))
    print(max(indices) + 1)
    return max(indices) + 1


def get_rdmsDer_rhf(norb, nchol, filename="en_der_afqmc.npz", nproc=4):
    rdm1 = []
    rdm2 = []
    weight = []
    for i in range(nproc):
        weighti = np.load(f"en_der_afqmc_{i}.npz")["weight"]
        # import pdb;pdb.set_trace()
        rdm1i = np.load(f"en_der_afqmc_{i}.npz")["rdm1"].reshape(
            weighti.shape[0], norb, norb
        )
        rdm2i = np.load(f"en_der_afqmc_{i}.npz")["rdm2"].reshape(
            weighti.shape[0], nchol, norb, norb
        )
        for j in range(rdm1i.shape[0]):
            rdm1.append(rdm1i[j])
            rdm2.append(rdm2i[j])
            weight.append(weighti[j])
    # import pdb;pdb.set_trace()
    return np.array(rdm1), np.array(rdm2), np.array(weight)


def get_rdmsDer_uhf(norb, nchol, filename="en_der_afqmc.npz", nproc=4):
    rdm1 = []
    rdm2 = []
    weight = []
    for i in range(nproc):
        weighti = np.load(f"en_der_afqmc_{i}.npz")["weight"]
        # import pdb;pdb.set_trace()
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
    # import pdb;pdb.set_trace()
    return np.array(rdm1), np.array(rdm2), np.array(weight)


def calculate_nuc_gradients(
    integral_der="Integral_der.npz",
    energy_der="en_der_afqmc.npz",
    printG=True,
    printQ=False,
    uhf=False,
):
    h1_der = np.load(integral_der)["array1"]  # (natm,3,norb,norb)
    h2_der = np.load(integral_der)["array2"]  # (natm,3,nchol,norb,norb)
    h0_der = np.load(integral_der)["array3"]  # (natm,3)

    if uhf:
        rdm1, rdm2, weights = get_rdmsDer_uhf(
            h1_der.shape[2], h2_der.shape[2], nproc=find_nproc()
        )
        obs1 = np.einsum("rxpq,npq->nrx", h1_der, rdm1[:, 0, :, :]) + np.einsum(
            "rxpq,npq->nrx", h1_der, rdm1[:, 1, :, :]
        )
        obs2 = np.einsum(
            "rxgpq,ngpq->nrx", h2_der, rdm2
        )  # [0]) + np.einsum("rxgpq,ngpq->nrx",h2_der,rdm2[1])
        # import pdb;pdb.set_trace()
        obs0 = h0_der
        gradients = obs1 + obs2 + obs0
    else:
        rdm1, rdm2, weights = get_rdmsDer_rhf(
            h1_der.shape[2], h2_der.shape[2], nproc=find_nproc()
        )
        obs1 = np.einsum("rxpq,npq->nrx", h1_der, rdm1)
        obs2 = np.einsum("rxgpq,ngpq->nrx", h2_der, rdm2)
        obs0 = h0_der
        gradients = obs1 + obs2 + obs0

    natm = gradients.shape[1]
    error = np.zeros((natm, 3))
    avg = np.zeros((natm, 3))
    for i in range(natm):
        for j in range(3):
            data, mask = reject_outliers(gradients[:, i, j], m=10)
            if np.sum(mask == False):
                print(f"({i},{j}) Outliers removed: {np.sum(mask == False)}")
            grad_avg, grad_err = stat_utils.blocking_analysis(
                weights[mask], data, neql=0, printQ=printQ, writeBlockedQ=False
            )
            if grad_avg is None:
                print(f"({i},{j}) Couldnt find average estimate")
                grad_avg = 0.0
            if grad_err is None:
                if grad_avg == 0.0:
                    grad_err = 0.0
                else:
                    print(f"({i},{j}) Couldnt find error estimate")
                    grad_err = 0.0
            avg[i, j] = grad_avg
            error[i, j] = grad_err
            # if error[i,j] is not None:
            #     error[i,j],sig_dec = round_to_1_sig_fig(error[i,j])
            #     avg[i,j] = round(avg[i,j],sig_dec)
            # else:
            #     avg[i,j] = f'{avg[i,j]:.8f}'
            #     error[i,j] = ""
    # import pdb;pdb.set_trace()
    if printG:
        print("Blocking analysis done")
        print("--------------- AFQMC Gradient ---------------\n")
        print(avg)
        print("--------------- Error ---------------\n")
        print(error)
        print("----------------------------------------------")
    return avg, error


def append_to_array(filename, array_data1, array_data2, array_data3):
    import os

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
