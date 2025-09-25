import numpy as np
import scipy as sp
from pyscf import fci, gto, scf, mcscf, ao2mo
from pyscf.fci import cistring

def large_ci(ci, norb, nelec, tol=0.1, return_strs=True):
    '''
    Search for the largest CI coefficients
    '''
    neleca, nelecb = nelec
    na = cistring.num_strings(norb, neleca)
    nb = cistring.num_strings(norb, nelecb)
    assert ci.size == na * nb

    ci = ci.reshape(na, nb)
    addra, addrb = np.where(abs(ci) > tol)

    if addra.size == 0:
        # No large CI coefficient > tol, search for the largest coefficient
        addra, addrb = np.unravel_index(np.argmax(abs(ci)), ci.shape)
        addra = numpy.asarray([addra])
        addrb = numpy.asarray([addrb])

    strsa = cistring.addrs2str(norb, neleca, addra)
    strsb = cistring.addrs2str(norb, nelecb, addrb)

    if return_strs:
        strsa = [bin(x) for x in strsa]
        strsb = [bin(x) for x in strsb]
        return list(zip(ci[addra,addrb], strsa, strsb))

    else:
        occslsta = cistring._strs2occslst(strsa, norb)
        occslstb = cistring._strs2occslst(strsb, norb)
        return list(zip(ci[addra,addrb], occslsta, occslstb))

def get_fci_state(ci_coeffs, norb, nelec, ndets=None, tol=1e-4):
    if ndets is None: ndets = int(ci_coeffs.size)
    coeffs, occ_a, occ_b = zip(
        *large_ci(ci_coeffs, norb, nelec, tol=tol, return_strs=False)
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

def create_msd_from_casci(casci, fci_state, wave_data, copy=True, update=False, verbose=False):
    # Create MSD state.
    mo_coeff = casci.mo_coeff
    nbsf = mo_coeff.shape[0]
    ncore = casci.ncore
    ncas = casci.ncas
    nextern = nbsf - ncore - ncas
    nelecas = casci.nelecas
    ndet = len(fci_state)

    wave_data_arr = []
    coeffs = []
    if verbose: print(f'\n# Note that determinant coefficients are unnormalized!\n')

    for det in fci_state:
        nocc_a = np.array(ncore*(1,) + det[0] + nextern*(0,))
        nocc_b = np.array(ncore*(1,) + det[1] + nextern*(0,))
        if verbose: print(f'# {det} --> {nocc_a}, {nocc_b} : {fci_state[det]}')

        wave_data_i = wave_data.copy()
        wave_data_i['mo_coeff'] = [mo_coeff[:, nocc_a>0], mo_coeff[:, nocc_b>0]]
        del wave_data_i['rdm1']
        wave_data_arr.append(wave_data_i)
        coeffs.append(fci_state[det])

    if copy: wave_data = wave_data.copy()
    for i in range(ndet): wave_data[f'{i}'] = wave_data_arr[i]

    # Check if coeffs are normalized.
    coeffs = np.array(coeffs)
    norm = np.sqrt(np.sum(coeffs**2))
    try: np.testing.assert_allclose(norm, 1.)
    except: coeffs /= norm
    wave_data['coeffs'] = np.array(coeffs)

    if update:
        mo_coeff = wave_data['0']['mo_coeff'].copy()
        rdm1 = [mo_coeff[0] @ mo_coeff[0].T.conj(), mo_coeff[1] @ mo_coeff[1].T.conj()]
        wave_data['mo_coeff'] = mo_coeff
        wave_data['rdm1'] = rdm1

    return wave_data

