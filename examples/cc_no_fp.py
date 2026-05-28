import numpy as np


def get_cc_no(mycc, *, e_core=0.0):
    from pyscf import cc

    mol = mycc.mol
    umf = mycc._scf

    dm1a_mo, dm1b_mo = mycc.make_rdm1()

    occ_a, Uno_a = np.linalg.eigh(dm1a_mo)
    occ_b, Uno_b = np.linalg.eigh(dm1b_mo)

    idx = occ_a.argsort()[::-1]
    occ_a = occ_a[idx]
    Uno_a = Uno_a[:, idx]
    idx = occ_b.argsort()[::-1]
    occ_b = occ_b[idx]
    Uno_b = Uno_b[:, idx]

    mo_a, mo_b = umf.mo_coeff
    Cno_a = mo_a @ Uno_a
    Cno_b = mo_b @ Uno_b

    nmo_a = mo_a.shape[1]
    nmo_b = mo_b.shape[1]
    mo_occ_a = np.zeros(nmo_a)
    mo_occ_b = np.zeros(nmo_b)

    nocca, noccb = mol.nelec
    mo_occ_a[:nocca] = 1.0
    mo_occ_b[:noccb] = 1.0

    umf_no = umf.copy()
    umf_no.mo_coeff = np.array([Cno_a, Cno_b])
    umf_no.mo_occ = np.array([mo_occ_a, mo_occ_b])
    print(f"\nNO energy: {e_core + umf_no.energy_tot():.8f}")

    mycc_no = cc.UCCSD(umf_no)
    mycc_no.frozen = mycc.frozen
    mycc_no.kernel()

    print(f"NO UCCSD energy: {e_core + mycc_no.e_tot}:.8f")
    print(f"Difference: {mycc_no.e_tot - mycc.e_tot:.8f}")
    print()
    print("Alpha/Beta NO occupations")
    for i, (oa, ob) in enumerate(zip(occ_a, occ_b)):
        print(f"{i:4d} {oa:12.8f} {ob:12.8f}")
    print()

    return mycc_no


if __name__ == "__main__":
    from pyscf import gto, scf, cc
    from trot.afqmc import AfqmcFp

    r = 2.5
    atomstring = f"""
        N 0 0 {-r/2}
        N 0 0 {r/2}
    """

    mol = gto.M(
        atom=atomstring,
        basis="6-31g",
        verbose=4,
    )

    umf = scf.UHF(mol)
    umf.kernel()

    for i in range(4):
        mo1 = umf.stability(external=True)[0]
        umf = umf.newton().run(mo1, umf.mo_occ)  # type: ignore
    umf.stability()

    mycc = cc.UCCSD(umf)
    mycc.frozen = 2
    mycc.kernel()

    mycc_no = get_cc_no(mycc)

    af = AfqmcFp(mycc_no)
    af.ene0 = mycc_no.e_tot
    af.n_walkers = 5
    af.dt = 0.025
    af.n_prop_steps = 40 * 5
    af.n_blocks = 1
    af.n_chunks = 1
    af.n_traj = 10
    af.walker_kind = "unrestricted"
    af.kernel()
