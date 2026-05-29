def prep_act_cc(mol, *, n_core, n_act=None):
    from pyscf import gto, scf, mcscf, ao2mo, cc
    import numpy as np

    if n_act is None:
        n_act = mol.nao - n_core

    # frozen = list(range(n_core)) + list(range(n_core + n_act, mol.nao))

    # RHF
    mf = scf.RHF(mol)
    mf.kernel()

    for i in range(4):
        mo1 = mf.stability(external=True)[0]
        mf = mf.newton().run(mo1, mf.mo_occ)  # type: ignore
    mf.stability()
    print(f"RHF Total energy: {mf.e_tot:.8f}\n")

    # UHF
    mol.symmetry = False
    umf = scf.UHF(mol)
    umf.max_cycle = 50
    umf.kernel()

    for i in range(4):
        mo1 = umf.stability(external=True)[0]
        umf = umf.newton().run(mo1, umf.mo_occ)  # type: ignore
    umf.stability()
    print(f"UHF Total energy: {umf.e_tot:.8f}")

    # Custom UHF in active space
    ncas = n_act
    nelecas = mol.nelectron - 2 * n_core
    mc = mcscf.CASSCF(mf, ncas, nelecas)

    h1e_cas, e_core = mc.get_h1eff()  # type: ignore
    h2e_cas = mc.get_h2eff(mc.mo_coeff)  # type: ignore
    h2e_cas_full = ao2mo.restore(1, h2e_cas, ncas)

    mol_as = gto.M()
    mol_as.verbose = mol.verbose
    mol_as.symmetry = False
    mol_as.nelectron = nelecas
    mol_as.incore_anyway = True

    nalpha = (nelecas + mol.spin) // 2
    nbeta = nelecas - nalpha
    mol_as.nelec = (nalpha, nbeta)

    act_umf = scf.UHF(mol_as)
    act_umf._custom_h0 = e_core  # type: ignore
    act_umf._custom_h1 = np.array(h1e_cas)  # type: ignore
    act_umf._custom_ovlp = np.array(np.eye(ncas))  # type: ignore
    act_umf._custom_eri = ao2mo.restore(4, np.array(h2e_cas), ncas)  # type: ignore

    def get_hcore(self, *args):
        return self._custom_h1

    def get_ovlp(self, *args):
        return self._custom_ovlp

    act_umf.get_hcore = get_hcore.__get__(act_umf)
    act_umf.get_ovlp = get_ovlp.__get__(act_umf)
    act_umf._eri = ao2mo.restore(8, h2e_cas_full, ncas)

    ## Building the guess
    Ccas = mc.mo_coeff  # type: ignore
    S = mf.get_ovlp()
    dm_ao = umf.make_rdm1()
    dm_mo_a = Ccas.T @ S @ dm_ao[0] @ S @ Ccas
    dm_mo_b = Ccas.T @ S @ dm_ao[1] @ S @ Ccas

    cas_idx = mc.ncore  # type: ignore
    cas_slice = slice(cas_idx, cas_idx + ncas)
    dm_a = dm_mo_a[cas_slice, cas_slice]
    dm_b = dm_mo_b[cas_slice, cas_slice]

    ## Symmetrize
    dm_a = 0.5 * (dm_a + dm_a.T)
    dm_b = 0.5 * (dm_b + dm_b.T)

    ## Re-idempotize
    def make_idempotent(dm, nelec):
        w, v = np.linalg.eigh(dm)
        idx = np.argsort(-w)
        occ = np.zeros_like(w)
        occ[idx[:nelec]] = 1.0
        return v @ np.diag(occ) @ v.T

    dm_a = make_idempotent(dm_a, nalpha)
    dm_b = make_idempotent(dm_b, nbeta)

    act_umf.kernel(dm0=(dm_a, dm_b))
    print(f"Act UHF Total energy: {e_core + act_umf.e_tot:.8f}")

    for i in range(4):
        mo1 = act_umf.stability(external=True)[0]
        act_umf = act_umf.newton().run(mo1, act_umf.mo_occ)  # type: ignore
    act_umf.stability()
    print(f"Act UHF Total energy: {e_core + act_umf.e_tot:.8f}")

    act_cc = cc.UCCSD(act_umf)
    act_cc.max_cycle = 200
    act_cc.kernel()
    print(f"Act UCCSD correlation energy: {act_cc.e_corr:.8f}")
    print(f"Act UCCSD Total energy: {e_core + act_cc.e_tot:.8f}\n")

    return act_cc

    # If needed this can be expressed in the original AO basis

    # mo_act = mc.mo_coeff[:, mc.ncore:mc.ncore + ncas]
    # uhf_mo_a, uhf_mo_b = act_umf.mo_coeff

    # mo_uhf_a_ao = mo_act @ uhf_mo_a
    # mo_uhf_b_ao = mo_act @ uhf_mo_b

    # mo_core = mc.mo_coeff[:, :mc.ncore]
    # mo_virt = mc.mo_coeff[:, mc.ncore + ncas:]

    # full_mo_a = np.hstack([mo_core, mo_uhf_a_ao, mo_virt])
    # full_mo_b = np.hstack([mo_core, mo_uhf_b_ao, mo_virt])

    # umf.mo_coeff = np.array([full_mo_a, full_mo_b])
    # umf.e_tot = umf.energy_tot()
    # print(f"\nUHF Total energy: {umf.e_tot:.8f}")

    # S = umf.get_ovlp()
    # Ca, Cb = umf.mo_coeff
    # print("Ortho a:", np.max(np.abs(Ca.T @ S @ Ca)))
    # print("Ortho b:", np.max(np.abs(Cb.T @ S @ Cb)))

    # mycc = cc.UCCSD(umf)
    # mycc.max_cycle=200
    # mycc.frozen = frozen
    # mycc.kernel()
    # print(f"UCCSD correlation energy: {mycc.e_corr:.8f}")
    # print(f"UCCSD Total energy: {mycc.e_tot:.8f}")

    # with open("cc_ao.pkl", "wb") as f:
    #    pickle.dump(mycc, f)


def stage_act_cc(act_cc, fname, chol_cut=1e-5):
    from trot.afqmc import AfqmcFp
    from trot.staging import HamInput
    import trot.staging

    import numpy as np
    from pyscf import ao2mo

    mol = act_cc.mol
    umf = act_cc._scf

    norb = umf.mo_coeff.shape[-1]
    mol.nao = norb

    h0 = umf._custom_h0
    h1 = umf._custom_h1
    eri = umf._custom_eri

    eri = ao2mo.restore(4, eri, norb)
    chol = trot.staging.modified_cholesky(eri, max_error=chol_cut)
    chol = chol.reshape((-1, norb, norb))

    nelec = mol.nelec

    Ca = umf.mo_coeff[0]
    h1 = Ca.T @ h1 @ Ca
    chol = np.einsum("gpq, ip, qj", chol, Ca.T, Ca)

    ham = HamInput(
        h0=h0,
        h1=h1,
        chol=chol,
        nelec=nelec,
        norb=norb,
        chol_cut=chol_cut,
        frozen=0,
        source_kind="cc",
        basis="restricted",
    )

    staged = trot.staging.stage(act_cc, ham=ham)

    def stage():
        return staged

    af = AfqmcFp(act_cc)
    af.stage = stage  # type: ignore
    af.save_staged(fname)


if __name__ == "__main__":

    # from pathlib import Path
    # import sys
    # import trot
    # trot_path = Path(trot.__file__).resolve().parent
    # path = trot_path.parent / "examples"
    # sys.path.append(str(path))
    # from active_space_fp import prep_act_cc, stage_act_cc

    from pyscf import gto

    r = 2.5
    atomstring = f"""
        N 0 0 {-r/2}
        N 0 0 {r/2}
    """

    mol = gto.M(
        atom=atomstring,
        basis="6-31g",
        symmetry="D2h",
        verbose=4,
    )

    # We want the same core orbitals for alpha and beta MOs
    act_cc = prep_act_cc(mol, n_core=2)
    stage_act_cc(act_cc, "n2_afqmc.h5")

    from trot.afqmc import AfqmcFp

    af = AfqmcFp.from_staged("n2_afqmc.h5")
    h0 = af._staged.ham.h0
    af.ene0 = h0 + act_cc.e_tot
    af.n_walkers = 5
    af.dt = 0.025
    af.n_prop_steps = 40 * 5
    af.n_blocks = 1
    af.n_chunks = 1
    af.n_traj = 10
    af.walker_kind = "unrestricted"
    af.seed = 514372
    af.kernel()
