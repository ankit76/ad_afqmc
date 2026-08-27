"""Standalone PySCF-Forge ULNO-CCSD check for Fe(H2O)6(2+)."""

from pyscf import gto, lno, mp, scf
from pyscf.data.elements import chemcore
from pyscf.lno import tools

from trot.lno_uhf import make_uhf_iao_fragments


LNO_THRESH = 1.0e-4
CANONICAL_UCCSD = -1729.416592804222


ATOM = """
Fe -0.64147387529051 0.51990405379180 0.11450483185168
O 0.33915564253394 2.18819453520299 -0.84159903476570
H 0.65823736209818 2.99799985286513 -0.40575899522591
O -1.29914529040912 -0.20924466129350 -1.80322550106302
H -2.11419692217693 0.03267722517314 -2.27701147100162
O 0.01237770373627 1.24396717111997 2.03533113813669
H 0.83097217272839 1.00756306178418 2.50577204385612
O -1.62284672685044 -1.15048542622041 1.07007991103536
H -1.94592790838151 -1.95766677806096 0.63237073377438
O -2.41866907277644 1.66373032346654 0.25658821238371
H -2.55381601393967 2.56024091790599 -0.09930153080408
O 1.13404022498401 -0.62623224778001 -0.02436092147139
H 1.26370551457750 -1.52400168391753 0.33034524187694
H -3.28110369794222 1.35873477131317 0.59094893393907
H -0.81032015579443 -0.82248643308581 -2.37966496653734
H 1.99795010138314 -0.32692539271158 -0.36001715946551
H 0.57998322602479 2.26642229799559 -1.78148703231078
H -0.47668197075682 1.85673194452426 2.61208351578631
H -1.85653031374812 -1.23356353207299 2.011352050005
"""


def main():
    mol = gto.M(
        atom=ATOM,
        basis={"default": "ccpvdz-dk", "Fe": "ccpwcvtz-dk"},
        unit="angstrom",
        symmetry=False,
        charge=2,
        spin=4,
        max_memory=20_000,
        verbose=4,
    )
    frozen = chemcore(mol)

    mf = scf.UHF(mol).x2c().density_fit().run()
    stable = False
    for i in range(10):
        if not stable:
            mo_i, _, stable,_ = mf.stability(return_status=True)
            dm = mf.make_rdm1(mo_i,mf.mo_occ)
            mf = mf.newton()
            mf.kernel(dm0=dm)
        elif stable:
            print(f'mf energy: {mf.e_tot}, stability {stable}')
            break

    atom_fragments = tools.autofrag_atom(mol, H2heavy=True)
    iaos = make_uhf_iao_fragments(mf, atom_fragments, frozen=frozen)
    spin_fragments = [[a.tolist(), b.tolist()] for a, b in iaos.fragments]

    ulno = lno.ULNOCCSD(
        mf,
        list(iaos.coeff),
        spin_fragments,
        lno_type=["1h", "1h"],
        lno_thresh=[10.0 * LNO_THRESH, LNO_THRESH],
        frozen=frozen,
    )
    ulno.verbose_imp = 4
    ulno.kernel()

    # This cheap full-space UMP2 energy estimates the correlation omitted by
    # the finite LNO spaces; Forge already accumulates the fragment MP2 terms.
    ump2 = mp.UMP2(mf, frozen=frozen).run()
    corrected = ulno.e_tot_ccsd_pt2corrected(ump2.e_corr)

    print("\nULNO-CCSD comparison")
    print(f"UHF                         = {mf.e_tot:.12f} Ha")
    print(f"ULNO-CCSD (raw)             = {ulno.e_tot:.12f} Ha")
    print(f"ULNO-CCSD + full-MP2 tail   = {corrected:.12f} Ha")
    print(f"Canonical UCCSD             = {CANONICAL_UCCSD:.12f} Ha")
    print(f"Raw ULNO error              = {(ulno.e_tot - CANONICAL_UCCSD) * 1000:+.3f} mHa")
    print(f"MP2-corrected ULNO error    = {(corrected - CANONICAL_UCCSD) * 1000:+.3f} mHa")


if __name__ == "__main__":
    main()
