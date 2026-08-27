"""Fe(H2O)6 heavy-atom LNO-AFQMC and canonical comparison."""

import math
from pathlib import Path

from pyscf import gto, mp, scf
from pyscf.data.elements import chemcore
from pyscf.lno import tools

from trot.lno_pt import format_lno_fragment_banner, setup_lno_pt
from trot.lno_uhf import (
    make_uhf_iao_fragments,
    prepare_uhf_canonical_hamiltonian,
    prepare_uhf_lno_fragment,
)


LNO_THRESH = 1.0e-5
T2_DISCARDED_NORM = 0.01


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


def make_mol(*, verbose: int = 4) -> gto.Mole:
    return gto.M(
        atom=ATOM,
        basis={"default": "ccpvdz-dk", "Fe": "ccpwcvtz-dk"},
        unit="angstrom",
        symmetry=False,
        charge=2,
        spin=4,
        max_memory=20_000,
        verbose=verbose,
    )


mol = make_mol()
frozen = chemcore(mol)
print(f"Frozen core orbitals: {frozen}")

scf_checkpoint = Path(__file__).with_suffix(".chk")
restarting = scf_checkpoint.exists()
mf = scf.UHF(mol).x2c().density_fit()
mf.chkfile = str(scf_checkpoint)
mf.init_guess = "chkfile" if restarting else "minao"
mf.run()

if restarting:
    print(f"Restarted SCF from {scf_checkpoint}; skipping stability analysis.")
else:
    for attempt in range(10):
        print(f"mf stability test {attempt + 1}")
        mo_coeff, _, stable, _ = mf.stability(return_status=True)
        if stable:
            break
        dm = mf.make_rdm1(mo_coeff, mf.mo_occ)
        mf = mf.newton()
        mf.chkfile = str(scf_checkpoint)
        mf.run(dm0=dm)

canonical_hamiltonian = prepare_uhf_canonical_hamiltonian(mf, frozen=frozen)

atom_fragments = tools.autofrag_atom(mol, H2heavy=True)
iaos = make_uhf_iao_fragments(mf, atom_fragments, frozen=frozen)
lno_correlation = 0.0
lno_variance = 0.0
lno_mp2_correlation = 0.0
lno_ccsd_correlation = 0.0
for fragment_index in range(len(iaos.fragments)):
    fragment = prepare_uhf_lno_fragment(
        mf,
        iaos,
        fragment_index,
        frozen=frozen,
        lno_thresh=LNO_THRESH,
    )
    print(
        format_lno_fragment_banner(
            fragment_index,
            fragment.ham,
            canonical_hamiltonian,
        )
    )
    fragment_job = setup_lno_pt(
        fragment,
        t2_discarded_norm=T2_DISCARDED_NORM,
    )
    fragment_result = fragment_job.kernel()
    correlation, error = fragment_job.fragment_correlation(fragment_result)
    lno_correlation += correlation
    lno_variance += error**2
    lno_mp2_correlation += fragment.meta["mp2_correlation_energy"]
    lno_ccsd_correlation += fragment.meta["uccsd_correlation_energy"]
    print(
        f"Fragment {fragment_index + 1} correlation = {correlation:.10f} "
        f"+/- {error:.3g} Ha"
    )

lno_energy = mf.e_tot + lno_correlation
lno_error = math.sqrt(lno_variance)
canonical_mp2_correlation = mp.UMP2(mf, frozen=frozen).run().e_corr
delta_mp2 = canonical_mp2_correlation - lno_mp2_correlation
lno_ccsd_energy = mf.e_tot + lno_ccsd_correlation

print("\nLNO energy summary")
print(f"LNO-AFQMC              = {lno_energy:.10f} +/- {lno_error:.3g} Ha")
print(
    f"LNO-AFQMC + delta-MP2  = {lno_energy + delta_mp2:.10f} "
    f"+/- {lno_error:.3g} Ha"
)
print(f"LNO-CCSD               = {lno_ccsd_energy:.10f} Ha")
print(f"LNO-CCSD + delta-MP2   = {lno_ccsd_energy + delta_mp2:.10f} Ha")
