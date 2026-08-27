"""Two-fragment heavy-H LNO-AFQMC and canonical comparison."""

import math

from pyscf import gto, scf
from pyscf.lno import tools

from trot.lno_pt import format_lno_fragment_banner, setup_lno_pt
from trot.lno_uhf import (
    make_uhf_iao_fragments,
    prepare_uhf_canonical,
    prepare_uhf_lno_fragment,
)


LNO_THRESH = 1.0e-4


mol = gto.M(
    atom="""
O  0.000000  0.000000  0.000000
H  0.758602  0.000000  0.504284
H  0.260455  0.000000 -0.872893
O  3.000000  0.500000  0.000000
H  3.758602  0.500000  0.504284
H  3.260455  0.500000 -0.872893
    """,
    basis="cc-pvdz",
    spin=2,
    verbose=3,
)

mf = scf.UHF(mol).density_fit().run()

atom_fragments = tools.autofrag_atom(mol, H2heavy=True)
iaos = make_uhf_iao_fragments(mf, atom_fragments, frozen=2)
canonical = prepare_uhf_canonical(mf, frozen=2)
lno_correlation = 0.0
lno_variance = 0.0
for fragment_index in range(len(iaos.fragments)):
    fragment = prepare_uhf_lno_fragment(
        mf,
        iaos,
        fragment_index,
        frozen=2,
        lno_thresh=LNO_THRESH,
    )
    print(format_lno_fragment_banner(fragment_index, fragment.ham, canonical.ham))
    fragment_job = setup_lno_pt(fragment)
    fragment_result = fragment_job.kernel()
    correlation, error = fragment_job.fragment_correlation(fragment_result)
    lno_correlation += correlation
    lno_variance += error**2
    print(
        f"Fragment {fragment_index + 1} correlation = {correlation:.10f} "
        f"+/- {error:.3g} Ha"
    )

lno_energy = mf.e_tot + lno_correlation
lno_error = math.sqrt(lno_variance)
print(f"LNO-AFQMC = {lno_energy:.10f} +/- {lno_error:.3g} Ha")

canonical_job = setup_lno_pt(canonical)
canonical_result = canonical_job.kernel()
canonical_correlation, canonical_error = canonical_job.fragment_correlation(
    canonical_result
)
canonical_energy = mf.e_tot + canonical_correlation
print(
    f"Canonical UHF-guide PT2-UCCSD AFQMC = {canonical_energy:.10f} "
    f"+/- {canonical_error:.3g} Ha"
)
print(f"LNO - canonical = {lno_energy - canonical_energy:.10f} Ha")
