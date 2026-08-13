from pyscf import gto, scf, cc
import dataclasses

from trot.afqmc import AfqmcFp
from trot.core.ops import k_energy
from trot.ham.chol import slice_ham_level
from trot.spin_proj import make_overlap_u_s2, make_energy_kernel_uw_rh_s2, quadrature_s2
from trot.meas.ucisd import energy_kernel_uw_rh, energy_kernel_gw_rh, slice_meas_ctx_chol
from trot.trial.ucisd import overlap_g, slice_trial_level
from trot.core.levels import TmpLevelSpec

import jax
import jax.numpy as jnp

from trot.prop.ml import make_level_pack
from trot.prop.blocks import make_block_ml_fp

mol = gto.M(
    atom="""
    O        0.0000000000      0.0000000000      1.0000000000
    H        0.9562300000      0.0000000000      0.0000000000
    H       -0.2353791634      0.9268076728      0.0000000000
    """,
    basis="cc-pvdz",
    verbose=3,
)

mf = scf.UHF(mol)
mf.kernel()

mo1 = mf.stability()[0]
dm1 = mf.make_rdm1(mo1, mf.mo_occ)
mf = mf.run(dm1)
mf.stability()

mycc = cc.UCCSD(mf)
mycc.kernel()

af = AfqmcFp(mycc)
af.n_walkers = 5
af.dt = 0.025
af.n_prop_steps = 40 * 5
af.n_blocks = 1
af.n_chunks = 1
af.n_traj = 10
af.ene0 = mycc.e_tot
af.walker_kind = "unrestricted"
af.mixed_precision = True
af.seed = 1000000

af.build_job()
job = af._job
job._prepare_runtime()
trial_data = job.trial_data
ham_data = job.ham_data
meas_ctx = job._runtime_meas_ctx

# Spin projection
## Data for the quadrature
target_spin = 0.0
betas, w_betas = quadrature_s2(
    target_spin,
    (job.sys.nup, job.sys.ndn),
    4,
)

## Overlap and energy with spin projection
overlap_u_s2 = make_overlap_u_s2(betas, w_betas, overlap_g)
energy_kernel_uw_rh_s2 = make_energy_kernel_uw_rh_s2(betas, w_betas, overlap_g, energy_kernel_gw_rh)


# Avoid computing the energy at 0 a.u. since it does not use the ml scheme
def always_zero(*args, **kwargs) -> jax.Array:
    return jnp.array(0.0)


## Trucation 1: No trunation + spin projection
level1 = TmpLevelSpec(norb_keep=None, nchol_keep=None)
p1 = make_level_pack(
    ham_data=ham_data,
    meas_ctx=meas_ctx,
    trial_data=trial_data,
    level=level1,
    e_kernel=energy_kernel_uw_rh_s2,
    fn_slice_ham=slice_ham_level,
    fn_slice_ctx=slice_meas_ctx_chol,
    fn_slice_trial=slice_trial_level,
)

## Truncation 2: Truncation, no spin projection
level2 = TmpLevelSpec(norb_keep=15, nchol_keep=50)
p2 = make_level_pack(
    ham_data=ham_data,
    meas_ctx=meas_ctx,
    trial_data=trial_data,
    level=level2,
    e_kernel=energy_kernel_uw_rh,
    fn_slice_ham=slice_ham_level,
    fn_slice_ctx=slice_meas_ctx_chol,
    fn_slice_trial=slice_trial_level,
)

block_fn = make_block_ml_fp(
    p1,
    p2,
)

job.meas_ops = dataclasses.replace(
    job.meas_ops,
    overlap=overlap_u_s2,
    kernels={
        k_energy: always_zero,
    },
)

af.build_job(
    force=True,
    block_fn=block_fn,
    meas_ops=job.meas_ops,
)

e, err = af.kernel()
