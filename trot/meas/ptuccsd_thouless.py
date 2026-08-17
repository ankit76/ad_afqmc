from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import tree_util

from ..core.ops import EstimatorOps, MeasOps, k_energy, k_force_bias
from ..core.system import System
from ..ham.chol import HamChol
from ..trial.ptuccsd_thouless import (
    PtuccsdThoulessTrial,
    greenp_from_green,
    greens_unrestricted,
    overlap_r,
    overlap_u,
    reference_overlap_r,
)
from .pt2ccsd import combine_first_order_energy

o_pt_components = "pt_components"


@dataclass(frozen=True)
class PtuccsdThoulessMeasCfg:
    memory_mode: str = "low"
    mixed_real_dtype: jnp.dtype = jnp.float64
    mixed_complex_dtype: jnp.dtype = jnp.complex128
    mixed_real_dtype_testing: jnp.dtype = jnp.float32
    mixed_complex_dtype_testing: jnp.dtype = jnp.complex64


@tree_util.register_pytree_node_class
@dataclass(frozen=True)
class PtuccsdThoulessMeasCtx:
    h1_b: jax.Array
    chol_b: jax.Array
    cfg: PtuccsdThoulessMeasCfg

    def tree_flatten(self):
        return (self.h1_b, self.chol_b), (self.cfg,)

    @classmethod
    def tree_unflatten(cls, aux, children):
        (cfg,) = aux
        h1_b, chol_b = children
        return cls(h1_b=h1_b, chol_b=chol_b, cfg=cfg)


def build_ptuccsd_thouless_meas_ctx(
    ham_data: HamChol,
    trial_data: PtuccsdThoulessTrial,
    cfg: PtuccsdThoulessMeasCfg = PtuccsdThoulessMeasCfg(),
) -> PtuccsdThoulessMeasCtx:
    if ham_data.basis != "restricted":
        raise ValueError(
            "PT2-UCCSD Thouless measurements require a restricted-basis Hamiltonian."
        )
    cb = trial_data.mo_coeff_b
    cbh = cb.conj().T
    h1_sym = 0.5 * (ham_data.h1 + ham_data.h1.T.conj())
    h1_b = cbh @ h1_sym @ cb
    chol_b = jnp.einsum("pi,gij,jq->gpq", cbh, ham_data.chol, cb, optimize="optimal")
    return PtuccsdThoulessMeasCtx(h1_b=h1_b, chol_b=chol_b, cfg=cfg)


def _green_blocks(
    walker: tuple[jax.Array, jax.Array],
    trial_data: PtuccsdThoulessTrial,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    green_a, green_b = greens_unrestricted(walker, trial_data)
    noa, nob = trial_data.nocc
    green_occ_a = green_a[:noa, noa:]
    green_occ_b = green_b[:nob, nob:]
    greenp_a = greenp_from_green(green_a, noa)
    greenp_b = greenp_from_green(green_b, nob)
    return green_a, green_b, green_occ_a, green_occ_b, greenp_a, greenp_b


def _chol_contract(chol: jax.Array, mat: jax.Array) -> jax.Array:
    return jnp.einsum("gij,ij->g", chol, mat, optimize="optimal")


def _theta2_force_terms(
    trial_data: PtuccsdThoulessTrial,
    green_occ_a: jax.Array,
    green_occ_b: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    t2aa, t2ab, t2bb = trial_data.t2aa, trial_data.t2ab, trial_data.t2bb
    t2g_a = jnp.einsum("ptqu,pt->qu", t2aa, green_occ_a, optimize="optimal")
    t2g_b = jnp.einsum("ptqu,pt->qu", t2bb, green_occ_b, optimize="optimal")
    theta2a = 0.5 * jnp.einsum("qu,qu->", t2g_a, green_occ_a, optimize="optimal")
    theta2b = 0.5 * jnp.einsum("qu,qu->", t2g_b, green_occ_b, optimize="optimal")
    t2g_ab_a = jnp.einsum(
        "ptqu,qu->pt", t2ab, green_occ_b, optimize="optimal"
    )
    theta2ab = jnp.einsum("pt,pt->", t2g_ab_a, green_occ_a, optimize="optimal")
    return theta2a + theta2b + theta2ab, t2g_a, t2g_b, t2g_ab_a


def force_bias_kernel_uw_rh(
    walker: tuple[jax.Array, jax.Array],
    ham_data: HamChol,
    meas_ctx: PtuccsdThoulessMeasCtx,
    trial_data: PtuccsdThoulessTrial,
) -> jax.Array:
    green_a, green_b, green_occ_a, green_occ_b, greenp_a, greenp_b = _green_blocks(
        walker, trial_data
    )
    noa, nob = trial_data.nocc
    chol_a = ham_data.chol
    chol_b = meas_ctx.chol_b

    lg_a = _chol_contract(chol_a, green_a)
    lg_b = _chol_contract(chol_b, green_b)
    f0 = lg_a + lg_b

    theta2, t2g_a, t2g_b, t2g_ab_a = _theta2_force_terms(
        trial_data, green_occ_a, green_occ_b
    )
    t2g_ab_b = jnp.einsum(
        "ptqu,pt->qu", trial_data.t2ab, green_occ_a, optimize="optimal"
    )
    t2_green_a = (greenp_a @ (t2g_a + t2g_ab_a).T) @ green_a[:noa, :]
    t2_green_b = (greenp_b @ (t2g_b + t2g_ab_b).T) @ green_b[:nob, :]

    fb_2_1 = theta2 * f0
    fb_2_2 = -_chol_contract(chol_a, t2_green_a) - _chol_contract(
        chol_b, t2_green_b
    )
    return f0 + fb_2_1 + fb_2_2 - f0 * theta2


def force_bias_kernel_rw_rh(
    walker: jax.Array,
    ham_data: HamChol,
    meas_ctx: PtuccsdThoulessMeasCtx,
    trial_data: PtuccsdThoulessTrial,
) -> jax.Array:
    noa, nob = trial_data.nocc
    return force_bias_kernel_uw_rh(
        (walker[:, :noa], walker[:, :nob]), ham_data, meas_ctx, trial_data
    )


def _energy_components_uw_rh(
    walker: tuple[jax.Array, jax.Array],
    ham_data: HamChol,
    meas_ctx: PtuccsdThoulessMeasCtx,
    trial_data: PtuccsdThoulessTrial,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    green_a, green_b, green_occ_a, green_occ_b, greenp_a, greenp_b = _green_blocks(
        walker, trial_data
    )
    t2aa, t2ab, t2bb = trial_data.t2aa, trial_data.t2ab, trial_data.t2bb
    noa, nob = trial_data.nocc
    cfg = meas_ctx.cfg

    h1_a = 0.5 * (ham_data.h1 + ham_data.h1.T.conj())
    h1_b = meas_ctx.h1_b
    chol_a = ham_data.chol
    chol_b = meas_ctx.chol_b

    e1_0 = jnp.einsum("ij,ij->", h1_a, green_a, optimize="optimal")
    e1_0 += jnp.einsum("ij,ij->", h1_b, green_b, optimize="optimal")

    t2g_a = 0.25 * jnp.einsum(
        "ptqu,pt->qu",
        t2aa.astype(cfg.mixed_real_dtype),
        green_occ_a.astype(cfg.mixed_complex_dtype),
        optimize="optimal",
    )
    t2g_b = 0.25 * jnp.einsum(
        "ptqu,pt->qu",
        t2bb.astype(cfg.mixed_real_dtype),
        green_occ_b.astype(cfg.mixed_complex_dtype),
        optimize="optimal",
    )
    t2g_ab_a = jnp.einsum(
        "ptqu,qu->pt",
        t2ab.astype(cfg.mixed_real_dtype),
        green_occ_b.astype(cfg.mixed_complex_dtype),
        optimize="optimal",
    )
    t2g_ab_b = jnp.einsum(
        "ptqu,pt->qu",
        t2ab.astype(cfg.mixed_real_dtype),
        green_occ_a.astype(cfg.mixed_complex_dtype),
        optimize="optimal",
    )
    theta2a = jnp.einsum("qu,qu->", t2g_a, green_occ_a, optimize="optimal")
    theta2b = jnp.einsum("qu,qu->", t2g_b, green_occ_b, optimize="optimal")
    theta2ab = jnp.einsum("pt,pt->", t2g_ab_a, green_occ_a, optimize="optimal")
    theta2 = 2.0 * (theta2a + theta2b) + theta2ab

    t2_green_a = (greenp_a @ t2g_a.T) @ green_a[:noa, :]
    t2_green_ab_a = (greenp_a @ t2g_ab_a.T) @ green_a[:noa, :]
    t2_green_b = (greenp_b @ t2g_b.T) @ green_b[:nob, :]
    t2_green_ab_b = (greenp_b @ t2g_ab_b.T) @ green_b[:nob, :]
    combo_a = 4.0 * t2_green_a + t2_green_ab_a
    combo_b = 4.0 * t2_green_b + t2_green_ab_b
    e1_2_1 = e1_0 * theta2
    e1_2_2 = -jnp.einsum("ij,ij->", h1_a, combo_a, optimize="optimal")
    e1_2_2 -= jnp.einsum("ij,ij->", h1_b, combo_b, optimize="optimal")
    e1_2 = e1_2_1 + e1_2_2

    lg_a = _chol_contract(chol_a, green_a)
    lg_b = _chol_contract(chol_b, green_b)
    lg = lg_a + lg_b
    e2_0_1 = 0.5 * (lg @ lg)
    gl1_a = jnp.einsum("pr,gqr->gpq", green_a, chol_a, optimize="optimal")
    gl1_b = jnp.einsum("pr,gqr->gpq", green_b, chol_b, optimize="optimal")
    e2_0_2 = -0.5 * (
        jnp.sum(gl1_a * jnp.swapaxes(gl1_a, -1, -2))
        + jnp.sum(gl1_b * jnp.swapaxes(gl1_b, -1, -2))
    )
    e2_0 = e2_0_1 + e2_0_2
    electronic_0 = e1_0 + e2_0

    e2_2_1 = e2_0 * theta2
    lt2g_a = _chol_contract(chol_a, 8.0 * t2_green_a + 2.0 * t2_green_ab_a)
    lt2g_b = _chol_contract(chol_b, 8.0 * t2_green_b + 2.0 * t2_green_ab_b)
    e2_2_2_1 = -0.5 * ((lt2g_a + lt2g_b) @ lg)
    combo2_a = 8.0 * t2_green_a + 2.0 * t2_green_ab_a
    combo2_b = 8.0 * t2_green_b + 2.0 * t2_green_ab_b

    def scan_over_chol(carry, x):
        e222_acc, e23_acc = carry
        chol_a_i, chol_b_i = x
        gl_a_i = jnp.einsum("pr,rq->pq", green_a, chol_a_i, optimize="optimal")
        gl_b_i = jnp.einsum("pr,rq->pq", green_b, chol_b_i, optimize="optimal")
        lt2_green_a_i = jnp.einsum(
            "pi,ji->pj", chol_a_i, combo2_a, optimize="optimal"
        )
        lt2_green_b_i = jnp.einsum(
            "pi,ji->pj", chol_b_i, combo2_b, optimize="optimal"
        )
        e222_acc += 0.5 * (
            jnp.einsum("pi,pi->", gl_a_i, lt2_green_a_i, optimize="optimal")
            + jnp.einsum("pi,pi->", gl_b_i, lt2_green_b_i, optimize="optimal")
        )
        glgp_a_i = jnp.einsum(
            "pi,it->pt", gl_a_i[:noa, :], greenp_a, optimize="optimal"
        ).astype(cfg.mixed_complex_dtype_testing)
        glgp_b_i = jnp.einsum(
            "pi,it->pt", gl_b_i[:nob, :], greenp_b, optimize="optimal"
        ).astype(cfg.mixed_complex_dtype_testing)
        l2t2_a = 0.5 * jnp.einsum(
            "pt,qu,ptqu->",
            glgp_a_i,
            glgp_a_i,
            t2aa.astype(cfg.mixed_real_dtype_testing),
            optimize="optimal",
        )
        l2t2_b = 0.5 * jnp.einsum(
            "pt,qu,ptqu->",
            glgp_b_i,
            glgp_b_i,
            t2bb.astype(cfg.mixed_real_dtype_testing),
            optimize="optimal",
        )
        l2t2_ab = jnp.einsum(
            "pt,qu,ptqu->",
            glgp_a_i,
            glgp_b_i,
            t2ab.astype(cfg.mixed_real_dtype_testing),
            optimize="optimal",
        )
        e23_acc += l2t2_a + l2t2_b + l2t2_ab
        return (e222_acc, e23_acc), None

    # The exchange-like term follows the dtype of the uncast Green's-function
    # contractions, while the quartic T2 term deliberately uses the lower
    # ``testing`` precision.  Keeping separate scan seeds is required when the
    # Hamiltonian and walkers are double precision but mixed precision is on.
    zero_e222 = jnp.zeros_like(e2_2_2_1)
    zero_e23 = jnp.array(0.0, dtype=cfg.mixed_complex_dtype_testing)
    (e2_2_2_2, e2_2_3), _ = jax.lax.scan(
        scan_over_chol, (zero_e222, zero_e23), (chol_a, chol_b)
    )
    e2_2 = e2_2_1 + e2_2_2_1 + e2_2_2_2 + e2_2_3

    h_t = e1_2 + e2_2
    return theta2, electronic_0, h_t


def components_ptuccsd_thouless_uw_rh(
    walker: tuple[jax.Array, jax.Array],
    ham_data: HamChol,
    meas_ctx: PtuccsdThoulessMeasCtx,
    trial_data: PtuccsdThoulessTrial,
) -> jax.Array:
    """Return guide-independent ``[theta, electronic_0, h_t]`` components."""
    return jnp.stack(_energy_components_uw_rh(walker, ham_data, meas_ctx, trial_data))


def components_ptuccsd_thouless_rw_rh(
    walker: jax.Array,
    ham_data: HamChol,
    meas_ctx: PtuccsdThoulessMeasCtx,
    trial_data: PtuccsdThoulessTrial,
) -> jax.Array:
    noa, nob = trial_data.nocc
    return components_ptuccsd_thouless_uw_rh(
        (walker[:, :noa], walker[:, :nob]), ham_data, meas_ctx, trial_data
    )


def energy_kernel_uw_rh(
    walker: tuple[jax.Array, jax.Array],
    ham_data: HamChol,
    meas_ctx: PtuccsdThoulessMeasCtx,
    trial_data: PtuccsdThoulessTrial,
) -> jax.Array:
    components = components_ptuccsd_thouless_uw_rh(
        walker, ham_data, meas_ctx, trial_data
    )
    return combine_first_order_energy(ham_data.h0, components)


def energy_kernel_rw_rh(
    walker: jax.Array,
    ham_data: HamChol,
    meas_ctx: PtuccsdThoulessMeasCtx,
    trial_data: PtuccsdThoulessTrial,
) -> jax.Array:
    components = components_ptuccsd_thouless_rw_rh(
        walker, ham_data, meas_ctx, trial_data
    )
    return combine_first_order_energy(ham_data.h0, components)


def make_ptuccsd_thouless_meas_ops(
    sys: System,
    memory_mode: str = "low",
    mixed_precision: bool = False,
    testing: bool = False,
) -> MeasOps:
    cfg = PtuccsdThoulessMeasCfg(
        memory_mode=memory_mode,
        mixed_real_dtype=jnp.float32 if mixed_precision else jnp.float64,
        mixed_complex_dtype=jnp.complex64 if mixed_precision else jnp.complex128,
        mixed_real_dtype_testing=jnp.float64 if testing else jnp.float32,
        mixed_complex_dtype_testing=jnp.complex128 if testing else jnp.complex64,
    )

    walker_kind = sys.walker_kind.lower()
    if walker_kind == "restricted":
        if sys.nup < sys.ndn:
            raise ValueError("Restricted PT2-UCCSD measurements require nup >= ndn.")
        overlap_fn = overlap_r
        kernels = {k_force_bias: force_bias_kernel_rw_rh, k_energy: energy_kernel_rw_rh}
        components_fn = components_ptuccsd_thouless_rw_rh
    elif walker_kind == "unrestricted":
        overlap_fn = overlap_u
        kernels = {k_force_bias: force_bias_kernel_uw_rh, k_energy: energy_kernel_uw_rh}
        components_fn = components_ptuccsd_thouless_uw_rh
    else:
        raise ValueError(
            "PT2-UCCSD Thouless measurements support restricted/unrestricted walkers, "
            f"got: {sys.walker_kind}"
        )

    return MeasOps(
        overlap=overlap_fn,
        build_meas_ctx=lambda ham_data, trial_data: build_ptuccsd_thouless_meas_ctx(
            ham_data, trial_data, cfg
        ),
        kernels=kernels,
        observables={o_pt_components: components_fn},
    )


def make_ptuccsd_thouless_estimator_ops(
    sys: System,
    memory_mode: str = "low",
    mixed_precision: bool = False,
    testing: bool = False,
) -> EstimatorOps:
    """Build a PT2-UCCSD estimator for a separately chosen propagation guide."""
    if sys.walker_kind.lower() != "restricted" or sys.nup < sys.ndn:
        raise ValueError("PT2-UCCSD estimators require a restricted walker with nup >= ndn.")

    cfg = PtuccsdThoulessMeasCfg(
        memory_mode=memory_mode,
        mixed_real_dtype=jnp.float32 if mixed_precision else jnp.float64,
        mixed_complex_dtype=jnp.complex64 if mixed_precision else jnp.complex128,
        mixed_real_dtype_testing=jnp.float64 if testing else jnp.float32,
        mixed_complex_dtype_testing=jnp.complex128 if testing else jnp.complex64,
    )
    return EstimatorOps(
        reference_overlap=reference_overlap_r,
        components=components_ptuccsd_thouless_rw_rh,
        combine_energy=combine_first_order_energy,
        component_names=("theta", "electronic_0", "h_t"),
        build_estimator_ctx=lambda ham_data, trial_data: build_ptuccsd_thouless_meas_ctx(
            ham_data, trial_data, cfg
        ),
    )


__all__ = [
    "PtuccsdThoulessMeasCfg",
    "PtuccsdThoulessMeasCtx",
    "build_ptuccsd_thouless_meas_ctx",
    "components_ptuccsd_thouless_rw_rh",
    "components_ptuccsd_thouless_uw_rh",
    "energy_kernel_rw_rh",
    "energy_kernel_uw_rh",
    "force_bias_kernel_rw_rh",
    "force_bias_kernel_uw_rh",
    "make_ptuccsd_thouless_estimator_ops",
    "make_ptuccsd_thouless_meas_ops",
    "o_pt_components",
]
