from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import tree_util

from ..core.ops import EstimatorOps, MeasOps, k_energy, k_force_bias
from ..core.system import System
from ..ham.chol import HamChol
from ..trial.ptccsd import PtccsdTrial, greens_restricted, overlap_pt_r
from ..trial.ptccsd import hf_overlap_r
from .pt2ccsd import combine_first_order_energy

o_pt_components = "pt_components"


@tree_util.register_pytree_node_class
@dataclass(frozen=True)
class PtccsdMeasCtx:
    rot_chol: jax.Array

    def tree_flatten(self):
        return (self.rot_chol,), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        (rot_chol,) = children
        return cls(rot_chol=rot_chol)


def build_ptccsd_meas_ctx(ham_data: HamChol, trial_data: PtccsdTrial) -> PtccsdMeasCtx:
    if ham_data.basis != "restricted":
        raise ValueError("PT-CCSD standalone kernels assume HamChol.basis == 'restricted'.")
    return PtccsdMeasCtx(rot_chol=ham_data.chol[:, : trial_data.nocc, :])


def _green_blocks(
    walker: jax.Array, trial_data: PtccsdTrial
) -> tuple[jax.Array, jax.Array, jax.Array]:
    green = greens_restricted(walker, trial_data)
    green_occ = green[:, trial_data.nocc :]
    greenp = jnp.vstack(
        [
            green_occ,
            -jnp.eye(trial_data.nvir, dtype=green.dtype),
        ]
    )
    return green, green_occ, greenp


def _chol_contract(chol: jax.Array, mat: jax.Array) -> jax.Array:
    return jnp.einsum("gij,ij->g", chol, mat, optimize="optimal")


def _t2_green_force_bias_terms(
    t2: jax.Array, green: jax.Array, green_occ: jax.Array, greenp: jax.Array
) -> tuple[jax.Array, jax.Array]:
    t2g_c = jnp.einsum("ptqu,pt->qu", t2, green_occ, optimize="optimal")
    t2g_e = jnp.einsum("ptqu,pu->qt", t2, green_occ, optimize="optimal")

    t2_green_c = (greenp @ t2g_c.T) @ green
    t2_green_e = (greenp @ t2g_e.T) @ green
    t2_green_corr = -4.0 * t2_green_c + 2.0 * t2_green_e

    t2g = 4.0 * t2g_c - 2.0 * t2g_e
    theta2 = 0.5 * jnp.einsum("ia,ia->", t2g, green_occ, optimize="optimal")
    return t2_green_corr, theta2


def _t2_green_energy_terms(
    t2: jax.Array, green: jax.Array, green_occ: jax.Array, greenp: jax.Array
) -> tuple[jax.Array, jax.Array]:
    t2g_c = jnp.einsum("ptqu,pt->qu", t2, green_occ, optimize="optimal")
    t2g_e = jnp.einsum("ptqu,pu->qt", t2, green_occ, optimize="optimal")

    t2_green_c = (greenp @ t2g_c.T) @ green
    t2_green_e = (greenp @ t2g_e.T) @ green
    t2_green = 2.0 * t2_green_c - t2_green_e

    t2g = 2.0 * t2g_c - t2g_e
    theta2 = jnp.einsum("ia,ia->", t2g, green_occ, optimize="optimal")
    return t2_green, theta2


def _energy_gl_batched(green: jax.Array, chol: jax.Array) -> jax.Array:
    return jnp.einsum("pr,gqr->gpq", green, chol, optimize="optimal")


def force_bias_pt_rw_rh(
    walker: jax.Array,
    ham_data: HamChol,
    meas_ctx: PtccsdMeasCtx,
    trial_data: PtccsdTrial,
) -> jax.Array:
    """
    Restricted first-order PT-CCSD force-bias ratio.

    Evaluates ``F0 + F_T - F0 * theta`` with a fixed HF reference.
    """
    green, green_occ, greenp = _green_blocks(walker, trial_data)
    lg = jnp.einsum("gpj,pj->g", meas_ctx.rot_chol, green, optimize="optimal")
    f0 = 2.0 * lg

    t1 = trial_data.t1
    theta1 = 2.0 * jnp.einsum("pt,pt->", t1, green_occ, optimize="optimal")
    t1gp = jnp.einsum("pt,it->pi", t1, greenp, optimize="optimal")
    gt1gp = jnp.einsum("pj,pi->ij", green, t1gp, optimize="optimal")

    t2_green_corr, theta2 = _t2_green_force_bias_terms(trial_data.t2, green, green_occ, greenp)

    theta = theta1 + theta2
    connected = _chol_contract(ham_data.chol, t2_green_corr - 2.0 * gt1gp)
    f_t = f0 * theta + connected
    return f0 + f_t - f0 * theta


def energy_pt_rw_rh(
    walker: jax.Array,
    ham_data: HamChol,
    meas_ctx: PtccsdMeasCtx,
    trial_data: PtccsdTrial,
) -> jax.Array:
    """
    Restricted first-order PT-CCSD local energy.

    Evaluates ``h0 + E0 + H_T - E0 * theta`` with a fixed HF reference.  The
    scalar ``ham_data.h0`` is kept outside the electronic numerator, matching
    the existing restricted CISD/RHF energy conventions.
    """
    t1, t2 = trial_data.t1, trial_data.t2
    nocc = trial_data.nocc

    green, green_occ, greenp = _green_blocks(walker, trial_data)
    h1 = ham_data.h1
    chol = ham_data.chol
    rot_chol = meas_ctx.rot_chol

    hg = jnp.einsum("pj,pj->", h1[:nocc, :], green, optimize="optimal")
    e1_0 = 2.0 * hg

    lg = jnp.einsum("gpj,pj->g", rot_chol, green, optimize="optimal")
    lg1 = jnp.einsum("gpj,qj->gpq", rot_chol, green, optimize="optimal")
    e2_0_1 = 2.0 * (lg @ lg)
    e2_0_2 = -jnp.sum(lg1 * jnp.swapaxes(lg1, -1, -2))
    e2_0 = e2_0_1 + e2_0_2
    electronic_0 = e1_0 + e2_0

    t1g = jnp.einsum("pt,pt->", t1, green_occ, optimize="optimal")
    theta1 = 2.0 * t1g
    gpt1 = greenp @ t1.T
    t1_green = gpt1 @ green

    e1_1_1 = 4.0 * t1g * hg
    e1_1_2 = -2.0 * jnp.einsum("ij,ij->", h1, t1_green, optimize="optimal")
    e1_1 = e1_1_1 + e1_1_2

    t2_green, theta2 = _t2_green_energy_terms(t2, green, green_occ, greenp)
    e1_2_1 = 2.0 * hg * theta2
    e1_2_2 = -2.0 * jnp.einsum("ij,ij->", h1, t2_green, optimize="optimal")
    e1_2 = e1_2_1 + e1_2_2

    e2_1_1 = e2_0 * theta1
    lt1g = _chol_contract(chol, t1_green)
    e2_1_2 = -2.0 * (lt1g @ lg)
    t1g1 = t1 @ green[:, nocc:].T
    l_t1 = jnp.einsum("git,pt->gip", chol[:, :, nocc:], t1, optimize="optimal")
    l_t1g = jnp.einsum("gia,qi->gaq", l_t1, green, optimize="optimal")
    e2_1_3_1 = jnp.einsum("gpq,gqa,ap->", lg1, lg1, t1g1, optimize="optimal")
    e2_1_3_2 = -jnp.einsum("gaq,gqa->", l_t1g, lg1, optimize="optimal")
    e2_1 = e2_1_1 + 2.0 * (e2_1_2 + e2_1_3_1 + e2_1_3_2)

    e2_2_1 = e2_0 * theta2
    lt2g = _chol_contract(chol, t2_green)
    e2_2_2_1 = -(lt2g @ lg)
    lt2_green = jnp.einsum("gpi,ji->gpj", rot_chol, t2_green, optimize="optimal")
    gl = _energy_gl_batched(green, chol)
    e2_2_2_2 = 0.5 * jnp.einsum("gpi,gpi->", gl, lt2_green, optimize="optimal")
    glgp = jnp.einsum("gpi,it->gpt", gl, greenp, optimize="optimal")
    l2t2_1 = jnp.einsum("gpt,gqu,ptqu->g", glgp, glgp, t2, optimize="optimal")
    l2t2_2 = jnp.einsum("gpu,gqt,ptqu->g", glgp, glgp, t2, optimize="optimal")
    e2_2_3 = jnp.sum(2.0 * l2t2_1 - l2t2_2)
    e2_2 = e2_2_1 + 4.0 * (e2_2_2_1 + e2_2_2_2) + e2_2_3

    theta = theta1 + theta2
    h_t = e1_1 + e1_2 + e2_1 + e2_2
    return ham_data.h0 + electronic_0 + h_t - electronic_0 * theta


def energy_components_pt_rw_rh(
    walker: jax.Array,
    ham_data: HamChol,
    meas_ctx: PtccsdMeasCtx,
    trial_data: PtccsdTrial,
) -> jax.Array:
    """
    Inverse-guide weighted components for an ensemble-level first-order PT-CCSD energy.

    The AFQMC state represented with the PT guide contains a denominator
    ``<psi0|phi> exp(theta)``.  Projection with ``psi0`` therefore requires the
    inverse guide factor ``r = exp(-theta)``.  Returns
    ``[r, r * theta, r * electronic_0, r * h_t]``; post-processing first divides
    the last three accumulated columns by the accumulated first column.
    """
    t1, t2 = trial_data.t1, trial_data.t2
    nocc = trial_data.nocc

    green, green_occ, greenp = _green_blocks(walker, trial_data)
    h1 = ham_data.h1
    chol = ham_data.chol
    rot_chol = meas_ctx.rot_chol

    hg = jnp.einsum("pj,pj->", h1[:nocc, :], green, optimize="optimal")
    e1_0 = 2.0 * hg

    lg = jnp.einsum("gpj,pj->g", rot_chol, green, optimize="optimal")
    lg1 = jnp.einsum("gpj,qj->gpq", rot_chol, green, optimize="optimal")
    e2_0_1 = 2.0 * (lg @ lg)
    e2_0_2 = -jnp.sum(lg1 * jnp.swapaxes(lg1, -1, -2))
    e2_0 = e2_0_1 + e2_0_2
    electronic_0 = e1_0 + e2_0

    t1g = jnp.einsum("pt,pt->", t1, green_occ, optimize="optimal")
    theta1 = 2.0 * t1g
    gpt1 = greenp @ t1.T
    t1_green = gpt1 @ green

    e1_1_1 = 4.0 * t1g * hg
    e1_1_2 = -2.0 * jnp.einsum("ij,ij->", h1, t1_green, optimize="optimal")
    e1_1 = e1_1_1 + e1_1_2

    t2_green, theta2 = _t2_green_energy_terms(t2, green, green_occ, greenp)
    e1_2_1 = 2.0 * hg * theta2
    e1_2_2 = -2.0 * jnp.einsum("ij,ij->", h1, t2_green, optimize="optimal")
    e1_2 = e1_2_1 + e1_2_2

    e2_1_1 = e2_0 * theta1
    lt1g = _chol_contract(chol, t1_green)
    e2_1_2 = -2.0 * (lt1g @ lg)
    t1g1 = t1 @ green[:, nocc:].T
    l_t1 = jnp.einsum("git,pt->gip", chol[:, :, nocc:], t1, optimize="optimal")
    l_t1g = jnp.einsum("gia,qi->gaq", l_t1, green, optimize="optimal")
    e2_1_3_1 = jnp.einsum("gpq,gqa,ap->", lg1, lg1, t1g1, optimize="optimal")
    e2_1_3_2 = -jnp.einsum("gaq,gqa->", l_t1g, lg1, optimize="optimal")
    e2_1 = e2_1_1 + 2.0 * (e2_1_2 + e2_1_3_1 + e2_1_3_2)

    e2_2_1 = e2_0 * theta2
    lt2g = _chol_contract(chol, t2_green)
    e2_2_2_1 = -(lt2g @ lg)
    lt2_green = jnp.einsum("gpi,ji->gpj", rot_chol, t2_green, optimize="optimal")
    gl = _energy_gl_batched(green, chol)
    e2_2_2_2 = 0.5 * jnp.einsum("gpi,gpi->", gl, lt2_green, optimize="optimal")
    glgp = jnp.einsum("gpi,it->gpt", gl, greenp, optimize="optimal")
    l2t2_1 = jnp.einsum("gpt,gqu,ptqu->g", glgp, glgp, t2, optimize="optimal")
    l2t2_2 = jnp.einsum("gpu,gqt,ptqu->g", glgp, glgp, t2, optimize="optimal")
    e2_2_3 = jnp.sum(2.0 * l2t2_1 - l2t2_2)
    e2_2 = e2_2_1 + 4.0 * (e2_2_2_1 + e2_2_2_2) + e2_2_3

    theta = theta1 + theta2
    h_t = e1_1 + e1_2 + e2_1 + e2_2
    guide_reweight = jnp.exp(-theta)
    return jnp.stack(
        [
            guide_reweight,
            guide_reweight * theta,
            guide_reweight * electronic_0,
            guide_reweight * h_t,
        ]
    )


def components_pt_rw_rh(
    walker: jax.Array,
    ham_data: HamChol,
    meas_ctx: PtccsdMeasCtx,
    trial_data: PtccsdTrial,
) -> jax.Array:
    """Return guide-independent ``[theta, electronic_0, h_t]`` components."""

    inverse_weighted = energy_components_pt_rw_rh(
        walker,
        ham_data,
        meas_ctx,
        trial_data,
    )
    return inverse_weighted[1:] / inverse_weighted[0]


def make_ptccsd_meas_ops(sys: System) -> MeasOps:
    if sys.nup != sys.ndn:
        raise ValueError("PT-CCSD measurements require nup == ndn.")
    if sys.walker_kind.lower() != "restricted":
        raise ValueError(
            "PT-CCSD measurements currently support only restricted walkers, "
            f"got: {sys.walker_kind}"
        )
    return MeasOps(
        overlap=overlap_pt_r,
        build_meas_ctx=build_ptccsd_meas_ctx,
        kernels={k_force_bias: force_bias_pt_rw_rh, k_energy: energy_pt_rw_rh},
        observables={o_pt_components: energy_components_pt_rw_rh},
    )


def make_ptccsd_estimator_ops(sys: System) -> EstimatorOps:
    """Build dense first-order PT-CCSD estimator operations for any guide."""

    if sys.nup != sys.ndn or sys.walker_kind.lower() != "restricted":
        raise ValueError("PT-CCSD estimators require a closed-shell restricted walker.")
    return EstimatorOps(
        reference_overlap=hf_overlap_r,
        components=components_pt_rw_rh,
        combine_energy=combine_first_order_energy,
        component_names=("theta", "electronic_0", "h_t"),
        build_estimator_ctx=build_ptccsd_meas_ctx,
    )
