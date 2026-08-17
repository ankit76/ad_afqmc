from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import lax, tree_util

from ..core.ops import MeasOps, k_energy, k_force_bias
from ..core.system import System
from ..ham.chol import HamChol
from ..trial.ptccsd_thouless import (
    PtccsdThoulessTrial,
    greenp_from_green,
    greens_restricted,
    overlap_ptccsd_thouless_r,
)


@tree_util.register_pytree_node_class
@dataclass(frozen=True)
class PtccsdThoulessMeasCtx:
    def tree_flatten(self):
        return (), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls()


def build_ptccsd_thouless_meas_ctx(
    ham_data: HamChol, trial_data: PtccsdThoulessTrial
) -> PtccsdThoulessMeasCtx:
    if ham_data.basis != "restricted":
        raise ValueError(
            "PT-CCSD Thouless standalone kernels assume HamChol.basis == 'restricted'."
        )
    return PtccsdThoulessMeasCtx()


def _green_blocks(
    walker: jax.Array, trial_data: PtccsdThoulessTrial
) -> tuple[jax.Array, jax.Array, jax.Array]:
    green = greens_restricted(walker, trial_data)
    green_occ = green[: trial_data.nocc, trial_data.nocc :]
    greenp = greenp_from_green(green, trial_data)
    return green, green_occ, greenp


def _chol_contract(chol: jax.Array, mat: jax.Array) -> jax.Array:
    return jnp.einsum("gij,ij->g", chol, mat, optimize="optimal")


def _t2_green_force_bias_terms(
    t2: jax.Array,
    green: jax.Array,
    green_occ: jax.Array,
    greenp: jax.Array,
    nocc: int,
) -> tuple[jax.Array, jax.Array]:
    t2g_c = jnp.einsum("ptqu,pt->qu", t2, green_occ, optimize="optimal")
    t2g_e = jnp.einsum("ptqu,pu->qt", t2, green_occ, optimize="optimal")

    t2_green_c = (greenp @ t2g_c.T) @ green[:nocc, :]
    t2_green_e = (greenp @ t2g_e.T) @ green[:nocc, :]
    t2_green_corr = -4.0 * t2_green_c + 2.0 * t2_green_e

    t2g = 4.0 * t2g_c - 2.0 * t2g_e
    theta2 = 0.5 * jnp.einsum("ia,ia->", t2g, green_occ, optimize="optimal")
    return t2_green_corr, theta2


def _t2_green_energy_terms(
    t2: jax.Array,
    green: jax.Array,
    green_occ: jax.Array,
    greenp: jax.Array,
    nocc: int,
) -> tuple[jax.Array, jax.Array]:
    t2g_c = jnp.einsum("ptqu,pt->qu", t2, green_occ, optimize="optimal")
    t2g_e = jnp.einsum("ptqu,pu->qt", t2, green_occ, optimize="optimal")

    t2_green_c = (greenp @ t2g_c.T) @ green[:nocc, :]
    t2_green_e = (greenp @ t2g_e.T) @ green[:nocc, :]
    t2_green = 2.0 * t2_green_c - t2_green_e

    t2g = 2.0 * t2g_c - t2g_e
    theta2 = jnp.einsum("ia,ia->", t2g, green_occ, optimize="optimal")
    return t2_green, theta2


def force_bias_pt_rw_rh(
    walker: jax.Array,
    ham_data: HamChol,
    meas_ctx: PtccsdThoulessMeasCtx,
    trial_data: PtccsdThoulessTrial,
) -> jax.Array:
    """
    Restricted first-order PT-CCSD force-bias ratio with exact T1.

    Evaluates ``F0 + F_T2 - F0 * theta2`` against the Thouless-rotated
    reference.  The input ``t2`` is the raw CCSD doubles tensor; no disconnected
    ``T1^2`` contribution is added.
    """
    del meas_ctx
    green, green_occ, greenp = _green_blocks(walker, trial_data)
    f0 = 2.0 * jnp.einsum("gpq,pq->g", ham_data.chol, green, optimize="optimal")

    t2_green_corr, theta2 = _t2_green_force_bias_terms(
        trial_data.t2, green, green_occ, greenp, trial_data.nocc
    )
    connected = _chol_contract(ham_data.chol, t2_green_corr)
    f_t = f0 * theta2 + connected
    return f0 + f_t - f0 * theta2


def energy_pt_rw_rh(
    walker: jax.Array,
    ham_data: HamChol,
    meas_ctx: PtccsdThoulessMeasCtx,
    trial_data: PtccsdThoulessTrial,
) -> jax.Array:
    """
    Restricted first-order PT-CCSD local energy with exact T1.

    Evaluates ``h0 + E0 + H_T2 - E0 * theta2``.  The scalar ``ham_data.h0`` is
    kept outside the electronic numerator, matching the existing restricted
    measurement convention.
    """
    del meas_ctx
    t2 = trial_data.t2
    nocc = trial_data.nocc

    green, green_occ, greenp = _green_blocks(walker, trial_data)
    h1 = ham_data.h1
    chol = ham_data.chol

    hg = jnp.einsum("pq,pq->", h1, green, optimize="optimal")
    e1_0 = 2.0 * hg

    t2_green, theta2 = _t2_green_energy_terms(t2, green, green_occ, greenp, nocc)
    e1_2_1 = 2.0 * hg * theta2
    e1_2_2 = -2.0 * jnp.einsum("pq,pq->", h1, t2_green, optimize="optimal")
    e1_2 = e1_2_1 + e1_2_2

    lg = jnp.einsum("gpq,pq->g", chol, green, optimize="optimal")
    lt2g = jnp.einsum("gpq,pq->g", chol, t2_green, optimize="optimal")
    e2_2_2_1 = -(lt2g @ lg)

    dtype_acc = jnp.result_type(walker, h1, chol, t2)
    zero = jnp.array(0.0, dtype=dtype_acc)

    def scan_over_chol(
        carry: tuple[jax.Array, jax.Array, jax.Array], chol_i: jax.Array
    ) -> tuple[tuple[jax.Array, jax.Array, jax.Array], None]:
        e20_acc, e222_acc, e223_acc = carry

        gl_i = jnp.einsum("pr,qr->pq", green, chol_i, optimize="optimal")
        e20_acc = e20_acc + 0.5 * (2.0 * jnp.trace(gl_i)) ** 2
        e20_acc = e20_acc - jnp.einsum("pq,qp->", gl_i, gl_i, optimize="optimal")

        lt2_green_i = jnp.einsum("pr,qr->pq", chol_i, t2_green, optimize="optimal")
        e222_acc = e222_acc + 0.5 * jnp.einsum("pq,pq->", gl_i, lt2_green_i, optimize="optimal")

        glgp_i = jnp.einsum("iq,qa->ia", gl_i[:nocc, :], greenp, optimize="optimal")
        l2t2_1 = jnp.einsum("ia,jb,iajb->", glgp_i, glgp_i, t2, optimize="optimal")
        l2t2_2 = jnp.einsum("ib,ja,iajb->", glgp_i, glgp_i, t2, optimize="optimal")
        e223_acc = e223_acc + 2.0 * l2t2_1 - l2t2_2
        return (e20_acc, e222_acc, e223_acc), None

    (e2_0, e2_2_2_2, e2_2_3), _ = lax.scan(scan_over_chol, (zero, zero, zero), chol)
    e2_2_1 = e2_0 * theta2
    e2_2 = e2_2_1 + 4.0 * (e2_2_2_1 + e2_2_2_2) + e2_2_3

    electronic_0 = e1_0 + e2_0
    h_t = e1_2 + e2_2
    return ham_data.h0 + electronic_0 + h_t - electronic_0 * theta2


def make_ptccsd_thouless_meas_ops(sys: System) -> MeasOps:
    if sys.nup != sys.ndn:
        raise ValueError("PT-CCSD Thouless measurements require nup == ndn.")
    if sys.walker_kind.lower() != "restricted":
        raise ValueError(
            "PT-CCSD Thouless measurements currently support only restricted walkers, "
            f"got: {sys.walker_kind}"
        )
    return MeasOps(
        overlap=overlap_ptccsd_thouless_r,
        build_meas_ctx=build_ptccsd_thouless_meas_ctx,
        kernels={k_force_bias: force_bias_pt_rw_rh, k_energy: energy_pt_rw_rh},
    )
