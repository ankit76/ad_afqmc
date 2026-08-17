from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import tree_util

from ..core.ops import EstimatorOps, MeasOps, k_energy, k_force_bias
from ..core.system import System
from ..ham.chol import HamChol
from ..trial.cisd_modes import mode_apply, mode_quadratic
from ..trial.ptccsd_modes import (
    PtccsdModeTrial,
    PtccsdThoulessModeTrial,
    det_overlap_thouless_r,
    greenp_thouless,
    greens_pt_r,
    greens_thouless_r,
    hf_overlap_r,
    overlap_pt_r,
    overlap_ptccsd_thouless_r,
)
from .cisd_modes import mode_quadratic_matrices
from .ptccsd import o_pt_components
from .pt2ccsd import combine_first_order_energy


@tree_util.register_pytree_node_class
@dataclass(frozen=True)
class PtccsdModeMeasCtx:
    rot_chol: jax.Array
    l_t1: jax.Array
    n_mode_chunks: int

    def tree_flatten(self):
        return (self.rot_chol, self.l_t1), (self.n_mode_chunks,)

    @classmethod
    def tree_unflatten(cls, aux, children):
        (n_mode_chunks,) = aux
        rot_chol, l_t1 = children
        return cls(rot_chol=rot_chol, l_t1=l_t1, n_mode_chunks=n_mode_chunks)


@tree_util.register_pytree_node_class
@dataclass(frozen=True)
class PtccsdThoulessModeMeasCtx:
    n_mode_chunks: int

    def tree_flatten(self):
        return (), (self.n_mode_chunks,)

    @classmethod
    def tree_unflatten(cls, aux, children):
        del children
        (n_mode_chunks,) = aux
        return cls(n_mode_chunks=n_mode_chunks)


def build_ptccsd_mode_meas_ctx(
    ham_data: HamChol,
    trial_data: PtccsdModeTrial,
    *,
    n_mode_chunks: int = 1,
) -> PtccsdModeMeasCtx:
    if ham_data.basis != "restricted":
        raise ValueError("PT-CCSD mode kernels require a restricted Hamiltonian.")
    if n_mode_chunks <= 0:
        raise ValueError("n_mode_chunks must be positive.")
    rot_chol = ham_data.chol[:, : trial_data.nocc, :]
    l_t1 = jnp.einsum(
        "git,pt->gip",
        ham_data.chol[:, :, trial_data.nocc :],
        trial_data.t1,
        optimize="optimal",
    )
    return PtccsdModeMeasCtx(
        rot_chol=rot_chol,
        l_t1=l_t1,
        n_mode_chunks=min(int(n_mode_chunks), trial_data.mode_rank),
    )


def build_ptccsd_thouless_mode_meas_ctx(
    ham_data: HamChol,
    trial_data: PtccsdThoulessModeTrial,
    *,
    n_mode_chunks: int = 1,
) -> PtccsdThoulessModeMeasCtx:
    if ham_data.basis != "restricted":
        raise ValueError("PT-CCSD Thouless mode kernels require a restricted Hamiltonian.")
    if n_mode_chunks <= 0:
        raise ValueError("n_mode_chunks must be positive.")
    return PtccsdThoulessModeMeasCtx(
        n_mode_chunks=min(int(n_mode_chunks), trial_data.mode_rank)
    )


def _chol_contract(chol: jax.Array, matrix: jax.Array) -> jax.Array:
    return jnp.einsum("gij,ij->g", chol, matrix, optimize="optimal")


def _pt_green_blocks(
    walker: jax.Array,
    trial_data: PtccsdModeTrial,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    green = greens_pt_r(walker, trial_data)
    green_occ = green[:, trial_data.nocc :]
    greenp = jnp.vstack(
        [
            green_occ,
            -jnp.eye(trial_data.nvir, dtype=green.dtype),
        ]
    )
    return green, green_occ, greenp


def _thouless_green_blocks(
    walker: jax.Array,
    trial_data: PtccsdThoulessModeTrial,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    green = greens_thouless_r(walker, trial_data)
    green_occ = green[: trial_data.nocc, trial_data.nocc :]
    return green, green_occ, greenp_thouless(green, trial_data)


def _mode_t2_green(
    trial_data,
    green: jax.Array,
    green_occ: jax.Array,
    greenp: jax.Array,
    *,
    green_rows: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    projections, kg = mode_apply(trial_data, green_occ)
    theta2 = mode_quadratic(trial_data, green_occ, projections)
    t2_green = (greenp @ kg.T) @ green_rows
    return t2_green, theta2


def force_bias_pt_rw_rh(
    walker: jax.Array,
    ham_data: HamChol,
    meas_ctx: PtccsdModeMeasCtx,
    trial_data: PtccsdModeTrial,
) -> jax.Array:
    green, green_occ, greenp = _pt_green_blocks(walker, trial_data)
    f0 = 2.0 * jnp.einsum("gpj,pj->g", meas_ctx.rot_chol, green, optimize="optimal")

    t1gp = jnp.einsum("pt,it->pi", trial_data.t1, greenp, optimize="optimal")
    gt1gp = jnp.einsum("pj,pi->ij", green, t1gp, optimize="optimal")
    t2_green, _ = _mode_t2_green(
        trial_data,
        green,
        green_occ,
        greenp,
        green_rows=green,
    )
    connected = _chol_contract(ham_data.chol, -2.0 * t2_green - 2.0 * gt1gp)
    return f0 + connected


def components_pt_rw_rh(
    walker: jax.Array,
    ham_data: HamChol,
    meas_ctx: PtccsdModeMeasCtx,
    trial_data: PtccsdModeTrial,
) -> jax.Array:
    """Return unweighted ``[theta, electronic_0, h_t]`` PT sufficient statistics."""

    green, green_occ, greenp = _pt_green_blocks(walker, trial_data)
    h1 = ham_data.h1
    chol = ham_data.chol
    nocc = trial_data.nocc

    hg = jnp.einsum("pj,pj->", h1[:nocc, :], green, optimize="optimal")
    e1_0 = 2.0 * hg
    lg = jnp.einsum("gpj,pj->g", meas_ctx.rot_chol, green, optimize="optimal")
    lg1 = jnp.einsum("gpj,qj->gpq", meas_ctx.rot_chol, green, optimize="optimal")
    e2_0 = 2.0 * (lg @ lg) - jnp.sum(lg1 * jnp.swapaxes(lg1, -1, -2))
    electronic_0 = e1_0 + e2_0

    t1g = jnp.einsum("pt,pt->", trial_data.t1, green_occ, optimize="optimal")
    theta1 = 2.0 * t1g
    t1_green = (greenp @ trial_data.t1.T) @ green
    e1_1 = 4.0 * t1g * hg - 2.0 * jnp.einsum(
        "ij,ij->", h1, t1_green, optimize="optimal"
    )

    t2_green, theta2 = _mode_t2_green(
        trial_data,
        green,
        green_occ,
        greenp,
        green_rows=green,
    )
    e1_2 = 2.0 * hg * theta2 - 2.0 * jnp.einsum(
        "ij,ij->", h1, t2_green, optimize="optimal"
    )

    lt1g = _chol_contract(chol, t1_green)
    t1g1 = trial_data.t1 @ green[:, nocc:].T
    l_t1g = jnp.einsum("gia,qi->gaq", meas_ctx.l_t1, green, optimize="optimal")
    e2_1_3_1 = jnp.einsum("gpq,gqa,ap->", lg1, lg1, t1g1, optimize="optimal")
    e2_1_3_2 = -jnp.einsum("gaq,gqa->", l_t1g, lg1, optimize="optimal")
    e2_1 = e2_0 * theta1 + 2.0 * (-2.0 * (lt1g @ lg) + e2_1_3_1 + e2_1_3_2)

    lt2g = _chol_contract(chol, t2_green)
    lt2_green = jnp.einsum("gpi,ji->gpj", meas_ctx.rot_chol, t2_green, optimize="optimal")
    gl = jnp.einsum("pr,gqr->gpq", green, chol, optimize="optimal")
    e2_2_2_2 = 0.5 * jnp.einsum("gpi,gpi->", gl, lt2_green, optimize="optimal")
    glgp = jnp.einsum("gpi,it->gpt", gl, greenp, optimize="optimal")
    e2_2_3 = jnp.sum(
        mode_quadratic_matrices(
            trial_data,
            glgp,
            n_mode_chunks=meas_ctx.n_mode_chunks,
        )
    )
    e2_2 = e2_0 * theta2 + 4.0 * (-(lt2g @ lg) + e2_2_2_2) + e2_2_3

    theta = theta1 + theta2
    h_t = e1_1 + e1_2 + e2_1 + e2_2
    return jnp.stack([theta, electronic_0, h_t])


def energy_pt_rw_rh(
    walker: jax.Array,
    ham_data: HamChol,
    meas_ctx: PtccsdModeMeasCtx,
    trial_data: PtccsdModeTrial,
) -> jax.Array:
    theta, electronic_0, h_t = components_pt_rw_rh(walker, ham_data, meas_ctx, trial_data)
    return ham_data.h0 + electronic_0 + h_t - theta * electronic_0


def inverse_guide_components_pt_rw_rh(
    walker: jax.Array,
    ham_data: HamChol,
    meas_ctx: PtccsdModeMeasCtx,
    trial_data: PtccsdModeTrial,
) -> jax.Array:
    components = components_pt_rw_rh(walker, ham_data, meas_ctx, trial_data)
    reweight = jnp.exp(-components[0])
    return jnp.concatenate([reweight[None], reweight * components])


def force_bias_pt_thouless_rw_rh(
    walker: jax.Array,
    ham_data: HamChol,
    meas_ctx: PtccsdThoulessModeMeasCtx,
    trial_data: PtccsdThoulessModeTrial,
) -> jax.Array:
    del meas_ctx
    green, green_occ, greenp = _thouless_green_blocks(walker, trial_data)
    f0 = 2.0 * jnp.einsum("gpq,pq->g", ham_data.chol, green, optimize="optimal")
    t2_green, _ = _mode_t2_green(
        trial_data,
        green,
        green_occ,
        greenp,
        green_rows=green[: trial_data.nocc, :],
    )
    return f0 + _chol_contract(ham_data.chol, -2.0 * t2_green)


def components_pt_thouless_rw_rh(
    walker: jax.Array,
    ham_data: HamChol,
    meas_ctx: PtccsdThoulessModeMeasCtx,
    trial_data: PtccsdThoulessModeTrial,
) -> jax.Array:
    """Return ``[theta2, electronic_0, h_t]`` for PT2/Thouless estimators."""

    green, green_occ, greenp = _thouless_green_blocks(walker, trial_data)
    h1 = ham_data.h1
    chol = ham_data.chol
    nocc = trial_data.nocc

    hg = jnp.einsum("pq,pq->", h1, green, optimize="optimal")
    e1_0 = 2.0 * hg
    t2_green, theta2 = _mode_t2_green(
        trial_data,
        green,
        green_occ,
        greenp,
        green_rows=green[:nocc, :],
    )
    e1_2 = 2.0 * hg * theta2 - 2.0 * jnp.einsum(
        "pq,pq->", h1, t2_green, optimize="optimal"
    )

    lg = jnp.einsum("gpq,pq->g", chol, green, optimize="optimal")
    gl = jnp.einsum("pr,gqr->gpq", green, chol, optimize="optimal")
    e2_0 = 2.0 * (lg @ lg) - jnp.sum(gl * jnp.swapaxes(gl, -1, -2))

    lt2g = _chol_contract(chol, t2_green)
    lt2_green = jnp.einsum("gpr,qr->gpq", chol, t2_green, optimize="optimal")
    e2_2_2_2 = 0.5 * jnp.einsum("gpq,gpq->", gl, lt2_green, optimize="optimal")
    glgp = jnp.einsum("gpi,it->gpt", gl[:, :nocc, :], greenp, optimize="optimal")
    e2_2_3 = jnp.sum(
        mode_quadratic_matrices(
            trial_data,
            glgp,
            n_mode_chunks=meas_ctx.n_mode_chunks,
        )
    )
    e2_2 = e2_0 * theta2 + 4.0 * (-(lt2g @ lg) + e2_2_2_2) + e2_2_3

    electronic_0 = e1_0 + e2_0
    h_t = e1_2 + e2_2
    return jnp.stack([theta2, electronic_0, h_t])


def energy_pt_thouless_rw_rh(
    walker: jax.Array,
    ham_data: HamChol,
    meas_ctx: PtccsdThoulessModeMeasCtx,
    trial_data: PtccsdThoulessModeTrial,
) -> jax.Array:
    theta2, electronic_0, h_t = components_pt_thouless_rw_rh(
        walker, ham_data, meas_ctx, trial_data
    )
    return ham_data.h0 + electronic_0 + h_t - theta2 * electronic_0


def inverse_guide_components_pt_thouless_rw_rh(
    walker: jax.Array,
    ham_data: HamChol,
    meas_ctx: PtccsdThoulessModeMeasCtx,
    trial_data: PtccsdThoulessModeTrial,
) -> jax.Array:
    components = components_pt_thouless_rw_rh(walker, ham_data, meas_ctx, trial_data)
    reweight = jnp.exp(-components[0])
    return jnp.concatenate([reweight[None], reweight * components])


def make_ptccsd_mode_meas_ops(
    sys: System,
    *,
    n_mode_chunks: int = 1,
) -> MeasOps:
    if sys.nup != sys.ndn or sys.walker_kind.lower() != "restricted":
        raise ValueError("PT-CCSD mode measurements require a closed-shell restricted walker.")
    return MeasOps(
        overlap=overlap_pt_r,
        build_meas_ctx=lambda ham_data, trial_data: build_ptccsd_mode_meas_ctx(
            ham_data,
            trial_data,
            n_mode_chunks=n_mode_chunks,
        ),
        kernels={k_force_bias: force_bias_pt_rw_rh, k_energy: energy_pt_rw_rh},
        observables={o_pt_components: inverse_guide_components_pt_rw_rh},
    )


def make_ptccsd_mode_estimator_ops(
    sys: System,
    *,
    n_mode_chunks: int = 1,
) -> EstimatorOps:
    if sys.nup != sys.ndn or sys.walker_kind.lower() != "restricted":
        raise ValueError("PT-CCSD mode estimators require a closed-shell restricted walker.")
    return EstimatorOps(
        reference_overlap=hf_overlap_r,
        components=components_pt_rw_rh,
        combine_energy=combine_first_order_energy,
        component_names=("theta", "electronic_0", "h_t"),
        build_estimator_ctx=lambda ham_data, trial_data: build_ptccsd_mode_meas_ctx(
            ham_data,
            trial_data,
            n_mode_chunks=n_mode_chunks,
        ),
    )


def make_ptccsd_thouless_mode_meas_ops(
    sys: System,
    *,
    n_mode_chunks: int = 1,
    exponentiate_t2: bool = True,
) -> MeasOps:
    if sys.nup != sys.ndn or sys.walker_kind.lower() != "restricted":
        raise ValueError(
            "PT-CCSD Thouless mode measurements require a closed-shell restricted walker."
        )
    overlap = overlap_ptccsd_thouless_r if exponentiate_t2 else det_overlap_thouless_r
    observables = (
        {o_pt_components: inverse_guide_components_pt_thouless_rw_rh}
        if exponentiate_t2
        else {}
    )
    return MeasOps(
        overlap=overlap,
        build_meas_ctx=lambda ham_data, trial_data: build_ptccsd_thouless_mode_meas_ctx(
            ham_data,
            trial_data,
            n_mode_chunks=n_mode_chunks,
        ),
        kernels={
            k_force_bias: force_bias_pt_thouless_rw_rh,
            k_energy: energy_pt_thouless_rw_rh,
        },
        observables=observables,
    )


def make_ptccsd_thouless_mode_estimator_ops(
    sys: System,
    *,
    n_mode_chunks: int = 1,
) -> EstimatorOps:
    """Build the common mode-native PT2/Thouless projected estimator."""

    if sys.nup != sys.ndn or sys.walker_kind.lower() != "restricted":
        raise ValueError(
            "PT-CCSD Thouless mode estimators require a closed-shell restricted walker."
        )
    return EstimatorOps(
        reference_overlap=det_overlap_thouless_r,
        components=components_pt_thouless_rw_rh,
        combine_energy=combine_first_order_energy,
        component_names=("theta", "electronic_0", "h_t"),
        build_estimator_ctx=lambda ham_data, trial_data: build_ptccsd_thouless_mode_meas_ctx(
            ham_data,
            trial_data,
            n_mode_chunks=n_mode_chunks,
        ),
    )
