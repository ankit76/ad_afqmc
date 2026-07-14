from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import lax, tree_util

from ..core.ops import MeasOps, k_energy, k_force_bias
from ..core.system import System
from ..ham.chol import HamChol
from ..trial.cisd_modes import CisdModeTrial, mode_apply, mode_quadratic
from ..trial.cisd_modes import overlap_r as cisd_mode_overlap_r
from .cisd import CisdMeasCfg, _energy_gl_batched_realimag, _force_bias_chol_contract_high_realimag

_CISD_MODE_MEAS_CFG_ATTR = "_cisd_mode_meas_cfg"


def _greens_restricted(walker: jax.Array, nocc: int) -> jax.Array:
    wocc = walker[:nocc, :]
    return jnp.linalg.solve(wocc.T, walker.T)


def _active_green_blocks(
    green: jax.Array,
    trial_data: CisdModeTrial,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    green_act = green[trial_data.occ_act_slice, :]
    green_occ = green[trial_data.occ_act_slice, trial_data.vir_act_slice]

    greenp = jnp.zeros((trial_data.norb, trial_data.nvir), dtype=green.dtype)
    greenp = greenp.at[: trial_data.nocc_full, :].set(green[:, trial_data.vir_act_slice])
    greenp = greenp.at[trial_data.vir_act_slice, :].set(
        -jnp.eye(trial_data.nvir, dtype=green.dtype)
    )
    return green_act, green_occ, greenp


@tree_util.register_pytree_node_class
@dataclass(frozen=True)
class CisdModeMeasCtx:
    rot_chol: jax.Array
    lci1: jax.Array
    cfg: CisdMeasCfg
    n_mode_chunks: int

    def tree_flatten(self):
        children = (self.rot_chol, self.lci1)
        aux = (self.cfg, self.n_mode_chunks)
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        cfg, n_mode_chunks = aux
        rot_chol, lci1 = children
        return cls(
            rot_chol=rot_chol,
            lci1=lci1,
            cfg=cfg,
            n_mode_chunks=n_mode_chunks,
        )


def get_cisd_mode_meas_cfg(meas_ops: MeasOps) -> CisdMeasCfg | None:
    cfg = getattr(meas_ops, _CISD_MODE_MEAS_CFG_ATTR, None)
    return cfg if isinstance(cfg, CisdMeasCfg) else None


def build_meas_ctx(
    ham_data: HamChol,
    trial_data: CisdModeTrial,
    *,
    cfg: CisdMeasCfg = CisdMeasCfg(memory_mode="high"),
    n_mode_chunks: int = 1,
) -> CisdModeMeasCtx:
    """Build full-Cholesky measurement intermediates.

    ``n_mode_chunks=1`` evaluates the complete mode axis in one batch. Larger
    values reduce mode-dependent temporary memory by scanning over that many
    partitions; the requested count is capped at the full mode rank.
    """
    if n_mode_chunks <= 0:
        raise ValueError("n_mode_chunks must be positive.")
    if cfg.memory_mode != "high":
        raise ValueError("Mode-native CISD measurements currently require memory_mode='high'.")

    if ham_data.basis != "restricted":
        raise ValueError("CISD mode MeasOps requires HamChol.basis == 'restricted'.")

    chol = ham_data.chol
    rot_chol = chol[:, : trial_data.nocc_full, :]
    lci1 = jnp.einsum(
        "git,pt->gip",
        chol[:, :, trial_data.vir_act_slice],
        trial_data.ci1,
        optimize="optimal",
    )
    return CisdModeMeasCtx(
        rot_chol=rot_chol,
        lci1=lci1,
        cfg=cfg,
        n_mode_chunks=min(int(n_mode_chunks), trial_data.mode_rank),
    )


def force_bias_kernel_rw_rh(
    walker: jax.Array,
    ham_data: HamChol,
    meas_ctx: CisdModeMeasCtx,
    trial_data: CisdModeTrial,
) -> jax.Array:
    green = _greens_restricted(walker, trial_data.nocc_full)
    green_act, green_occ, greenp = _active_green_blocks(green, trial_data)

    lg = jnp.einsum("gpj,pj->g", meas_ctx.rot_chol, green, optimize="optimal")
    ci1g = jnp.einsum("pt,pt->", trial_data.ci1, green_occ, optimize="optimal")
    ci1gp = jnp.einsum("pt,it->pi", trial_data.ci1, greenp, optimize="optimal")
    gci1gp = jnp.einsum("pj,pi->ij", green_act, ci1gp, optimize="optimal")

    projections, kg = mode_apply(trial_data, green_occ)
    gkg = mode_quadratic(trial_data, green_occ, projections)
    overlap = 1.0 + 2.0 * ci1g + gkg

    cisd_green = -2.0 * (greenp @ kg.T) @ green_act
    correction = _force_bias_chol_contract_high_realimag(
        ham_data.chol,
        cisd_green - 2.0 * gci1gp,
        meas_ctx.cfg,
    )
    return (2.0 * lg + 4.0 * ci1g * lg + 2.0 * lg * gkg + correction) / overlap


def _mode_project_pair_batch(modes: jax.Array, matrix: jax.Array) -> jax.Array:
    """Project one mode chunk against a complex active pair-space matrix."""
    matrix_r = jnp.real(matrix).astype(modes.dtype)
    result_r = jnp.einsum("rpt,pt->r", modes, matrix_r, optimize="optimal")
    if not jnp.issubdtype(matrix.dtype, jnp.complexfloating):
        return result_r.astype(jnp.float64)

    matrix_i = jnp.imag(matrix).astype(modes.dtype)
    result_i = jnp.einsum("rpt,pt->r", modes, matrix_i, optimize="optimal")
    return result_r.astype(jnp.complex128) + 1.0j * result_i.astype(jnp.complex128)


def _transform_modes(modes: jax.Array, greenp: jax.Array) -> jax.Array:
    """Form ``v_r @ greenp.T`` at the stored eigenvector precision."""
    greenp_r = jnp.real(greenp).astype(modes.dtype)
    transformed_r = jnp.einsum("rpt,it->rpi", modes, greenp_r, optimize="optimal")
    if not jnp.issubdtype(greenp.dtype, jnp.complexfloating):
        return transformed_r.astype(jnp.float64)

    greenp_i = jnp.imag(greenp).astype(modes.dtype)
    transformed_i = jnp.einsum("rpt,it->rpi", modes, greenp_i, optimize="optimal")
    complex_dtype = jnp.complex64 if modes.dtype == jnp.float32 else jnp.complex128
    return transformed_r.astype(complex_dtype) + 1.0j * transformed_i.astype(complex_dtype)


def _mode_energy_chunk(
    eigenvalues: jax.Array,
    modes: jax.Array,
    *,
    green_occ: jax.Array,
    greenp: jax.Array,
    h_projector: jax.Array,
    chol_projector: jax.Array,
    exchange_projector: jax.Array,
    gl_active: jax.Array,
    hg: jax.Array,
    e20: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Return one mode batch's energy numerator and overlap doubles.

    The complete Cholesky axis is evaluated at once. Only the mode axis is
    partitioned when ``n_mode_chunks > 1``.
    """
    amplitudes = _mode_project_pair_batch(modes, green_occ)
    h_modes = _mode_project_pair_batch(modes, h_projector)
    chol_modes = _mode_project_pair_batch(modes, chol_projector)
    exchange_modes = _mode_project_pair_batch(modes, exchange_projector)

    transformed_modes = _transform_modes(modes, greenp)
    mode_complex_dtype = jnp.complex64 if modes.dtype == jnp.float32 else jnp.complex128
    mode_gl = jnp.einsum(
        "gpi,rpi->gr",
        gl_active.astype(mode_complex_dtype),
        transformed_modes,
        optimize="optimal",
    )
    e223_modes = jnp.sum(mode_gl * mode_gl, axis=0, dtype=jnp.complex128)

    values = eigenvalues.astype(jnp.float64)
    doubles = jnp.sum(values * amplitudes * amplitudes, dtype=jnp.complex128)
    contribution = values * (
        (2.0 * hg + e20) * amplitudes * amplitudes
        + amplitudes * (-2.0 * h_modes - 4.0 * chol_modes + 2.0 * exchange_modes)
        + e223_modes
    )
    numerator = jnp.sum(contribution, dtype=jnp.complex128)
    return numerator, doubles


def _mode_energy_numerator_and_doubles(
    trial_data: CisdModeTrial,
    meas_ctx: CisdModeMeasCtx,
    *,
    green_occ: jax.Array,
    greenp: jax.Array,
    h_projector: jax.Array,
    chol_projector: jax.Array,
    exchange_projector: jax.Array,
    gl_active: jax.Array,
    hg: jax.Array,
    e20: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Accumulate all modes, optionally scanning over mode-only partitions."""
    rank = trial_data.mode_rank
    zero = jnp.asarray(0.0, dtype=jnp.complex128)
    if rank == 0:
        return zero, zero

    def evaluate_chunk(eigenvalues, modes):
        return _mode_energy_chunk(
            eigenvalues,
            modes,
            green_occ=green_occ,
            greenp=greenp,
            h_projector=h_projector,
            chol_projector=chol_projector,
            exchange_projector=exchange_projector,
            gl_active=gl_active,
            hg=hg,
            e20=e20,
        )

    n_mode_chunks = min(meas_ctx.n_mode_chunks, rank)
    if n_mode_chunks == 1:
        return evaluate_chunk(trial_data.eigenvalues, trial_data.modes)

    # Scan over exactly n_mode_chunks contiguous, balanced partitions. Shorter
    # partitions have one masked entry, avoiding a padded copy of the complete
    # mode tensor while keeping one static batch shape inside lax.scan.
    base_chunk_size = rank // n_mode_chunks
    n_larger_chunks = rank % n_mode_chunks
    chunk_size = base_chunk_size + int(n_larger_chunks > 0)
    chunk_offsets = jnp.arange(chunk_size, dtype=jnp.int32)

    def scan_body(carry, chunk_index):
        is_larger = chunk_index < n_larger_chunks
        chunk_length = base_chunk_size + is_larger.astype(jnp.int32)
        start = chunk_index * base_chunk_size + jnp.minimum(chunk_index, n_larger_chunks)
        indices = start + chunk_offsets
        valid = chunk_offsets < chunk_length
        indices = jnp.minimum(indices, rank - 1)
        eigenvalues = jnp.where(valid, trial_data.eigenvalues[indices], 0.0)
        modes = trial_data.modes[indices]
        numerator_i, doubles_i = evaluate_chunk(eigenvalues, modes)
        numerator, doubles = carry
        return (numerator + numerator_i, doubles + doubles_i), None

    (numerator, doubles), _ = lax.scan(
        scan_body,
        (zero, zero),
        jnp.arange(n_mode_chunks, dtype=jnp.int32),
    )
    return numerator, doubles


def energy_kernel_rw_rh(
    walker: jax.Array,
    ham_data: HamChol,
    meas_ctx: CisdModeMeasCtx,
    trial_data: CisdModeTrial,
) -> jax.Array:
    """Complete deterministic local energy from every stored K mode."""
    green = _greens_restricted(walker, trial_data.nocc_full)
    green_act, green_occ, greenp = _active_green_blocks(green, trial_data)

    h1 = ham_data.h1
    chol = ham_data.chol
    rot_chol = meas_ctx.rot_chol

    hg = jnp.einsum("pj,pj->", h1[: trial_data.nocc_full, :], green, optimize="optimal")
    ci1g = jnp.einsum("pt,pt->", trial_data.ci1, green_occ, optimize="optimal")
    ci1_green = (greenp @ trial_data.ci1.T) @ green_act
    e1_nonci2 = (
        2.0 * hg + 4.0 * ci1g * hg - 2.0 * jnp.einsum("ij,ij->", h1, ci1_green, optimize="optimal")
    )

    lg = jnp.einsum("gpj,pj->g", rot_chol, green, optimize="optimal")
    lg1 = jnp.einsum("gpj,qj->gpq", rot_chol, green, optimize="optimal")
    e20 = 2.0 * (lg @ lg) - jnp.sum(lg1 * jnp.swapaxes(lg1, -1, -2))

    lci1g = _force_bias_chol_contract_high_realimag(chol, ci1_green, meas_ctx.cfg)
    e2_1_2 = -2.0 * (lci1g @ lg)
    ci1g1 = trial_data.ci1 @ green[:, trial_data.vir_act_slice].T
    e2_1_3_1 = jnp.einsum(
        "gpq,gqa,ap->",
        lg1,
        lg1[:, :, trial_data.occ_act_slice],
        ci1g1,
        optimize="optimal",
    )
    lci1g_mat = jnp.einsum("gia,qi->gaq", meas_ctx.lci1, green, optimize="optimal")
    e2_1_3_2 = -jnp.einsum(
        "gaq,gqa->",
        lci1g_mat,
        lg1[:, :, trial_data.occ_act_slice],
        optimize="optimal",
    )
    e2_1 = 2.0 * e20 * ci1g + 2.0 * (e2_1_2 + e2_1_3_1 + e2_1_3_2)
    nonci2_numerator = e1_nonci2 + e20 + e2_1

    h_projector = jnp.einsum("ij,iu,qj->qu", h1, greenp, green_act, optimize="optimal")
    chol_lg = jnp.einsum("g,gij->ij", lg, chol, optimize="optimal")
    chol_projector = jnp.einsum("ij,iu,qj->qu", chol_lg, greenp, green_act, optimize="optimal")

    gl = _energy_gl_batched_realimag(green, chol, meas_ctx.cfg)
    exchange_matrix = jnp.einsum("gpj,gpi->ji", gl, rot_chol, optimize="optimal")
    exchange_projector = jnp.einsum(
        "ji,ju,qi->qu", exchange_matrix, greenp, green_act, optimize="optimal"
    )

    ci2_numerator, doubles = _mode_energy_numerator_and_doubles(
        trial_data,
        meas_ctx,
        green_occ=green_occ,
        greenp=greenp,
        h_projector=h_projector,
        chol_projector=chol_projector,
        exchange_projector=exchange_projector,
        gl_active=gl[:, trial_data.occ_act_slice, :],
        hg=hg,
        e20=e20,
    )
    overlap = 1.0 + 2.0 * ci1g + doubles
    return ham_data.h0 + (nonci2_numerator + ci2_numerator) / overlap


def make_cisd_mode_meas_ops(
    sys: System,
    *,
    mixed_precision: bool = True,
    n_mode_chunks: int = 1,
) -> MeasOps:
    """Build exact full-rank mode measurements with batched Cholesky terms.

    Only the mode axis is optionally partitioned. All Cholesky vectors remain
    batched, matching the dense CISD high-memory measurement policy.
    """
    if sys.walker_kind.lower() != "restricted":
        raise ValueError(
            "CISD mode MeasOps currently supports only restricted walkers, "
            f"got: {sys.walker_kind}"
        )
    if n_mode_chunks <= 0:
        raise ValueError("n_mode_chunks must be positive.")

    cfg = CisdMeasCfg(
        memory_mode="high",
        mixed_real_dtype=jnp.float32 if mixed_precision else jnp.float64,
        mixed_complex_dtype=jnp.complex64 if mixed_precision else jnp.complex128,
        mixed_real_dtype_testing=jnp.float32 if mixed_precision else jnp.float64,
        mixed_complex_dtype_testing=jnp.complex64 if mixed_precision else jnp.complex128,
    )
    meas_ops = MeasOps(
        overlap=cisd_mode_overlap_r,
        build_meas_ctx=lambda ham_data, trial_data: build_meas_ctx(
            ham_data,
            trial_data,
            cfg=cfg,
            n_mode_chunks=n_mode_chunks,
        ),
        kernels={k_force_bias: force_bias_kernel_rw_rh, k_energy: energy_kernel_rw_rh},
        observables={},
    )
    object.__setattr__(meas_ops, _CISD_MODE_MEAS_CFG_ATTR, cfg)
    return meas_ops
