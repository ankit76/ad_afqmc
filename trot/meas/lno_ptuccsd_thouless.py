from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from ..core.ops import EstimatorOps
from ..core.system import System
from ..ham.chol import HamCholUhf
from ..trial.ptuccsd_thouless import PtuccsdThoulessTrial, reference_overlap_u


class LnoEstimatorContext(NamedTuple):
    """T1-transformed Hamiltonian and W-projected connected doubles."""

    exp_t1_a: jax.Array
    exp_t1_b: jax.Array
    h1bar_a: jax.Array
    h1bar_b: jax.Array
    cholbar_a: jax.Array
    cholbar_b: jax.Array
    fockbar_a: jax.Array
    fockbar_b: jax.Array
    t2aa_f: jax.Array
    t2ab_f: jax.Array
    t2ba_f: jax.Array
    t2bb_f: jax.Array
    weight_a: jax.Array
    weight_b: jax.Array
    e0t1_f: jax.Array


def _thouless_transform(mo_t: jax.Array, nocc: int) -> tuple[jax.Array, jax.Array]:
    """Return exp(T1) and exp(-T1) in the occupied/virtual MO ordering."""

    norb = int(mo_t.shape[0])
    bra_orbitals = mo_t.conj().T
    exp_t1 = jnp.eye(norb, dtype=mo_t.dtype)
    exp_t1 = exp_t1.at[:nocc, nocc:].set(bra_orbitals[:, nocc:])
    exp_mt1 = jnp.eye(norb, dtype=mo_t.dtype)
    exp_mt1 = exp_mt1.at[:nocc, nocc:].set(-bra_orbitals[:, nocc:])
    return exp_t1, exp_mt1


def _ufock(
    h1_a: jax.Array,
    h1_b: jax.Array,
    chol_a: jax.Array,
    chol_b: jax.Array,
    noa: int,
    nob: int,
) -> tuple[jax.Array, jax.Array]:
    trace_a = jnp.trace(chol_a[:, :noa, :noa], axis1=1, axis2=2)
    trace_b = jnp.trace(chol_b[:, :nob, :nob], axis1=1, axis2=2)
    trace_sum = trace_a + trace_b
    coulomb_a = jnp.einsum("gpq,g->pq", chol_a, trace_sum, optimize="optimal")
    coulomb_b = jnp.einsum("gpq,g->pq", chol_b, trace_sum, optimize="optimal")
    exchange_a = jnp.einsum(
        "gpj,gjq->pq", chol_a[:, :, :noa], chol_a[:, :noa, :], optimize="optimal"
    )
    exchange_b = jnp.einsum(
        "gpj,gjq->pq", chol_b[:, :, :nob], chol_b[:, :nob, :], optimize="optimal"
    )
    return h1_a + coulomb_a - exchange_a, h1_b + coulomb_b - exchange_b


def _fragment_t1_reference_energy(
    t1a: jax.Array,
    t1b: jax.Array,
    chol_a: jax.Array,
    chol_b: jax.Array,
    weight_a: jax.Array,
    weight_b: jax.Array,
    noa: int,
    nob: int,
) -> jax.Array:
    lt1a = jnp.einsum(
        "ia,gja->gij", t1a, chol_a[:, :noa, noa:], optimize="optimal"
    )
    lt1b = jnp.einsum(
        "ia,gja->gij", t1b, chol_b[:, :nob, nob:], optimize="optimal"
    )
    aa = jnp.einsum("gik,ik,gjj->", lt1a, weight_a, lt1a, optimize="optimal")
    aa -= jnp.einsum("gij,gjk,ik->", lt1a, lt1a, weight_a, optimize="optimal")
    ab = jnp.einsum("gik,ik,gjj->", lt1a, weight_a, lt1b, optimize="optimal")
    ba = jnp.einsum("gik,ik,gjj->", lt1b, weight_b, lt1a, optimize="optimal")
    bb = jnp.einsum("gik,ik,gjj->", lt1b, weight_b, lt1b, optimize="optimal")
    bb -= jnp.einsum("gij,gjk,ik->", lt1b, lt1b, weight_b, optimize="optimal")
    return 0.5 * (aa + ab + ba + bb)


def _build_context(
    ham_data: HamCholUhf,
    trial_data: PtuccsdThoulessTrial,
    *,
    weight_a: jax.Array,
    weight_b: jax.Array,
) -> LnoEstimatorContext:
    """Build the transformed Hamiltonian used by historical ``upt2ccsd``."""

    noa, nob = trial_data.nocc
    weight_a, weight_b = jnp.asarray(weight_a), jnp.asarray(weight_b)
    if weight_a.shape != (noa, noa) or weight_b.shape != (nob, nob):
        raise ValueError("LNO weights must match the alpha and beta occupied spaces.")
    exp_t1_a, exp_mt1_a = _thouless_transform(trial_data.mo_t_a, noa)
    exp_t1_b, exp_mt1_b = _thouless_transform(trial_data.mo_t_b, nob)

    h1bar_a = exp_t1_a @ ham_data.h1_a @ exp_mt1_a
    h1bar_b = exp_t1_b @ ham_data.h1_b @ exp_mt1_b
    cholbar_a = jnp.einsum(
        "pr,grs,sq->gpq", exp_t1_a, ham_data.chol_a, exp_mt1_a, optimize="optimal"
    )
    cholbar_b = jnp.einsum(
        "pr,grs,sq->gpq", exp_t1_b, ham_data.chol_b, exp_mt1_b, optimize="optimal"
    )
    fockbar_a, fockbar_b = _ufock(
        h1bar_a, h1bar_b, cholbar_a, cholbar_b, noa, nob
    )

    # Historical LNO localization acts on only the first occupied index.  The
    # resulting same-spin tensors are not antisymmetric in i<->j, hence the
    # four explicit aa/ab/ba/bb blocks retained by the measurement kernel.
    t2aa_f = jnp.einsum(
        "iajb,ik->kajb", trial_data.t2aa, weight_a, optimize="optimal"
    )
    t2ab_f = jnp.einsum(
        "iajb,ik->kajb", trial_data.t2ab, weight_a, optimize="optimal"
    )
    t2ba_f = jnp.einsum(
        "jbia,ik->kajb", trial_data.t2ab, weight_b, optimize="optimal"
    )
    t2bb_f = jnp.einsum(
        "iajb,ik->kajb", trial_data.t2bb, weight_b, optimize="optimal"
    )

    t1a = exp_t1_a[:noa, noa:]
    t1b = exp_t1_b[:nob, nob:]
    e0t1_f = _fragment_t1_reference_energy(
        t1a,
        t1b,
        ham_data.chol_a,
        ham_data.chol_b,
        weight_a,
        weight_b,
        noa,
        nob,
    )
    return LnoEstimatorContext(
        exp_t1_a=exp_t1_a,
        exp_t1_b=exp_t1_b,
        h1bar_a=h1bar_a,
        h1bar_b=h1bar_b,
        cholbar_a=cholbar_a,
        cholbar_b=cholbar_b,
        fockbar_a=fockbar_a,
        fockbar_b=fockbar_b,
        t2aa_f=t2aa_f,
        t2ab_f=t2ab_f,
        t2ba_f=t2ba_f,
        t2bb_f=t2bb_f,
        weight_a=weight_a,
        weight_b=weight_b,
        e0t1_f=e0t1_f,
    )


def _half_green(transformed_walker: jax.Array, nocc: int) -> jax.Array:
    return jnp.linalg.solve(
        transformed_walker[:nocc].T,
        transformed_walker.T,
    )


def _components(
    walker: tuple[jax.Array, jax.Array],
    ham_data: HamCholUhf,
    meas_ctx: LnoEstimatorContext,
    trial_data: PtuccsdThoulessTrial,
) -> jax.Array:
    """Return ``[theta_F, e0_F, hT_F, e0]`` for one walker.

    Every component is normalized by the exact-T1 determinant overlap.  The
    nonlinear subtraction is intentionally deferred until population means
    have been formed.
    """

    del ham_data
    noa, nob = trial_data.nocc
    norb = trial_data.norb
    nva, nvb = norb - noa, norb - nob
    walker_a, walker_b = walker
    walker_bar_a = meas_ctx.exp_t1_a @ walker_a
    walker_bar_b = meas_ctx.exp_t1_b @ walker_b
    green_a = _half_green(walker_bar_a, noa)
    green_b = _half_green(walker_bar_b, nob)
    green_occ_a = green_a[:, noa:]
    green_occ_b = green_b[:, nob:]
    greenp_a = jnp.vstack(
        (green_occ_a, -jnp.eye(nva, dtype=green_occ_a.dtype))
    )
    greenp_b = jnp.vstack(
        (green_occ_b, -jnp.eye(nvb, dtype=green_occ_b.dtype))
    )

    h1a, h1b = meas_ctx.h1bar_a, meas_ctx.h1bar_b
    t2aa = meas_ctx.t2aa_f
    t2ab = meas_ctx.t2ab_f
    t2ba = meas_ctx.t2ba_f
    t2bb = meas_ctx.t2bb_f

    e1_0 = jnp.einsum("pj,pj->", h1a[:noa], green_a, optimize="optimal")
    e1_0 += jnp.einsum("pj,pj->", h1b[:nob], green_b, optimize="optimal")

    t2g_aa_c = 0.25 * jnp.einsum(
        "iajb,ia->jb", t2aa, green_occ_a, optimize="optimal"
    )
    t2g_aa_e = 0.25 * jnp.einsum(
        "iajb,ja->ib", t2aa, green_occ_a, optimize="optimal"
    )
    t2g_bb_c = 0.25 * jnp.einsum(
        "iajb,ia->jb", t2bb, green_occ_b, optimize="optimal"
    )
    t2g_bb_e = 0.25 * jnp.einsum(
        "iajb,ja->ib", t2bb, green_occ_b, optimize="optimal"
    )
    t2g_ab_a = 0.5 * jnp.einsum(
        "iajb,ia->jb", t2ab, green_occ_a, optimize="optimal"
    )
    t2g_ab_b = 0.5 * jnp.einsum(
        "iajb,jb->ia", t2ab, green_occ_b, optimize="optimal"
    )
    t2g_ba_a = 0.5 * jnp.einsum(
        "iajb,jb->ia", t2ba, green_occ_a, optimize="optimal"
    )
    t2g_ba_b = 0.5 * jnp.einsum(
        "iajb,ia->jb", t2ba, green_occ_b, optimize="optimal"
    )
    theta_aa = jnp.einsum("jb,jb->", t2g_aa_c, green_occ_a, optimize="optimal")
    theta_bb = jnp.einsum("jb,jb->", t2g_bb_c, green_occ_b, optimize="optimal")
    theta_ab = jnp.einsum("jb,jb->", t2g_ab_a, green_occ_b, optimize="optimal")
    theta_ba = jnp.einsum("jb,jb->", t2g_ba_b, green_occ_a, optimize="optimal")
    theta_f = 2.0 * (theta_aa + theta_bb) + theta_ab + theta_ba

    t2_green_aaa_c = jnp.einsum(
        "pb,jb,jq->pq", greenp_a, t2g_aa_c, green_a, optimize="optimal"
    )
    t2_green_aaa_e = jnp.einsum(
        "pb,ib,iq->pq", greenp_a, t2g_aa_e, green_a, optimize="optimal"
    )
    t2_green_bbb_c = jnp.einsum(
        "pb,jb,jq->pq", greenp_b, t2g_bb_c, green_b, optimize="optimal"
    )
    t2_green_bbb_e = jnp.einsum(
        "pb,ib,iq->pq", greenp_b, t2g_bb_e, green_b, optimize="optimal"
    )
    t2_green_aba = jnp.einsum(
        "pa,ia,iq->pq", greenp_a, t2g_ab_b, green_a, optimize="optimal"
    )
    t2_green_baa = jnp.einsum(
        "pb,jb,jq->pq", greenp_a, t2g_ba_b, green_a, optimize="optimal"
    )
    t2_green_bab = jnp.einsum(
        "pa,ia,iq->pq", greenp_b, t2g_ba_a, green_b, optimize="optimal"
    )
    t2_green_abb = jnp.einsum(
        "pb,jb,jq->pq", greenp_b, t2g_ab_a, green_b, optimize="optimal"
    )
    t2_green_aaa = 2.0 * (t2_green_aaa_c - t2_green_aaa_e)
    t2_green_bbb = 2.0 * (t2_green_bbb_c - t2_green_bbb_e)
    t2_green_a = t2_green_aaa + t2_green_aba + t2_green_baa
    t2_green_b = t2_green_bbb + t2_green_bab + t2_green_abb
    e1_2 = theta_f * e1_0
    e1_2 -= jnp.einsum("pq,pq->", t2_green_a, h1a, optimize="optimal")
    e1_2 -= jnp.einsum("pq,pq->", t2_green_b, h1b, optimize="optimal")

    t2_green_a_tot = 2.0 * t2_green_aaa + 2.0 * (t2_green_aba + t2_green_baa)
    t2_green_b_tot = 2.0 * t2_green_bbb + 2.0 * (t2_green_bab + t2_green_abb)
    result_dtype = jnp.result_type(green_a, green_b, h1a, h1b, jnp.complex64)
    zero = jnp.zeros((), dtype=result_dtype)

    def scan_chol(carry, chol_pair):
        e2_0_acc, h2_direct_acc, h2_exchange_acc, h2_t2_acc, e2_f_acc = carry
        chol_a_i, chol_b_i = chol_pair
        gl_a = jnp.einsum("ir,pr->ip", green_a, chol_a_i, optimize="optimal")
        gl_b = jnp.einsum("ir,pr->ip", green_b, chol_b_i, optimize="optimal")
        tr_a = jnp.trace(gl_a[:, :noa])
        tr_b = jnp.trace(gl_b[:, :nob])
        tr_sum = tr_a + tr_b
        e2_0_i = 0.5 * tr_sum * tr_sum
        e2_0_i -= 0.5 * jnp.einsum(
            "ij,ji->", gl_a[:, :noa], gl_a[:, :noa], optimize="optimal"
        )
        e2_0_i -= 0.5 * jnp.einsum(
            "ij,ji->", gl_b[:, :nob], gl_b[:, :nob], optimize="optimal"
        )

        lt2g_a = jnp.einsum(
            "pq,pq->", chol_a_i, t2_green_a_tot, optimize="optimal"
        )
        lt2g_b = jnp.einsum(
            "pq,pq->", chol_b_i, t2_green_b_tot, optimize="optimal"
        )
        h2_direct_i = -0.5 * (lt2g_a + lt2g_b) * tr_sum

        lt2_green_a = jnp.einsum(
            "pi,ji->pj", chol_a_i[:noa], t2_green_a_tot, optimize="optimal"
        )
        lt2_green_b = jnp.einsum(
            "pi,ji->pj", chol_b_i[:nob], t2_green_b_tot, optimize="optimal"
        )
        h2_exchange_i = 0.5 * (
            jnp.einsum("ip,ip->", gl_a, lt2_green_a, optimize="optimal")
            + jnp.einsum("ip,ip->", gl_b, lt2_green_b, optimize="optimal")
        )

        glgp_a = jnp.einsum(
            "ip,pa->ia", gl_a, greenp_a, optimize="optimal"
        )
        glgp_b = jnp.einsum(
            "ip,pa->ia", gl_b, greenp_b, optimize="optimal"
        )
        l2t2_aa = 0.5 * jnp.einsum(
            "ia,iajb,jb->", glgp_a, t2aa, glgp_a, optimize="optimal"
        )
        l2t2_ab = 0.5 * jnp.einsum(
            "ia,iajb,jb->", glgp_a, t2ab, glgp_b, optimize="optimal"
        )
        l2t2_ba = 0.5 * jnp.einsum(
            "ia,iajb,jb->", glgp_b, t2ba, glgp_a, optimize="optimal"
        )
        l2t2_bb = 0.5 * jnp.einsum(
            "ia,iajb,jb->", glgp_b, t2bb, glgp_b, optimize="optimal"
        )
        h2_t2_i = l2t2_aa + l2t2_ab + l2t2_ba + l2t2_bb

        d_a = jnp.einsum(
            "ia,ka->ik", chol_a_i[:noa, noa:], green_occ_a, optimize="optimal"
        )
        d_b = jnp.einsum(
            "ia,ka->ik", chol_b_i[:nob, nob:], green_occ_b, optimize="optimal"
        )
        trace_d = jnp.trace(d_a) + jnp.trace(d_b)
        weighted_trace = jnp.einsum(
            "ik,ik->", meas_ctx.weight_a, d_a, optimize="optimal"
        )
        weighted_trace += jnp.einsum(
            "ik,ik->", meas_ctx.weight_b, d_b, optimize="optimal"
        )
        e2_f_i = 0.5 * trace_d * weighted_trace
        e2_f_i -= 0.5 * jnp.einsum(
            "ij,jk,ik->", d_a, d_a, meas_ctx.weight_a, optimize="optimal"
        )
        e2_f_i -= 0.5 * jnp.einsum(
            "ij,jk,ik->", d_b, d_b, meas_ctx.weight_b, optimize="optimal"
        )
        return (
            e2_0_acc + e2_0_i,
            h2_direct_acc + h2_direct_i,
            h2_exchange_acc + h2_exchange_i,
            h2_t2_acc + h2_t2_i,
            e2_f_acc + e2_f_i,
        ), None

    (e2_0, h2_direct, h2_exchange, h2_t2, e2_f), _ = jax.lax.scan(
        scan_chol,
        (zero, zero, zero, zero, zero),
        (meas_ctx.cholbar_a, meas_ctx.cholbar_b),
    )
    e0 = e1_0 + e2_0
    ht_f = e1_2 + e2_0 * theta_f + h2_direct + h2_exchange + h2_t2
    e1_f = jnp.einsum(
        "ia,ik,ka->",
        green_occ_a,
        meas_ctx.weight_a,
        meas_ctx.fockbar_a[:noa, noa:],
        optimize="optimal",
    )
    e1_f += jnp.einsum(
        "ia,ik,ka->",
        green_occ_b,
        meas_ctx.weight_b,
        meas_ctx.fockbar_b[:nob, nob:],
        optimize="optimal",
    )
    e0_f = meas_ctx.e0t1_f + e1_f + e2_f
    return jnp.stack((theta_f, e0_f, ht_f, e0))


def _combine(h0: jax.Array, components: jax.Array) -> jax.Array:
    """Combine population means as e0_F + hT_F - theta_F * e0."""

    del h0
    theta_f, e0_f, ht_f, e0 = jnp.moveaxis(components, -1, 0)
    return e0_f + ht_f - theta_f * e0


def make_lno_estimator_ops(
    sys: System,
    weight_a: jax.Array,
    weight_b: jax.Array,
) -> EstimatorOps:
    """Build the historical exact-T1, connected-T2 split-LNO estimator."""

    if sys.walker_kind.lower() != "unrestricted":
        raise ValueError("split LNO PT-UCCSD estimators require unrestricted walkers.")
    return EstimatorOps(
        reference_overlap=reference_overlap_u,
        components=_components,
        combine_energy=_combine,
        component_names=("theta_F", "e0_F", "hT_F", "e0"),
        build_estimator_ctx=lambda ham_data, trial_data: (
            _build_context(
                ham_data,
                trial_data,
                weight_a=weight_a,
                weight_b=weight_b,
            )
        ),
        use_for_population_control=False,
    )


__all__ = ["make_lno_estimator_ops"]
