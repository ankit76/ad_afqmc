from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .config import configure_once

configure_once()

import jax.numpy as jnp

from .core.system import System
from .driver import MixedEstimatorQmcResult, run_mixed_estimator_qmc
from .meas.lno_ptuccsd_modes import make_lno_ptuccsd_mode_estimator_ops
from .meas.ptuccsd_modes import PtuccsdModePairSamplingCfg
from .meas.uhf import make_uhf_meas_ops
from .prop.blocks import block_mixed_estimator
from .prop.types import QmcParams
from .setup import Job, setup
from .staging import StagedInputs
from .trial.ptuccsd_modes import make_split_ptuccsd_thouless_mode_trial_data
from .trial.uhf import UhfTrial, make_uhf_trial_ops


def format_lno_fragment_banner(
    fragment_index: int,
    fragment_ham: Any,
    total_ham: Any,
) -> str:
    """Format local LIS dimensions beside canonical active-space totals."""
    local_norb = fragment_ham.norb_spin
    total_norb = total_ham.norb_spin
    if local_norb is None or total_norb is None:
        raise ValueError("fragment banners require spin-resolved orbital dimensions.")
    local_occ = tuple(int(value) for value in fragment_ham.nelec)
    total_occ = tuple(int(value) for value in total_ham.nelec)
    local_vir = tuple(int(local_norb[s] - local_occ[s]) for s in range(2))
    total_vir = tuple(int(total_norb[s] - total_occ[s]) for s in range(2))
    local_nchol = int(fragment_ham.chol.shape[0])
    total_nchol = int(total_ham.chol.shape[0])
    return "\n".join(
        (
            f"FRAGMENT-{fragment_index + 1}",
            f"  alpha occ  : {local_occ[0]} [total {total_occ[0]}]",
            f"  beta occ   : {local_occ[1]} [total {total_occ[1]}]",
            f"  alpha virt : {local_vir[0]} [total {total_vir[0]}]",
            f"  beta virt  : {local_vir[1]} [total {total_vir[1]}]",
            f"  nchol      : {local_nchol} [total {total_nchol}]",
        )
    )


@dataclass(slots=True)
class LnoPtJob:
    """UHF-guided LNO-AFQMC with retained connected-T2 modes."""

    guide_job: Job
    estimator_data: Any
    estimator_ops: Any

    def fragment_correlation(
        self,
        result: MixedEstimatorQmcResult,
    ) -> tuple[float, float]:
        """Combine population means into the fragment correlation energy."""
        return result.estimator_mean_energy, result.estimator_stderr_energy

    def kernel(self) -> MixedEstimatorQmcResult:
        job = self.guide_job
        if not isinstance(job.params, QmcParams):
            raise TypeError("LNO-PT requires QmcParams.")
        return run_mixed_estimator_qmc(
            sys=job.sys,
            params=job.params,
            ham_data=job.ham_data,
            guide_data=job.trial_data,
            guide_ops=job.trial_ops,
            guide_prop_ops=job.prop_ops,
            guide_meas_ops=job.meas_ops,
            estimator_data=self.estimator_data,
            estimator_ops=self.estimator_ops,
            mixed_block_fn=block_mixed_estimator,
            mesh=job.mesh,
        )


def setup_lno_pt(
    source: StagedInputs,
    *,
    t2_discarded_norm: float = 0.01,
    params: QmcParams | None = None,
) -> LnoPtJob:
    """Set up mode-truncated, semistochastic LNO-AFQMC with a UHF guide."""
    if source.ham.basis != "unrestricted" or source.trial.kind.lower() != "pt2uccsd":
        raise ValueError("LNO-PT requires a split Hamiltonian and PT2-UCCSD trial.")
    if not 0.0 <= t2_discarded_norm < 1.0:
        raise ValueError("t2_discarded_norm must lie in [0, 1).")
    if source.ham.norb_spin is None:
        raise ValueError("LNO-PT requires spin-resolved orbital dimensions.")

    sys = System(source.ham.norb, source.ham.nelec, walker_kind="unrestricted")
    trial = source.trial.data
    weight_a = jnp.asarray(trial["weight_a"], dtype=jnp.float64)
    weight_b = jnp.asarray(trial["weight_b"], dtype=jnp.float64)
    estimator_data = make_split_ptuccsd_thouless_mode_trial_data(
        trial,
        source.ham.norb_spin,
        sys,
        discarded_norm_target=t2_discarded_norm,
    )
    nchol = int(source.ham.chol.shape[0])
    component_sampling = (
        PtuccsdModePairSamplingCfg(
            chol_head_size=round(nchol / 8),
            pair_sample_size=4096,
            rank_head_by_guide=True,
            guide_chol_batch_size=16,
            head_chol_batch_size=16,
            tail_probability_uniform_mix=0.01,
            track_half_sample_diagnostic=True,
            walker_guide_policy="head_rms",
            walker_guide_weight_mix=0.1,
        )
        if nchol > 1
        else None
    )
    if component_sampling is None:
        print("[lno-pt] UHF guide; PT2 estimator with deterministic Cholesky sum")
    else:
        print(
            "[lno-pt] UHF guide; PT2 semistochastic sampling: "
            f"chol_head={component_sampling.chol_head_size}/{nchol}, "
            f"walker_chol_pairs={component_sampling.pair_sample_size}"
        )
    estimator_ops = make_lno_ptuccsd_mode_estimator_ops(
        sys,
        weight_a,
        weight_b,
        mixed_precision=False,
        component_sampling=component_sampling,
    )

    identity = jnp.eye(sys.norb, dtype=jnp.float64)
    guide_data = UhfTrial(identity[:, : sys.nup], identity[:, : sys.ndn])
    if params is None:
        params = QmcParams(n_chunks=16, auto_n_chunks=True)
    guide_job = setup(
        source,
        walker_kind="unrestricted",
        params=params,
        trial_data=guide_data,
        trial_ops=make_uhf_trial_ops(sys),
        meas_ops=make_uhf_meas_ops(sys),
    )
    return LnoPtJob(guide_job, estimator_data, estimator_ops)


__all__ = [
    "LnoPtJob",
    "format_lno_fragment_banner",
    "setup_lno_pt",
]
