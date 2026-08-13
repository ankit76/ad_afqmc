from typing import Any, Callable
import jax

from ..ham.chol import HamChol
from ..core.levels import TmpLevelSpec, TmpLevelPack


def make_level_pack(
    *,
    ham_data: HamChol,
    meas_ctx: Any,
    trial_data: Any,
    level: TmpLevelSpec,
    e_kernel: Callable[..., jax.Array],
    fn_slice_ham: Callable[..., HamChol],
    fn_slice_ctx: Callable[..., Any],
    fn_slice_trial: Callable[..., Any],
) -> TmpLevelPack:

    norb = trial_data.norb
    na, nb = trial_data.nocc
    nchol = ham_data.chol.shape[0]

    norb_keep = level.norb_keep
    nchol_keep = level.nchol_keep

    norb_keep = norb if norb_keep is None else norb_keep
    if norb_keep > norb:
        raise ValueError(f"norb_keep ({norb_keep}) must be <= norb ({norb}).")
    if norb_keep < na or norb_keep < nb:
        raise ValueError(f"norb_keep ({norb_keep}) must be >= nocc ({trial_data.nocc}).")

    nchol_keep = nchol if nchol_keep is None else nchol_keep
    if nchol_keep > nchol:
        raise ValueError(f"nchol_keep ({nchol_keep}) must be <= nchol ({nchol}).")

    level = TmpLevelSpec(norb_keep, nchol_keep)

    tr_ham = fn_slice_ham(ham_data, norb_keep=norb_keep, nchol_keep=nchol_keep)
    tr_ctx = fn_slice_ctx(meas_ctx, norb_keep, nchol_keep)
    tr_trial = fn_slice_trial(trial_data, norb_keep)

    return TmpLevelPack(
        level=level,
        ham_data=tr_ham,
        meas_ctx=tr_ctx,
        trial_data=tr_trial,
        e_kernel=e_kernel,
    )
