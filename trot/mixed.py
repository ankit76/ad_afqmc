from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from .core.ops import MeasOps
from .core.system import System, WalkerKind
from .meas.pt2ccsd import make_pt2ccsd_meas_ops
from .prop.blocks import MixedBlockFn, block_mixed
from .staging import TrialInput, stage_pt2ccsd_trial
from .stat_utils import pt2ccsd_blocking
from .trial.pt2ccsd import make_pt2ccsd_trial_data

# Recipes for mixed guide/trial AFQMC, where the walkers propagate under one wavefunction
# (the guide) and the energy is measured against another (the trial).
#
# The class and job layers are generic; everything that differs between combinations is
# collected here. Picking a trial picks its whole pipeline -- staging, measurement ops,
# block function and blocking analysis -- so a trial can never be paired with the wrong
# estimator.


@dataclass(frozen=True)
class MixedRecipe:
    """
    One (guide, trial) combination.

    name:                 registry key, e.g. "pt2ccsd"
    guide_kind:           mean field that stages the guide, e.g. "rhf"
    walker_kind:          walker representation the trial supports

    stage_trial:          pyscf cc object -> TrialInput
    make_trial_data:      TrialInput.data -> trial pytree
    make_trial_meas_ops:  (sys, ...) -> MeasOps for the trial estimator

    mixed_block_fn:       per block propagate(guide) + measure(trial)
    blocking_fn:          combines the block components into (energy, stderr)

    mixed_block_fn and blocking_fn belong together: the block function decides which
    components come out of each block, and the blocking function is the only thing that
    knows how to recombine them. They are chosen as a pair, never independently.
    """

    name: str
    guide_kind: str
    walker_kind: WalkerKind

    stage_trial: Callable[..., TrialInput]
    make_trial_data: Callable[[dict, System], Any]
    make_trial_meas_ops: Callable[..., MeasOps]

    mixed_block_fn: MixedBlockFn
    blocking_fn: Callable[..., Any]


MIXED_RECIPES: dict[str, MixedRecipe] = {
    "pt2ccsd": MixedRecipe(
        name="pt2ccsd",
        guide_kind="rhf",
        walker_kind="restricted",
        stage_trial=stage_pt2ccsd_trial,
        make_trial_data=make_pt2ccsd_trial_data,
        make_trial_meas_ops=make_pt2ccsd_meas_ops,
        # the pt2CCSD energy kernel returns (t2, e0, e1) per walker and the energy is
        # h0 + e0 + e1 - t2 * e0, which is nonlinear in the block averages, so these two
        # must stay matched
        mixed_block_fn=block_mixed,
        blocking_fn=pt2ccsd_blocking,
    ),
}


def get_mixed_recipe(name: str) -> MixedRecipe:
    try:
        return MIXED_RECIPES[name]
    except KeyError:
        avail = ", ".join(sorted(MIXED_RECIPES))
        raise ValueError(f"unknown mixed recipe {name!r}; available: {avail}") from None


def available_mixed_recipes() -> tuple[str, ...]:
    return tuple(sorted(MIXED_RECIPES))
