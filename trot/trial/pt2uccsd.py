"""Public PT2-UCCSD trial API.

The implementation keeps the historical ``ptuccsd_thouless`` names used on
the development branch while exposing names parallel to ``trial.pt2ccsd``.
"""

from .ptuccsd_thouless import (
    PtuccsdThoulessTrial,
    get_rdm1,
    greenp_from_green,
    greens_unrestricted,
    make_ptuccsd_thouless_trial_data,
    make_ptuccsd_thouless_trial_ops,
    overlap_r,
    overlap_u,
    reference_overlap_r,
    reference_overlap_u,
    theta_t2_from_greens,
    theta_t2_u,
    thouless_mo_from_t1,
)

Pt2uccsdTrial = PtuccsdThoulessTrial
make_pt2uccsd_trial_data = make_ptuccsd_thouless_trial_data
make_pt2uccsd_trial_ops = make_ptuccsd_thouless_trial_ops

__all__ = [
    "Pt2uccsdTrial",
    "get_rdm1",
    "greenp_from_green",
    "greens_unrestricted",
    "make_pt2uccsd_trial_data",
    "make_pt2uccsd_trial_ops",
    "overlap_r",
    "overlap_u",
    "reference_overlap_r",
    "reference_overlap_u",
    "theta_t2_from_greens",
    "theta_t2_u",
    "thouless_mo_from_t1",
]
