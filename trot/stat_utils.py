from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, cast

import jax
import numpy as np
import jax.numpy as jnp


def _pick_plateau_with_status(
    Bs: np.ndarray,
    SEs: np.ndarray,
    Gs: np.ndarray,
    *,
    min_blocks: int = 20,
    min_rise: float = 0.20,
    flat_tol: float = 0.03,
    k: int = 3,
) -> tuple[int, float, int, bool, str]:
    assert Bs.size > 0
    Bs, SEs, Gs = map(np.asarray, (Bs, SEs, Gs))
    ok = Gs >= min_blocks
    Bs2, SEs2, Gs2 = Bs[ok], SEs[ok], Gs[ok]
    if Bs2.size == 0:
        return int(Bs[0]), float(SEs[0]), int(Gs[0]), False, "insufficient_blocks"
    rise_ok = SEs2 >= (1.0 + min_rise) * SEs2[0]
    for i in range(0, Bs2.size - k):
        if not rise_ok[i]:
            continue
        window = SEs2[i : i + k + 1]
        if np.all(np.abs(np.diff(window)) <= flat_tol * window[:-1]):
            return int(Bs2[i]), float(SEs2[i]), int(Gs2[i]), True, "plateau"
    finite = np.isfinite(SEs2)
    if not np.any(finite):
        return int(Bs[0]), float(SEs[0]), int(Gs[0]), False, "nonfinite_se_curve"
    jmax = int(np.where(finite, SEs2, -np.inf).argmax())
    thresh = 0.95 * SEs2[jmax]
    candidates = np.where(SEs2 >= thresh)[0]
    j = int(candidates[0]) if candidates.size > 0 else jmax
    return int(Bs2[j]), float(SEs2[j]), int(Gs2[j]), False, "near_maximum_fallback"


def _pick_plateau(
    Bs: np.ndarray,
    SEs: np.ndarray,
    Gs: np.ndarray,
    *,
    min_blocks: int = 20,
    min_rise: float = 0.20,
    flat_tol: float = 0.03,
    k: int = 3,
) -> tuple[int, float, int]:
    """Backward-compatible plateau selection without status metadata."""

    B_star, se_star, G_star, _, _ = _pick_plateau_with_status(
        Bs,
        SEs,
        Gs,
        min_blocks=min_blocks,
        min_rise=min_rise,
        flat_tol=flat_tol,
        k=k,
    )
    return B_star, se_star, G_star


def blocking_analysis_ratio(
    ene: np.ndarray | jax.Array,
    wt: np.ndarray | jax.Array,
    block_grid: Iterable[int] | None = None,
    *,
    min_blocks: int = 20,
    min_rise: float = 0.20,
    flat_tol: float = 0.03,
    k: int = 3,
    bins: int | str = "fd",
    figsize: tuple[float, float] = (12, 4.2),
    title: str | None = None,
    print_q: bool = True,
    plot_q: bool = False,
    exact: float | None = None,
) -> Dict[str, Any]:
    """Blocking analysis for mu = sum(wt*ene)/sum(wt)"""
    ene = np.asarray(ene, float).ravel()
    wt = np.asarray(wt, float).ravel()
    n = ene.size
    assert wt.size == n

    S = wt * ene
    N = wt
    S_tot, N_tot = S.sum(), N.sum()
    mu_full = S_tot / N_tot

    if block_grid is None:
        raw = np.unique(np.rint(np.geomspace(1, max(2, n // min_blocks), 18)).astype(int))
        block_grid = [int(b) for b in raw if b >= 1 and (n // b) >= min_blocks]
        if (n // raw[-1]) >= 5 and raw[-1] not in block_grid:
            block_grid.append(int(raw[-1]))

    Bs_list: list[int] = []
    SEs_list: list[float] = []
    Gs_list: list[int] = []
    LOO_cache: dict[int, tuple[np.ndarray, float, int]] = {}
    for B in block_grid:
        G = n // B
        if G < 5:
            continue
        usable = G * B
        Sg = S[:usable].reshape(G, B).sum(axis=1)
        Ng = N[:usable].reshape(G, B).sum(axis=1)
        St, Nt = Sg.sum(), Ng.sum()

        denom_loo = Nt - Ng
        safe = np.abs(denom_loo) > 1e-18
        mu_loo = np.where(safe, (St - Sg) / denom_loo, St / Nt)

        mu_bar = mu_loo.mean()
        var = (G - 1) / G * np.sum((mu_loo - mu_bar) ** 2)
        se = float(np.sqrt(max(var, 0.0)))

        Bs_list.append(B)
        SEs_list.append(se)
        Gs_list.append(G)
        LOO_cache[B] = (mu_loo, mu_bar, G)

    Bs = np.array(Bs_list, int)
    SEs = np.array(SEs_list, float)
    Gs = np.array(Gs_list, int)
    ci95: tuple[float, float] | None = None
    if Bs.size == 0:
        B_star: int | None = None
        se_star: float | None = None
        G_star: int | None = None
        plateau_found = False
        selection_reason = "unavailable"
    else:
        B_star, se_star, G_star, plateau_found, selection_reason = _pick_plateau_with_status(
            Bs,
            SEs,
            Gs,
            min_blocks=min_blocks,
            min_rise=min_rise,
            flat_tol=flat_tol,
            k=k,
        )
        ci95 = (mu_full - 1.96 * se_star, mu_full + 1.96 * se_star)

    if B_star is None:
        # Blocking analysis not possible
        out = {
            "mu": float(mu_full),
            "block_sizes": None,
            "se_curve": None,
            "n_blocks": None,
            "B_star": None,
            "se_star": None,
            "plateau_found": False,
            "selection_reason": selection_reason,
            "ci95_star": (None, None),
            "estimator_scale_samples": None,
            "bias": None,
            "z_score": None,
        }
        return out

    se_star = cast(float, se_star)
    ci95 = cast(tuple[float, float], ci95)
    mu_loo, mu_bar, G = LOO_cache[B_star]
    est_samples = mu_full + (G - 1) / np.sqrt(G) * (mu_loo - mu_bar)

    bias = z = None
    if exact is not None and np.isfinite(se_star) and se_star > 0:
        bias = float(mu_full - exact)
        z = float((mu_full - exact) / se_star)

    out = {
        "mu": float(mu_full),
        "block_sizes": Bs,
        "se_curve": SEs,
        "n_blocks": Gs,
        "B_star": int(B_star),
        "se_star": float(se_star),
        "plateau_found": plateau_found,
        "selection_reason": selection_reason,
        "ci95_star": (float(ci95[0]), float(ci95[1])),
        "estimator_scale_samples": est_samples,
        "bias": bias,
        "z_score": z,
    }

    if print_q:
        print(f"mu: {out['mu']:.16g}  SE*: {out['se_star']:.16g}  95% CI: {out['ci95_star']}")
        print(f"selection: {out['selection_reason']}  " f"plateau_found: {out['plateau_found']}")
        if out["z_score"] is not None:
            print(f"bias: {out['bias']:.16g}  z: {out['z_score']:.6g}")

        # table: block size vs SE, mark chosen B*
        se0 = float(SEs[0]) if SEs.size else float("nan")
        print("\nBlocking SE curve (ratio LOO):")
        print(f"{'':1s}{'B':>6s} {'G':>6s} {'SE':>14s} {'SE/SE(B=1)':>12s}")
        for B, G, se in zip(Bs, Gs, SEs):
            mark = "*" if int(B) == int(B_star) else " "
            rel = (float(se) / se0) if (se0 > 0 and np.isfinite(se0)) else float("nan")
            print(f"{mark}{int(B):6d} {int(G):6d} {float(se):14.6e} {rel:12.3f}")
        print("")  # trailing newline

    if plot_q:
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # SE curve
        ax1.plot(Bs, SEs, marker="o", lw=1.6)
        ax1.axvline(B_star, ls="--", color="k", alpha=0.85, label=f"chosen B = {B_star}")
        if exact is not None:
            ax1.set_title(
                (title or "Blocking SE for ratio estimator")
                + "\n"
                + rf"$\mu$={mu_full:.6f}, SE*={se_star:.3e}, bias={bias:.3e}, z={z:.2f}"
            )
        else:
            ax1.set_title(title or "Blocking SE for ratio estimator")
        ax1.set_xscale("log")
        ax1.set_xlabel("block size B (walkers)")
        ax1.set_ylabel(r"SE[$\mu$]")
        ax1.grid(True, alpha=0.25)
        ax1.legend()

        # estimator-scale histogram
        ax2.hist(est_samples, bins=bins, density=True, alpha=0.6, edgecolor="white")
        xs = np.linspace(mu_full - 6 * se_star, mu_full + 6 * se_star, 400)
        pdf = (1.0 / (se_star * np.sqrt(2 * np.pi))) * np.exp(
            -0.5 * ((xs - mu_full) / se_star) ** 2
        )
        ax2.plot(xs, pdf, lw=2.0, color="#f58518", label="Normal(SE*)")
        ax2.axvline(mu_full, ls="--", color="k", lw=1.2, label=r"$\hat\mu$")
        ax2.axvline(ci95[0], ls=":", color="k", lw=1.2, label="95% CI")
        ax2.axvline(ci95[1], ls=":", color="k", lw=1.2)
        if exact is not None:
            ax2.axvline(exact, ls="--", color="red", lw=1.4, label="exact/target")
        ax2.set_xlabel("estimator-scale (rescaled LOO)")
        ax2.set_ylabel("density")
        ax2.legend()
        fig.tight_layout()

    return out


def blocking_analysis_components(
    h0: float | complex,
    weights: np.ndarray | jax.Array,
    components: np.ndarray | jax.Array,
    combine_energy: Callable[[Any, Any], Any],
    block_grid: Iterable[int] | None = None,
    *,
    min_blocks: int = 20,
    min_rise: float = 0.20,
    flat_tol: float = 0.03,
    k: int = 3,
    print_q: bool = True,
) -> Dict[str, Any]:
    """Blocking/jackknife analysis for a nonlinear energy of component ratios.

    Each input row contains a block-level component mean and ``weights``
    contains the corresponding denominator. The full estimate first forms

    ``c = sum_b weights[b] * components[b] / sum_b weights[b]``

    and then evaluates ``combine_energy(h0, c)``. Leave-one-superblock-out
    estimates apply the same nonlinear combination, retaining all covariance
    among components.
    """

    weights_array = np.asarray(weights).reshape(-1)
    components_array = np.asarray(components)
    if components_array.ndim != 2:
        raise ValueError(
            "components must have shape (n_blocks, n_components); "
            f"got {components_array.shape}."
        )
    if components_array.shape[0] != weights_array.shape[0]:
        raise ValueError(
            f"weights length {weights_array.shape[0]} does not match components "
            f"length {components_array.shape[0]}."
        )
    finite = np.isfinite(weights_array) & np.all(np.isfinite(components_array), axis=1)
    weights_array = weights_array[finite]
    components_array = components_array[finite]
    n = int(weights_array.shape[0])
    if n == 0:
        raise ValueError("No finite component blocks are available for analysis.")

    weighted_components = weights_array[:, None] * components_array
    total_weight = weights_array.sum()
    if abs(total_weight) <= 1.0e-18:
        raise ValueError("The total component-estimator weight is numerically zero.")
    total_components = weighted_components.sum(axis=0)
    mean_components = total_components / total_weight
    mean_energy = float(
        np.real(np.asarray(combine_energy(h0, mean_components)).reshape(()))
    )

    if block_grid is None:
        raw = np.unique(np.rint(np.geomspace(1, max(2, n // min_blocks), 18)).astype(int))
        block_grid = [int(b) for b in raw if b >= 1 and (n // b) >= min_blocks]
        if raw.size > 0 and (n // raw[-1]) >= 5 and raw[-1] not in block_grid:
            block_grid.append(int(raw[-1]))

    block_sizes: list[int] = []
    block_errors: list[float] = []
    group_counts: list[int] = []
    for block_size in block_grid:
        n_groups = n // int(block_size)
        if n_groups < 5:
            continue
        usable = n_groups * int(block_size)
        group_weights = weights_array[:usable].reshape(n_groups, int(block_size)).sum(axis=1)
        group_components = (
            weighted_components[:usable]
            .reshape(n_groups, int(block_size), components_array.shape[1])
            .sum(axis=1)
        )
        used_weight = group_weights.sum()
        used_components = group_components.sum(axis=0)
        loo_weight = used_weight - group_weights
        safe = np.abs(loo_weight) > 1.0e-18
        fallback = used_components / used_weight
        loo_components = np.where(
            safe[:, None],
            (used_components[None, :] - group_components) / loo_weight[:, None],
            fallback[None, :],
        )
        loo_energy = np.real(np.asarray(combine_energy(h0, loo_components))).reshape(n_groups)
        loo_mean = float(np.mean(loo_energy))
        variance = (n_groups - 1) / n_groups * np.sum((loo_energy - loo_mean) ** 2)
        block_sizes.append(int(block_size))
        block_errors.append(float(np.sqrt(max(float(variance), 0.0))))
        group_counts.append(n_groups)

    sizes = np.asarray(block_sizes, dtype=int)
    errors = np.asarray(block_errors, dtype=float)
    groups = np.asarray(group_counts, dtype=int)
    if sizes.size == 0:
        result = {
            "mu": mean_energy,
            "mean_components": mean_components,
            "block_sizes": None,
            "se_curve": None,
            "n_blocks": None,
            "B_star": None,
            "se_star": None,
            "plateau_found": False,
            "selection_reason": "unavailable",
        }
    else:
        block_star, error_star, _, plateau_found, selection_reason = _pick_plateau_with_status(
            sizes,
            errors,
            groups,
            min_blocks=min_blocks,
            min_rise=min_rise,
            flat_tol=flat_tol,
            k=k,
        )
        result = {
            "mu": mean_energy,
            "mean_components": mean_components,
            "block_sizes": sizes,
            "se_curve": errors,
            "n_blocks": groups,
            "B_star": int(block_star),
            "se_star": float(error_star),
            "plateau_found": bool(plateau_found),
            "selection_reason": str(selection_reason),
        }

    if print_q:
        error = result["se_star"]
        error_text = f"{float(error):.6e}" if error is not None else "unavailable"
        print(
            "component estimator: "
            f"energy={mean_energy:.12f}, SE={error_text}, "
            f"selection={result['selection_reason']}."
        )
    return result


def component_estimator_outlier_mask(
    h0: float | complex,
    weights: np.ndarray | jax.Array,
    components: np.ndarray | jax.Array,
    combine_energy: Callable[[Any, Any], Any],
    *,
    zeta: float | None = 20.0,
    min_scale: float = 1.0e-10,
) -> tuple[np.ndarray, np.ndarray]:
    """Return block proxy energies and the robust component-estimator keep mask.

    The proxy applies ``combine_energy`` independently to each block's component
    means.  For finite blocks, the historical PT2-CCSD rule keeps blocks whose
    absolute deviation from the median is less than ``zeta`` times the median
    absolute deviation.  ``zeta=None`` disables robust rejection while still
    excluding nonfinite weights, components, and proxy energies.

    The proxy is used only to identify exceptional blocks.  Final energies must
    still be formed from the retained weighted component ratios so nonlinear
    component covariance is preserved.
    """

    weights_array = np.asarray(weights).reshape(-1)
    components_array = np.asarray(components)
    if components_array.ndim != 2:
        raise ValueError(
            "components must have shape (n_blocks, n_components); "
            f"got {components_array.shape}."
        )
    if components_array.shape[0] != weights_array.shape[0]:
        raise ValueError(
            f"weights length {weights_array.shape[0]} does not match components "
            f"length {components_array.shape[0]}."
        )
    if zeta is not None and (not np.isfinite(zeta) or zeta <= 0.0):
        raise ValueError("zeta must be positive and finite, or None.")
    if not np.isfinite(min_scale) or min_scale <= 0.0:
        raise ValueError("min_scale must be positive and finite.")

    proxy_energies = np.real(np.asarray(combine_energy(h0, components_array))).reshape(-1)
    if proxy_energies.shape[0] != weights_array.shape[0]:
        raise ValueError(
            "combine_energy must return one proxy energy per component block; "
            f"got {proxy_energies.shape}."
        )
    finite = (
        np.isfinite(weights_array)
        & np.all(np.isfinite(components_array), axis=1)
        & np.isfinite(proxy_energies)
    )
    if zeta is None or not np.any(finite):
        return proxy_energies, finite

    finite_energies = proxy_energies[finite]
    deviations = np.abs(proxy_energies - np.median(finite_energies))
    median_absolute_deviation = float(np.median(deviations[finite]))
    scale = max(median_absolute_deviation, min_scale)
    keep = finite & (deviations / scale < zeta)
    return proxy_energies, keep


def _autocovariance_fft(data: np.ndarray, max_lag: int) -> np.ndarray:
    """Return autocovariances through ``max_lag`` using the ``n - lag`` normalization."""
    data = np.asarray(data, dtype=float).ravel()
    n = data.size
    centered = data - data.mean()
    n_fft = 1 << (2 * n - 1).bit_length()
    spectrum = np.fft.rfft(centered, n=n_fft)
    autocovariance_sums = np.fft.irfft(spectrum * spectrum.conjugate(), n=n_fft)
    return autocovariance_sums[: max_lag + 1] / np.arange(n, n - max_lag - 1, -1)


def gamma_analysis_ratio(
    ene: np.ndarray | jax.Array,
    wt: np.ndarray | jax.Array,
    *,
    s_tau: float = 1.5,
    max_lag: int | None = None,
    min_effective_samples: float = 20.0,
    figsize: tuple[float, float] = (12, 4.2),
    title: str | None = None,
    print_q: bool = True,
    plot_q: bool = False,
    exact: float | None = None,
) -> Dict[str, Any]:
    r"""Automatic-window Gamma-method analysis of a weighted ratio.

    The estimator is

    .. math::

        \hat\mu = \frac{\sum_t w_t E_t}{\sum_t w_t}.

    Its projected fluctuation (influence) series is

    .. math::

        x_t = \frac{w_t(E_t - \hat\mu)}{\bar w}.

    The autocovariance of ``x_t`` therefore includes fluctuations of the
    numerator and denominator as well as their cross-correlation.  The
    summation window is selected by the automatic procedure of U. Wolff,
    Comput. Phys. Commun. 156, 143 (2004), with the paper's default
    ``s_tau=1.5``.  The search is capped at half of the trajectory.

    The standard AFQMC driver evaluates this method alongside blocking and
    uses it as the primary error estimate by default.
    """

    ene_raw = np.asarray(ene)
    wt_raw = np.asarray(wt)
    if np.iscomplexobj(ene_raw) and np.any(np.abs(np.imag(ene_raw)) > 0.0):
        raise ValueError("ene must be real-valued")
    if np.iscomplexobj(wt_raw) and np.any(np.abs(np.imag(wt_raw)) > 0.0):
        raise ValueError("wt must be real-valued")

    ene_np = np.asarray(np.real(ene_raw), dtype=float).ravel()
    wt_np = np.asarray(np.real(wt_raw), dtype=float).ravel()
    n = ene_np.size
    if wt_np.size != n:
        raise ValueError("ene and wt must contain the same number of samples")
    if n < 4:
        raise ValueError("Gamma analysis requires at least four samples")
    if not np.all(np.isfinite(ene_np)) or not np.all(np.isfinite(wt_np)):
        raise ValueError("ene and wt must contain only finite values")
    if not np.isfinite(s_tau) or s_tau <= 0.0:
        raise ValueError("s_tau must be positive and finite")
    if not np.isfinite(min_effective_samples) or min_effective_samples < 0.0:
        raise ValueError("min_effective_samples must be nonnegative and finite")

    search_limit = n // 2
    if max_lag is not None:
        if isinstance(max_lag, bool) or int(max_lag) != max_lag or max_lag < 1:
            raise ValueError("max_lag must be a positive integer")
        search_limit = min(search_limit, int(max_lag))

    weight_sum = float(wt_np.sum())
    weight_abs_sum = float(np.abs(wt_np).sum())
    denominator_scale = max(weight_abs_sum, np.finfo(float).tiny)
    if abs(weight_sum) <= 10.0 * np.finfo(float).eps * denominator_scale:
        raise ValueError("sum(wt) is zero or numerically ill-conditioned")

    mu = float(np.dot(wt_np, ene_np) / weight_sum)
    mean_weight = weight_sum / n
    influence = wt_np * (ene_np - mu) / mean_weight
    autocovariance = _autocovariance_fft(influence, search_limit)
    variance_naive = float(autocovariance[0])
    denominator_fraction = abs(weight_sum) / denominator_scale

    if not np.isfinite(variance_naive) or variance_naive <= 0.0:
        warnings = (
            "projected ratio fluctuations have zero variance; autocorrelation is undefined",
        )
        out = {
            "mu": mu,
            "se_gamma": 0.0,
            "ci95_gamma": (mu, mu),
            "window": 0,
            "window_found": False,
            "tau_int": None,
            "tau_int_error": None,
            "effective_sample_size": None,
            "se_error": None,
            "long_run_variance": 0.0,
            "variance_naive": variance_naive,
            "autocovariance": autocovariance,
            "autocorrelation": np.full_like(autocovariance, np.nan),
            "tau_int_curve": np.full_like(autocovariance, np.nan),
            "tau_exp_curve": np.full_like(autocovariance, np.nan),
            "window_criterion": np.full_like(autocovariance, np.nan),
            "influence": influence,
            "denominator_fraction": denominator_fraction,
            "reliable": False,
            "warnings": warnings,
            "bias": (mu - exact) if exact is not None else None,
            "z_score": None,
        }
        if print_q:
            print(f"mu: {mu:.16g}  Gamma SE: 0  (degenerate projected series)")
        return out

    autocorrelation = autocovariance / variance_naive
    correlation_sum = autocovariance[0] + 2.0 * np.concatenate(
        ([0.0], np.cumsum(autocovariance[1:]))
    )
    tau_curve = correlation_sum / (2.0 * variance_naive)
    window_criterion = np.full(search_limit + 1, np.nan)
    tau_exp_curve = np.full(search_limit + 1, np.nan)

    window = search_limit
    window_found = False
    tiny_tau = np.finfo(float).tiny
    for lag in range(1, search_limit + 1):
        tau_lag = float(tau_curve[lag])
        if not np.isfinite(tau_lag) or tau_lag <= 0.5:
            tau_exp = tiny_tau
            criterion = -tiny_tau
        else:
            ratio = (2.0 * tau_lag + 1.0) / (2.0 * tau_lag - 1.0)
            tau_exp = s_tau / np.log(ratio)
            criterion = np.exp(-lag / tau_exp) - tau_exp / np.sqrt(lag * n)
        tau_exp_curve[lag] = tau_exp
        window_criterion[lag] = criterion
        # A negative truncated variance can occur for strongly alternating
        # finite series.  Keep searching instead of returning an invalid SE.
        if criterion < 0.0 and correlation_sum[lag] > 0.0:
            window = lag
            window_found = True
            break

    long_run_variance_raw = float(correlation_sum[window])
    warnings_list: list[str] = []
    if not window_found:
        warnings_list.append(
            f"automatic window was not found before the maximum lag ({search_limit})"
        )

    if not np.isfinite(long_run_variance_raw) or long_run_variance_raw <= 0.0:
        warnings_list.append("truncated autocovariance sum is not positive")
        se_gamma = float("nan")
        ci95 = (float("nan"), float("nan"))
        tau_int = float("nan")
        tau_int_error = float("nan")
        effective_sample_size = float("nan")
        se_error = float("nan")
        long_run_variance = float("nan")
    else:
        # Wolff Eq. (49): cancel the leading finite-N bias caused by using the
        # sample mean in the autocovariance estimator.
        bias_correction = 1.0 + (2.0 * window + 1.0) / n
        long_run_variance = long_run_variance_raw * bias_correction
        se_gamma = float(np.sqrt(long_run_variance / n))
        ci95 = (mu - 1.96 * se_gamma, mu + 1.96 * se_gamma)
        tau_int = float(tau_curve[window])
        effective_sample_size = float(n / (2.0 * tau_int))
        tau_error_term = max(window + 0.5 - tau_int, 0.0)
        tau_int_error = float(2.0 * tau_int * np.sqrt(tau_error_term / n))
        se_error = float(se_gamma * np.sqrt((window + 0.5) / n))
        if effective_sample_size < min_effective_samples:
            warnings_list.append(
                "effective sample size "
                f"({effective_sample_size:.1f}) is below the requested minimum "
                f"({min_effective_samples:.1f})"
            )

    if denominator_fraction < 0.1:
        warnings_list.append(
            "positive and negative weights strongly cancel in the ratio denominator"
        )

    reliable = bool(
        window_found
        and np.isfinite(se_gamma)
        and np.isfinite(effective_sample_size)
        and effective_sample_size >= min_effective_samples
        and denominator_fraction >= 0.1
    )

    bias = z_score = None
    if exact is not None:
        bias = float(mu - exact)
        if np.isfinite(se_gamma) and se_gamma > 0.0:
            z_score = float(bias / se_gamma)

    out = {
        "mu": mu,
        "se_gamma": se_gamma,
        "ci95_gamma": (float(ci95[0]), float(ci95[1])),
        "window": int(window),
        "window_found": window_found,
        "tau_int": tau_int,
        "tau_int_error": tau_int_error,
        "effective_sample_size": effective_sample_size,
        "se_error": se_error,
        "long_run_variance": long_run_variance,
        "variance_naive": variance_naive,
        "autocovariance": autocovariance,
        "autocorrelation": autocorrelation,
        "tau_int_curve": tau_curve,
        "tau_exp_curve": tau_exp_curve,
        "window_criterion": window_criterion,
        "influence": influence,
        "denominator_fraction": denominator_fraction,
        "reliable": reliable,
        "warnings": tuple(warnings_list),
        "bias": bias,
        "z_score": z_score,
    }

    if print_q:
        print(f"mu: {mu:.16g}  Gamma SE: {se_gamma:.16g}  " f"95% CI: {out['ci95_gamma']}")
        print(
            f"window: {window}  tau_int: {tau_int:.6g} +/- {tau_int_error:.3g}  "
            f"N_eff: {effective_sample_size:.1f}  reliable: {reliable}"
        )
        for warning in warnings_list:
            print(f"warning: {warning}")

    if plot_q:
        import matplotlib.pyplot as plt

        lags = np.arange(search_limit + 1)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        ax1.plot(lags, autocorrelation, marker="o", ms=3, lw=1.2)
        ax1.axhline(0.0, color="k", lw=0.8, alpha=0.5)
        ax1.axvline(window, ls="--", color="k", alpha=0.85, label=f"W = {window}")
        ax1.set_xlabel("lag (AFQMC blocks)")
        ax1.set_ylabel(r"$\rho_x(t)$")
        ax1.set_title(title or "Projected ratio autocorrelation")
        ax1.grid(True, alpha=0.25)
        ax1.legend()

        ax2.plot(lags, tau_curve, marker="o", ms=3, lw=1.2)
        ax2.axvline(window, ls="--", color="k", alpha=0.85, label=f"W = {window}")
        if np.isfinite(tau_int):
            ax2.axhline(tau_int, ls=":", color="k", alpha=0.7)
        ax2.set_xlabel("summation window W")
        ax2.set_ylabel(r"$\tau_\mathrm{int}(W)$")
        ax2.set_title("Integrated autocorrelation time")
        ax2.grid(True, alpha=0.25)
        ax2.legend()
        fig.tight_layout()

    return out


def gamma_analysis_components(
    h0: float | complex,
    weights: np.ndarray | jax.Array,
    components: np.ndarray | jax.Array,
    combine_energy: Callable[[Any, Any], Any],
    *,
    s_tau: float = 1.5,
    max_lag: int | None = None,
    min_effective_samples: float = 20.0,
    figsize: tuple[float, float] = (12, 4.2),
    title: str | None = None,
    print_q: bool = True,
    plot_q: bool = False,
    exact: float | None = None,
) -> Dict[str, Any]:
    r"""Gamma-method analysis for a nonlinear function of component ratios.

    For block components :math:`c_t`, possibly complex weights :math:`w_t`,
    and a real-valued final energy :math:`E=f(\bar c)`, the component mean is

    .. math::

        \bar c = \frac{\sum_t w_t c_t}{\sum_t w_t}.

    The multivariate delta method projects the correlated component and weight
    fluctuations onto the scalar influence series

    .. math::

        x_t = Df_{\bar c}\left[\frac{w_t(c_t-\bar c)}{\bar w}\right].

    The usual automatic-window Gamma analysis is then applied to ``x_t``.
    Complex components are differentiated with respect to their real and
    imaginary parts, and all covariance among component numerators and the
    shared denominator is retained.  The per-block proxy ``f(c_t)`` is not
    used because averaging it would not represent the nonlinear estimator.
    """

    weights_array = np.asarray(weights).reshape(-1)
    components_array = np.asarray(components)
    if components_array.ndim != 2:
        raise ValueError(
            "components must have shape (n_blocks, n_components); "
            f"got {components_array.shape}."
        )
    if components_array.shape[0] != weights_array.shape[0]:
        raise ValueError(
            f"weights length {weights_array.shape[0]} does not match components "
            f"length {components_array.shape[0]}."
        )
    finite = np.isfinite(weights_array) & np.all(np.isfinite(components_array), axis=1)
    weights_array = weights_array[finite]
    components_array = components_array[finite]
    n = int(weights_array.shape[0])
    if n < 4:
        raise ValueError("Gamma analysis requires at least four finite component samples")

    weight_sum = weights_array.sum()
    weight_abs_sum = float(np.abs(weights_array).sum())
    denominator_scale = max(weight_abs_sum, np.finfo(float).tiny)
    if abs(weight_sum) <= 10.0 * np.finfo(float).eps * denominator_scale:
        raise ValueError("sum(weights) is zero or numerically ill-conditioned")
    denominator_fraction = float(abs(weight_sum) / denominator_scale)

    weighted_components = weights_array[:, None] * components_array
    mean_components = weighted_components.sum(axis=0) / weight_sum
    mean_energy = float(
        np.real(np.asarray(combine_energy(h0, mean_components)).reshape(()))
    )

    n_components = int(mean_components.size)
    packed_mean = np.concatenate(
        (np.real(mean_components), np.imag(mean_components))
    ).astype(float, copy=False)

    def real_energy(packed_components):
        component_values = (
            packed_components[:n_components]
            + 1.0j * packed_components[n_components:]
        )
        return jnp.real(combine_energy(h0, component_values))

    component_gradient = np.asarray(
        jax.grad(real_energy)(jnp.asarray(packed_mean)),
        dtype=float,
    )
    mean_weight = weight_sum / n
    component_influence = (
        weights_array[:, None]
        * (components_array - mean_components[None, :])
        / mean_weight
    )
    packed_influence = np.concatenate(
        (np.real(component_influence), np.imag(component_influence)),
        axis=1,
    )
    energy_influence = np.asarray(packed_influence @ component_gradient, dtype=float)

    analysis = gamma_analysis_ratio(
        energy_influence,
        np.ones(n),
        s_tau=s_tau,
        max_lag=max_lag,
        min_effective_samples=min_effective_samples,
        figsize=figsize,
        title=title or "Projected component-estimator autocorrelation",
        print_q=False,
        plot_q=plot_q,
    )
    warnings = list(analysis["warnings"])
    if denominator_fraction < 0.1:
        warnings.append(
            "complex weights strongly cancel in the component-ratio denominator"
        )
    se_gamma = float(analysis["se_gamma"])
    bias = z_score = None
    if exact is not None:
        bias = float(mean_energy - exact)
        if np.isfinite(se_gamma) and se_gamma > 0.0:
            z_score = float(bias / se_gamma)
    analysis.update(
        {
            "mu": mean_energy,
            "mean_components": mean_components,
            "component_gradient": component_gradient,
            "component_influence": component_influence,
            "influence": energy_influence,
            "ci95_gamma": (
                float(mean_energy - 1.96 * se_gamma),
                float(mean_energy + 1.96 * se_gamma),
            ),
            "denominator_fraction": denominator_fraction,
            "reliable": bool(analysis["reliable"] and denominator_fraction >= 0.1),
            "warnings": tuple(warnings),
            "bias": bias,
            "z_score": z_score,
        }
    )

    if print_q:
        tau_int = analysis["tau_int"]
        tau_error = analysis["tau_int_error"]
        tau_text = (
            f"{float(tau_int):.6g} +/- {float(tau_error):.3g}"
            if tau_int is not None and tau_error is not None
            else "unavailable"
        )
        effective_samples = analysis["effective_sample_size"]
        effective_samples_text = (
            f"{float(effective_samples):.1f}"
            if effective_samples is not None
            else "unavailable"
        )
        print(
            f"mu: {mean_energy:.16g}  Gamma SE: {se_gamma:.16g}  "
            f"95% CI: {analysis['ci95_gamma']}"
        )
        print(
            f"window: {analysis['window']}  tau_int: {tau_text}  "
            f"N_eff: {effective_samples_text}  "
            f"reliable: {analysis['reliable']}"
        )
        for warning in warnings:
            print(f"warning: {warning}")

    return analysis


def reject_outliers(
    data: np.ndarray | jax.Array,
    obs: int,
    m: float = 10.0,
    min_threshold: float = 1e-5,
) -> tuple[Any, Any]:
    target = data[:, obs]
    median_val = np.median(target)
    d = np.abs(target - median_val)
    mdev = np.median(d)
    q1, q3 = np.percentile(target, [25, 75])
    iqr = q3 - q1
    normalized_iqr = iqr / 1.349
    dispersion = max(mdev, normalized_iqr, min_threshold)
    s = d / dispersion
    mask = s < m
    return data[mask], mask


def jackknife_ratios(
    num: np.ndarray,
    denom: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Jackknife mean and standard error for a ratio estimator with array valued numerator.

    Parameters
    ----------
    num : np.ndarray
        Numerator samples, shape (n_samples, *obs_shape).
    denom : np.ndarray
        Denominator samples, shape (n_samples,).

    Returns
    -------
    mean : np.ndarray
        Jackknife estimate of the ratio mean, shape (*obs_shape,).
    sigma : np.ndarray
        Jackknife standard error, shape (*obs_shape,).
    """
    num = np.asarray(num)
    denom = np.asarray(denom).ravel()
    n = num.shape[0]
    assert denom.shape[0] == n

    num_sum = num.sum(axis=0)
    denom_sum = denom.sum()

    # leave one out sums
    loo_num = (num_sum - num) / (n - 1)  # (n, *obs_shape)
    d_shape = (n,) + (1,) * (num.ndim - 1)
    loo_denom = (denom_sum - denom).reshape(d_shape) / (n - 1)  # (n, 1, ...)

    loo_ratio = (loo_num / loo_denom).real  # (n, *obs_shape)
    mean = loo_ratio.mean(axis=0)
    sigma = np.sqrt((n - 1) * np.var(loo_ratio, axis=0))
    return mean, sigma


def rebin_observable(
    obs: np.ndarray,
    weights: np.ndarray,
    block_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Rebin block-level observable data into larger super-blocks.

    Parameters
    ----------
    obs : np.ndarray
        Per-block weighted-mean observable, shape ``(n_blocks, *obs_shape)``.
    weights : np.ndarray
        Per-block total weights, shape ``(n_blocks,)``.
    block_size : int
        Number of original blocks per super-block.

    Returns
    -------
    num : np.ndarray
        Super-block numerator sums, shape ``(n_groups, *obs_shape)``.
    denom : np.ndarray
        Super-block denominator sums, shape ``(n_groups,)``.
    """
    obs = np.asarray(obs)
    weights = np.asarray(weights).ravel()
    n = obs.shape[0]
    n_groups = n // block_size
    usable = n_groups * block_size

    w = weights[:usable].reshape(n_groups, block_size)
    w_shape = (n_groups, block_size) + (1,) * (obs.ndim - 1)
    o = obs[:usable].reshape((n_groups, block_size) + obs.shape[1:])

    denom = w.sum(axis=1)  # (n_groups,)
    num = (w.reshape(w_shape) * o).sum(axis=1)  # (n_groups, *obs_shape)
    return num, denom


def clean_pt2ccsd(ept_sp, wt_sp, t2_sp, e0_sp, e1_sp, zeta=20):
    # print(f'Clean AFQMC/pt2CCSD Observation...')
    d = jnp.abs(ept_sp - jnp.median(ept_sp))
    d_med = jnp.median(d)
    d_med = jnp.where(d_med == 0, 1e-10, d_med)
    z = d / d_med
    mask = z < zeta
    print(
        f"Remove outlier blocks zeta {z[~mask]} \n"
        f"                    energy {ept_sp[~mask]} \n"
        f"                    weight {wt_sp.real[~mask]} "
    )

    wt_clean = wt_sp[mask]
    t2_clean = t2_sp[mask]
    e0_clean = e0_sp[mask]
    e1_clean = e1_sp[mask]

    return (wt_clean, t2_clean, e0_clean, e1_clean)


def pt2ccsd_blocking(
    h0, weights, t2_sp, e0_sp, e1_sp, printQ=False, min_blocks=5, plateau_window=2, plateau_tol=0.04
):
    nsample = len(weights)
    max_size = max(1, nsample // min_blocks)

    block_errs = []
    block_means = []
    block_sizes = []

    for block_size in range(1, max_size + 1):
        n_blocks = nsample // block_size
        if n_blocks < min_blocks:
            break

        wt_truncated = weights[: n_blocks * block_size]
        t2_truncated = t2_sp[: n_blocks * block_size]
        e0_truncated = e0_sp[: n_blocks * block_size]
        e1_truncated = e1_sp[: n_blocks * block_size]

        wt_t2 = wt_truncated * t2_truncated
        wt_e0 = wt_truncated * e0_truncated
        wt_e1 = wt_truncated * e1_truncated

        wt = wt_truncated.reshape(n_blocks, block_size)
        wt_t2 = wt_t2.reshape(n_blocks, block_size)
        wt_e0 = wt_e0.reshape(n_blocks, block_size)
        wt_e1 = wt_e1.reshape(n_blocks, block_size)

        block_wt = jnp.sum(wt, axis=1)
        block_t2 = jnp.sum(wt_t2, axis=1) / block_wt
        block_e0 = jnp.sum(wt_e0, axis=1) / block_wt
        block_e1 = jnp.sum(wt_e1, axis=1) / block_wt

        block_energy = (h0 + block_e0 + block_e1 - block_t2 * block_e0).real
        block_mean = jnp.mean(block_energy)
        block_error = jnp.std(block_energy, ddof=1) / jnp.sqrt(n_blocks)

        block_sizes.append(block_size)
        block_means.append(block_mean)
        block_errs.append(block_error)

    # --- Plateau detection ---
    errs = jnp.array(block_errs)
    plateau_idx = None

    if len(errs) >= plateau_window + 1:
        for i in range(1, len(errs) - plateau_window + 1):
            window = errs[i : i + plateau_window]
            rel_changes = jnp.abs(jnp.diff(window) / window[:-1])
            if jnp.all(rel_changes < plateau_tol):
                plateau_idx = i
                break

    if plateau_idx is not None:
        err = jnp.mean(errs[plateau_idx : plateau_idx + plateau_window])
    else:
        err = errs.max()

    # --- Overall energy ---
    wt_avg = jnp.mean(weights)
    t2_avg = jnp.mean(weights * t2_sp) / wt_avg
    e0_avg = jnp.mean(weights * e0_sp) / wt_avg
    e1_avg = jnp.mean(weights * e1_sp) / wt_avg
    energy_avg = h0 + e0_avg + e1_avg - t2_avg * e0_avg

    # --- Printing ---
    if printQ:
        print("Performing Blocking Analysis for AFQMC/pt2CCSD energy...")
        print(f"{'Bsz':>4s}  {'NB':>4s}  {'Nsp':>4s}  {'Energy':>11s}  {'Error':>8s}")

        if plateau_idx is not None:
            print_end = min(len(block_errs), plateau_idx + plateau_window + 3)
        else:
            print_end = len(block_errs)

        for i in range(print_end):
            bs = block_sizes[i]
            nb = nsample // bs
            marker = "  <--" if (plateau_idx is not None and i == plateau_idx) else ""
            print(
                f"{bs:4d}  {nb:4d}  {bs*nb:4d}  {block_means[i]:11.6f}  {block_errs[i]:8.6f}{marker}"
            )

        if plateau_idx is not None:
            print(f"Plateau found at block size {block_sizes[plateau_idx]}, error = {err.real:.6f}")
        else:
            print(f"No plateau found, using max error = {err.real:.6f}")

    return energy_avg.real, err.real
