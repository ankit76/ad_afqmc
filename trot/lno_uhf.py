from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .staging import STAGE_FORMAT_VERSION, HamInput, StagedInputs, TrialInput


_FORGE_INSTALL = (
    "PySCF-Forge ULNO is required; run `python -m pip install -e '.[lno]'` or "
    "`python -m pip install -e /path/to/pyscf-forge`."
)

_CHOL_CUT = 1.0e-5


_CCPVXZ = re.compile(r"^(daug|aug)?ccp(w?c)?v[dtq56]z$")
_RELATIVISTIC_BASIS = re.compile(r"(dk\d?$|dkh\d?|^x2c|anorcc|^zora|^dyall)")


def _basis_names_match(
    basis: Any,
    pattern: re.Pattern[str],
    *,
    require_all: bool,
) -> bool:
    def matches(value: Any) -> bool:
        if not isinstance(value, str):
            return False
        name = re.sub(r"[\s\-_]", "", value.lower())
        return bool(pattern.search(name))

    if isinstance(basis, str):
        return matches(basis)
    if not isinstance(basis, dict) or not basis:
        return False
    values = [matches(value) for value in basis.values()]
    return all(values) if require_all else any(values)


def _is_standard_ccpvxz_basis(basis: Any) -> bool:
    return _basis_names_match(basis, _CCPVXZ, require_all=True)


def _is_relativistic_basis(basis: Any) -> bool:
    return _basis_names_match(basis, _RELATIVISTIC_BASIS, require_all=False)


def free_atom_minao(
    mol: Any,
    *,
    occ_tol: float = 1.0e-6,
    sv_tol: float = 1.0e-8,
    x2c: bool | None = None,
) -> dict[str, list[Any]]:
    """Build an occupied free-atom reference basis for IAO construction."""
    from pyscf import gto
    from pyscf.scf import atom_hf

    if np.asarray(mol._ecpbas).ndim != 2:
        mol._ecpbas = np.zeros((0, gto.BAS_SLOTS), dtype=np.int32)
    if x2c is None:
        x2c = _is_relativistic_basis(mol.basis)

    elements = tuple(dict.fromkeys(mol.elements))
    uncontracted = {symbol: gto.uncontract(mol._basis[symbol]) for symbol in elements}
    ecp = getattr(mol, "_ecp", None) or {}
    first_atom = {}
    for atom_index, symbol in enumerate(mol.elements):
        first_atom.setdefault(symbol, atom_index)

    reference_basis: dict[str, list[Any]] = {}
    for symbol in elements:
        atom_ecp = {symbol: ecp[symbol]} if symbol in ecp else {}
        nelec = gto.charge(symbol) - mol.atom_nelec_core(first_atom[symbol])
        atom = gto.M(
            atom=f"{symbol} 0 0 0",
            basis={symbol: uncontracted[symbol]},
            ecp=atom_ecp,
            spin=nelec % 2,
            verbose=0,
        )
        if atom.nelectron != nelec:
            raise RuntimeError(
                f"free atom {symbol} has {atom.nelectron} electrons; expected {nelec}."
            )

        ao_loc = atom.ao_loc_nr()
        shells_by_l: dict[int, list[tuple[float, int]]] = defaultdict(list)
        for shell in range(atom.nbas):
            shells_by_l[atom.bas_angular(shell)].append(
                (float(atom.bas_exp(shell)[0]), int(ao_loc[shell]))
            )

        atomic_hf = (
            atom_hf.AtomHF1e(atom)
            if atom.nelectron == 1
            else atom_hf.AtomSphAverageRHF(atom)
        )
        if x2c:
            atomic_hf = atomic_hf.sfx2c1e()
        atomic_hf.run()
        occupied = np.asarray(atomic_hf.mo_coeff)[
            :, np.asarray(atomic_hf.mo_occ) > occ_tol
        ]

        reference_shells: list[Any] = []
        for angular_momentum in sorted(shells_by_l):
            entries = shells_by_l[angular_momentum]
            exponents = np.asarray([exponent for exponent, _ in entries])
            starts = [start for _, start in entries]
            ncomp = 2 * angular_momentum + 1
            radial = np.stack(
                [occupied[start : start + ncomp] for start in starts]
            ).reshape(len(entries), -1)
            left, singular_values, _ = np.linalg.svd(radial, full_matrices=False)
            if singular_values.size == 0:
                continue
            for index in np.flatnonzero(singular_values > sv_tol * singular_values[0]):
                contraction = left[:, index]
                reference_shells.append(
                    [angular_momentum]
                    + [
                        [float(exponents[k]), float(contraction[k])]
                        for k in range(len(entries))
                    ]
                )
        reference_basis[symbol] = reference_shells
    return reference_basis


def default_iao_minao(mol: Any) -> str | dict[str, list[Any]]:
    """Match the historical LNO IAO reference choice for the molecular basis."""
    if _is_standard_ccpvxz_basis(mol.basis):
        return "minao"
    x2c = _is_relativistic_basis(mol.basis)
    label = "scalar-X2C free-atom" if x2c else "free-atom"
    print(f"[lno] building {label} IAO reference basis.")
    return free_atom_minao(mol, x2c=x2c)


@dataclass(frozen=True, slots=True)
class UhfIaoFragments:
    coeff: tuple[NDArray, NDArray]
    fragments: tuple[tuple[NDArray[np.int64], NDArray[np.int64]], ...]


@dataclass(frozen=True, slots=True)
class JointDfCholesky:
    chol_a: NDArray
    chol_b: NDArray
    residual_max: float


def make_uhf_iao_fragments(
    mf: Any,
    atom_fragments: Sequence[Sequence[int]],
    *,
    frozen: Any = 0,
) -> UhfIaoFragments:
    """Build spin-resolved IAOs and map reference IAOs to atom fragments."""
    try:
        from pyscf import lo, mp
        from pyscf.lno import tools as lno_tools
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(_FORGE_INSTALL) from exc

    mo_coeff = np.asarray(mf.mo_coeff)
    mo_occ = np.asarray(mf.mo_occ)
    if mo_coeff.ndim != 3 or mo_coeff.shape[0] != 2:
        raise TypeError("make_uhf_iao_fragments requires a UHF reference.")

    minao = default_iao_minao(mf.mol)
    frozen_mask = mp.UMP2(mf, frozen=frozen).get_frozen_mask()
    overlap = np.asarray(mf.get_ovlp())
    coeff: list[NDArray] = []
    for spin in range(2):
        occupied = (mo_occ[spin] > 1e-10) & frozen_mask[spin]
        iao = lo.iao.iao(mf.mol, mo_coeff[spin][:, occupied], minao=minao)
        coeff.append(np.asarray(lo.orth.vec_lowdin(iao, overlap)))

    reference = lo.iao.reference_mol(mf.mol, minao=minao)
    indices = lno_tools.autofrag_iao(reference, frag_atmlist=atom_fragments)
    if (
        coeff[0].shape[1] != reference.nao_nr()
        or coeff[1].shape[1] != reference.nao_nr()
    ):
        raise RuntimeError("IAO count does not match the minimal reference basis.")

    fragments = tuple(
        (
            np.asarray(fragment, dtype=np.int64),
            np.asarray(fragment, dtype=np.int64),
        )
        for fragment in indices
    )
    return UhfIaoFragments(coeff=(coeff[0], coeff[1]), fragments=fragments)


def _as_packed_df(df: ArrayLike) -> NDArray:
    array = np.asarray(df)
    if array.ndim == 2:
        return array
    if array.ndim != 3 or array.shape[1] != array.shape[2]:
        raise ValueError("DF tensors must have shape (naux,npair) or (naux,norb,norb).")
    rows, cols = np.tril_indices(array.shape[1])
    return array[:, rows, cols]


def _unpack_symmetric(packed: NDArray, norb: int) -> NDArray:
    expected = norb * (norb + 1) // 2
    if packed.ndim != 2 or packed.shape[1] != expected:
        raise ValueError(f"packed tensor must have {expected} columns, got {packed.shape}.")
    output = np.zeros((packed.shape[0], norb, norb), dtype=packed.dtype)
    rows, cols = np.tril_indices(norb)
    output[:, rows, cols] = packed
    output[:, cols, rows] = packed
    return output


def joint_df_pair_cholesky(
    df_a: ArrayLike,
    df_b: ArrayLike,
    *,
    chol_cut: float = 1e-5,
    max_chol: int | None = None,
) -> JointDfCholesky:
    """Factor the joint A/B pair Gram matrix without forming that matrix."""
    a = _as_packed_df(df_a)
    b = _as_packed_df(df_b)
    if a.shape[0] != b.shape[0]:
        raise ValueError("alpha and beta DF tensors must use the same auxiliary basis.")
    if chol_cut <= 0:
        raise ValueError("chol_cut must be positive.")

    naux = int(a.shape[0])
    npa, npb = int(a.shape[1]), int(b.shape[1])
    npair = npa + npb
    rank_limit = min(naux, npair) if max_chol is None else min(int(max_chol), npair)
    if rank_limit <= 0:
        raise ValueError("max_chol must be positive.")

    diag = np.concatenate(
        (
            np.einsum("pi,pi->i", a.conj(), a, optimize=True).real,
            np.einsum("pi,pi->i", b.conj(), b, optimize=True).real,
        )
    )
    capacity = min(rank_limit, 64)
    factors = np.empty((npair, capacity), dtype=np.result_type(a, b))
    rank = 0

    while rank < rank_limit:
        pivot = int(np.argmax(diag))
        if diag[pivot] <= chol_cut:
            break
        pivot_df = a[:, pivot] if pivot < npa else b[:, pivot - npa]
        column = np.concatenate((a.conj().T @ pivot_df, b.conj().T @ pivot_df))
        if rank:
            column -= factors[:, :rank] @ factors[pivot, :rank].conj()
        pivot_residual = float(column[pivot].real)
        if pivot_residual <= chol_cut:
            diag[pivot] = max(0.0, pivot_residual)
            continue

        if rank == factors.shape[1]:
            new_capacity = min(rank_limit, max(rank + 1, 2 * rank))
            grown = np.empty((npair, new_capacity), dtype=factors.dtype)
            grown[:, :rank] = factors
            factors = grown
        factors[:, rank] = column / np.sqrt(pivot_residual)
        diag -= np.abs(factors[:, rank]) ** 2
        np.maximum(diag, 0.0, out=diag)
        rank += 1

    residual_max = float(np.max(diag, initial=0.0))
    if residual_max > chol_cut and rank == rank_limit:
        raise RuntimeError(
            f"joint Cholesky reached rank {rank_limit} with residual {residual_max:.3e}."
        )

    rows = factors[:, :rank].T
    na = _pair_dimension(npa)
    nb = _pair_dimension(npb)
    return JointDfCholesky(
        chol_a=_unpack_symmetric(rows[:, :npa], na),
        chol_b=_unpack_symmetric(rows[:, npa:], nb),
        residual_max=residual_max,
    )


def _pair_dimension(npair: int) -> int:
    norb = int((np.sqrt(8 * npair + 1) - 1) // 2)
    if norb * (norb + 1) // 2 != npair:
        raise ValueError(f"{npair} is not a triangular pair dimension.")
    return norb


def _df_block_to_pairs(block: NDArray, coeff: NDArray) -> NDArray:
    nao, norb = coeff.shape
    if np.iscomplexobj(coeff):
        from pyscf import lib

        ao = lib.unpack_tril(block) if block.shape[1] != nao * nao else block.reshape(-1, nao, nao)
        transformed = np.einsum(
            "Pmn,mp,nq->Ppq", ao, coeff.conj(), coeff, optimize=True
        )
    else:
        from pyscf.ao2mo import _ao2mo

        if block.shape[1] == nao * (nao + 1) // 2:
            mo = np.asarray(coeff, order="F")
            transformed = _ao2mo.nr_e2(
                block,
                mo,
                (0, norb, 0, norb),
                aosym="s2",
                mosym="s1",
            ).reshape(-1, norb, norb)
        else:
            ao = block.reshape(-1, nao, nao)
            transformed = np.einsum("Pmn,mp,nq->Ppq", ao, coeff, coeff, optimize=True)
    rows, cols = np.tril_indices(norb)
    return np.asarray(transformed[:, rows, cols])


def build_joint_df_cholesky(
    mf: Any,
    coeff: tuple[NDArray, NDArray],
    *,
    chol_cut: float = 1e-5,
    max_chol: int | None = None,
) -> JointDfCholesky:
    """Transform molecular DF tensors and jointly factor the two pair spaces."""
    if any(np.iscomplexobj(block) for block in coeff):
        raise NotImplementedError(
            "split-LIS AFQMC currently requires real alpha and beta orbital coefficients."
        )
    with_df = getattr(mf, "with_df", None)
    if with_df is None:
        raise TypeError("UHF LNO preparation requires a density-fitted mean field.")
    naux = int(with_df.get_naoaux())
    npa = coeff[0].shape[1] * (coeff[0].shape[1] + 1) // 2
    npb = coeff[1].shape[1] * (coeff[1].shape[1] + 1) // 2
    dtype = np.result_type(coeff[0], coeff[1], np.float64)
    df_a = np.empty((naux, npa), dtype=dtype)
    df_b = np.empty((naux, npb), dtype=dtype)
    offset = 0
    for raw_block in with_df.loop():
        block = np.asarray(raw_block)
        stop = offset + block.shape[0]
        df_a[offset:stop] = _df_block_to_pairs(block, coeff[0])
        df_b[offset:stop] = _df_block_to_pairs(block, coeff[1])
        offset = stop
    if offset != naux:
        raise RuntimeError(f"DF iterator yielded {offset} auxiliaries; expected {naux}.")
    return joint_df_pair_cholesky(df_a, df_b, chol_cut=chol_cut, max_chol=max_chol)


def spin_frozen_core_h1(
    mf: Any,
    core_coeff: tuple[NDArray, NDArray],
    active_coeff: tuple[NDArray, NDArray],
) -> tuple[float, tuple[NDArray, NDArray]]:
    """Fold spin-dependent frozen occupied spaces into h0 and h1."""
    hcore = np.asarray(mf.get_hcore())
    if hcore.ndim == 2:
        hcore_spin = (hcore, hcore)
    elif hcore.shape[0] == 2:
        hcore_spin = (hcore[0], hcore[1])
    else:
        raise ValueError(f"unexpected hcore shape {hcore.shape}.")

    density = tuple(c @ c.conj().T for c in core_coeff)
    veff = np.asarray(mf.get_veff(mf.mol, np.asarray(density)))
    if veff.shape[0] != 2:
        raise ValueError(f"unexpected UHF core potential shape {veff.shape}.")

    h0 = complex(mf.energy_nuc())
    h1: list[NDArray] = []
    for spin in range(2):
        h0 += np.einsum("ij,ji->", density[spin], hcore_spin[spin])
        h0 += 0.5 * np.einsum("ij,ji->", density[spin], veff[spin])
        c = active_coeff[spin]
        h1.append(np.asarray(c.conj().T @ (hcore_spin[spin] + veff[spin]) @ c))
    return float(np.real_if_close(h0)), (h1[0], h1[1])


def _pad_orbitals(array: ArrayLike, size: int) -> NDArray:
    array = np.asarray(array)
    padding = [(0, 0)] * array.ndim
    padding[-2:] = [(0, size - array.shape[-2]), (0, size - array.shape[-1])]
    return np.pad(array, padding)


def _pad_uccsd_amplitudes(
    t1: tuple[ArrayLike, ArrayLike],
    t2: tuple[ArrayLike, ArrayLike, ArrayLike],
    norb_spin: tuple[int, int],
) -> tuple[tuple[NDArray, NDArray], tuple[NDArray, NDArray, NDArray]]:
    """Zero-pad the virtual axes to the shared runtime orbital dimension."""
    pad_a = max(norb_spin) - norb_spin[0]
    pad_b = max(norb_spin) - norb_spin[1]
    t1a, t1b = map(np.asarray, t1)
    t2aa, t2ab, t2bb = map(np.asarray, t2)
    return (
        (
            np.pad(t1a, ((0, 0), (0, pad_a))),
            np.pad(t1b, ((0, 0), (0, pad_b))),
        ),
        (
            np.pad(t2aa, ((0, 0), (0, 0), (0, pad_a), (0, pad_a))),
            np.pad(t2ab, ((0, 0), (0, 0), (0, pad_a), (0, pad_b))),
            np.pad(t2bb, ((0, 0), (0, 0), (0, pad_b), (0, pad_b))),
        ),
    )


def _make_pt2uccsd_trial(
    t1: tuple[ArrayLike, ArrayLike],
    t2: tuple[ArrayLike, ArrayLike, ArrayLike],
    norb_spin: tuple[int, int],
    uocc_loc: tuple[ArrayLike, ArrayLike],
) -> TrialInput:
    """Build an exact-T1, linear-connected-T2 trial for a split LNO space."""
    t1_padded, t2_padded = _pad_uccsd_amplitudes(t1, t2, norb_spin)
    size = max(norb_spin)
    dtype = np.result_type(*t1_padded, *t2_padded)
    weight_a, weight_b = (
        np.asarray(u) @ np.asarray(u).conj().T for u in uocc_loc
    )
    return TrialInput(
        kind="pt2uccsd",
        data={
            "t1a": t1_padded[0],
            "t1b": t1_padded[1],
            "t2aa": t2_padded[0],
            "t2ab": t2_padded[1],
            "t2bb": t2_padded[2],
            "mo_coeff_b": np.eye(size, dtype=dtype),
            "weight_a": weight_a,
            "weight_b": weight_b,
        },
        frozen=0,
        source_kind="cc",
    )


def _threshold_pair(lno_thresh: float | Sequence[float]) -> tuple[float, float]:
    if np.isscalar(lno_thresh):
        threshold = float(lno_thresh)
        if threshold <= 0.0:
            raise ValueError("lno_thresh must be positive.")
        return 10.0 * threshold, threshold
    values = tuple(float(value) for value in lno_thresh)
    if len(values) != 2:
        raise ValueError("lno_thresh must be a scalar or (occupied,virtual).")
    if values[0] <= 0.0 or values[1] <= 0.0:
        raise ValueError("LNO occupied and virtual thresholds must be positive.")
    return values[0], values[1]


def _frozen_indices(value: Any, nmo: int) -> NDArray[np.int64]:
    array = np.asarray(value)
    if array.ndim == 0:
        count = int(array)
        if count < 0 or count > nmo:
            raise ValueError(f"invalid frozen count {count} for {nmo} orbitals.")
        return np.arange(count, dtype=np.int64)
    indices = np.asarray(array, dtype=np.int64).reshape(-1)
    if np.any(indices < 0) or np.any(indices >= nmo):
        raise ValueError("frozen orbital index is out of range.")
    return np.unique(indices)


def prepare_uhf_lno_fragment(
    mf: Any,
    iaos: UhfIaoFragments,
    fragment_index: int,
    *,
    frozen: Any = 0,
    lno_thresh: float | Sequence[float] = 1e-5,
) -> StagedInputs:
    """Prepare one split-space exact-T1/linear-T2 LNO calculation."""
    try:
        from pyscf.lno.ulnoccsd import UCCSD, ULNOCCSD, get_fragment_energy
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(_FORGE_INSTALL) from exc

    if not 0 <= fragment_index < len(iaos.fragments):
        raise IndexError(
            f"fragment_index={fragment_index} is outside [0, {len(iaos.fragments)})."
        )
    if any(np.iscomplexobj(block) for block in np.asarray(mf.mo_coeff)):
        raise NotImplementedError(
            "split-LIS AFQMC currently requires real alpha and beta orbital coefficients."
        )
    occ_threshold, vir_threshold = _threshold_pair(lno_thresh)
    mlno = ULNOCCSD(
        mf,
        list(iaos.coeff),
        [[list(a), list(b)] for a, b in iaos.fragments],
        lno_type=["1h", "1h"],
        lno_thresh=[occ_threshold, vir_threshold],
        frozen=frozen,
    )
    eris = mlno.ao2mo()
    index_a, index_b = iaos.fragments[fragment_index]
    orbloc = [iaos.coeff[0][:, index_a], iaos.coeff[1][:, index_b]]
    lno_param = [
        [
            {"thresh": occ_threshold, "pct_occ": None, "norb": None},
            {"thresh": vir_threshold, "pct_occ": None, "norb": None},
        ]
        for _ in range(2)
    ]
    full_coeff, frozen_lno, uocc_loc, _ = mlno.make_las(
        eris, orbloc, ["1h", "1h"], lno_param
    )

    mo_occ = np.asarray(mf.mo_occ)
    active_coeff: list[NDArray] = []
    core_coeff: list[NDArray] = []
    cc_frozen: list[list[int]] = []
    for spin in range(2):
        indices = _frozen_indices(frozen_lno[spin], full_coeff[spin].shape[1])
        active = np.ones(full_coeff[spin].shape[1], dtype=bool)
        active[indices] = False
        occupied = mo_occ[spin] > 1e-10
        core_coeff.append(np.asarray(full_coeff[spin][:, ~active & occupied]))
        active_coeff.append(np.asarray(full_coeff[spin][:, active]))
        cc_frozen.append(indices.tolist())

    mlno._precompute()
    mcc = UCCSD(mf, mo_coeff=full_coeff, frozen=tuple(cc_frozen)).set(
        verbose=mlno.verbose_imp
    )
    mcc._s1e = mlno._s1e
    mcc._h1e = mlno._h1e
    mcc._vhf = mlno._vhf
    cc_eris = mcc.ao2mo()
    _, t1_mp2, t2_mp2 = mcc.init_amps(eris=cc_eris)
    projector = [np.asarray(u).T.conj() for u in uocc_loc]
    mp2_correlation = get_fragment_energy(
        cc_eris, t1_mp2, t2_mp2, projector
    ).real
    _, t1_raw, t2_raw = mcc.kernel(
        eris=cc_eris, t1=t1_mp2, t2=t2_mp2
    )
    if not mcc.converged:
        raise RuntimeError("fragment UCCSD did not converge.")
    ccsd_correlation = get_fragment_energy(
        cc_eris, t1_raw, t2_raw, projector
    ).real
    t1 = (np.asarray(t1_raw[0]), np.asarray(t1_raw[1]))
    t2 = (np.asarray(t2_raw[0]), np.asarray(t2_raw[1]), np.asarray(t2_raw[2]))

    active_pair = (active_coeff[0], active_coeff[1])
    core_pair = (core_coeff[0], core_coeff[1])
    h0, h1 = spin_frozen_core_h1(mf, core_pair, active_pair)
    joint = build_joint_df_cholesky(mf, active_pair, chol_cut=_CHOL_CUT)
    norb_spin = (active_pair[0].shape[1], active_pair[1].shape[1])
    size = max(norb_spin)
    nelec = (t1[0].shape[0], t1[1].shape[0])
    ham = HamInput(
        h0=h0,
        h1=_pad_orbitals(h1[0], size),
        h1_b=_pad_orbitals(h1[1], size),
        chol=_pad_orbitals(joint.chol_a, size),
        chol_b=_pad_orbitals(joint.chol_b, size),
        nelec=nelec,
        norb=size,
        norb_spin=norb_spin,
        chol_cut=_CHOL_CUT,
        frozen=0,
        source_kind="cc",
        basis="unrestricted",
    )
    uocc_pair = (np.asarray(uocc_loc[0]), np.asarray(uocc_loc[1]))
    trial = _make_pt2uccsd_trial(t1, t2, norb_spin, uocc_pair)
    return StagedInputs(
        ham=ham,
        trial=trial,
        meta={
            "format_version": STAGE_FORMAT_VERSION,
            "source_kind": "lno_uccsd",
            "fragment_index": int(fragment_index),
            "lno_thresh": [occ_threshold, vir_threshold],
            "mp2_correlation_energy": float(mp2_correlation),
            "uccsd_correlation_energy": float(ccsd_correlation),
            "cholesky_residual_max": joint.residual_max,
        },
    )


def _canonical_uhf_spaces(
    mf: Any,
    frozen: int,
) -> tuple[
    int,
    tuple[NDArray, NDArray],
    tuple[NDArray, NDArray],
    tuple[int, int],
    tuple[int, int],
]:
    """Validate and partition canonical UHF orbitals into core and active spaces."""
    if not isinstance(frozen, (int, np.integer)) or int(frozen) < 0:
        raise ValueError("canonical UHF preparation requires a nonnegative frozen count.")
    frozen = int(frozen)
    mo_coeff = np.asarray(mf.mo_coeff)
    mo_occ = np.asarray(mf.mo_occ)
    if mo_coeff.ndim != 3 or mo_coeff.shape[0] != 2:
        raise TypeError("canonical UHF preparation requires a UHF reference.")
    if np.iscomplexobj(mo_coeff):
        raise NotImplementedError(
            "split canonical AFQMC currently requires real alpha and beta orbitals."
        )
    if frozen > min(int(np.count_nonzero(mo_occ[0])), int(np.count_nonzero(mo_occ[1]))):
        raise ValueError("frozen core exceeds the occupied alpha or beta space.")

    active_pair = (
        np.asarray(mo_coeff[0][:, frozen:]),
        np.asarray(mo_coeff[1][:, frozen:]),
    )
    core_pair = (
        np.asarray(mo_coeff[0][:, :frozen]),
        np.asarray(mo_coeff[1][:, :frozen]),
    )
    nelec = tuple(
        int(np.count_nonzero(mo_occ[spin, frozen:] > 1.0e-10)) for spin in range(2)
    )
    norb_spin = (active_pair[0].shape[1], active_pair[1].shape[1])
    return frozen, active_pair, core_pair, nelec, norb_spin


def _build_canonical_uhf_hamiltonian(
    mf: Any,
    active_pair: tuple[NDArray, NDArray],
    core_pair: tuple[NDArray, NDArray],
    nelec: tuple[int, int],
    norb_spin: tuple[int, int],
    *,
    source_kind: str,
) -> HamInput:
    h0, h1 = spin_frozen_core_h1(mf, core_pair, active_pair)
    joint = build_joint_df_cholesky(mf, active_pair, chol_cut=_CHOL_CUT)
    size = max(norb_spin)
    return HamInput(
        h0=h0,
        h1=_pad_orbitals(h1[0], size),
        h1_b=_pad_orbitals(h1[1], size),
        chol=_pad_orbitals(joint.chol_a, size),
        chol_b=_pad_orbitals(joint.chol_b, size),
        nelec=nelec,
        norb=size,
        norb_spin=norb_spin,
        chol_cut=_CHOL_CUT,
        frozen=0,
        source_kind=source_kind,
        basis="unrestricted",
    )


def prepare_uhf_canonical_hamiltonian(
    mf: Any,
    *,
    frozen: int = 0,
) -> HamInput:
    """Prepare the canonical split-UHF Hamiltonian without running UCCSD."""
    frozen, active_pair, core_pair, nelec, norb_spin = _canonical_uhf_spaces(
        mf, frozen
    )
    return _build_canonical_uhf_hamiltonian(
        mf,
        active_pair,
        core_pair,
        nelec,
        norb_spin,
        source_kind="mf",
    )


def prepare_uhf_canonical(
    mf: Any,
    *,
    frozen: int = 0,
) -> StagedInputs:
    """Prepare full canonical UCCSD inputs in split UHF orbital bases."""
    from pyscf import cc

    frozen, active_pair, core_pair, nelec, norb_spin = _canonical_uhf_spaces(
        mf, frozen
    )
    mcc = cc.UCCSD(mf, frozen=frozen)
    e_corr, t1_raw, t2_raw = mcc.kernel()
    if not mcc.converged:
        raise RuntimeError("canonical UCCSD did not converge.")
    t1 = (np.asarray(t1_raw[0]), np.asarray(t1_raw[1]))
    t2 = (np.asarray(t2_raw[0]), np.asarray(t2_raw[1]), np.asarray(t2_raw[2]))

    if any(
        t1[spin].shape != (nelec[spin], norb_spin[spin] - nelec[spin])
        for spin in range(2)
    ):
        raise RuntimeError("canonical UCCSD amplitudes do not match the active orbital spaces.")

    ham = _build_canonical_uhf_hamiltonian(
        mf,
        active_pair,
        core_pair,
        nelec,
        norb_spin,
        source_kind="cc",
    )
    uocc_pair = (np.eye(nelec[0]), np.eye(nelec[1]))
    trial = _make_pt2uccsd_trial(t1, t2, norb_spin, uocc_pair)
    meta = {
        "format_version": STAGE_FORMAT_VERSION,
        "source_kind": "canonical_uccsd",
        "uccsd_correlation_energy": float(e_corr),
    }
    return StagedInputs(ham=ham, trial=trial, meta=meta)


__all__ = [
    "JointDfCholesky",
    "UhfIaoFragments",
    "build_joint_df_cholesky",
    "default_iao_minao",
    "free_atom_minao",
    "joint_df_pair_cholesky",
    "make_uhf_iao_fragments",
    "prepare_uhf_canonical",
    "prepare_uhf_canonical_hamiltonian",
    "prepare_uhf_lno_fragment",
    "spin_frozen_core_h1",
]
