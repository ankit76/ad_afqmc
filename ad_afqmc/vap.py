from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
import optax
from jax import jit


def apply_sz_projector(
    input_ket: jnp.ndarray, m: float, n_grid: int
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return sz projected determinants and coefficients for a single ket."""
    norb = input_ket.shape[0] // 2

    gammas = 2.0 * jnp.pi * jnp.arange(n_grid) / n_grid
    exp_factors = jnp.exp(1.0j * gammas / 2.0)
    exp_neg_factors = jnp.exp(-1.0j * gammas / 2.0)
    m_factors = jnp.exp(-1.0j * gammas * m) / n_grid

    def projector_for_gamma(ig):
        rot_diag = jnp.concatenate(
            [jnp.ones(norb) * exp_factors[ig], jnp.ones(norb) * exp_neg_factors[ig]]
        )
        return input_ket * rot_diag[:, None], m_factors[ig]

    return jax.vmap(projector_for_gamma, in_axes=0, out_axes=(0, 0))(jnp.arange(n_grid))


def build_rotchol(psi_t_conj: jnp.ndarray, chol: jnp.ndarray) -> jnp.ndarray:
    """
    Precompute the Cholesky-rotated matrices used for two-body energy evaluation.

    Args:
        psi_t_conj: Trial determinant conjugate transpose, shape (n_occ, 2 * n_orb).
        chol: Cholesky tensors, shape (n_chol, n_orb, n_orb).

    Returns:
        rotchol: Array with shape (2, n_chol, n_occ, n_orb) containing alpha/beta blocks.
    """
    _, norb, _ = chol.shape

    psi_alpha = psi_t_conj[:, :norb]
    psi_beta = psi_t_conj[:, norb:]

    rot_alpha = jnp.einsum("ij,njk->nik", psi_alpha, chol)
    rot_beta = jnp.einsum("ij,njk->nik", psi_beta, chol)

    rotchol = jnp.stack([rot_alpha, rot_beta], axis=0)
    return rotchol


def _evaluate_linear_expansion_energy(
    psi: jnp.ndarray, kets: jnp.ndarray, coeffs: jnp.ndarray, ham: hamiltonian
) -> jnp.ndarray:
    """Compute projected energy for a linear combination of determinants."""
    context = ham.prepare(psi)
    psi_t_conj = psi.T.conj()

    def compute_overlap(ket):
        ovlp_mat = psi_t_conj @ ket
        return jsp.linalg.det(ovlp_mat)

    def compute_energy(ket):
        return ham.mixed_energy(psi, ket, context)

    overlaps = jax.vmap(compute_overlap)(kets)
    energies = jax.vmap(compute_energy)(kets)

    weighted_terms = coeffs * overlaps * energies
    enum = jnp.sum(weighted_terms)
    ovlp = jnp.sum(coeffs * overlaps)

    return (enum / ovlp).real


def get_energy_chol(
    bra: jnp.ndarray,
    ket: jnp.ndarray,
    h1: jnp.ndarray,
    rotchol: jnp.ndarray,
    enuc: float,
) -> jnp.ndarray:
    """Return the mixed estimator <bra|H|ket>/<bra|ket> for a pair of determinants."""
    norb = h1[0].shape[0]
    n_chol = rotchol[0].shape[0]
    nocc = bra.shape[1]

    # Calculate Green's function.
    Ghalf = ket @ jnp.linalg.inv(bra.T.conj() @ ket)
    Ghalfa = Ghalf[:norb]
    Ghalfb = Ghalf[norb:]
    G = Ghalf @ bra.T.conj()

    # Core energy.
    energy = enuc

    # One-body energy,
    e1 = jnp.trace(h1[0] @ G[:norb, :norb]) + jnp.trace(h1[1] @ G[norb:, norb:])
    energy += e1

    # Two-body energy.
    def scan_fun(i, carry):
        ej, ek = carry
        W = jnp.zeros((2, nocc, nocc), dtype=jnp.complex128)
        W0 = rotchol[0, i] @ Ghalfa
        W1 = rotchol[1, i] @ Ghalfb
        ej += 0.5 * (jnp.trace(W0) + jnp.trace(W1)) ** 2
        ek -= 0.5 * (
            jnp.sum(W0 * W0.T)
            + jnp.sum(W0 * W1.T)
            + jnp.sum(W1 * W0.T)
            + jnp.sum(W1 * W1.T)
        )
        return ej, ek

    init = (0.0 + 0.0j, 0.0 + 0.0j)

    ej, ek = jax.lax.fori_loop(0, n_chol, scan_fun, init)
    energy += ej + ek
    return energy


def _mixed_green(bra: jnp.ndarray, ket: jnp.ndarray) -> jnp.ndarray:
    return (ket @ jnp.linalg.inv(bra.T.conj() @ ket) @ bra.T.conj()).T


def get_energy_hubbard(
    bra: jnp.ndarray,
    ket: jnp.ndarray,
    h1: jnp.ndarray,
    u: float,
    enuc: float = 0.0,
) -> jnp.ndarray:
    """Return the mixed estimator <bra|H|ket>/<bra|ket> for a pair of determinants."""
    norb = h1[0].shape[0]
    green = _mixed_green(bra, ket)
    energy_1 = jnp.sum(green[:norb, :norb] * h1[0]) + jnp.sum(
        green[norb:, norb:] * h1[1]
    )
    energy_2 = u * (
        jnp.sum(green[:norb, :norb].diagonal() * green[norb:, norb:].diagonal())
        - jnp.sum(green[:norb, norb:].diagonal() * green[norb:, :norb].diagonal())
    )
    return energy_1 + energy_2 + enuc


class hamiltonian:
    """Abstract Hamiltonian interface."""

    def prepare(self, bra: jnp.ndarray) -> dict:
        """Precompute any data derived from the bra determinant."""
        return {}

    def mixed_energy(
        self, bra: jnp.ndarray, ket: jnp.ndarray, context: dict | None = None
    ) -> jnp.ndarray:
        """Return <bra|H|ket>/<bra|ket> using optional cached context."""
        raise NotImplementedError


@dataclass
class cholesky_hamiltonian(hamiltonian):
    """Hamiltonian defined by one-body integrals, Cholesky vectors, and core energy."""

    h1: jnp.ndarray
    chol: jnp.ndarray
    enuc: float

    def prepare(self, bra: jnp.ndarray) -> dict:
        norb = self.h1[0].shape[0]
        psi_t_conj = bra.T.conj()
        chol = self.chol.reshape((-1, norb, norb))
        rotchol = build_rotchol(psi_t_conj, chol)
        return {"rotchol": rotchol}

    def mixed_energy(
        self, bra: jnp.ndarray, ket: jnp.ndarray, context: dict | None = None
    ) -> jnp.ndarray:
        assert context is not None, "cholesky_hamiltonian requires cached context."
        return get_energy_chol(bra, ket, self.h1, context["rotchol"], self.enuc)


@dataclass
class hubbard_hamiltonian(hamiltonian):
    """Hubbard Hamiltonian with on-site interaction U."""

    h1: jnp.ndarray
    u: float
    enuc: float = 0.0

    def mixed_energy(
        self, bra: jnp.ndarray, ket: jnp.ndarray, context: dict | None = None
    ) -> jnp.ndarray:
        return get_energy_hubbard(bra, ket, self.h1, self.u, self.enuc)


def _spin_rot_elements(
    alpha: float, beta: float, gamma: float
) -> tuple[jnp.ndarray, ...]:
    """
    Spin-1/2 rotation (z-y-z Euler angles):
    U = e^{-i alpha sigma_z/2} e^{-i beta sigma_y/2} e^{-i gamma sigma_z/2}
      = [[e^{-i(alpha+gamma)/2} cos(beta/2),   -e^{-i(alpha-gamma)/2} sin(beta/2)],
         [e^{ i(alpha-gamma)/2} sin(beta/2),    e^{ i(alpha+gamma)/2} cos(beta/2)]]
    Returns the four scalar elements u11, u12, u21, u22.
    """
    half = 0.5
    cb = jnp.cos(beta * half)
    sb = jnp.sin(beta * half)
    e_p = jnp.exp(-0.5j * (alpha + gamma))
    e_m = jnp.exp(-0.5j * (alpha - gamma))
    u11 = e_p * cb
    u12 = -e_m * sb
    u21 = jnp.conj(e_m) * sb
    u22 = jnp.conj(e_p) * cb
    return u11, u12, u21, u22


def apply_s0_projector(
    input_ket: jnp.ndarray, n_alpha: int = 8, n_beta: int = 8, n_gamma: int = 8
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    SU(2) projection specialized to S=0 on a general (noncollinear) GHF state.

    Args:
      input_ket : jax.Array, shape (2*norb, nocc)   (GHF columns are occupied spinors)
      n_alpha, n_beta, n_gamma : Euler angle quadrature counts

    Returns:
      kets   : (Ngrid, 2*norb, nocc) rotated kets R(omega)|phi⟩
      coeffs : (Ngrid,) complex weights including (1/8pi^2) * sin beta * volume element
    """
    norb = input_ket.shape[0] // 2

    # uniform alpha, gamma in [0, 2pi), midpoint beta in (0, pi) with sin beta weight
    dalpha = 2.0 * jnp.pi / n_alpha
    dgamma = 2.0 * jnp.pi / n_gamma
    alpha = jnp.arange(n_alpha) * dalpha
    gamma = jnp.arange(n_gamma) * dgamma
    beta_edges = jnp.linspace(0.0, jnp.pi, n_beta + 1)
    beta = 0.5 * (beta_edges[:-1] + beta_edges[1:])
    dbeta = jnp.pi / n_beta

    A, B, G = jnp.meshgrid(alpha, beta, gamma, indexing="ij")
    A = A.reshape(-1)
    B = B.reshape(-1)
    G = G.reshape(-1)

    prefac = (1.0 / (8.0 * jnp.pi**2)) * dalpha * dbeta * dgamma

    def rotate_one(angles):
        a, b, g = angles
        u11, u12, u21, u22 = _spin_rot_elements(a, b, g)
        alpha_block = u11 * input_ket[:norb, :] + u12 * input_ket[norb:, :]
        beta_block = u21 * input_ket[:norb, :] + u22 * input_ket[norb:, :]
        ket_rot = jnp.vstack([alpha_block, beta_block])
        weight = prefac * jnp.sin(b)
        return ket_rot, weight

    kets, coeffs = jax.vmap(rotate_one)(jnp.stack([A, B, G], axis=1))
    return kets, coeffs


def _apply_pg_to_ket(ket: jnp.ndarray, U: jnp.ndarray) -> jnp.ndarray:
    """U acts in orbital space on both spin blocks."""
    norb = ket.shape[0] // 2
    up = U @ ket[:norb, :]
    dn = U @ ket[norb:, :]
    return jnp.vstack([up, dn])


class projector:
    """Abstract projector that maps a batch of determinants to a new linear combination."""

    def apply(
        self, kets: jnp.ndarray, coeffs: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        raise NotImplementedError


@dataclass
class sz_projector(projector):
    """Project onto fixed Sz using discrete rotations."""

    m: float = 0.0
    n_grid: int = 8

    def apply(
        self, kets: jnp.ndarray, coeffs: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        def rotate_single(ket: jnp.ndarray):
            return apply_sz_projector(ket, self.m, self.n_grid)

        rot_kets, rot_coeffs = jax.vmap(rotate_single, in_axes=0)(kets)
        batch, nterms, norb2, nocc = rot_kets.shape
        new_kets = rot_kets.reshape(batch * nterms, norb2, nocc)
        new_coeffs = (coeffs[:, None] * rot_coeffs).reshape(batch * nterms)
        return new_kets, new_coeffs


@dataclass
class singlet_projector(projector):
    """Project onto total spin S=0 using SU(2) quadrature."""

    n_alpha: int = 8
    n_beta: int = 8
    n_gamma: int = 8

    def apply(
        self, kets: jnp.ndarray, coeffs: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        def rotate_single(ket):
            return apply_s0_projector(
                ket, n_alpha=self.n_alpha, n_beta=self.n_beta, n_gamma=self.n_gamma
            )

        rot_kets, rot_coeffs = jax.vmap(rotate_single, in_axes=0)(kets)
        batch, nterms, norb2, nocc = rot_kets.shape
        new_kets = rot_kets.reshape(batch * nterms, norb2, nocc)
        new_coeffs = (coeffs[:, None] * rot_coeffs).reshape(batch * nterms)
        return new_kets, new_coeffs


@dataclass
class point_group_projector(projector):
    """Project onto a point-group irrep defined by orbital operators."""

    pg_ops: list
    pg_chars: list | None = None

    def __post_init__(self):
        if self.pg_chars is None:
            chars = jnp.ones(len(self.pg_ops), dtype=jnp.complex128)
        else:
            chars = jnp.asarray(self.pg_chars, dtype=jnp.complex128)

        self.pg_ops = [jnp.asarray(U) for U in self.pg_ops]
        self._chars = chars
        self._norm = 1.0 / len(self.pg_ops)

    def apply(
        self, kets: jnp.ndarray, coeffs: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        rotated = []
        for U in self.pg_ops:
            rotated.append(jax.vmap(lambda ket: _apply_pg_to_ket(ket, U))(kets))

        rotated = jnp.stack(rotated, axis=0)  # (ngroup, batch, 2*norb, nocc)
        new_kets = rotated.reshape(-1, *kets.shape[1:])

        weight_factors = (jnp.conj(self._chars) * self._norm)[:, None]
        new_coeffs = (weight_factors * coeffs[None, :]).reshape(-1)
        return new_kets, new_coeffs


def _apply_projector_sequence(
    psi: jnp.ndarray, projectors: tuple[projector, ...]
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Apply projectors sequentially to generate a linear combination of Slater determinants.

    Returns:
        Tuple of (kets, coeffs) describing the expanded state.
    """
    kets = psi[jnp.newaxis, ...]
    if jnp.issubdtype(psi.dtype, jnp.complexfloating):
        coeff_dtype = psi.dtype
    else:
        coeff_dtype = jnp.complex128
    coeffs = jnp.array([1.0 + 0.0j], dtype=coeff_dtype)

    for projector in projectors:
        kets, coeffs = projector.apply(kets, coeffs)

    return kets, coeffs


def _evaluate_projected_energy(
    psi: jnp.ndarray, ham: hamiltonian, projectors: tuple[projector, ...]
) -> jnp.ndarray:
    """Evaluate the projected energy after applying a sequence of projectors."""
    kets, coeffs = _apply_projector_sequence(psi, projectors)
    return _evaluate_linear_expansion_energy(psi, kets, coeffs, ham)


def calculate_projected_energy(
    psi: jnp.ndarray,
    ham: hamiltonian,
    projectors: tuple[projector, ...],
) -> jnp.ndarray:
    """
    Calculate the projected energy of a GHF determinant under optional symmetry projections.

    Args:
        psi: GHF determinant.
        ham: Hamiltonian object implementing the mixed-energy interface.
        projectors: Sequence of projector objects applied to |psi>.

    Returns:
        Projected energy as a scalar jax.Array.
    """

    @jax.jit
    def energy_function():
        return _evaluate_projected_energy(psi, ham, projectors)

    return energy_function()


def optimize(
    psi: jnp.ndarray,
    ham: hamiltonian,
    projectors: tuple[projector, ...],
    maxiter: int = 100,
    step: float = 0.01,
) -> tuple[float, np.ndarray]:
    """
    Variationally optimize a GHF determinant under optional symmetry projections.

    Args:
        psi: Initial GHF determinant.
        ham: Hamiltonian object implementing the mixed-energy interface.
        projectors: Sequence of projector objects applied to |psi>.
        maxiter: Number of Optax steps for Optax.
        step: AMSGrad learning rate.
    """
    psi_var = jnp.asarray(psi)
    psi_dtype = psi_var.dtype

    def objective_function(current_psi: jnp.ndarray) -> jnp.ndarray:
        energy = _evaluate_projected_energy(current_psi, ham, projectors)
        return jnp.real(energy)

    objective_function = jit(objective_function)
    value_and_grad = jit(jax.value_and_grad(objective_function))
    optimizer = optax.amsgrad(step, b2=0.99)
    opt_state = optimizer.init(psi_var)

    energy = float(np.array(objective_function(psi_var)))
    print(f"\n# Initial projected energy = {energy}")
    print("Starting optimization with Optax AMSGrad...")

    for iteration in range(maxiter):
        energy_val, grads = value_and_grad(psi_var)
        grads_real = jnp.real(grads)
        grad_norm = float(np.array(jnp.linalg.norm(jnp.ravel(grads_real))))
        grads_typed = grads_real.astype(psi_dtype)
        updates, opt_state = optimizer.update(grads_typed, opt_state, psi_var)
        psi_var = optax.apply_updates(psi_var, updates)
        energy = energy_val
        print(
            f"Iteration {iteration + 1}: Energy = {float(np.array(energy_val))}, "
            f"Grad norm = {grad_norm}"
        )

    return float(np.array(energy)), np.array(psi_var)


def _evaluate_projected_property(
    psi: jnp.ndarray,
    projectors: tuple[projector, ...],
    property_fn: callable,
) -> jnp.ndarray:
    """
    Generic evaluation of projected observables using weighted sums over
    projector-generated determinants.
    """
    kets, coeffs = _apply_projector_sequence(psi, projectors)

    def calc_overlap(bra, ket):
        return jnp.linalg.det(bra.T.conj() @ ket)

    overlaps = jax.vmap(lambda ket: jax.vmap(lambda bra: calc_overlap(bra, ket))(kets))(
        kets
    )
    values = jax.vmap(lambda ket: jax.vmap(lambda bra: property_fn(bra, ket))(kets))(
        kets
    )
    weights = coeffs.conj()[:, None] * coeffs[None, :] * overlaps
    value_num = jnp.tensordot(weights, values, axes=([0, 1], [0, 1]))
    value_denom = jnp.sum(weights)
    return (value_num / value_denom).real


def calculate_projected_1rdm(
    psi: jnp.ndarray,
    projectors: tuple[projector, ...],
) -> np.ndarray:
    """
    Calculate the projected one-body reduced density matrix (1RDM)
    of a GHF determinant under optional symmetry projections.

    Args:
        psi: GHF determinant.
        projectors: Sequence of projector objects applied to |psi>.

    Returns:
        Projected 1RDM as a numpy ndarray.
    """

    @jax.jit
    def rdm1_function(psi_var: jnp.ndarray) -> jnp.ndarray:
        return _evaluate_projected_property(psi_var, projectors, _mixed_green)

    rdm1 = rdm1_function(psi)
    return np.array(rdm1)


def calculate_projected_density_correlations(
    psi: jnp.ndarray,
    projectors: tuple[projector, ...],
) -> np.ndarray:
    """
    Calculate the projected density-density correlation matrix
    of a GHF determinant under optional symmetry projections.

    Args:
        psi: GHF determinant.
        projectors: Sequence of projector objects applied to |psi>.

    Returns:
        Projected density-density correlation matrix as a numpy ndarray.
    """

    @jax.jit
    def density_corr_function(psi_var: jnp.ndarray) -> jnp.ndarray:
        def calc_density_corr(bra, ket):
            green = _mixed_green(bra, ket)
            green_diag = jnp.diagonal(green)
            density_corr = (
                green_diag[:, None] * green_diag[None, :]
                - green * green.T
                + jnp.diag(green_diag)
            )
            return density_corr

        return _evaluate_projected_property(psi_var, projectors, calc_density_corr)

    density_corr = density_corr_function(psi)
    return np.array(density_corr)


def _SzSz_from_G(G):
    n = G.shape[0] // 2
    Guu = G[:n, :n]
    Gdd = G[n:, n:]
    Gud = G[:n, n:]
    Gdu = G[n:, :n]

    def nn_same(Gblk):
        occ = jnp.diag(Gblk)
        return occ[:, None] * occ[None, :] - Gblk * Gblk.T + jnp.diag(occ)

    Euu = nn_same(Guu)
    Edd = nn_same(Gdd)
    Eud = jnp.diag(Guu)[:, None] * jnp.diag(Gdd)[None, :] - Gud * Gdu.T
    Edu = jnp.diag(Gdd)[:, None] * jnp.diag(Guu)[None, :] - Gdu * Gud.T

    SzSz = 0.25 * (Euu + Edd - Eud - Edu)
    return SzSz


def _SpSm_from_G(G):
    n = G.shape[0] // 2
    Guu = G[:n, :n]
    Gdd = G[n:, n:]
    Gud = G[:n, n:]
    Gdu = G[n:, :n]
    coh = jnp.outer(jnp.diag(Gud), jnp.diag(Gdu))
    diag = jnp.diag(jnp.diag(Guu))
    exch = Guu * Gdd.T
    return coh + diag - exch


def _SmSp_from_G(G):
    n = G.shape[0] // 2
    Guu = G[:n, :n]
    Gdd = G[n:, n:]
    Gud = G[:n, n:]
    Gdu = G[n:, :n]
    coh = jnp.outer(jnp.diag(Gdu), jnp.diag(Gud))
    diag = jnp.diag(jnp.diag(Gdd))
    exch = Gdd * Guu.T
    return coh + diag - exch


def _SdotS_from_G(G):
    return _SzSz_from_G(G) + 0.5 * (_SpSm_from_G(G) + _SmSp_from_G(G))


def calculate_projected_sz_correlations(psi, projectors):
    @jax.jit
    def evaluate():
        def prop_fn(bra, ket):
            return _SzSz_from_G(_mixed_green(bra, ket))

        return _evaluate_projected_property(psi, projectors, prop_fn)

    corr = evaluate()
    return np.array(corr)


def calculate_projected_s_correlations(psi, projectors):
    @jax.jit
    def evaluate():
        def prop_fn(bra, ket):
            return _SdotS_from_G(_mixed_green(bra, ket))

        return _evaluate_projected_property(psi, projectors, prop_fn)

    corr = evaluate()
    return np.array(corr)


def calculate_projected_s2(psi, projectors):
    @jax.jit
    def evaluate():
        def prop_fn(bra, ket):
            G = _mixed_green(bra, ket)
            return jnp.sum(_SdotS_from_G(G))

        return _evaluate_projected_property(psi, projectors, prop_fn)

    s2 = evaluate()
    return float(s2)
