from functools import partial

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
import scipy as sp
from jax import grad, jit
from scipy.optimize import minimize


def get_wigner_d(s, m, k, beta):
    """
    Wigner small d-matrix.
    """
    nmin = int(max(0, k - m))
    nmax = int(min(s + k, s - m))
    fac = np.sqrt(
        sp.special.factorial(s + k)
        * sp.special.factorial(s - k)
        * sp.special.factorial(s + m)
        * sp.special.factorial(s - m)
    )
    cumsum = 0.0

    for n in range(nmin, nmax + 1):
        denom = (
            sp.special.factorial(s + k - n)
            * sp.special.factorial(s - n - m)
            * sp.special.factorial(n - k + m)
            * sp.special.factorial(n)
        )

        sign = (-1.0) ** (n - k + m)
        cos = (np.cos(beta / 2.0)) ** (2 * s - 2 * n + k - m)
        sin = (np.sin(beta / 2.0)) ** (2 * n - k + m)
        num = sign * cos * sin
        cumsum += num / denom

    return fac * cumsum


def apply_Sz_projector(input_ket, m, ngrid):
    norb = input_ket.shape[0] // 2
    nocc = input_ket.shape[1]
    kets = np.zeros((ngrid, 2 * norb, nocc), dtype=np.complex128)
    coeffs = np.zeros(ngrid, dtype=np.complex128)

    # Integrate gamma \in [0, 2pi) by quadrature.
    for ig in range(ngrid):
        gamma = 2 * np.pi * ig / ngrid

        # Coefficients.
        coeffs[ig] = np.exp(-1.0j * gamma * m)

        # Rotation matrix about the z-axis. Only need to consider the diagonal.
        rot_diag = np.ones(2 * norb, dtype=np.complex128)
        rot_diag[:norb] *= np.exp(1.0j * gamma / 2.0)  # Acting on spin up.
        rot_diag[norb:] *= np.exp(-1.0j * gamma / 2.0)  # Acting on spin down.
        rot_mat = np.diag(rot_diag)
        kets[ig] = rot_mat @ input_ket

    coeffs /= ngrid  # For numerical integration.
    return kets, coeffs


def apply_Sz_projector_jax(input_ket, m, ngrid):
    # input_ket should be a jax array, shape (2*norb, nocc).
    norb = input_ket.shape[0] // 2

    # We'll define a function that, for a single gamma, gives (ket, coeff).
    def projector_for_gamma(ig):
        gamma = 2.0 * jnp.pi * ig / ngrid

        # Coefficients.
        coeff_ig = jnp.exp(-1.0j * gamma * m) / ngrid

        # Rotation matrix about the z-axis. Only need to consider the diagonal.
        rot_diag_a = jnp.ones(norb) * jnp.exp(1.0j * gamma / 2.0)
        rot_diag_b = jnp.ones(norb) * jnp.exp(-1.0j * gamma / 2.0)
        rot_diag = jnp.concatenate([rot_diag_a, rot_diag_b])
        rot_mat = jnp.diag(rot_diag)
        ket_ig = rot_mat @ input_ket
        return ket_ig, coeff_ig

    # Vectorize over ig = 0..ngrid-1
    igs = jnp.arange(ngrid)
    kets, coeffs = jax.vmap(projector_for_gamma, in_axes=0, out_axes=(0, 0))(igs)
    return kets, coeffs


@partial(jit, static_argnums=2)
def apply_Sz_projector_jax_fast(input_ket, m, ngrid):
    norb = input_ket.shape[0] // 2

    # Pre-compute all the gammas and exponentials
    gammas = 2.0 * jnp.pi * jnp.arange(ngrid) / ngrid
    exp_factors = jnp.exp(1.0j * gammas / 2.0)
    exp_neg_factors = jnp.exp(-1.0j * gammas / 2.0)
    m_factors = jnp.exp(-1.0j * gammas * m) / ngrid

    def projector_for_gamma(ig):
        rot_diag = jnp.concatenate(
            [jnp.ones(norb) * exp_factors[ig], jnp.ones(norb) * exp_neg_factors[ig]]
        )
        return input_ket * rot_diag[:, None], m_factors[ig]

    return jax.vmap(projector_for_gamma, in_axes=0, out_axes=(0, 0))(jnp.arange(ngrid))


def get_real_wavefunction(kets, coeffs):
    ngrid, norbx2, nocc = kets.shape
    real_kets = np.zeros((2 * ngrid, norbx2, nocc), dtype=np.complex128)
    real_coeffs = np.zeros(2 * ngrid, dtype=np.complex128)
    real_kets[:ngrid] = kets.copy()
    real_kets[ngrid:] = np.conj(kets)
    real_coeffs[:ngrid] = coeffs.copy() / np.sqrt(2.0)
    real_coeffs[ngrid:] = np.conj(coeffs) / np.sqrt(2.0)
    return real_kets, real_coeffs


def get_projected_energy(psi, m, h1, chol, enuc, ngrid):
    norb = h1[0].shape[0]
    nchol = chol.shape[0]
    nocc = psi.shape[1]

    kets, coeffs = apply_Sz_projector(psi, m, ngrid)
    # kets, coeffs = get_real_wavefunction(kets, coeffs)
    psiTconj = psi.T.conj()
    rotchol = np.zeros((2, nchol, nocc, norb), dtype=np.complex128)

    # TODO: parallelize.
    for i in range(nchol):
        rotchol[0, i] = psiTconj[:nocc, :norb] @ chol[i].reshape((norb, norb))
        rotchol[1, i] = psiTconj[:nocc, norb:] @ chol[i].reshape((norb, norb))

    enum, ovlp = 0.0, 0.0

    for i in range(kets.shape[0]):
        enum_i = get_energy(psi, kets[i], h1, rotchol, enuc)
        ovlp_mat = psiTconj @ kets[i]
        ovlp_i = sp.linalg.det(ovlp_mat)
        enum += coeffs[i] * enum_i * ovlp_i
        ovlp += coeffs[i] * ovlp_i

    enum = enum.real
    ovlp = ovlp.real
    return enum / ovlp


@jit
def build_rotchol(psiTconj, chol):
    """
    psiTconj: shape (2*norb, nocc) presumably,
              but you only use psiTconj[:nocc,:norb] and psiTconj[:nocc,norb:].
    chol: shape (nchol, norb, norb) or (nchol, norb**2) that you can reshape.

    We want rotchol of shape (2, nchol, nocc, norb).
    """

    nocc = psiTconj.shape[0]  # assuming your code snippet used psiTconj[:nocc, ...]
    # norb from chol
    # let's assume chol is (nchol, norb, norb)
    nchol, norb, _ = chol.shape

    # Split alpha and beta blocks from psiTconj
    psi_alpha = psiTconj[:nocc, :norb]  # shape (nocc, norb)
    psi_beta = psiTconj[:nocc, norb:]  # shape (nocc, norb)

    def multiply_chol(chol_i):
        # chol_i: shape (norb, norb)
        # Multiply alpha slice
        alpha_part = psi_alpha @ chol_i  # (nocc,norb) x (norb,norb) -> (nocc,norb)
        beta_part = psi_beta @ chol_i  # same shape
        return alpha_part, beta_part

    # Apply over axis 0 of chol (i.e. loop i=0..nchol-1)
    rot_alpha, rot_beta = jax.vmap(multiply_chol, in_axes=(0,))(chol)
    # rot_alpha, rot_beta each shape (nchol, nocc, norb)

    # Stack them along a new first dimension for the spin index
    rotchol = jnp.stack([rot_alpha, rot_beta], axis=0)
    # rotchol has shape (2, nchol, nocc, norb)
    return rotchol


@jit
def build_rotchol_fast(psiTconj, chol):
    """
    psiTconj: shape (nocc, 2*norb)
    chol: shape (nchol, norb, norb)
    Returns: rotchol shape (2, nchol, nocc, norb)
    """
    _, norb, _ = chol.shape

    psi_alpha = psiTconj[:, :norb]
    psi_beta = psiTconj[:, norb:]

    rot_alpha = jnp.einsum("ij,njk->nik", psi_alpha, chol)
    rot_beta = jnp.einsum("ij,njk->nik", psi_beta, chol)

    rotchol = jnp.stack([rot_alpha, rot_beta], axis=0)
    return rotchol


def get_projected_energy_jax(psi, m, h1, chol, enuc, ngrid):
    norb = h1[0].shape[0]
    nchol = chol.shape[0]
    nocc = psi.shape[1]

    kets, coeffs = apply_Sz_projector_jax(psi, m, ngrid)
    # kets, coeffs = get_real_wavefunction(kets, coeffs)
    psiTconj = psi.T.conj()
    rotchol = build_rotchol(psiTconj, chol.reshape((-1, norb, norb)))

    def scan_fun(i, carry):
        enum, ovlp = carry
        enum_i = get_energy_jax(psi, kets[i], h1, rotchol, enuc)
        ovlp_mat = psiTconj @ kets[i]
        ovlp_i = jsp.linalg.det(ovlp_mat)
        enum += coeffs[i] * enum_i * ovlp_i
        ovlp += coeffs[i] * ovlp_i
        return enum, ovlp

    # Initial values.
    init = (0.0 + 0.0j, 0.0 + 0.0j)

    # Run the loop from i = 0 to i = ngrid.
    enum, ovlp = jax.lax.fori_loop(0, kets.shape[0], scan_fun, init)
    return enum.real / ovlp.real


@partial(jit, static_argnums=5)
def get_projected_energy_jax_fast(psi, m, h1, chol, enuc, ngrid):
    norb = h1[0].shape[0]

    kets, coeffs = apply_Sz_projector_jax(psi, m, ngrid)
    psiTconj = psi.T.conj()
    rotchol = build_rotchol(psiTconj, chol.reshape((-1, norb, norb)))

    def compute_overlap(ket):
        ovlp_mat = psiTconj @ ket
        return jsp.linalg.det(ovlp_mat)

    def compute_energy(ket):
        return get_energy_jax(psi, ket, h1, rotchol, enuc)

    overlaps = jax.vmap(compute_overlap)(kets)
    energies = jax.vmap(compute_energy)(kets)

    weighted_terms = coeffs * overlaps * energies
    enum = jnp.sum(weighted_terms)
    ovlp = jnp.sum(coeffs * overlaps)

    return enum.real / ovlp.real


# Hamiltonian matrix element < GHF | H | GHF > / < GHF | GHF >
def get_energy(bra, ket, h1, rotchol, enuc):
    norb = h1[0].shape[0]
    nchol = rotchol[0].shape[0]
    nocc = bra.shape[1]

    # Calculate Green's function.
    Ghalf = ket @ np.linalg.inv(bra.T.conj() @ ket)
    Ghalfa = Ghalf[:norb]
    Ghalfb = Ghalf[norb:]
    G = Ghalf @ bra.T.conj()

    # Core energy.
    energy = enuc

    # One-body energy,
    e1 = np.trace(h1[0] @ G[:norb, :norb]) + np.trace(h1[1] @ G[norb:, norb:])
    energy += e1

    # Two-body energy.
    ej, ek = 0.0, 0.0
    W = np.zeros((2, nocc, nocc), dtype=np.complex128)

    for i in range(nchol):
        W[0] = rotchol[0, i] @ Ghalfa
        W[1] = rotchol[1, i] @ Ghalfb
        ej += 0.5 * (np.trace(W[0]) + np.trace(W[1])) ** 2
        ek -= 0.5 * (
            np.sum(W[0] * W[0].T)
            + np.sum(W[0] * W[1].T)
            + np.sum(W[1] * W[0].T)
            + np.sum(W[1] * W[1].T)
        )

    energy += ej + ek
    return energy


# Hamiltonian matrix element < GHF | H | GHF > / < GHF | GHF >
@jit
def get_energy_jax(bra, ket, h1, rotchol, enuc):
    norb = h1[0].shape[0]
    nchol = rotchol[0].shape[0]
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

    # Initial values.
    init = (0.0 + 0.0j, 0.0 + 0.0j)

    # Run the loop from i = 0 to i = nchol.
    ej, ek = jax.lax.fori_loop(0, nchol, scan_fun, init)
    energy += ej + ek
    return energy


def gradient_descent(psi, nelec, h1, chol, enuc, ngrid=10, maxiter=100, step=0.01):
    norb = h1[0].shape[0]
    nchol = chol.shape[0]
    na, nb = nelec
    nocc = sum(nelec)
    m = 0.5 * (na - nb)

    energy_init = get_projected_energy(psi, m, h1, chol, enuc, ngrid)
    print(f"\n# Initial projected energy = {energy_init}")

    def objective_function(x):
        psi = x.reshape(2 * norb, nocc)
        return get_projected_energy(psi, m, h1, chol, enuc, ngrid)

    def gradient(x, *args):
        return np.array(grad(objective_function)(x, *args), dtype=np.float64)

    x = psi.flatten()

    for i in range(maxiter):
        grads = gradient(x)
        x -= step * grads

    psi = x.reshape(2 * norb, nocc)
    energy = get_projected_energy(psi, m, h1, chol, enuc, ngrid)
    return energy, psi


from scipy.optimize import OptimizeResult


def amsgrad(
    fun,
    x0,
    args=(),
    jac=None,
    callback=None,
    learning_rate=0.01,
    beta1=0.9,
    beta2=0.99,
    epsilon=1e-8,
    maxiter=1000,
    tol=1e-5,
    hess=None,
    hessp=None,
    bounds=None,
    constraints=(),
    **options,
):
    """
    AMSGrad optimizer implementation for scipy.optimize.minimize.

    Parameters:
    -----------
    fun : callable
        Objective function to be minimized: f(x, *args) -> float
    x0 : ndarray
        Initial guess
    args : tuple, optional
        Extra arguments passed to objective function
    jac : callable, optional
        Gradient of objective function: jac(x, *args) -> array_like
    callback : callable, optional
        Called after each iteration: callback(xk) -> None
    learning_rate : float, optional
        Learning rate (default: 0.001)
    beta1 : float, optional
        Exponential decay rate for first moment (default: 0.9)
    beta2 : float, optional
        Exponential decay rate for second moment (default: 0.999)
    epsilon : float, optional
        Small constant for numerical stability (default: 1e-8)
    maxiter : int, optional
        Maximum number of iterations (default: 1000)
    tol : float, optional
        Tolerance for convergence (default: 1e-5)

    Returns:
    --------
    OptimizeResult
        Result object with optimization outcome
    """
    print(
        f"step = {learning_rate}, beta1 = {beta1}, beta2 = {beta2}, epsilon = {epsilon}"
    )
    x = np.asarray(x0).flatten()
    if jac is None:
        from scipy.optimize import approx_fprime

        jac = lambda x, *args: approx_fprime(x, fun, epsilon, *args)

    # Initialize moment vectors
    m = np.zeros_like(x)  # First moment
    v = np.zeros_like(x)  # Second moment
    v_hat = np.zeros_like(x)  # Maximum of past squared gradients

    # Initialize optimization variables
    t = 0
    fx = fun(x, *args)
    nfev = 1
    ngev = 0

    for iteration in range(maxiter):
        t += 1

        # Compute gradient
        grad = jnp.array(jac(x, *args))
        ngev += 1

        # Update biased first moment estimate
        m = beta1 * m + (1 - beta1) * grad

        # Update biased second raw moment estimate
        v = beta2 * v + (1 - beta2) * np.square(grad)

        # Update maximum of past squared gradients
        v_hat = np.maximum(v_hat, v)

        # Compute bias-corrected first moment estimate
        # m_hat = m / (1 - beta1**t)
        m_hat = m

        # Compute bias-corrected second raw moment estimate
        # v_hat_corrected = v_hat / (1 - beta2**t)
        v_hat_corrected = v_hat

        # Update parameters
        step = learning_rate * m_hat / (np.sqrt(v_hat_corrected) + epsilon)
        x_new = x - step

        # Calculate new function value
        fx_new = fun(x_new, *args)
        nfev += 1

        print(
            f"Iteration {iteration + 1}: Energy = {fx_new}, Grad norm = {np.linalg.norm(grad)}"
        )

        # Check for convergence
        if np.all(np.abs(step) < tol):
            break

        # Update current position
        x = x_new
        fx = fx_new

        if callback is not None:
            callback(x)

    return OptimizeResult(
        x=x,
        success=True,
        nit=iteration + 1,
        nfev=nfev,
        ngev=ngev,
        fun=fx,
        jac=grad,
        message="Optimization terminated successfully.",
        status=0,
    )


def optimize(psi, nelec, h1, chol, enuc, ngrid=10, maxiter=100, step=0.01):
    norb = h1[0].shape[0]
    nchol = chol.shape[0]
    na, nb = nelec
    nocc = sum(nelec)
    m = 0.5 * (na - nb)

    energy_init = get_projected_energy_jax(psi, m, h1, chol, enuc, ngrid)
    print(f"\n# Initial projected energy = {energy_init}")

    @jit
    def objective_function(x):
        psi = x.reshape(2 * norb, nocc)
        return get_projected_energy_jax(psi, m, h1, chol, enuc, ngrid)

    @jit
    def gradient(x, *args):
        return (grad(objective_function)(x, *args)).real

    x = psi.flatten()
    print("Starting optimization...")
    res = minimize(
        objective_function,
        x,
        jac=gradient,
        tol=1e-10,
        method=amsgrad,
        # method="BFGS",
        options={
            # "maxls": 20,
            # "gtol": 1e-10,
            # "eps": 1e-10,
            # "maxiter": 15000,
            # "ftol": 1.0e-10,
            # "maxcor": 1000,
            # "maxfun": 15000,
            "disp": True,
        },
    )

    energy = res.fun
    psi = res.x.reshape(2 * norb, nocc)
    return energy, psi
