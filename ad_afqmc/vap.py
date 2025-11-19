import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jax import grad, jit

import numpy as np
import scipy as sp
from scipy.optimize import minimize, OptimizeResult
from functools import partial


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


# -----------------------------------------------------------------------------
# Projectors.
# -----------------------------------------------------------------------------
def apply_sz_projector(input_ket, s, sz, ngrid):
    norb = input_ket.shape[0] // 2
    nocc = input_ket.shape[1]
    kets = np.zeros((ngrid, 2*norb, nocc), dtype=np.complex128)
    coeffs = np.zeros(ngrid, dtype=np.complex128)

    # Integrate gamma \in [0, 2pi) by quadrature.
    for ig in range(ngrid):
        gamma = 2 * np.pi * ig / ngrid

        # Coefficients.
        coeffs[ig] = np.exp(1.j * sz * gamma)
        ket_a = np.exp(-1.j * gamma/2.) * input_ket[:norb]
        ket_b = np.exp(1.j * gamma/2.) * input_ket[norb:]
        kets[ig] = np.vstack([ket_a, ket_b])

    coeffs *= (2*s+1)/2. * np.pi/ngrid
    return kets, coeffs

@partial(jit, static_argnums=(1, 2, 3))
def apply_sz_projector_jax(input_ket, s, sz, ngrid):
    norb = input_ket.shape[0] // 2

    # Pre-compute all the gammas and exponentials.
    gammas = jnp.linspace(0., 2*jnp.pi, ngrid, endpoint=False)
    coeffs = jnp.exp(1.j * sz * gammas) * (2*s+1)/2. * jnp.pi/ngrid

    def rot_gamma(gamma):
        ket_a = jnp.exp(-1.j * gamma/2.) * input_ket[:norb]
        ket_b = jnp.exp(1.j * gamma/2.) * input_ket[norb:]
        return jnp.vstack([ket_a, ket_b])

    return jax.vmap(rot_gamma)(gammas), coeffs

def apply_s2_singlet_projector_jax(input_ket, nalpha=8, nbeta=8, ngamma=8):
    """
    Wigner small-d matrix = 1.
    """
    norb = input_ket.shape[0] // 2

    # Uniform grid over alpha, beta, gamma.
    #da = 2. * jnp.pi / nalpha
    #dg = 2. * jnp.pi / ngamma
    #alphas = jnp.arange(nalpha) * da
    #gammas = jnp.arange(ngamma) * dg

    # Midpoint rule in beta avoids endpoints; include sin(beta) Jacobian explicitly.
    db = jnp.pi / nbeta
    beta_edges = jnp.linspace(0.0, jnp.pi, nbeta+1)
    betas = 0.5 * (beta_edges[:-1] + beta_edges[1:])

    alphas, da = jnp.linspace(0., 2*jnp.pi, nalpha, endpoint=False, retstep=True)
    #betas, db = jnp.linspace(0., jnp.pi, nbeta, endpoint=False, retstep=True)
    gammas, dg = jnp.linspace(0., 2*jnp.pi, ngamma, endpoint=False, retstep=True)

    # Build the full tensor grid and flatten
    A, B, G = jnp.meshgrid(alphas, betas, gammas, indexing="ij")
    A = A.reshape(-1)
    B = B.reshape(-1)
    G = G.reshape(-1)

    def rotate(solid_angle):
        alpha, beta, gamma = solid_angle
        aa = jnp.exp(-1.j * (alpha + gamma)/2.) * jnp.cos(beta/2.)
        bb = jnp.exp(1.j * (alpha + gamma)/2.) * jnp.cos(beta/2.)
        ab = -jnp.exp(-1.j * (alpha - gamma)/2.) * jnp.sin(beta/2.)
        ba = jnp.exp(1.j * (alpha - gamma)/2.) * jnp.sin(beta/2.)

        ket_a = aa * input_ket[:norb] + ab * input_ket[norb:]
        ket_b = ba * input_ket[:norb] + bb * input_ket[norb:]
        rot_ket = jnp.vstack([ket_a, ket_b])
        rot_coeff = jnp.sin(beta)
        return rot_ket, rot_coeff

    prefac = 1./(8*jnp.pi**2) * da * db * dg
    kets, coeffs = jax.vmap(rotate)(jnp.stack([A, B, G], axis=-1))
    return kets, prefac * coeffs

def get_real_wavefunction(kets, coeffs):
    ngrid, norbx2, nocc = kets.shape
    real_kets = np.zeros((2 * ngrid, norbx2, nocc), dtype=np.complex128)
    real_coeffs = np.zeros(2 * ngrid, dtype=np.complex128)
    real_kets[:ngrid] = kets.copy()
    real_kets[ngrid:] = np.conj(kets)
    real_coeffs[:ngrid] = coeffs.copy() / np.sqrt(2.0)
    real_coeffs[ngrid:] = np.conj(coeffs) / np.sqrt(2.0)
    return real_kets, real_coeffs

# -----------------------------------------------------------------------------
# Energies.
# -----------------------------------------------------------------------------
def get_energy(bra, ket, h1, rotchol, enuc):
    """
    Hamiltonian matrix element < GHF | H | GHF > / < GHF | GHF >.
    """
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

@jit
def get_energy_jax(bra, ket, h1, rotchol, enuc):
    """
    Hamiltonian matrix element < GHF | H | GHF > / < GHF | GHF >.
    """
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

def get_sz_projected_energy(psi, h1, chol, enuc, s, sz, ngrid):
    norb = h1[0].shape[0]
    nchol = chol.shape[0]
    nocc = psi.shape[1]

    kets, coeffs = apply_sz_projector(psi, s, sz, ngrid)
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


@partial(jit, static_argnums=(4, 5, 6))
def get_sz_projected_energy_jax(psi, h1, chol, enuc, s, sz, ngrid):
    norb = h1[0].shape[0]
    kets, coeffs = apply_sz_projector_jax(psi, s, sz, ngrid)
    psiTconj = psi.T.conj()
    rotchol = build_rotchol(psiTconj, chol.reshape((-1, norb, norb)))

    def get_overlap(ket):
        ovlp_mat = psiTconj @ ket
        return jsp.linalg.det(ovlp_mat)

    def get_energy(ket):
        return get_energy_jax(psi, ket, h1, rotchol, enuc)

    overlaps = jax.vmap(get_overlap)(kets)
    energies = jax.vmap(get_energy)(kets)
    num = jnp.sum(coeffs * overlaps * energies)
    denom = jnp.sum(coeffs * overlaps)
    return num.real / denom.real

def get_s2_singlet_projected_energy_jax(
        psi, h1, chol, enuc, nalpha=8, nbeta=8, ngamma=8
    ):
    norb = h1[0].shape[0]

    kets, coeffs = apply_s2_singlet_projector_jax(psi, nalpha, nbeta, ngamma)
    psiTconj = psi.T.conj()
    rotchol = build_rotchol(psiTconj, chol.reshape((-1, norb, norb)))

    def get_overlap(ket):
        ovlp_mat = psiTconj @ ket
        return jsp.linalg.det(ovlp_mat)

    def get_energy(ket):
        return get_energy_jax(psi, ket, h1, rotchol, enuc)

    overlaps = jax.vmap(get_overlap)(kets)
    energies = jax.vmap(get_energy)(kets)

    num = jnp.sum(coeffs * overlaps * energies)
    denom = jnp.sum(coeffs * overlaps)
    return num.real / denom.real

def get_ext_sz_projected_energy_jax(
        psi, h1, chol, enuc, s, sz, ext_ops, ext_chars=None, ngrid=8
    ):
    """
    Assumes spinless external projectors.
    """
    def _apply_ext_rotation(input_ket, U):
        """U acts in orbital space on both spin blocks."""
        norb = input_ket.shape[0] // 2
        ket_a = U @ input_ket[:norb]
        ket_b = U @ input_ket[norb:]
        return jnp.vstack([ket_a, ket_b])

    if ext_chars is None: ext_chars = [1.] * len(ext_ops)
    norb = h1[0].shape[0]
    
    # Apply spin rotations that commute with the spinless external projectors.
    base_kets, coeffs = apply_sz_projector_jax(psi, s, sz, ngrid)
    psiTconj = psi.T.conj()
    rotchol = build_rotchol(psiTconj, chol.reshape((-1, norb, norb)))

    def get_overlap(ket):
        ovlp_mat = psiTconj @ ket
        return jsp.linalg.det(ovlp_mat)
        #return jsp.linalg.slogdet(ovlp_mat)

    def get_energy(ket):
        return get_energy_jax(psi, ket, h1, rotchol, enuc)

    def apply_ext_rotation(Ug, char_g):
        # Apply external rotations Ug to each spin-rotated ket.
        # Shape (base_kets.shape).
        kets_g = jax.vmap(lambda ket: _apply_ext_rotation(ket, Ug))(base_kets)
        overlaps = jax.vmap(get_overlap)(kets_g)
        energies = jax.vmap(get_energy)(kets_g)
        char_g_conj = jnp.conj(jnp.array(char_g))
        num = char_g_conj * jnp.sum(coeffs * overlaps * energies)
        denom = char_g_conj * jnp.sum(coeffs * overlaps)
        
        #signs, log_overlaps = jax.vmap(get_overlap)(kets_g)
        #num = char_g_conj * jnp.sum(
        #        signs * jnp.exp(jnp.log(coeffs) + log_overlaps + jnp.log(energies))
        #    )

        #denom = char_g_conj * jnp.sum(
        #        signs * jnp.exp(jnp.log(coeffs) + log_overlaps)
        #    )
        
        return num, denom
    
    # [[U1_num, U1_denom], [U2_num, U2_denom], ...] 
    num_denom_arr = jnp.array(
        [apply_ext_rotation(Ug, char_g) for Ug, char_g in zip(ext_ops, ext_chars)]
    )
    num, denom = jnp.sum(num_denom_arr, axis=0) / len(ext_ops)
    return num.real / denom.real

def get_ext_s2_singlet_projected_energy_jax(
        psi, h1, chol, enuc, ext_ops, ext_chars=None, nalpha=8, nbeta=8, ngamma=8
    ):
    """
    Assumes spinless external projectors.
    """
    def _apply_ext_rotation(input_ket, U):
        """U acts in orbital space on both spin blocks."""
        norb = input_ket.shape[0] // 2
        ket_a = U @ input_ket[:norb]
        ket_b = U @ input_ket[norb:]
        return jnp.vstack([ket_a, ket_b])

    if ext_chars is None: ext_chars = [1.] * len(ext_ops)
    norb = h1[0].shape[0]
    
    # Apply spin rotations that commute with the spinless external projectors.
    base_kets, coeffs = apply_s2_singlet_projector_jax(psi, nalpha, nbeta, ngamma)
    psiTconj = psi.T.conj()
    rotchol = build_rotchol(psiTconj, chol.reshape((-1, norb, norb)))

    def get_overlap(ket):
        ovlp_mat = psiTconj @ ket
        return jsp.linalg.det(ovlp_mat)
        #return jsp.linalg.slogdet(ovlp_mat)

    def get_energy(ket):
        return get_energy_jax(psi, ket, h1, rotchol, enuc)

    def apply_ext_rotation(Ug, char_g):
        # Apply external rotations Ug to each spin-rotated ket.
        # Shape (base_kets.shape).
        kets_g = jax.vmap(lambda ket: _apply_ext_rotation(ket, Ug))(base_kets)
        overlaps = jax.vmap(get_overlap)(kets_g)
        energies = jax.vmap(get_energy)(kets_g)
        char_g_conj = jnp.conj(jnp.array(char_g))
        num = char_g_conj * jnp.sum(coeffs * overlaps * energies)
        denom = char_g_conj * jnp.sum(coeffs * overlaps)
        
        #signs, log_overlaps = jax.vmap(get_overlap)(kets_g)
        #num = char_g_conj * jnp.sum(
        #        signs * jnp.exp(jnp.log(coeffs) + log_overlaps + jnp.log(energies))
        #    )

        #denom = char_g_conj * jnp.sum(
        #        signs * jnp.exp(jnp.log(coeffs) + log_overlaps)
        #    )
        
        return num, denom
    
    # [[U1_num, U1_denom], [U2_num, U2_denom], ...] 
    num_denom_arr = jnp.array(
        [apply_ext_rotation(Ug, char_g) for Ug, char_g in zip(ext_ops, ext_chars)]
    )
    num, denom = jnp.sum(num_denom_arr, axis=0) / len(ext_ops)

    #jax.debug.print('num = {x}', x=num)
    #jax.debug.print('denom = {x}', x=denom)

    return num.real / denom.real

# -----------------------------------------------------------------------------
# Optimization.
# -----------------------------------------------------------------------------
def gradient_descent(psi, nelec, h1, chol, enuc, ngrid=10, maxiter=100, step=0.01):
    norb = h1[0].shape[0]
    na, nb = nelec
    nocc = sum(nelec)
    s = 0
    sz = 0.5 * (na - nb)

    energy_init = get_sz_projected_energy(psi, s, sz, h1, chol, enuc, ngrid)
    print(f"\n# Initial projected energy = {energy_init}")

    def objective_function(x):
        psi = x.reshape(2 * norb, nocc)
        #psi, _ = jnp.linalg.qr(psi) # Orthonormalize.
        return get_sz_projected_energy(psi, s, sz, h1, chol, enuc, ngrid)

    def gradient(x, *args):
        return np.array(grad(objective_function)(x, *args), dtype=np.float64)

    x = psi.flatten()

    for i in range(maxiter):
        grads = gradient(x)
        x -= step * grads

    psi = x.reshape(2 * norb, nocc)
    energy = get_sz_projected_energy(psi, s, sz, h1, chol, enuc, ngrid)
    return energy, psi


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
        f"# step = {learning_rate}, beta1 = {beta1}, beta2 = {beta2}, epsilon = {epsilon}"
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
            f"# Iteration {iteration + 1}: Energy = {fx_new:.9e}, Grad norm = {np.linalg.norm(grad):.9e}"
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

def optimize(
    psi,
    nelec,
    h1,
    chol,
    enuc,
    nalpha=8,
    nbeta=8,
    ngamma=8,
    maxiter=100,
    step=0.01,
    projector="s2",
    ext_ops=None,
    ext_chars=None,
):
    norb = h1[0].shape[0]
    nchol = chol.shape[0]
    na, nb = nelec
    nocc = sum(nelec)
    s = 0.
    sz = 0.5 * (na - nb)

    print(f"\n# maxiter: {maxiter}")
    print(f"# projector: {projector}")
    print(f"# quadrature nalpha, nbeta, ngamma: {nalpha, nbeta, ngamma}")
    if ext_ops is not None: print(f"# external projectors: {list(ext_ops.keys())}")

    if "s2" in projector:
        assert na == nb, "\n# S^2 projection only implemented for singlet states."

        if ("ext" in projector) and (ext_ops is not None):
            energy_init = get_ext_s2_singlet_projected_energy_jax(
                psi,
                h1,
                chol,
                enuc,
                list(ext_ops.values())[0], # TODO: Only works for 1 group now!
                list(ext_chars.values())[0],
                nalpha=nalpha,
                nbeta=nbeta,
                ngamma=ngamma,
            )

        else:
            energy_init = get_s2_singlet_projected_energy_jax(
                psi, h1, chol, enuc, nalpha=ngamma, nbeta=nbeta, ngamma=ngamma
            )

    elif "sz" in projector:
        if ("ext" in projector) and (ext_ops is not None):
            energy_init = get_ext_sz_projected_energy_jax(
                psi,
                h1,
                chol,
                enuc,
                s,
                sz,
                list(ext_ops.values())[0], # TODO: Only works for 1 group now!
                list(ext_chars.values())[0],
                ngrid=nbeta,
            )

        else:
            energy_init = get_sz_projected_energy_jax(psi, h1, chol, enuc, s, sz, nbeta)

    print(f"\n# Initial projected energy = {energy_init}")

    @jit
    def objective_function(x):
        psi = x.reshape(2*norb, nocc)

        if "s2" in projector:
            if ("ext" in projector) and (ext_ops is not None):
                return get_ext_s2_singlet_projected_energy_jax(
                    psi,
                    h1,
                    chol,
                    enuc,
                    list(ext_ops.values())[0], # TODO: Only works for 1 group now!
                    list(ext_chars.values())[0],
                    nalpha=nalpha,
                    nbeta=nbeta,
                    ngamma=ngamma,
                )

            else:
                return get_s2_singlet_projected_energy_jax(
                    psi, h1, chol, enuc, nalpha=nalpha, nbeta=nbeta, ngamma=ngamma
                )

        elif "sz" in projector:
            if ("ext" in projector) and (ext_ops is not None):
                return get_ext_sz_projected_energy_jax(
                    psi,
                    h1,
                    chol,
                    enuc,
                    s,
                    sz,
                    list(ext_ops.values())[0], # TODO: Only works for 1 group now!
                    list(ext_chars.values())[0],
                    ngrid=nbeta,
                )
            else:
                return get_sz_projected_energy_jax(psi, h1, chol, enuc, s, sz, nbeta)

    @jit
    def gradient(x, *args):
        return (grad(objective_function)(x, *args)).real

    print("\n# Starting optimization...")
    x = psi.flatten()
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
            "maxiter": maxiter,
            # "ftol": 1.0e-10,
            # "maxcor": 1000,
            # "maxfun": 15000,
            "disp": True,
        },
    )
    
    if res.success: print(f'\n# {res.message}')
    else: print(f'\n# Unable to converge optimization')
    energy = res.fun
    psi = res.x.reshape(2*norb, nocc)
    return energy, psi
