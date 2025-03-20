import numpy as np
import scipy as sp
from scipy.optimize import minimize

import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jax.scipy.special import factorial
from jax import grad, jit

def get_wigner_d(s, m, k, beta):
    """
    Wigner small d-matrix.
    """
    nmin = np.maximum(0, k - m).astype(int)
    nmax = np.minimum(s + k, s - m).astype(int)
    fac = np.sqrt(sp.special.factorial(s+k) * sp.special.factorial(s-k) *
                   sp.special.factorial(s+m) * sp.special.factorial(s-m))
    cumsum = 0.

    for n in range(nmin, nmax+1):
        denom = (sp.special.factorial(s+k-n) * sp.special.factorial(s-n-m) *
                 sp.special.factorial(n-k+m) * sp.special.factorial(n))
        sign = (-1.)**(n - k + m)
        cos = (np.cos(beta/2.))**(2*s - 2*n + k - m)
        sin = (np.sin(beta/2.))**(2*n - k + m)
        num = sign * cos * sin
        cumsum += (num / denom)
    
    return fac * cumsum

def get_wigner_d_jax(s, m, k, beta):
    """
    Wigner small d-matrix.
    """
    nmin = jnp.maximum(0, k - m)
    nmax = jnp.minimum(s + k, s - m)
    fac = jnp.sqrt(factorial(s+k) * factorial(s-k) * factorial(s+m) * factorial(s-m))
    nrange = jnp.arange(2*s+1, dtype=jnp.int32)

    def summand(n):
        # Condition to keep only n in [nmin..nmax].
        cond = (n >= nmin) & (n <= nmax)
        denom = (factorial(s+k-n) * factorial(s-n-m) *
                 factorial(n-k+m) * factorial(n))
        sign = (-1.)**(n - k + m)
        cos = (jnp.cos(beta/2.))**(2*s - 2*n + k - m)
        sin = (jnp.sin(beta/2.))**(2*n - k + m)
        num = sign * cos * sin
        val = num / denom
        return jnp.where(cond, val, 0.0)

    # Vectorize 'summand' over the array nrange and sum.
    cumsum = jnp.sum(jax.vmap(summand)(nrange))
    return fac * cumsum

def apply_Sz_projector(input_ket, m, ngrid):
    norb = input_ket.shape[0] // 2
    nocc = input_ket.shape[1]
    kets = np.zeros((ngrid, 2*norb, nocc), dtype=np.complex128)
    coeffs = np.zeros(ngrid, dtype=np.complex128)
    
    # Integrate gamma \in [0, 2pi) by quadrature.
    for ig in range(ngrid):
        gamma = 2 * np.pi * ig / ngrid
        
        # Coefficients.
        coeffs[ig] = np.exp(1.j * gamma * m)

        # Rotation matrix about the z-axis. Only need to consider the diagonal.
        rot_diag = np.ones(2*norb, dtype=np.complex128)
        rot_diag[:norb] *= np.exp(-1.j * gamma / 2.) # Acting on spin up.
        rot_diag[norb:] *= np.exp(1.j * gamma / 2.) # Acting on spin down.
        rot_mat = np.diag(rot_diag)
        kets[ig] = rot_mat @ input_ket
    
    coeffs /= ngrid # For numerical integration.
    return kets, coeffs

def apply_Sz_projector_jax(input_ket, m, ngrid):
    # input_ket should be a jax array, shape (2*norb, nocc).
    norb = input_ket.shape[0] // 2
    nocc = input_ket.shape[1]

    # We'll define a function that, for a single gamma, gives (ket, coeff).
    @jit
    def one_point(ig):
        gamma = 2. * jnp.pi * ig / ngrid
        
        # Coefficients.
        coeff_ig = jnp.exp(1.j * gamma * m) / ngrid

        # Rotation matrix about the z-axis. Only need to consider the diagonal.
        rot_diag_a = jnp.ones(norb) * jnp.exp(-1.j * gamma / 2.)
        rot_diag_b = jnp.ones(norb) * jnp.exp(1.j * gamma / 2.)
        rot_diag = jnp.concatenate([rot_diag_a, rot_diag_b])
        rot_mat = jnp.diag(rot_diag)
        ket_ig = rot_mat @ input_ket
        return ket_ig, coeff_ig

    # Vectorize over ig = 0..ngrid-1
    igs = jnp.arange(ngrid)
    kets, coeffs = jax.vmap(one_point, in_axes=0, out_axes=(0, 0))(igs)
    return kets, coeffs

def apply_S2_projector(input_ket, s, sz, ngrid_z, ngrid_y):
    """
    Project a broken symmetry HF state onto the manifold with quantum numbers
    s, sz.
    """
    norb = input_ket.shape[0] // 2
    nocc = input_ket.shape[1]
    kets = np.zeros((2*s+1, ngrid_z, ngrid_y, ngrid_z, 2*norb, nocc), dtype=np.complex128)
    coeffs = np.zeros((2*s+1, ngrid_z, ngrid_y, ngrid_z), dtype=np.complex128)
    
    # Gauss-Legendre quadrature for integral over beta.
    # x = cos(beta) in [-1, 1]
    xs, ws = np.polynomial.legendre.leggauss(ngrid_y)
    betas = np.arccos(xs)
    sorted_inds = np.argsort(betas)
    betas.sort()
    ws = ws[sorted_inds]
    
    kz_range = np.arange(-s, s+1, 1)
    prefactor = (2*s+1)/2. * 1./ngrid_z**2
    
    # TODO: Really only works for s = sz = 0 for now.
    for ikz, kz in enumerate(kz_range):
        for ig_alpha in range(ngrid_z):
            for ig_beta in range(ngrid_y):
                for ig_gamma in range(ngrid_z):
                    alpha = 2 * np.pi * ig_alpha / ngrid_z # Rotation around z-axis.
                    beta = betas[ig_beta] # Rotation around y-axis.
                    gamma = 2 * np.pi * ig_gamma / ngrid_z # Rotation around z-axis.

                    # Coefficients.
                    coeff = np.exp(1.j*alpha*sz) * np.exp(1.j*gamma*kz) * ws[ig_beta]
                    wignerd = get_wigner_d(s, sz, kz, beta)
                    coeffs[ikz, ig_alpha, ig_beta, ig_gamma] = coeff * wignerd

                    phi_alpha = np.zeros(2*norb, dtype=np.complex128)
                    phi_beta = np.zeros((2*norb, 2*norb), dtype=np.complex128)
                    phi_gamma = np.zeros(2*norb, dtype=np.complex128)

                    # Sz. Rotation around z-axis.
                    phi_gamma[:norb] = np.exp(-1.j * gamma/2.)
                    phi_gamma[norb:] = np.exp(1.j * gamma/2.)

                    # Sy. Rotation around y-axis.
                    phi_beta[:norb, :norb] = np.cos(beta/2.) * np.eye(norb)
                    phi_beta[:norb, norb:] = np.sin(beta/2.) * np.eye(norb)
                    phi_beta[norb:, :norb] = -np.sin(beta/2.) * np.eye(norb)
                    phi_beta[norb:, norb:] = np.cos(beta/2.) * np.eye(norb)

                    # Sz. Rotation around z-axis.
                    phi_alpha[:norb] = np.exp(-1.j * alpha/2.)
                    phi_alpha[norb:] = np.exp(1.j * alpha/2.)

                    rotmat = np.diag(phi_alpha) @ phi_beta @ np.diag(phi_gamma)
                    kets[ikz, ig_alpha, ig_beta, ig_gamma] = rotmat @ input_ket

    coeffs *= prefactor
    return kets, coeffs

def apply_S2_projector_jax(input_ket, s, sz, ngrid_z, ngrid_y):
    # We'll define a function that, for a single gamma, gives (ket, coeff).
    norb = input_ket.shape[0] // 2
    nocc = input_ket.shape[1]
    
    @jit
    def build_rotation_matrix(alpha, beta, gamma):
        # Sz. Rotation around z-axis.
        phi_alpha = jnp.concatenate([
            jnp.exp(-1j * alpha/2.) * jnp.ones(norb, dtype=jnp.complex128),
            jnp.exp(1j * alpha/2.) * jnp.ones(norb, dtype=jnp.complex128)])
        phi_gamma = jnp.concatenate([
            jnp.exp(-1j * gamma/2.) * jnp.ones(norb, dtype=jnp.complex128),
            jnp.exp(1j * gamma/2.) * jnp.ones(norb, dtype=jnp.complex128)])

        # Sy. Rotation around y-axis.
        c = jnp.cos(beta/2.)
        s = jnp.sin(beta/2.)
        top_left  =  c * jnp.eye(norb, dtype=jnp.complex128)
        top_right =  s * jnp.eye(norb, dtype=jnp.complex128)
        bot_left  = -s * jnp.eye(norb, dtype=jnp.complex128)
        bot_right =  c * jnp.eye(norb, dtype=jnp.complex128)
        top = jnp.concatenate([top_left, top_right], axis=1) # (norb, 2*norb)
        bot = jnp.concatenate([bot_left, bot_right], axis=1) # (norb, 2*norb)
        phi_beta = jnp.concatenate([top, bot], axis=0)       # (2*norb, 2*norb)

        rotmat = jnp.diag(phi_alpha) @ phi_beta @ jnp.diag(phi_gamma)
        return rotmat

    @jit
    def one_point(kz, alpha, beta, gamma, wbeta):
        """
        Compute the rotated ket & coefficient for one combination
        of (kz, alpha, beta, gamma).
        """
        rotmat = build_rotation_matrix(alpha, beta, gamma)
        rotket = rotmat @ input_ket # shape (2*norb, nocc)

        wignerd = get_wigner_d_jax(s, sz, kz, beta)
        coeff = jnp.exp(1j*alpha*sz) * jnp.exp(1j*gamma*kz) * wignerd * wbeta
        return rotket, coeff

    # Alpha, gamma, kz arrays.
    alphas = jnp.arange(ngrid_z) * 2*np.pi / ngrid_z
    gammas = jnp.arange(ngrid_z) * 2*np.pi / ngrid_z
    kz_range = jnp.arange(-s, s+1, 1)

    # Gauss-Legendre quadrature for integral over beta.
    # x = cos(beta) in [-1, 1]
    xs, ws = np.polynomial.legendre.leggauss(ngrid_y)
    betas = np.arccos(xs)
    sorted_inds = np.argsort(betas)
    betas.sort()
    ws = ws[sorted_inds]
    betas = jnp.array(betas)
    ws = jnp.array(ws)
    
    # Make a 4D mesh for (kz, alpha, beta, gamma) with shape
    # (2s+1, ngrid_z, ngrid_y, ngrid_z).
    kz_grid, alpha_grid, beta_grid, gamma_grid = jnp.meshgrid(
        kz_range, alphas, betas, gammas, indexing='ij')

    # Also broadcast the beta weights ws across the same shape.
    # ws has shape (ngrid_y,). We'll inject dimensions to match 
    # (2s+1, ngrid_z, ngrid_y, ngrid_z).
    ws_grid = ws[None, None, :, None] # shape (1, 1, ngrid_y, 1).
    # shape (2s+1, ngrid_z, ngrid_y, ngrid_z).
    ws_grid = jnp.broadcast_to(ws_grid, (2*s+1, ngrid_z, ngrid_y, ngrid_z))

    # Flatten all these grids so we can vmap over a 1D array.
    kz_range = kz_grid.ravel()
    alphas = alpha_grid.ravel()
    betas = beta_grid.ravel()
    gammas = gamma_grid.ravel()
    ws = ws_grid.ravel()

    # Vectorize over all flattened points.
    kets, coeffs = jax.vmap(one_point)(kz_range, alphas, betas, gammas, ws)
    kets = kets.reshape(2*s+1, ngrid_z, ngrid_y, ngrid_z, 2*norb, nocc)
    prefactor = (2*s+1)/2. * 1./ngrid_z**2
    coeffs = coeffs.reshape(2*s+1, ngrid_z, ngrid_y, ngrid_z)
    return kets, coeffs * prefactor

def get_real_wavefunction(kets, coeffs):
    ndim = kets.ndim

    if ndim == 3: # Sz projector.
        _, norbx2, nocc = kets.shape
        real_kets = np.zeros((2, *kets.shape), dtype=np.complex128)
        real_coeffs = np.zeros((2, *coeffs.shape), dtype=np.complex128)
        real_kets[0] = kets.copy()
        real_kets[1] = np.conj(kets)
        real_coeffs[0] = coeffs.copy() / np.sqrt(2.)
        real_coeffs[1] = np.conj(coeffs) / np.sqrt(2.)

    elif ndim == 6: # S2 projector.
        _, _, _, _, norbx2, nocc = kets.shape
        real_kets = np.zeros((2, *kets.shape), dtype=np.complex128)
        real_coeffs = np.zeros((2, *coeffs.shape), dtype=np.complex128)
        real_kets[0] = kets.copy()
        real_kets[1] = np.conj(kets)
        real_coeffs[0] = coeffs.copy() / np.sqrt(2.)
        real_coeffs[1] = np.conj(coeffs) / np.sqrt(2.)
    
    real_kets = real_kets.reshape(-1, norbx2, nocc)
    real_coeffs = real_coeffs.reshape(-1)
    return real_kets, real_coeffs

def get_real_wavefunction_jax(kets, coeffs):
    ndim = kets.ndim

    if ndim == 3: # Sz projector.
        _, norbx2, nocc = kets.shape
        real_kets = jnp.array([kets, kets.conj()])
        real_coeffs = jnp.array([coeffs, coeffs.conj()]) / jnp.sqrt(2.)

    elif ndim == 6: # S2 projector.
        _, _, _, _, norbx2, nocc = kets.shape
        real_kets = jnp.array([kets, kets.conj()])
        real_coeffs = jnp.array([coeffs, coeffs.conj()]) / jnp.sqrt(2.)
    
    real_kets = real_kets.reshape(-1, norbx2, nocc)
    real_coeffs = real_coeffs.reshape(-1)
    return real_kets, real_coeffs

def get_projected_energy(psi, sz, h1, chol, enuc, ngrid, projector='Sz'):
    norb = h1[0].shape[0]
    nchol = chol.shape[0]
    nocc = psi.shape[1]
    
    if projector == 'Sz':
        kets, coeffs = apply_Sz_projector(psi, sz, ngrid)
        #kets, coeffs = get_real_wavefunction(kets, coeffs)

    elif projector == 'S2':
        kets, coeffs = apply_S2_projector(psi, sz, ngrid)
        kets, coeffs = get_real_wavefunction(kets, coeffs)

    psiTconj = psi.T.conj()
    rotchol = np.zeros((2, nchol, nocc, norb), dtype=np.complex128)
    
    # TODO: parallelize.
    for i in range(nchol):
        rotchol[0, i] = psiTconj[:nocc, :norb] @ chol[i].reshape((norb, norb))
        rotchol[1, i] = psiTconj[:nocc, norb:] @ chol[i].reshape((norb, norb))

    enum, ovlp = 0., 0.

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
    psiTconj: shape (nocc, 2*norb).
    chol: shape (nchol, norb, norb).

    We want rotchol of shape (2, nchol, nocc, norb).
    """

    nocc = psiTconj.shape[0]   # assuming your code snippet used psiTconj[:nocc, ...]
    # norb from chol
    # let's assume chol is (nchol, norb, norb)
    nchol, norb, _ = chol.shape

    # Split alpha and beta blocks from psiTconj
    psi_alpha = psiTconj[:nocc, :norb]   # shape (nocc, norb)
    psi_beta  = psiTconj[:nocc, norb:]   # shape (nocc, norb)

    def multiply_chol(chol_i):
        # chol_i: shape (norb, norb)
        # Multiply alpha slice
        alpha_part = psi_alpha @ chol_i   # (nocc,norb) x (norb,norb) -> (nocc,norb)
        beta_part  = psi_beta  @ chol_i   # same shape
        return alpha_part, beta_part

    # Apply over axis 0 of chol (i.e. loop i=0..nchol-1)
    rot_alpha, rot_beta = jax.vmap(multiply_chol, in_axes=(0,))(chol)
    # rot_alpha, rot_beta each shape (nchol, nocc, norb)

    # Stack them along a new first dimension for the spin index
    rotchol = jnp.stack([rot_alpha, rot_beta], axis=0)
    # rotchol has shape (2, nchol, nocc, norb)
    return rotchol

def get_Sz_projected_energy_jax(psi, m, h1, chol, enuc, ngrid):
    norb = h1[0].shape[0]
    nchol = chol.shape[0]
    nocc = psi.shape[1]

    kets, coeffs = apply_Sz_projector_jax(psi, m, ngrid)
    #kets, coeffs = get_real_wavefunction(kets, coeffs)
    psiTconj = psi.T.conj()
    rotchol = build_rotchol(psiTconj, chol.reshape((-1, norb, norb)))
    
    def scan_fun(i, carry):
        enum, ovlp = carry
        enum_i = get_energy_jax(psi, kets[i], h1, rotchol, enuc)
        ovlp_mat = psiTconj @ kets[i]
        ovlp_i = jsp.linalg.det(ovlp_mat)
        enum += coeffs[i] * enum_i * ovlp_i
        ovlp += coeffs[i] * ovlp_i
        return jnp.array([enum.real, ovlp.real])

    # Initial values.
    init = jnp.zeros(2)

    # Run the loop from i = 0 to i = ngrid.
    enum, ovlp = jax.lax.fori_loop(0, kets.shape[0], scan_fun, init)
    return enum / ovlp

def get_S2_projected_energy_jax(psi, s, sz, h1, chol, enuc, ngrid_z, ngrid_y):
    norb = h1[0].shape[0]
    nchol = chol.shape[0]
    nocc = psi.shape[1]

    kets, coeffs = apply_S2_projector_jax(psi, s, sz, ngrid_z, ngrid_y)
    kets, coeffs = get_real_wavefunction_jax(kets, coeffs)
    psiTconj = psi.T.conj()
    rotchol = build_rotchol(psiTconj, chol.reshape((-1, norb, norb)))
    
    def scan_fun(i, carry):
        enum, ovlp = carry
        enum_i = get_energy_jax(psi, kets[i], h1, rotchol, enuc)
        ovlp_mat = psiTconj @ kets[i]
        ovlp_i = jsp.linalg.det(ovlp_mat)
        enum += coeffs[i] * enum_i * ovlp_i
        ovlp += coeffs[i] * ovlp_i
        return jnp.array([enum.real, ovlp.real])

    # Initial values.
    init = jnp.zeros(2)

    # Run the loop from i = 0 to i = ngrid.
    enum, ovlp = jax.lax.fori_loop(0, kets.shape[0], scan_fun, init)
    return enum / ovlp

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
    energy = enuc + 0.j

    # One-body energy,
    e1 = np.trace(h1[0] @ G[:norb, :norb]) + np.trace(h1[1] @ G[norb:, norb:])
    energy += e1

    # Two-body energy.
    ej, ek = 0., 0.
    W = np.zeros((2, nocc, nocc), dtype=np.complex128)

    for i in range(nchol):
        W[0] = rotchol[0, i] @ Ghalfa
        W[1] = rotchol[1, i] @ Ghalfb
        ej += 0.5 * (np.trace(W[0]) + np.trace(W[1]))**2
        ek -= 0.5 * (np.sum(W[0] * W[0].T) + np.sum(W[0] * W[1].T) + 
                     np.sum(W[1] * W[0].T) + np.sum(W[1] * W[1].T))
    
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
    energy = enuc + 0.j

    # One-body energy,
    e1 = jnp.trace(h1[0] @ G[:norb, :norb]) + jnp.trace(h1[1] @ G[norb:, norb:])
    energy += e1

    # Two-body energy.
    def scan_fun(i, carry):
        ej, ek = carry
        W = jnp.zeros((2, nocc, nocc), dtype=jnp.complex128)
        W0 = rotchol[0, i] @ Ghalfa
        W1 = rotchol[1, i] @ Ghalfb
        ej += 0.5 * (jnp.trace(W0) + jnp.trace(W1))**2
        ek -= 0.5 * (jnp.sum(W0 * W0.T) + jnp.sum(W0 * W1.T) + 
                       jnp.sum(W1 * W0.T) + jnp.sum(W1 * W1.T))
        return jnp.array([ej, ek])
    
    # Initial values.
    init = jnp.zeros(2, dtype=jnp.complex128)

    # Run the loop from i = 0 to i = nchol.
    ej, ek = jax.lax.fori_loop(0, nchol, scan_fun, init)
    energy += ej + ek
    return energy

def optimize(psi, nelec, h1, chol, enuc, s=0, ngrid_z=10, ngrid_y=None, 
             store_iters=False, maxiter=100, step=0.01, method='BFGS', 
             projector='Sz', verbose=False):
    norb = h1[0].shape[0]
    nchol = chol.shape[0]
    na, nb = nelec
    nocc = sum(nelec)
    sz = 0.5 * (na - nb)
    x = psi.flatten()
    energy_iters = []

    if projector == 'Sz':
        energy_init = get_Sz_projected_energy_jax(psi, sz, h1, chol, enuc, ngrid_z)
        print(f'\n# Initial projected energy = {energy_init}')

        @jit
        def objective_function(x):
            psi = x.reshape(2*norb, nocc)
            return get_Sz_projected_energy_jax(psi, sz, h1, chol, enuc, ngrid_z)

    elif projector == 'S2':
        if ngrid_y is None: ngrid_y = ngrid_z
        energy_init = get_S2_projected_energy_jax(
                            psi, s, sz, h1, chol, enuc, ngrid_z, ngrid_y)
        print(f'\n# Initial projected energy = {energy_init}')

        @jit
        def objective_function(x):
            psi = x.reshape(2*norb, nocc)
            return get_S2_projected_energy_jax(
                        psi, s, sz, h1, chol, enuc, ngrid_z, ngrid_y)

    @jit
    def gradient(x, *args):
        return jnp.array(grad(objective_function)(x, *args), dtype=np.float64)

    def callback(x, *args):
        energy = objective_function(x)
        print(f'projected E = {energy}')
        energy_iters.append(energy)

    res = minimize(
            objective_function,
            x,
            jac=gradient,
            tol=1e-8,
            method=method,
            callback=callback,
            options={
                "gtol": 1e-8,
                "eps": 1e-8,
                "ftol": 1e-8,
                "maxiter": 15000,
                "disp": verbose,
            }
        )

    energy = res.fun
    psi = res.x.reshape(2*norb, nocc)
    if store_iters: return energy, psi, energy_iters
    return energy, psi
