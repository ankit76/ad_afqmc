from functools import partial

import jax.numpy as jnp
import numpy as np
from jax import custom_jvp, jit, lax, vmap

print = partial(print, flush=True)


@custom_jvp
@jit
def detach(x):
    return x


@detach.defjvp
def detach_grad(primals, tangents):
    return primals[0], 0.0 * tangents[0]


@custom_jvp
def _eigh(a):
    w, v = jnp.linalg.eigh(a)
    return w, v


@_eigh.defjvp
def _eigh_jvp(primals, tangents):
    a = primals[0]
    at = tangents[0]
    w, v = primal_out = _eigh(*primals)

    deg_thresh = 1.0e-5
    eji = w[..., np.newaxis, :] - w[..., np.newaxis]
    eji = jnp.array(jnp.where(eji == 0.0, 1.0, eji))
    eji = jnp.where(abs(eji) < deg_thresh, 1.0e200, eji)
    eye_n = jnp.eye(a.shape[-1])
    Fmat = jnp.reciprocal(eji) - eye_n
    dw, dv = _eigh_jvp_jitted_nob(v, Fmat, at)
    return primal_out, (dw, dv)


@jit
def _eigh_jvp_jitted_nob(v, Fmat, at):
    vt_at_v = jnp.dot(v.conj().T, jnp.dot(at, v))
    dw = jnp.diag(vt_at_v)
    dv = jnp.dot(v, jnp.multiply(Fmat, vt_at_v))
    return dw, dv


@jit
def qr_vmap_restricted(walkers):
    walkers, r = vmap(jnp.linalg.qr)(walkers)
    norm_factors = vmap(lambda x: jnp.prod(jnp.diag(x)))(r)
    return walkers, norm_factors


@jit
def qr_vmap_unrestricted(walkers):
    walkers[0], r_0 = vmap(jnp.linalg.qr)(walkers[0])
    walkers[1], r_1 = vmap(jnp.linalg.qr)(walkers[1])
    norm_factors_0 = vmap(lambda x: jnp.prod(jnp.diag(x)))(r_0)
    norm_factors_1 = vmap(lambda x: jnp.prod(jnp.diag(x)))(r_1)
    return walkers, jnp.array([norm_factors_0, norm_factors_1])


# modified cholesky for a given matrix
@partial(jit, static_argnums=(1, 2))
def modified_cholesky(mat, norb, nchol_max):
    diag = mat.diagonal()
    size = mat.shape[0]
    # norb = (jnp.sqrt(1+8*size).astype('int8') - 1)//2
    # nchol_max = size
    nu = jnp.argmax(diag)
    delta_max = diag[nu]

    def scanned_fun(carry, x):
        carry["Mapprox"] += carry["chol_vecs"][x] * carry["chol_vecs"][x]
        delta = diag - carry["Mapprox"]
        nu = jnp.argmax(jnp.abs(delta))
        delta_max = jnp.abs(delta[nu])
        # R = (carry['chol_vecs'][:x + 1, nu]).dot(carry['chol_vecs'][:x + 1, :])
        R = (carry["chol_vecs"][:, nu]).dot(carry["chol_vecs"])
        carry["chol_vecs"] = (
            carry["chol_vecs"].at[x + 1].set((mat[nu] - R) / (delta_max) ** 0.5)
        )
        return carry, x

    carry = {}
    carry["Mapprox"] = jnp.zeros(size)
    carry["chol_vecs"] = jnp.zeros((size, size))
    carry["chol_vecs"] = carry["chol_vecs"].at[0].set(mat[nu] / delta_max**0.5)
    carry, x = lax.scan(scanned_fun, carry, jnp.arange(nchol_max - 1))

    # tril = jnp.tril_indices(norb)

    # def scanned_fun_1(carry, x):
    #     chol_mat = jnp.zeros((norb, norb))
    #     chol_mat = chol_mat.at[tril].set(x)
    #     chol_mat = chol_mat.T.at[tril].set(x)
    #     return carry, chol_mat.reshape(-1)

    # _, chol_vecs = lax.scan(scanned_fun_1, 1.0, carry["chol_vecs"][:nchol_max])
    # return chol_vecs
    return carry["chol_vecs"][:nchol_max]
