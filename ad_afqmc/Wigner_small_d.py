from jax.scipy.special import gamma
from jax import vmap, lax
import jax.numpy as jnp

def wigner_small_d(j, mp, m, beta):
    """
    Compute small Wigner d-matrix element
    """
    j, mp, m = jnp.array(j), jnp.array(mp), jnp.array(m)
    if abs(m) > j or abs(mp) > j:
        return 0.0
    
    prefactor = jnp.sqrt(gamma((j + m + 1).astype(int)) * gamma((j - m + 1).astype(int)) * 
                        gamma((j + mp + 1).astype(int)) * gamma((j - mp + 1).astype(int)))
    
    k_min = (jnp.maximum(0, m - mp)).astype(int)
    k_max = (jnp.minimum(j + m, j - mp)).astype(int)

    def wignerSummation(carry, k):
        numerator = (-1)**(k - m + mp)
        denominator = (gamma(k + 1) * gamma((j + m - k + 1).astype(int)) * 
                        gamma((j - mp - k + 1).astype(int)) * gamma((mp - m + k + 1).astype(int)))
        
        cos_term = (jnp.cos(beta/2))**(2*j + m - mp - 2*k)
        sin_term = (jnp.sin(beta/2))**(mp - m + 2*k)
        
        carry += numerator / denominator * cos_term * sin_term
        return carry, None

    sum_term, _ = lax.scan(wignerSummation, 0., jnp.arange(k_min, k_max + 1))
    
    return prefactor * sum_term
