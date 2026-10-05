import jax.numpy as jnp
from jax import lax


def variance(x):
    """Population variance, written explicitly to avoid jnp.var's NaN branch."""
    centered = x - jnp.mean(x)
    return jnp.mean(centered * centered)


def prefix_sum(x):
    return lax.scan(lambda carry, a: (carry + a, carry + a), jnp.array(0.0), x)
