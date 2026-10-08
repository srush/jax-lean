"""Shared array helpers for the examples."""
import jax.numpy as jnp


def arange(n, *, dtype=jnp.float32):
    return jnp.arange(n, dtype=dtype)
