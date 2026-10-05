"""Ordinary executable JAX. This module has no jaxlean dependency."""
import jax
import jax.numpy as jnp


def sample_times_100(key):
    draw = jax.random.randint(key, shape=(), minval=0, maxval=6)
    return draw.astype(jnp.float32) * 100.0


def sample_scaled(key, scale):
    draw = jax.random.randint(key, shape=(), minval=0, maxval=6)
    return draw.astype(jnp.float32) * scale
