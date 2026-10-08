"""Ordinary JAX. Norm bounds live in Lean, not in annotations on this code."""
import jax
import jax.numpy as jnp


def vector_norm(x):
    return jnp.linalg.norm(x)


def linear_clip(x, weights, radius):
    return jnp.clip(x @ weights, -radius, radius)


def norm_clip(x, radius):
    # The theorem requires radius > 0, so zero vectors have a safe denominator.
    return x * (radius / jnp.maximum(radius, jnp.linalg.norm(x)))


def batch_norm_clip(x, radius):
    return jax.vmap(norm_clip, in_axes=(0, None))(x, radius)
