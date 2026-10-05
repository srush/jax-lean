import jax
import jax.numpy as jnp


def quadratic(x):
    return x * x + 3.0 * x


def jvp_tangent(x, v):
    _, tangent = jax.jvp(quadratic, (x,), (v,))
    return tangent


def gradient(x):
    return jvp_tangent(x, jnp.ones_like(x))
