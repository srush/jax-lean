"""Standard JAX only. Callables and n specialize when JAX builds the Jaxpr."""
import jax
import jax.numpy as jnp


def uniform_six(key):
    return jax.random.randint(key, (), 0, 6).astype(jnp.float32)


def times_100(value):
    return value * 100.0


def square_plus_one(value):
    return value * value + 1.0


def monte_carlo(key, sample, f, n):
    keys = jax.random.split(key, n)
    draws = jax.vmap(sample)(keys)
    values = jax.vmap(f)(draws)
    return jnp.mean(values)


def monte_carlo_4(key):
    return monte_carlo(key, uniform_six, times_100, 4)


def monte_carlo_square_8(key):
    return monte_carlo(key, uniform_six, square_plus_one, 8)


def sample_square_plus_one(key):
    return square_plus_one(uniform_six(key))
