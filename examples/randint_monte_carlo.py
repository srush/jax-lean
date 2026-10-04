"""Monte Carlo over explicit discrete values, using only standard JAX.

The first estimator averages 16 six-sided dice. The second averages eight
squared values on {0, 1/4, 1/2, 3/4}; it estimates that finite-grid mean,
not the continuous integral of x**2.
"""
import jax
import jax.numpy as jnp


def die_draw(key):
    return jax.random.randint(key, (), 1, 7).astype(jnp.float32)


def grid_index(key):
    return jax.random.randint(key, (), 0, 4).astype(jnp.float32)


@jax.jit
def grid_square(index):
    return (index / 4.0) ** 2


def die_estimate(draws):
    return jnp.mean(draws)


def grid_estimate(indices):
    return jnp.mean(grid_square(indices))


def mc_die_16(key):
    keys = jax.random.split(key, 16)
    return die_estimate(jax.vmap(die_draw)(keys))


def mc_grid_square_8(key):
    keys = jax.random.split(key, 8)
    return grid_estimate(jax.vmap(grid_index)(keys))
