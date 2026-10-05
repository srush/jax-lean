import jax.numpy as jnp


def mlp(x, w1, b1, w2, b2):
    """A 2 → 3 → 2 forward pass; every row uses the same parameters."""
    hidden = jnp.maximum(x @ w1 + b1, 0.0)
    return hidden @ w2 + b2


def mlp_inputs(batch):
    # These values specify shapes/dtypes for tracing. Weights remain arguments.
    return (jnp.ones((batch, 2)), jnp.ones((2, 3)), jnp.ones(3),
            jnp.ones((3, 2)), jnp.ones(2))


def residual(x, w1, b1, w2, b2):
    """A different graph: 3 → 4 → 3, square activation, and a skip connection."""
    hidden = jnp.square(x @ w1 + b1)
    return x + (hidden @ w2 + b2)


def residual_inputs(batch):
    return (jnp.ones((batch, 3)), jnp.ones((3, 4)), jnp.ones(4),
            jnp.ones((4, 3)), jnp.ones(3))
