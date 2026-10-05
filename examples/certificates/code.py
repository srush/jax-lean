import jax.numpy as jnp


def mean(x):
    return jnp.mean(x)


def relu_layer(x, weights, bias):
    return jnp.maximum(x @ weights + bias, 0.0)


def gram(x):
    return x.T @ x
