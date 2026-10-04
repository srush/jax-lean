"""Ordinary JAX scatter functions; only their Jaxprs enter the transpiler."""
import jax.numpy as jnp


def replace_middle(x, values):
    return x.at[1:3].set(values)


def replace_column(x, values):
    return x.at[:, 1].set(values)


def replace_selected(x, values):
    return x.at[jnp.array([0, 2])].set(values)


def accumulate_selected(x, values):
    return x.at[jnp.array([1, 1, 3])].add(values)
