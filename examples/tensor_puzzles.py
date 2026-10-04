"""JAX adaptations of Tensor Puzzles 2, 3, 11, and 17.

https://github.com/srush/Tensor-Puzzles
These illustrate the implementation-versus-pseudocode proof workflow; they
are not entries in the original restricted-operator competition.
"""
import jax
import jax.numpy as jnp


@jax.jit
def puzzle_sum(a):
    return a @ jnp.ones_like(a)


@jax.jit
def puzzle_outer(a, b):
    return a[:, None] * b[None, :]


@jax.jit
def puzzle_flip(a):
    return a[::-1]


@jax.jit
def puzzle_flatten(a):
    return a.reshape(-1)


def outer_flatten(a, b):
    return puzzle_flatten(puzzle_outer(a, b))
