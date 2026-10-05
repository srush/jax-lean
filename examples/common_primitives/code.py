"""Unmodified JAX functions covering common typed and tensor operations."""
import jax
import jax.numpy as jnp


def masked_values(x, mask):
    return jnp.where(mask & (x > 0), x, 0.)


def batched_scores(q, k):
    return q @ jnp.swapaxes(k, -1, -2)


def row_softmax(x):
    return jax.nn.softmax(x, axis=-1)


def embeddings(table, ids):
    return jnp.take(table, ids, axis=0, mode='clip')


def shifted_embeddings(table, ids):
    return jnp.take(table, ids + jnp.int32(1), axis=0, mode='clip')


def rearrange(x, y):
    return jnp.concatenate((x, y), axis=1).transpose(2, 0, 1)


def extrema(x):
    return jnp.max(x, axis=(0, 2)) - jnp.min(x, axis=(0, 2))
