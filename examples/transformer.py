"""A small bidirectional transformer, following srush/lean-transformer.

Ordinary JAX only. jit preserves useful function boundaries in Jaxpr.
We retain the article's 1 + ReLU normalization and pre-attention feedforward
layer. No positional features, masks, layer norm, residuals, or multiple heads.
"""
import jax
import jax.numpy as jnp


@jax.jit
def project(x, weight):
    return x @ weight


@jax.jit
def forward(x, weight):
    return jnp.maximum(project(x, weight), 0.0)


@jax.jit
def normalize(scores):
    weights = 1.0 + jnp.maximum(scores, 0.0)
    return weights / jnp.sum(weights, axis=1, keepdims=True)


@jax.jit
def attention(q, k, v):
    scores = q @ k.T
    return normalize(scores) @ v


@jax.jit
def transformer_block(x, weight, wq, wk, wv):
    h = forward(x, weight)
    q = project(h, wq)
    k = project(h, wk)
    v = project(h, wv)
    return attention(q, k, v)


def transformer(x, w0, q0, k0, v0, w1, q1, k1, v1):
    h = transformer_block(x, w0, q0, k0, v0)
    return transformer_block(h, w1, q1, k1, v1)
