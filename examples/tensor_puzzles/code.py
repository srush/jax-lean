"""JAX adaptations of 20 Tensor Puzzles (compression deferred).

https://github.com/srush/Tensor-Puzzles
These illustrate the implementation-versus-pseudocode proof workflow; they
are not entries in the original restricted-operator competition.
"""
import jax
import jax.numpy as jnp

from examples.arrays import arange


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


def loop_sum(a):
    out = jnp.zeros((1,), dtype=a.dtype)
    for i in range(a.shape[0]):
        out = out.at[0].set(out[0] + a[i])
    return out[0]


def loop_outer(a, b):
    out = jnp.zeros((a.shape[0], b.shape[0]), dtype=a.dtype)
    for i in range(a.shape[0]):
        for j in range(b.shape[0]):
            out = out.at[i, j].set(a[i] * b[j])
    return out


def loop_flip(a):
    out = jnp.zeros_like(a)
    for i in range(a.shape[0]):
        out = out.at[i].set(a[a.shape[0] - 1 - i])
    return out


def loop_flatten(a):
    out = jnp.zeros((a.size,), dtype=a.dtype)
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            out = out.at[i * a.shape[1] + j].set(a[i, j])
    return out


def puzzle_ones(n):
    return jnp.ones((n,), dtype=jnp.float32)


def loop_ones(n):
    out = jnp.zeros((n,), dtype=jnp.float32)
    for i in range(n):
        out = out.at[i].set(1.)
    return out


def puzzle_diag(a):
    return jnp.sum(a * puzzle_eye(a.shape[0]), axis=1)


def loop_diag(a):
    out = jnp.zeros((a.shape[0],), dtype=a.dtype)
    for i in range(a.shape[0]):
        out = out.at[i].set(a[i, i])
    return out


def puzzle_eye(n):
    i = arange(n)
    return jnp.where(i[:, None] == i[None, :], 1., 0.)


def loop_eye(n):
    out = jnp.zeros((n, n), dtype=jnp.float32)
    for i in range(n):
        out = out.at[i, i].set(1.)
    return out


def puzzle_triu(n):
    i = arange(n)
    return jnp.where(i[:, None] <= i[None, :], 1., 0.)


def loop_triu(n):
    out = jnp.zeros((n, n), dtype=jnp.float32)
    for i in range(n):
        for j in range(n):
            if i <= j:
                out = out.at[i, j].set(1.)
    return out


@jax.jit
def puzzle_cumsum(a):
    i = arange(a.shape[0])
    return jnp.sum(jnp.where(i[:, None] >= i[None, :], a[None, :], 0.), axis=1)


def loop_cumsum(a):
    out = jnp.zeros_like(a)
    total = 0.
    for i in range(a.shape[0]):
        total = total + a[i]
        out = out.at[i].set(total)
    return out


def puzzle_diff(a):
    return jnp.concatenate((a[:1], a[1:] - a[:-1]))


def loop_diff(a):
    out = jnp.zeros_like(a)
    out = out.at[0].set(a[0])
    for i in range(1, a.shape[0]):
        out = out.at[i].set(a[i] - a[i - 1])
    return out


def puzzle_vstack(a, b):
    return jnp.concatenate((a[None, :], b[None, :]), axis=0)


def loop_vstack(a, b):
    out = jnp.zeros((2, a.shape[0]), dtype=a.dtype)
    for i in range(a.shape[0]):
        out = out.at[0, i].set(a[i])
        out = out.at[1, i].set(b[i])
    return out


def puzzle_roll(a):
    return jnp.roll(a, -1)


def loop_roll(a):
    out = jnp.zeros_like(a)
    for i in range(a.shape[0]):
        out = out.at[i].set(a[(i + 1) % a.shape[0]])
    return out


def puzzle_pad_to(a, n):
    return jnp.concatenate((a, jnp.zeros((max(0, n - a.shape[0]),), dtype=a.dtype)))[:n]


def loop_pad_to(a, n):
    out = jnp.zeros((n,), dtype=a.dtype)
    for i in range(min(n, a.shape[0])):
        out = out.at[i].set(a[i])
    return out


def puzzle_sequence_mask(values, length):
    columns = arange(values.shape[1])
    return jnp.where(columns[None, :] < length[:, None].astype(jnp.float32), values, 0.)


@jax.jit
def loop_sequence_row(values, length):
    out = jnp.zeros_like(values)
    for j in range(values.shape[0]):
        out = out.at[j].set(jnp.where(j < length.astype(jnp.float32), values[j], 0.))
    return out


def loop_sequence_mask(values, length):
    out = jnp.zeros_like(values)
    for i in range(values.shape[0]):
        out = out.at[i].set(loop_sequence_row(values[i], length[i]))
    return out


def puzzle_bincount(a, n):
    bins = arange(n)
    return jnp.sum(jnp.where(a.astype(jnp.float32)[None, :] == bins[:, None], 1., 0.), axis=1)


@jax.jit
def loop_masked_sum(values, mask):
    total = 0.
    for i in range(values.shape[0]):
        total = total + jnp.where(mask[i], values[i], 0.)
    return total


def loop_bincount(a, n):
    out = jnp.zeros((n,), dtype=jnp.float32)
    for j in range(n):
        out = out.at[j].set(loop_masked_sum(jnp.ones(a.shape), a.astype(jnp.float32) == j))
    return out


def puzzle_scatter_add(values, link, n):
    bins = arange(n)
    return jnp.sum(jnp.where(link.astype(jnp.float32)[None, :] == bins[:, None], values[None, :], 0.), axis=1)


def loop_scatter_add(values, link, n):
    out = jnp.zeros((n,), dtype=values.dtype)
    for j in range(n):
        out = out.at[j].set(loop_masked_sum(values, link.astype(jnp.float32) == j))
    return out


def puzzle_linspace(start, stop, n):
    return start + (stop - start) * arange(n) / max(1, n - 1)


def loop_linspace(start, stop, n):
    out = jnp.zeros((n,), dtype=jnp.float32)
    for i in range(n):
        out = out.at[i].set(start + (stop - start) * i / max(1, n - 1))
    return out


def puzzle_heaviside(a, b):
    return jnp.where(a == 0., b, jnp.where(a > 0., 1., 0.))


def loop_heaviside(a, b):
    out = jnp.zeros_like(a)
    for i in range(a.shape[0]):
        out = out.at[i].set(jnp.where(a[i] == 0., b[i], jnp.where(a[i] > 0., 1., 0.)))
    return out


def puzzle_repeat(a, n):
    return jnp.broadcast_to(a, (n, a.shape[0]))


def loop_repeat(a, n):
    out = jnp.zeros((n, a.shape[0]), dtype=a.dtype)
    for i in range(n):
        for j in range(a.shape[0]):
            out = out.at[i, j].set(a[j])
    return out


def puzzle_bucketize(v, boundaries):
    positions = arange(boundaries.shape[0]) + 1.
    return jnp.max(jnp.where(v[:, None] >= boundaries[None, :], positions[None, :], 0.), axis=1)


def loop_bucketize(v, boundaries):
    out = jnp.zeros_like(v)
    for i in range(v.shape[0]):
        bucket = 0.
        for j in range(boundaries.shape[0]):
            bucket = jnp.where(v[i] >= boundaries[j], float(j + 1), bucket)
        out = out.at[i].set(bucket)
    return out
