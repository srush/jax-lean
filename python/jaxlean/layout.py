"""Bounded coordinate maps for fixed Jaxpr shapes."""
from math import prod
from .jaxpr import coords, index, lean_shape, TranslationError


def reduction(src, axes, flat=False):
    axes = tuple(axes)
    if len(set(axes)) != len(axes) or any(d < 0 or d >= len(src) for d in axes):
        raise TranslationError("invalid reduction axes")
    reduced = tuple(src[d] for d in axes)
    dst = tuple(n for d, n in enumerate(src) if d not in axes)
    remaining = iter(coords(dst))
    if flat:
        cs = coords(reduced, f"((Index.equivFin {lean_shape(reduced)}).symm j)")
    else:
        cs = coords(reduced, "j")
    ix = [cs[axes.index(d)] if d in axes else next(remaining) for d in range(len(src))]
    return reduced, prod(reduced), f"(fun i j => {index(ix)})"


def contraction(lhs, rhs, dimensions):
    (lc, rc), (lb, rb) = dimensions
    dst = tuple(lhs[d] for d in lb) + tuple(lhs[d] for d in range(len(lhs)) if d not in (*lc, *lb)) + tuple(rhs[d] for d in range(len(rhs)) if d not in (*rc, *rb))
    reduced = tuple(lhs[d] for d in lc)
    out = iter(coords(dst))
    li, ri = [None] * len(lhs), [None] * len(rhs)
    for l, r in zip(lb, rb, strict=True):
        li[l] = ri[r] = next(out)
    for c, l, r in zip(coords(reduced, "j"), lc, rc, strict=True):
        li[l] = ri[r] = c
    for ix in (li, ri):
        for d, v in enumerate(ix):
            if v is None:
                ix[d] = next(out)
    return reduced, f"(fun i j => {index(li)})", f"(fun i j => {index(ri)})"


def identity_extreme(eq):
    # JAX softmax emits max(-inf, finite_reduce_max). Erase only this identity,
    # before parsing literals; infinities anywhere else remain unsupported.
    from jax.extend import core
    import numpy as np
    op = eq.primitive.name
    if op not in ("max", "min") or len(eq.invars) != 2:
        return None
    for i, atom in enumerate(eq.invars):
        if isinstance(atom, core.Literal) and np.shape(atom.val) == ():
            identity = np.isneginf(atom.val) if op == "max" else np.isposinf(atom.val)
            other = eq.invars[1-i]
            if identity and not isinstance(other, core.Literal) and other.aval.shape == eq.outvars[0].aval.shape:
                return other
    return None


def gather_map(eq):
    """Jaxpr gather with clipped starts and no explicit batching dimensions."""
    from .jaxpr import shape
    p = eq.params
    d = p['dimension_numbers']
    src, idx, out = (shape(v.aval) for v in (*eq.invars, eq.outvars[0]))
    sizes = tuple(p['slice_sizes'])
    if p['mode'].name != 'CLIP':
        raise TranslationError("gather requires explicit mode='clip'; promise/fill modes are not modeled")
    if d.operand_batching_dims or d.start_indices_batching_dims:
        raise TranslationError('gather batching dimensions are not yet supported')
    if len(sizes) != len(src) or any(n <= 0 or n > s for n, s in zip(sizes, src)):
        raise TranslationError('gather requires nonempty in-bounds slice sizes')
    window_axes = [axis for axis in range(len(src)) if axis not in d.collapsed_slice_dims]
    output_batch = [c for axis, c in enumerate(coords(out)) if axis not in d.offset_dims]
    if not idx or idx[-1] != len(d.start_index_map):
        raise TranslationError('invalid gather index vector')
    components = []
    for axis, n in enumerate(src):
        offset = '0' if axis in d.collapsed_slice_dims else coords(out)[d.offset_dims[window_axes.index(axis)]] + '.val'
        if axis in d.start_index_map:
            value = f"ix {index([*output_batch, str(d.start_index_map.index(axis))])}"
            start = f"(Tensor.clipStart {n-sizes[axis]} ({value}))"
            components.append(f"⟨{start}.val + {offset}, by have := {start}.isLt; omega⟩")
        else:
            components.append(f"⟨{offset}, by omega⟩")
    return f"(fun ix i => {index(components)})"


def uses_indices(jp):
    from .jaxpr import kind
    return any(kind(v.aval) == 'int' for v in jp.invars) or any(
        eq.primitive.name == 'gather' or
        (eq.primitive.name == 'jit' and uses_indices(eq.params['jaxpr'].jaxpr))
        for eq in jp.eqns)


def concatenate_map(left, right, axis):
    out = list(left)
    out[axis] += right[axis]
    lc, rc = coords(out), coords(out)
    c = coords(out)[axis]
    lc[axis] = f"⟨{c}.val, by omega⟩"
    rc[axis] = f"⟨{c}.val - {left[axis]}, by omega⟩"
    mapping = f"(fun i => if h : {c}.val < {left[axis]} then Sum.inl {index(lc)} else Sum.inr {index(rc)})"
    return tuple(out), mapping
