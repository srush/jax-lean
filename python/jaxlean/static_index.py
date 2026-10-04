"""Decode literal integer index scaffolding from Jaxpr, never Python source."""
import numpy as np
from jax.extend import core


def integer_value(atom, env):
    if isinstance(atom, core.Literal) and np.asarray(atom.val).dtype.kind in 'iub':
        return np.asarray(atom.val)
    return env.get(atom) if not isinstance(atom, core.Literal) else None


def integer_equation(eq, env):
    if len(eq.outvars) != 1:
        return None
    try:
        dtype = np.dtype(eq.outvars[0].aval.dtype)
    except TypeError:
        return None
    if dtype.kind not in 'iub':
        return None
    values = [integer_value(v, env) for v in eq.invars]
    if any(v is None for v in values):
        return None
    op, p = eq.primitive.name, eq.params
    if op == 'broadcast_in_dim':
        layout = [1] * len(p['shape'])
        for axis, n in zip(p['broadcast_dimensions'], values[0].shape, strict=True):
            layout[axis] = n
        return np.broadcast_to(values[0].reshape(layout), p['shape']).copy()
    if op == 'concatenate':
        return np.concatenate(values, axis=p['dimension'])
    if op == 'convert_element_type':
        return values[0].astype(dtype)
    if op == 'lt':
        return np.less(*values)
    if op == 'add':
        return np.add(*values, dtype=dtype)
    if op == 'select_n' and len(values) == 3 and values[0].dtype.kind == 'b':
        return np.where(values[0], values[2], values[1]).astype(dtype)
    return None


def scatter_plan(eq, env):
    """Expand static scatter dimension numbers into (destination, update) pairs.

    All windows must be in bounds. Replacement destinations must be disjoint;
    additive collisions are allowed in the ideal-real model. Flags such as
    unique_indices are promises, not evidence, so we check actual coordinates.
    """
    from .jaxpr import TranslationError, shape, kind
    p = eq.params
    if len(eq.invars) != 3 or len(eq.outvars) != 1:
        raise TranslationError('scatter requires one operand and one update tensor')
    operand, indices, update = eq.invars
    s, u = shape(operand.aval), shape(update.aval)
    dims = p['dimension_numbers']
    additive = eq.primitive.name == 'scatter-add'
    combiner = p.get('update_jaxpr')
    if additive:
        # Validate the scalar combiner as well as the primitive name.
        if (combiner is None or combiner.constvars or combiner.effects
            or len(combiner.invars) != 2 or len(combiner.eqns) != 1
            or combiner.eqns[0].primitive.name != 'add'
            or tuple(combiner.eqns[0].invars) != tuple(combiner.invars)
            or tuple(combiner.eqns[0].outvars) != tuple(combiner.outvars)):
            raise TranslationError('scatter-add requires the scalar addition combiner')
    elif combiner is not None:
        raise TranslationError('scatter requires replacement semantics')
    if (not s or shape(eq.outvars[0].aval) != s
        or any(kind(v.aval) != 'real' for v in (operand, update, eq.outvars[0]))
        or dims.operand_batching_dims or dims.scatter_indices_batching_dims
        or p.get('update_consts', ())
        or p['mode'].name not in ('FILL_OR_DROP', 'PROMISE_IN_BOUNDS', 'CLIP')):
        raise TranslationError('unsupported scatter layout or mode')
    value = integer_value(indices, env)
    if value is None or value.dtype.kind not in 'iu' or value.ndim < 1:
        raise TranslationError('scatter requires statically known integer indices')
    window = tuple(dims.update_window_dims)
    inserted = tuple(dims.inserted_window_dims)
    mapped = tuple(dims.scatter_dims_to_operand_dims)
    operand_window = tuple(d for d in range(len(s)) if d not in inserted)
    batch = tuple(d for d in range(len(u)) if d not in window)
    if (window != tuple(sorted(set(window))) or any(d < 0 or d >= len(u) for d in window)
        or inserted != tuple(sorted(set(inserted))) or any(d < 0 or d >= len(s) for d in inserted)
        or len(set(mapped)) != len(mapped) or any(d < 0 or d >= len(s) for d in mapped)
        or len(window) != len(operand_window) or value.shape[-1] != len(mapped)
        or tuple(u[d] for d in batch) != value.shape[:-1]
        or any(u[w] > s[d] for w, d in zip(window, operand_window))):
        raise TranslationError('invalid scatter dimension numbers or window shape')
    if np.prod(u, dtype=object) > 256:
        raise TranslationError('static scatter exceeds the 256-element expansion limit')
    plan, seen = [], set()
    for source in np.ndindex(u):
        start = value[tuple(source[d] for d in batch)]
        dest = [0] * len(s)
        for axis, offset in zip(mapped, start, strict=True):
            dest[axis] = int(offset)
        for axis, w in zip(operand_window, window, strict=True):
            dest[axis] += source[w]
        dest = tuple(dest)
        if any(i < 0 or i >= n for i, n in zip(dest, s, strict=True)):
            raise TranslationError('scatter indices and entire windows must be statically in bounds')
        if not additive and dest in seen:
            raise TranslationError('scatter replacement destinations must not overlap')
        seen.add(dest)
        plan.append((dest, source))
    return plan
