"""Recognize a pinned standard JAX call before abstracting its implementation.

No monkey-patching, source-code rewriting, or matching by function name alone.
The accepted callee must structurally match JAX 0.8.0's scalar int32 randint,
or its vmap over a one-dimensional batch of keys.
Its replacement is explicitly an ideal distribution specification.
"""
from functools import lru_cache

import jax
import jax.numpy as jnp
from jax.extend import core
import numpy as np

from ..jaxpr import TranslationError


def _signature(closed):
    """Alpha-invariant structural comparison; ignore source/debug metadata only."""
    jp = closed.jaxpr
    variables = {}

    def aval(a):
        return (tuple(a.shape), str(a.dtype), bool(getattr(a, "weak_type", False)))

    def atom(v):
        if isinstance(v, core.Literal):
            return ("literal", aval(v.aval), np.asarray(v.val).tobytes())
        if v not in variables:
            variables[v] = len(variables)
        return ("variable", variables[v], aval(v.aval))

    def param(v):
        if isinstance(v, core.ClosedJaxpr):
            return _signature(v)
        if isinstance(v, dict):
            return tuple((k, param(x)) for k, x in sorted(v.items()))
        if isinstance(v, (tuple, list)):
            return tuple(param(x) for x in v)
        if isinstance(v, np.ndarray):
            return (str(v.dtype), v.shape, v.tobytes())
        # Primitive parameters here are immutable dtype/sharding/mesh values.
        return v

    inputs = tuple(atom(v) for v in [*jp.constvars, *jp.invars])
    eqns = tuple((e.primitive, tuple(atom(v) for v in e.invars),
                  tuple(atom(v) for v in e.outvars), param(e.params), e.effects) for e in jp.eqns)
    return inputs, eqns, tuple(atom(v) for v in jp.outvars), param(closed.consts), jp.effects


@lru_cache(maxsize=32)
def _randint_signature(batch=None):
    fn = lambda key: jax.random.randint(key, (), 0, 6, dtype=jnp.int32)
    arg = jax.random.key(0)
    if batch is not None:
        fn = jax.vmap(fn)
        arg = jax.ShapeDtypeStruct((batch,), arg.dtype)
    ref = jax.make_jaxpr(fn)(arg)
    return _signature(ref.jaxpr.eqns[0].params["jaxpr"])


def uniform_bounds(eq):
    output_shape = tuple(eq.outvars[0].aval.shape) if eq.outvars else None
    if (len(eq.invars) != 3 or len(eq.outvars) != 1
            or output_shape is None or len(output_shape) > 1
            or np.dtype(eq.outvars[0].aval.dtype) != np.dtype("int32")):
        raise TranslationError("uniform model supports scalar int32 randint, optionally vmapped over split keys")
    batch = output_shape[0] if output_shape else None
    if tuple(eq.invars[0].aval.shape) != output_shape:
        raise TranslationError("vector draws require vmap over split keys, not randint(shape=...)")
    if _signature(eq.params["jaxpr"]) != _randint_signature(batch):
        raise TranslationError("randint callee does not match the pinned standard JAX implementation")
    bounds = eq.invars[1:]
    if any(not isinstance(v, core.Literal) for v in bounds):
        raise TranslationError("uniform model requires literal randint bounds")
    lo, hi = (int(v.val) for v in bounds)
    if not -(2**31) <= lo < hi <= 2**31 - 1:
        raise TranslationError("uniform model requires nonempty, in-range int32 bounds")
    return lo, hi, batch


def has_user_arrays(jp):
    """Choose presentation using Jaxpr values, excluding abstracted PRNG internals."""
    if any(v.aval.shape for v in [*jp.constvars, *jp.invars, *jp.outvars]):
        return True
    for eq in jp.eqns:
        if any(v.aval.shape for v in eq.outvars):
            return True
        if eq.primitive.name == "jit" and eq.params.get("name") == "_randint":
            continue
        if any(has_user_arrays(v.jaxpr) for v in eq.params.values() if isinstance(v, core.ClosedJaxpr)):
            return True
    return False
