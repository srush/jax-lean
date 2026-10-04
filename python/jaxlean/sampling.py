"""Finite scalar laws, JAX execution, and exact Lean moment claims.

Weights specify an ideal rational law. Sampling uses JAX's floating-point PRNG
API; no theorem certifies that implementation. `moments` enumerates the law
numerically; `to_lean` proves supplied exact claims about transpiled arithmetic.
"""
from dataclasses import dataclass
from fractions import Fraction
from itertools import product
from math import prod
from numbers import Integral
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np

from .translate import translate, TranslationError


@dataclass(frozen=True)
class _Source:
    values: tuple[float, ...]
    weights: tuple[int, ...]


@dataclass(frozen=True, init=False)
class Discrete:
    """A finite scalar random variable with explicit independent source draws.

    Discrete([-1, 2], weights=[1, 3]) has probabilities 1/4 and 3/4.
    Values are stored as float32; weights are exact nonnegative integers.
    `map` reuses a draw. `independent_map2` explicitly creates independent draws,
    even if passed the same Discrete object twice. There is no implicit resampling.
    """

    _sources: tuple[_Source, ...]
    _fn: Callable

    def __init__(self, values, weights=None):
        raw = np.asarray(values)
        if raw.ndim != 1 or raw.size == 0 or raw.dtype.kind not in "fiu":
            raise ValueError("values must be a nonempty one-dimensional real numeric sequence")
        with np.errstate(over="ignore", invalid="ignore"):
            stored = raw.astype(np.float32)
        if not np.all(np.isfinite(stored)):
            raise ValueError("values must be finite when stored as float32")
        ws = tuple(1 for _ in stored) if weights is None else tuple(weights)
        if len(ws) != len(stored):
            raise ValueError("one weight is required per value")
        if any(isinstance(w, (bool, np.bool_)) or not isinstance(w, Integral) or w < 0 for w in ws):
            raise ValueError("weights must be nonnegative integers")
        ws = tuple(int(w) for w in ws)
        if sum(ws) == 0:
            raise ValueError("at least one weight must be positive")
        object.__setattr__(self, "_sources", (_Source(tuple(map(float, stored)), ws),))
        object.__setattr__(self, "_fn", lambda x: x)

    @classmethod
    def _derived(cls, sources, fn):
        result = object.__new__(cls)
        object.__setattr__(result, "_sources", sources)
        object.__setattr__(result, "_fn", fn)
        # Fail closed on extra randomness, effects, unsupported primitives, or
        # non-scalar outputs. The Jaxpr is inspected, not sampled or optimized.
        closed = jax.make_jaxpr(fn)(*(jnp.array(0.0, dtype=jnp.float32) for _ in sources))
        if (len(closed.out_avals) != 1 or closed.out_avals[0].shape != ()
                or np.dtype(closed.out_avals[0].dtype).kind != "f"):
            raise ValueError("operations must return one floating scalar")
        translate(closed)
        return result

    def map(self, fn):
        """Transform the SAME draw; arbitrary supported scalar JAX arithmetic."""
        return self._derived(self._sources, lambda *xs: fn(self._fn(*xs)))

    def independent_map2(self, other, fn):
        """Combine independent draws, with exact product probabilities.

        For dependent reuse, write x.map(lambda v: fn(v, v)) instead.
        """
        if not isinstance(other, Discrete):
            raise TypeError("other must be a Discrete random variable")
        split = len(self._sources)
        return self._derived(self._sources + other._sources,
                             lambda *xs: fn(self._fn(*xs[:split]), other._fn(*xs[split:])))

    def _worlds(self, max_outcomes):
        count = prod(len(s.values) for s in self._sources)
        if not isinstance(max_outcomes, Integral) or max_outcomes < 1:
            raise ValueError("max_outcomes must be a positive integer")
        if count > max_outcomes:
            raise ValueError(f"enumeration has {count} outcomes; limit is {max_outcomes}")
        indices = list(product(*(range(len(s.values)) for s in self._sources)))
        columns = tuple(jnp.asarray([s.values[i[k]] for i in indices], dtype=jnp.float32)
                        for k, s in enumerate(self._sources))
        total = prod(sum(s.weights) for s in self._sources)
        probabilities = tuple(Fraction(prod(s.weights[i[k]] for k, s in enumerate(self._sources)), total)
                              for i in indices)
        return columns, probabilities

    def enumerate(self, *, max_outcomes=256):
        """Return JAX outcome values and exact Fraction probabilities.

        Duplicate output values keep separate worlds. Arithmetic on values uses
        JAX floats, not the exact real semantics of the exported Lean program.
        """
        columns, probabilities = self._worlds(max_outcomes)
        return jax.vmap(self._fn)(*columns), probabilities

    def moments(self, *, max_outcomes=256):
        """Numerical distribution mean/variance by enumeration, NOT sample estimates."""
        values, probabilities = self.enumerate(max_outcomes=max_outcomes)
        p = jnp.asarray([float(q) for q in probabilities], dtype=values.dtype)
        mean = jnp.sum(p * values)
        return mean, jnp.sum(p * (values - mean) ** 2)

    def sample(self, key, shape=()):
        """JIT-compatible pointwise execution with explicit JAX keys.

        Each independent source receives its own split key. Runtime probabilities
        and arithmetic are floating-point approximations to the specified law.
        The sample shape must be static under jit, as in jax.random.choice.
        """
        shape = tuple(shape)
        if any(isinstance(n, bool) or not isinstance(n, Integral) or n < 0 for n in shape):
            raise ValueError("shape must contain nonnegative integers")
        keys = jax.random.split(key, len(self._sources))
        draws = []
        for source, source_key in zip(self._sources, keys, strict=True):
            p = jnp.asarray([float(Fraction(w, sum(source.weights))) for w in source.weights],
                            dtype=jnp.float32)
            draws.append(jax.random.choice(source_key, jnp.asarray(source.values, dtype=jnp.float32),
                                           shape=(prod(shape),), p=p))
        return jax.vmap(self._fn)(*draws).reshape(shape)

    def to_lean(self, *, name="sampled", namespace="JaxLean.Generated",
                mean=None, variance=None, max_outcomes=256):
        """Export the law, transpiled outcomes, and optional exact moment proofs.

        Claims accept int, Fraction, or a rational string, NOT rounded floats.
        Lean must check the returned module. An incorrect claim fails checking.
        Rational/polynomial examples work with norm_num; other supported functions
        may require a user-written proof. No claim is silently admitted.
        """
        columns, probabilities = self._worlds(max_outcomes)
        closed = jax.make_jaxpr(lambda: jax.vmap(self._fn)(*columns))()
        # translate validates names, namespaces, and the complete arithmetic.
        program = translate(closed, name=name + "Outcomes", namespace=namespace)
        if len(closed.out_avals) != 1 or closed.out_avals[0].shape != (len(probabilities),):
            raise TranslationError("expected one scalar output per world")
        program = program.replace("import JaxLean.RealOps", "import JaxLean.RealOps\nimport JaxLean.FiniteLaw")
        n = len(probabilities)

        def rational(q):
            if isinstance(q, (float, np.floating)):
                raise ValueError("exact claims require int, Fraction, or a rational string")
            q = Fraction(q)
            return f"({q.numerator} / {q.denominator} : ℝ)"

        mass = ", ".join(rational(q) for q in probabilities)
        ns = ".".join(f"«{part}»" for part in namespace.split("."))
        lines = [program, f"namespace {ns}",
                 f"noncomputable def {name}Law : JaxLean.FiniteLaw (Fin {n}) where",
                 f"  mass := ![{mass}]",
                 "  nonneg := by intro i; fin_cases i <;> norm_num",
                 "  total := by norm_num [Fin.sum_univ_succ]", "",
                 f"noncomputable def {name}Value (i : Fin {n}) : ℝ := {name}Outcomes (i, ())", ""]
        for operation, claim in (("mean", mean), ("variance", variance)):
            if claim is not None:
                lines.append(f"theorem {name}_{operation} : {name}Law.{operation} {name}Value = {rational(claim)} := by")
                if operation == "variance":
                    lines.append("  rw [JaxLean.FiniteLaw.variance_eq_second_moment]")
                lines += ["  norm_num [JaxLean.FiniteLaw.mean,",
                          f"    {name}Law, {name}Value, {name}Outcomes,",
                          "    JaxLean.Tensor.map, JaxLean.Tensor.map₂, JaxLean.Tensor.scalar,",
                          "    JaxLean.Tensor.reindex, JaxLean.Tensor.ofArray,",
                          "    JaxLean.Index.equivFin_vector_val, Fin.sum_univ_succ]", ""]
        return "\n".join([*lines, f"end {ns}", ""])
