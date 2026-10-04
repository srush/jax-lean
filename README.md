# JaxLean

A minimal Jaxpr → Lean translator for **ideal real arithmetic**. It takes a JAX
IR object and emits ordinary pure Lean functions that you can read, execute over
rationals, and prove theorems about over `ℝ`.

The core is about 330 lines of Python and 90 lines of Lean. There is no second
instruction language, optimizer, custom tactic, or dependency on Aeneas itself.

## Start here

Requires Python 3.11+ and [Lean's elan installer](https://github.com/leanprover/elan).
JAX is pinned to **0.8.0**, Lean/mathlib to **4.33.0**; these are the tested versions,
not a promise to track JAX's latest internal APIs.

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
lake exe cache get
python -m examples.generate
lake build
lake env lean JaxLean/Run.lean
python -m pytest -q
```

`lake build` checks the generated definitions and all proofs. `Run.lean` prints:

```text
[2]                    -- mean of [1, 3]
[1]                    -- population variance of [1, 3]
[1, 4, 2, 5, 3, 6]     -- transpose, flattened in row-major order
[4, 5, 10, 11]         -- matrix product
([6], [1, 3, 6])       -- scan: final carry and prefix sums
```

Read these files in order:

1. [The JAX examples](examples/generate.py).
2. [Generated mean](JaxLean/Generated/Mean.lean): two equations, two let-bindings.
3. [Tensor semantics](JaxLean/Tensor.lean): bounded indices and coordinate maps.
4. [Proofs](JaxLean/Proofs.lean): specifications and proofs about generated code.
5. [Translator](python/jaxlean/translate.py): the entire compiler in one file.

## Translate your own Jaxpr

```python
from pathlib import Path
import jax
import jax.numpy as jnp
from jaxlean import translate

closed = jax.make_jaxpr(lambda x: jnp.sum(x * x))(jnp.ones(3))
Path("Energy.lean").write_text(translate(closed, name="energy"))
```

Then run `lake env lean Energy.lean`. This produces a definition equivalent to:

```lean
def energy {R : Type} [Field R] (x : Tensor R [3]) : Tensor R [] :=
  let squared : Tensor R [3] := fun i => x i * x i
  let total : Tensor R [] := fun _ => ∑ k : Fin 3, squared (k, ())
  total
```

The API accepts a `ClosedJaxpr`, or a bare `Jaxpr` with `consts=...`. It does not
parse pretty-printed Jaxpr text. Arguments/results use Jaxpr's flattened order;
one result is a tensor, multiple results a Lean tuple, and no results `Unit`.
Shapes are specialized to the trace. Pytrees are not reconstructed.

## Design, following Aeneas

[Aeneas's overview](https://github.com/AeneasVerif/aeneas/blob/main/documentation/aeneas-overview.md)
and [tutorial](https://github.com/AeneasVerif/icfp-tutorial) provide the main pattern:
extract an IR, translate it to pure functions, then keep specifications/proofs
separate from generated definitions. We adopt that layout and proof workflow.
Jaxpr already has explicit functional let-bindings; it needs none of Aeneas's
machinery for Rust borrows. Because this fragment is total mathematical code,
we also do not need Aeneas's error/divergence monad. Unsupported source programs
fail during translation.

The representation is small:

```text
Index []          = Unit
Index (n :: rest) = Fin n × Index rest
Tensor R shape    = Index shape → R
```

An index into `[2, 3]` is `(row : Fin 2, column : Fin 3, ())`.
Transpose, broadcast, squeeze, reverse, and slicing reindex a function.
Reductions and contractions are finite sums/products. Reshape uses a proved
row-major equivalence with `Fin shape.prod`; `reshape_roundtrip` is proved.
Lean checks generated shapes and index bounds. No operation returns a default
element on a bad index.

We inline pure calls. Small `scan`s are statically unrolled, with a default
limit of 32 iterations (`max_scan_length=`); reverse scans preserve JAX's output
ordering. This avoids a loop logic in the first version. The bound applies per
scan; nested scans can still produce large output.

## The initial numerical subset

This is an **80/20 target**, not a measured claim of 80% Jaxpr coverage.

| Area | Supported |
| --- | --- |
| Arithmetic | `add`, `add_any`, `sub`, `mul`, `div`, `neg`, `square`, `integer_pow` (including negative exponents) |
| Ordering | `min`, `max`, `abs`, six comparisons; Boolean `and`, `or`, `xor`, `not`; Boolean `select_n` |
| Smooth functions | `exp`, `log`, `sqrt`, `rsqrt`, `sin`, `cos`, `tanh`, interpreted using `RealOps ℝ` |
| Shape/indexing | `reshape` (including dimension permutations), `transpose`, `broadcast_in_dim`, `squeeze`, static `slice`, `rev`, `concatenate` |
| Reductions | `reduce_sum`, `reduce_prod`, including empty reductions; means lowered to sum/division |
| Linear algebra | `dot_general`: batch dimensions and multiple contraction axes |
| Other | floating `iota`, float-to-float casts, integer constants cast to floats, `copy`, `stop_gradient` |
| Structure | closed constants, literal/multiple/unused outputs, pure `jit`, primal `custom_jvp_call`, short `scan` |

Already transformed `grad`/`vmap` programs work **when their emitted primitives
fall in this table**. We do not reimplement or verify JAX differentiation.

Rejected: runtime integer/complex arithmetic, integer selectors, NaN/infinite
constants, PRNG keys/bit operations, effects, `while`/`cond`, dynamic shapes and
indexing, gather/scatter, convolutions, sorting, general floating exponents,
and unrecognized primitives. `bfloat16` is not supported yet. Supported numeric
storage dtypes are float16/32/64; Boolean tensors retain Boolean semantics.

`jnp.var` currently includes a NaN fallback branch even for ordinary inputs.
The translator rejects it. [The example](examples/generate.py) writes the
population variance formula explicitly; no hidden special case drops that branch.

## Reals, execution, and future precision

Generated algebraic functions are polymorphic over a field. Instantiate them at
`ℝ` for theorems, or `ℚ` for exact `#eval`. This is one generated program with two
interpretations; rational testing does not establish real-valued correctness.
Transcendentals additionally require [RealOps](JaxLean/RealOps.lean). Its real
instance is noncomputable, and there is deliberately no fake rational instance.

Finite floating literals are lifted to their **exact stored binary rational**.
For example, a traced float32 `0.1` is not the real number `1/10`. Intermediate
rounding and float casts are erased in the real model. Comparisons compare
ideal values, so even branches may differ from floating execution.

The model uses mathlib's total operations: division by zero is zero, square root
of a negative real is zero, and `Real.log` is total (including nonpositive inputs).
To relate a theorem to usual numerical formulas, require nonzero divisors,
nonnegative square-root arguments, and positive logarithm arguments as needed.
No theorem here covers overflow, NaN propagation, signed zeros, or rounding error.

**A floating-point backend cannot be an instance of `Field`**: rounded addition
is not associative. Add a separate scalar-operation interface/semantics and a
refinement or error relation to the real model. Keep today's real theorems as
specifications. Do not assert field laws about machine floats.

## What is verified

[Proofs.lean](JaxLean/Proofs.lean) checks, without `sorry` or custom axioms:

- The generated two-sample mean equals `(x₀ + x₁)/2` and is invariant under swap.
- Transposing twice is identity, and summation is invariant under transpose.
- The generated population variance equals `(x₀ - x₁)²/4`, is nonnegative,
  and is invariant under a common shift.
- For independent real random variables with finite second moments, the
  **generated** mean has variance `(Var X + Var Y)/4`. If both variances are
  at most `v`, the estimate's variance is at most `v/2`.

The last proof uses mathlib probability, with independence and integrability
as explicit assumptions. It models supplied random samples, not JAX's PRNG.
The population variance statistic and the variance of an estimator are different
quantities; the examples cover each separately.

The Python translator, JAX tracing, and our choice of primitive semantics remain
trusted. Type-checking prevents ill-typed shapes/indexes; it does **not** prove
that every generated permutation or primitive implements JAX correctly. Tests
execute the emitted Lean over rationals and compare with JAX on finite examples.
Transcendentals have type-checking tests, not a rational execution comparison.

JAX's [IR documentation](https://docs.jax.dev/en/latest/jaxpr.html) and
[interpreter guide](https://docs.jax.dev/en/latest/notebooks/Writing_custom_interpreters_in_Jax.html)
describe its structure and execution. We do not assume those documents provide
a machine-checked specification. The compiler calls JAX's structural checker,
then Lean checks the emitted terms. **There is no end-to-end compiler correctness
theorem yet.**

Next useful steps: symbolic-size tensors and an `n`-sample variance theorem;
compositional scan translation; then a small deep embedding with denotational
semantics and checked per-program translation certificates. Add precision only
after choosing the intended relationship to this real semantics.

## Development

Edit Python examples or the translator, regenerate, and keep proofs separate.

```sh
python -m examples.generate --check   # verify generated files are current
lake build                          # kernel-check specifications and proofs
python -m pytest -q                 # compile/run differential and rejection tests
```

The differential suite leaves its emitted Lean in `tests/_generated/` for
inspection. Only project-local `.venv/` and `.lake/` are needed at runtime;
there are no absolute paths to another checkout in the package configuration.
