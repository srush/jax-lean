# Semantic references and correspondence boundary

Reviewed 2026-10-04. The implementation stays pinned to JAX 0.8.0.
These references guide the specification; this file is not a conformance proof.

## What is published?

JAX publishes a [Jaxpr language description](https://docs.jax.dev/en/latest/601/jaxpr.html)
and [Autodidax](https://docs.jax.dev/en/latest/autodidax.html). These describe
its typed expression language and interpreter architecture. The current web
documentation can describe newer IR APIs than this project's pinned version.

The [interpreter tutorial](https://docs.jax.dev/en/latest/notebooks/Writing_custom_interpreters_in_Jax.html)
gives an executable environment-based interpretation: read atoms, evaluate an
equation's primitive, bind its results, then read outputs. Its simplified
interpreter does not cover all higher-order primitives. For our version, the
concrete implementation reference is
[`eval_jaxpr` in JAX 0.8.0](https://github.com/jax-ml/jax/blob/jax-v0.8.0/jax/_src/core.py).
The corresponding primitive implementation and lowering reference is
[`lax.py` in JAX 0.8.0](https://github.com/jax-ml/jax/blob/jax-v0.8.0/jax/_src/lax/lax.py).

JAX's [`lax` documentation](https://docs.jax.dev/en/latest/jax.lax.html) points to
[XLA operation semantics](https://openxla.org/xla/operation_semantics), while
noting that some JAX operations differ from XLA.

The [StableHLO specification](https://openxla.org/stablehlo/spec) provides
operation formulas, shape constraints and execution descriptions. It also has
a [reference interpreter](https://openxla.org/stablehlo/interpreter_status).
StableHLO is a different IR; neither that specification nor its interpreter
establishes correctness of JAX-to-StableHLO lowering or our real abstraction.

This review did not identify a complete machine-checked semantics of all Jaxpr
that we can import as an authoritative equivalence theorem. The practical
reference is the pinned JAX implementation plus its published explanations,
with StableHLO/XLA providing additional operation-level specifications.

## Correspondence table for the certifiable core

The rules below are **our** real-valued definitions. Reference links identify
where to inspect related operational behavior, not already-proved agreement.

| Jaxpr feature | Our Lean interpretation | Reference |
| --- | --- | --- |
| Pure single-result `jit` calls | Evaluate a separately imported callee in an environment of typed actual arguments, then bind its result | [JAX evaluator](https://github.com/jax-ml/jax/blob/jax-v0.8.0/jax/_src/core.py) |
| Variables, literals, let equations | Shape-typed environment lookup and sequential binding | [JAX evaluator](https://github.com/jax-ml/jax/blob/jax-v0.8.0/jax/_src/core.py) |
| `add`, `add_any`, `sub`, `mul`, `div`, `neg` | Pointwise exact real field operations | [JAX primitives](https://github.com/jax-ml/jax/blob/jax-v0.8.0/jax/_src/lax/lax.py) |
| `square`, `integer_pow` | Pointwise natural square or integer exponentiation | [JAX primitives](https://github.com/jax-ml/jax/blob/jax-v0.8.0/jax/_src/lax/lax.py) |
| `abs`, `min`, `max` | Pointwise real absolute value/order | [JAX primitives](https://github.com/jax-ml/jax/blob/jax-v0.8.0/jax/_src/lax/lax.py) |
| `broadcast_in_dim` subset | Scalar replication, prepending one axis, singleton first-axis expansion, vector→column, rank-two singleton last-axis expansion | [broadcast specification](https://openxla.org/stablehlo/spec#broadcast_in_dim) |
| Static `reshape` without permutation | Preserve the row-major flat coordinate between equal-size shapes | [reshape specification](https://openxla.org/stablehlo/spec#reshape) |
| Vector `rev` | Read input coordinate `n-1-i` | [reverse specification](https://openxla.org/stablehlo/spec#reverse) |
| Rank-two transpose | Swap the two coordinates | [transpose specification](https://openxla.org/stablehlo/spec#transpose) |
| Matrix, vector–matrix, and vector inner-product `dot_general` | Contract matching coordinates with a finite sum | [dot specification](https://openxla.org/stablehlo/spec#dot_general) |
| `reduce_sum` subset | Exact real finite sum over the leading axis or rank-two trailing axis | [reduce specification](https://openxla.org/stablehlo/spec#reduce) |
| Float-to-float `convert_element_type` | Identity in the real abstraction; rounding discarded | [conversion specification](https://openxla.org/stablehlo/spec#convert) |
| `copy`, `stop_gradient` | Identity in the forward-value model | [JAX APIs](https://docs.jax.dev/en/latest/jax.lax.html) |

## Deliberate differences

The StableHLO specification explicitly discusses implementation-defined reduction
schedules and non-associativity of floating addition. Our finite real sum has
neither rounding nor schedule-dependent results. See its
[reduction section](https://openxla.org/stablehlo/spec#reduce).

Our arithmetic omits finite precision, overflow, NaNs and signed zeros. Division
by zero uses the total real-field convention; integer powers follow the same
field model. Matrix precision settings are not a precision guarantee in this
model. Min/max operate on real values, with no NaN behavior. Casts preserve the
mathematical value: a test deliberately exhibits a JAX float32→float16→float32
rounding difference while its real-model identity certificate passes.

Jaxpr dtypes and shapes are checked before import, but the Lean theorem quantifies
over all real values of those shapes, not only representable floating values.
Random operations, effects, general control flow and non-jit calls are outside
this certificate fragment. `certify_module` handles pure single-result jit calls
whose bodies are recursively supported. No randomness claim is derived from these references.

## What checking establishes

1. JAX's structural checker accepts the input Jaxpr.
2. The trusted importer constructs an explicit Lean IR from that object.
3. Lean checks local semantic rules and each generated equality certificate.
4. Differential tests compare supported generated operations with JAX on finite
   examples; corruption tests verify that changed target operations fail proof.

The checked equality is `eval(imported_IR, inputs) = generated_function(inputs)`.
The importer, our selection of real semantics, and the relationship to JAX's
runtime remain outside that theorem. Merely adding a reference link does not
close any of those gaps. A future floating-point refinement would need an
explicit relation to the pinned JAX primitive/lowering behavior, including
precision and reduction-order allowances.

## Typed indices and clipped gather

- [JAX gather](https://docs.jax.dev/en/latest/_autosummary/jax.lax.gather.html)
  and [StableHLO gather](https://openxla.org/stablehlo/spec#gather) guide the
  slice/window dimension mapping. This implementation accepts explicit clip mode:
  each signed start is clamped to `[0, operand_size - slice_size]` before adding
  the output window offset. Explicit batching dimensions, empty windows, and
  promise/fill modes are rejected.
- Runtime index values use Lean `Int32`, with signed comparisons and wrapping
  addition/subtraction/multiplication. Conversion to real still erases machine
  float rounding. Tests include signed overflow followed by clipped gather.
- Nonempty `reduce_max`/`reduce_min` use finite folds over the reduction domain.
  The only admitted infinite-literal patterns are eliminated identities
  `max(-inf, x)` and `min(+inf, x)` in the finite-real model. This is needed by
  the pinned standard softmax Jaxpr; there is no general infinity or NaN model.
