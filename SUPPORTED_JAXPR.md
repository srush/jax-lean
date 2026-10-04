# Supported Jaxpr

This is the support inventory for JAX **0.8.0**. The translator consumes Jaxpr
objects, not Python source or printed IR. Support means translation into the
stated mathematical model, not verified equivalence to machine execution.
Unknown primitives and unsupported cases fail translation.

Update this file whenever extending the translator, with the restrictions and
a test or example exercising the extension. Python API names alone do not define
coverage: inspect the Jaxpr produced for the concrete input shapes and dtypes.

## Deterministic primitives

| Jaxpr primitive | Supported cases / interpretation |
| --- | --- |
| `add`, `add_any`, `sub`, `mul`, `div`, `neg`, `square`, `integer_pow` | Real field arithmetic; static integer powers, including negative exponents |
| `min`, `max`, `abs`, `eq`, `ne`, `lt`, `le`, `gt`, `ge` | Ideal real ordering and comparisons |
| `and`, `or`, `xor`, `not` | Boolean operands only |
| `select_n` | Boolean selector with two branches |
| `exp`, `log`, `sqrt`, `rsqrt`, `sin`, `cos`, `tanh` | `RealOps` semantics; mathlib's total real functions |
| `reshape`, `transpose`, `broadcast_in_dim`, `squeeze` | Static shapes; reshape supports dimension permutations |
| `slice`, `rev`, `concatenate` | Static indices, axes and strides |
| `reduce_sum`, `reduce_prod` | Static axes, including empty reductions; `mean` lowers to sum/division. Leading-axis `reduce_sum(axes=(0,))` emits named `Tensor.sumFirst`; other axes use coordinate expressions |
| `dot_general` | Batch dimensions and multiple contraction axes; standard matrix multiplication emits `Tensor.matmul`, row-vector–matrix multiplication emits `Tensor.vecmat` |
| `iota` | Floating result only |
| `convert_element_type` | Float-to-float, integer constants to float, and modeled sampled integers to float; rounding erased |
| `copy`, `stop_gradient` | Identity in the forward semantics |
| `jit` | Inline supported pure nested Jaxpr; special sampler boundary below |
| `custom_jvp_call` | Inline primal Jaxpr only |
| `scan` | Static unrolling, default maximum 32 iterations per scan; reverse scans supported |

Closed constants and literal, unused, multiple or empty outputs are supported.
Inputs/results follow flattened Jaxpr order; pytrees are not reconstructed.
Shapes are fixed by tracing. Numeric storage dtypes are float16/32/64; Boolean
tensors keep Boolean semantics. Already transformed `grad` and `vmap` programs
work when their emitted primitives fit this inventory; differentiation itself
is not verified.

Evidence: [translation tests](tests/test_translation.py),
[examples](examples/generate.py), [tensor semantics](JaxLean/Tensor.lean).

## Opt-in sampling: `random_model="uniform"`

| Jaxpr operation / boundary | Accepted form | Emitted model |
| --- | --- | --- |
| `jit` callee named `_randint` | Structural match against pinned JAX scalar int32 sampler; scalar typed key; literal nonempty bounds | `Rand.uniformInt lo (hi-lo)` |
| `random_split` | One scalar root key split once into a positive static one-dimensional count `n` | Tracked split-key batch; declares independent children under the selected model |
| `jit` callee named `_randint`, batched | Structural match against `vmap` of the scalar sampler; consumes exactly the tracked batch | `Rand.iid n (Rand.uniformInt lo (hi-lo))` |
| Subsequent conversion, arithmetic, shape operations and reduction | Supported deterministic operations above, after sampled integers are cast to float | Ordinary transpiled continuation inside `Rand.map` |

`_randint` is a **modeled library-call boundary**, not a Jaxpr primitive. Its
name alone is insufficient: the nested Jaxpr's primitives, parameters, shapes,
dtypes, literals and effects must match the pinned implementation. Its internal
`random_bits`, bit operations, remainders, etc. are **not** generally supported.
The currently matched key implementation is JAX's default typed Threefry key.

Each random program takes one scalar typed root key and returns one real scalar.
It has one sampling call, either scalar or vmapped over the split batch. Pure
jit wrappers and real parameters are supported. Repeating a sampled value
retains its dependence; broadcasting one draw does not create independent draws.

Rejected: raw/legacy key arrays, input key batches, repeated or nested splitting,
using the parent key after splitting, multiple sampling calls/key reuse, direct
`randint(key, shape=(n,))`, multidimensional key batches, dynamic bounds, empty
ranges, integer arithmetic on sampled values, random array/multiple outputs,
and other distributions. General key/dataflow semantics remain future work.

The product law is an **explicit ideal sampling specification**, not a proof
that the actual PRNG has exact uniformity or independence. Integer-to-float
conversion and subsequent operations also use ideal real arithmetic. The
translator never infers independence from the Python function's name or from
a reduction. It emits the reduction itself; separate proofs apply stdlib rules.

Evidence: [scalar tests](tests/test_random_translation.py),
[Monte Carlo tests](tests/test_monte_carlo.py),
[JAX example](examples/monte_carlo.py),
[generated code](JaxLean/Generated/MonteCarlo4.lean),
[application proofs](JaxLean/MonteCarloProofs.lean).

## Certifiable transpilation subset

`certify(closed_jaxpr, name=..., namespace=...)` emits the normal tensor function,
a separately imported shape-typed IR, and an equality theorem for all real
inputs. The artifact is certified **only after Lean checks that theorem**.
Ordinary `translate` and `transpile` do not automatically produce certificates.

| Jaxpr feature | Certificate coverage |
| --- | --- |
| `add`, `add_any`, `sub`, `mul`, `div`, `min`, `max` | Equal shapes, scalar broadcasting, expansion of a singleton leading axis, or a singleton last axis in rank two |
| `neg`, `abs`, `square`, `integer_pow` | Real pointwise operations; static integer powers, including negative ones |
| `broadcast_in_dim` | Scalar replication; identity; prepending one leading axis; expanding a singleton first axis; vector→column (`[n]`→`[n,1]`); rank-two singleton last-axis expansion |
| `transpose` | Rank-two swap, or identity at any rank |
| `dot_general` | Standard matrix–matrix, row-vector–matrix, and vector inner-product contractions; no batch dimensions |
| `reshape` | Any fixed source/target shapes with equal element count, including empty shapes; row-major order; `dimensions=None` |
| `rev` | Rank-one tensors with `dimensions=(0,)` |
| `reduce_sum` | `axes=(0,)` with arbitrary fixed trailing dimensions; `axes=(1,)` for rank-two tensors |
| `convert_element_type` | Float-to-float identity in the real abstraction, not a rounding proof |
| `copy`, `stop_gradient` | Forward-value identity |
| Values | Floating inputs and finite scalar literals; one output; unused values and identity outputs permitted |
| Binding | Explicit SSA environments; multiple inputs; shape-indexed references; literal outputs |
| `jit` via `certify_module` | Pure single-output nested ClosedJaxpr bodies, each satisfying this table; named calls with typed arguments and child certificates |

Float16/32/64 inputs are modeled as real inputs, not constrained to representable
machine values. Literals preserve their stored binary rational value. Division
uses total real-field semantics. Shapes are specialized, including empty axes.

Other cases fail closed: captured array constants, other broadcast layouts,
other reduction axes, reshape dimension permutations, rank-two-or-higher reversal,
nontrivial rank-three-or-higher transpose, batched/general
contractions, non-jit calls, scans, Boolean comparisons/selection, transcendentals,
random operations, multiple outputs, integer or Boolean inputs. The single-body
`certify` API still rejects jit; use `certify_module` for call graphs. Every child
is checked recursively, and unsupported children fail the whole module. The
call rule changes environments according to typed actual arguments. Call names
are presentation metadata, not a whitelist of trusted mathematical operations.
Both APIs consume only Jaxpr, and preserve its fixed shapes.

The model's reference sources and deliberate differences are recorded in
[SEMANTICS_REFERENCES.md](SEMANTICS_REFERENCES.md). These references guide the
model; there is no claim of verified conformance to JAX, XLA or StableHLO.

Evidence: [IR semantics and local rules](JaxLean/Jaxpr.lean),
[certified mean](JaxLean/Generated/CertifiedMean.lean),
[certified dense ReLU layer](JaxLean/Generated/CertifiedLayer.lean),
[certified Gram matrix](JaxLean/Generated/CertifiedGram.lean),
[transformer call graph](JaxLean/Generated/Transformer.lean),
[function-boundary tests](tests/test_transformer.py),
[tensor puzzle certificates and Monte Carlo kernels](examples/certify_more.py),
[pseudocode and sampling tests](tests/test_more_examples.py),
[certificate tests](tests/test_certification.py), including deliberately corrupted
arithmetic and references rejected by Lean. The Python Jaxpr importer remains
trusted. The certificate does not prove that the imported IR matches the Python
object, nor that real semantics matches floating-point/PRNG execution.

The new randint examples use the existing scalar/vmapped sampler specification;
there is no extension to PRNG support. Their pure `die_estimate` and `grid_estimate`
kernels are separately certified by `certify_module`. The theorem connecting a
random wrapper to its deterministic kernel is checked in
[RandintMonteCarloProofs](JaxLean/RandintMonteCarloProofs.lean).

## Reusable theorem coverage

This is distinct from primitive coverage. A supported primitive does not imply
that every desired property has an automatic proof.

- [Pseudocode](JaxLean/Pseudocode.lean): arbitrary-size accumulator-loop/finite-sum
  equality and row-major reshape/quotient-remainder equality, including empty
  dimensions. [TensorPuzzleProofs](JaxLean/TensorPuzzleProofs.lean) applies these
  to the generated functions and composes their boundary contracts.
- [MatrixRules](JaxLean/MatrixRules.lean): arbitrary row/column selection through
  query–key products, permutation of contracted axes, row normalization under
  row selection and column permutation, nonnegative normalized weights and
  row sum one for strictly positive scoring functions on nonempty rows.
  [TransformerProofs](JaxLean/TransformerProofs.lean) applies these rules to
  the traced two-block transformer; no transformer-specific primitive is added.
- [NormRules](JaxLean/NormRules.lean) and [TensorNorm](JaxLean/TensorNorm.lean):
  explicit L2 norm, coordinatewise domination, scaling, pointwise contraction,
  composition, matrix bounds using the Frobenius norm, symmetric componentwise
  clipping (`radius ≥ 0`) and radial clipping (`radius > 0`). The generic rules
  apply to all finite dimensions; the examples specialize shapes during tracing.
  Evidence: [JAX source](examples/vector_norms.py),
  [generated-program proofs](JaxLean/NormProofs.lean),
  [norm tests](tests/test_norm_rules.py).
  Vector `jnp.linalg.norm` and symmetric `jnp.clip` use existing lowered
  primitives; this does not claim support for every norm API/order or a new
  norm/clipping Jaxpr primitive. No spectral-norm computation or floating-point
  error bound is provided.
- [Batch](JaxLean/Batch.lean) and [BatchRules](JaxLean/BatchRules.lean): pure
  vmap composition/locality, linear maps commuting with sums, bilinear maps
  expanding one or both reductions, tensor and shared-matrix instances. No
  probability assumptions. Evidence: [batch tests](tests/test_batch_rules.py),
  including generated deterministic functions and a rejected diagonal-only
  bilinear expansion.
- [BatchProbability](JaxLean/BatchProbability.lean): measurable coordinate maps
  preserve an independent family under an arbitrary measure, without `Rand`.
  This preserves independence; it does not establish it from a vmap alone.
- `FiniteLaw.meanLinear`, `FiniteLaw.mean_sum`, `Rand.mean_map_sum`: expectation
  commutes with sums via the generic linear rule, including dependent terms.
- [SelectionRules](JaxLean/SelectionRules.lean): row selection through maps,
  pointwise binary operations, shared matrix multiplication and bias broadcast.
- [FiniteLaw](JaxLean/FiniteLaw.lean) and [Random](JaxLean/Random.lean): exact
  finite-law expectation, variance, covariance, affine transformations and
  independent sums.
- [Random](JaxLean/Random.lean): expression-level propagation through addition,
  scaling, division and constant shifts. Addition preserves a covariance term;
  nonlinear variance exposes second-moment obligations.
- [Independent](JaxLean/Independent.lean): expectation and variance of sums over
  independent coordinates, with a different function allowed at each coordinate.
  Proved by induction without enumerating the product sample space. These are
  reduction rules, not whole-estimator theorems.

Monte Carlo averaging is an application in `MonteCarloProofs.lean`: division
propagates variance with a squared denominator, the independent-sum rule adds
variances, and algebra closes the result. No `monteCarlo` definition or
`variance_monteCarlo` theorem is provided by the stdlib. The same rules apply to
weighted sums. Plain Lean rewriting currently drives propagation; arbitrary
nonlinear programs may require more moments or additional proofs.

The rules quantify over the number of coordinates and their functions. A
transpiled Jaxpr specializes Python callables and `n` during tracing; it does
not yet emit a higher-order or shape-polymorphic function. These rules add no new Jaxpr primitive coverage; leading-axis sums now preserve
a named operation for rewriting. JAX lowers vmap before translation, so source
vmap syntax is not reconstructed.

## Other unsupported cases and trust boundary

Runtime integer/complex arithmetic, integer selectors, bfloat16, nonfinite
constants, general PRNG/bit operations, effects, `while`, `cond`, dynamic shapes
and indexing, gather/scatter, convolutions, sorting, general floating exponents,
and unknown primitives are rejected. `jnp.var`'s NaN fallback currently prevents
translation; write the explicit population variance formula instead.

Floating literals preserve their exact stored rational value, but intermediate
rounding, overflow, NaN propagation and signed zeros are not modeled. Division
and transcendental functions use total field/mathlib semantics. The translator
and JAX tracing remain trusted; ordinary translation has no general preservation theorem or
machine-float refinement proof. The optional certified subset above does have
per-program equality proofs relative to the separately imported Lean IR. Lean checks the generated definitions and the
mathematical proofs about them.
