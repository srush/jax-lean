# Supported Jaxpr

This is the support inventory for JAX **0.8.0**. The translator consumes Jaxpr
objects, not Python source or printed IR. Support means translation into the
stated mathematical model, not verified equivalence to machine execution.
Unknown primitives and unsupported cases fail translation.
Use the [primitive index](docs/primitive-index.md) to find the definition,
import rule, and evaluator for each operation.

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
| `scatter`, `scatter-add` | Static in-bounds scalar, window, and multiple-index updates; arbitrary valid window/scatter dimension mappings without explicit operand/index batching dimensions; at most 256 update elements. Replacement destinations must be disjoint; additive collisions accumulate in the real model. `FILL_OR_DROP`, `PROMISE_IN_BOUNDS`, and `CLIP` agree for accepted in-bounds windows |
| Integer index scaffolding | Integer/bool literals and captured constants; static `broadcast_in_dim`, `concatenate`, `lt`, integer `add`, boolean `select_n`, and integer/bool `convert_element_type`. No runtime integer arithmetic |
| `reduce_max`, `reduce_min` | Any static axes with nonempty reduction domains; finite real extrema. Empty reductions are rejected |
| `gather` | Real operand, signed int32 indices, explicit `CLIP` mode, positive static slice sizes; general offset/collapsed/start-index maps without explicit batching dimensions |
| Runtime indices | Signed `int32` inputs/results, wrapped `add`/`sub`/`mul`, signed comparisons, Boolean selection, shape operations, and conversion to ideal real values. Other widths and bit operations are rejected |
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
tensors keep Boolean semantics. Signed int32 indices use Lean `Int32`, including wraparound; they are not real numbers. Already transformed `grad` and `vmap` programs
work when their emitted primitives fit this inventory; differentiation itself
is not verified.

Evidence: [translation tests](tests/test_translation.py),
[examples](examples/generate.py), [tensor semantics](JaxLean/Core/Tensor.lean).

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
[application proofs](JaxLean/Examples/MonteCarloProofs.lean).

## Certifiable transpilation subset

`certify(closed_jaxpr, name=..., namespace=...)` emits the normal tensor function,
a separately imported shape-typed IR, and an equality theorem for all real
inputs. The artifact is certified **only after Lean checks that theorem**.
Ordinary `translate` and `transpile` do not automatically produce certificates.

| Jaxpr feature | Certificate coverage |
| --- | --- |
| `add`, `add_any`, `sub`, `mul`, `div`, `min`, `max` | Equal shapes, scalar broadcasting, or equal-rank singleton-axis broadcasting |
| `neg`, `abs`, `square`, `integer_pow` | Real pointwise operations; static integer powers, including negative ones |
| `broadcast_in_dim` | General static dimension maps, including singleton expansion |
| `transpose` | Any static permutation |
| `dot_general` | General fixed-shape contractions, including multiple contracted axes and batch dimensions. Existing matrix helpers remain named; other layouts use `Tensor.contract` |
| `reshape` | Any fixed source/target shapes with equal element count, including empty shapes; real tensors also support dimension permutations |
| `slice`, `squeeze` | Static in-bounds positive-stride slices; removing singleton dimensions. Imported as a shape-typed coordinate map |
| `scatter`, `scatter-add` | Same static subset above; imported as scalar `.set` / `.addAt` steps with `.reindex` extraction from the original update tensor. Independently evaluated with coordinate equality tests |
| Integer index scaffolding | Same static subset above, resolved during Jaxpr import; integer/bool captured constants are metadata only, not real-valued SSA variables |
| `rev` | Real tensors, any static axes |
| `concatenate` | Real tensors, any static axis and number of inputs |
| `reduce_sum` | Any static axes, including empty reduction domains |
| `reduce_max`, `reduce_min` | Any static axes with nonempty reduction domains |
| `exp`, `log`, `sqrt`, `rsqrt`, `sin`, `cos`, `tanh` | Mathlib real functions, matching the existing `RealOps ℝ` instance |
| `eq`, `ne`, `lt`, `le`, `gt`, `ge` | Real or signed int32 comparisons, with Boolean results |
| `and`, `or`, `xor`, `not`, `select_n` | Boolean logic; Boolean selection of real, Boolean, or int32 tensors with broadcasting |
| `gather` | Same clipped, non-batched subset above; bounded coordinate map dependent on runtime int32 indices |
| Integer SSA | Signed int32 input/output, wrapped `add`/`sub`/`mul`, comparisons, selection, reindexing, reshape without permutation, and int32-to-real casts |
| `convert_element_type` | Float-to-float identity and int32-to-real cast in the real abstraction, not a rounding proof; same-type integer/Boolean identity |
| `copy`, `stop_gradient` | Forward-value identity |
| Values | Real, Boolean, and signed int32 tensors; finite scalar literals; one output; unused values and identity outputs permitted. Captured integer/Boolean constants are allowed as typed literals or static metadata |
| Binding | Explicit SSA environments; multiple inputs; shape-indexed references; literal outputs |
| `jit` via `certify_module` | Pure single-output nested ClosedJaxpr bodies, each satisfying this table; named calls with typed arguments and child certificates |

Float16/32/64 inputs are modeled as real inputs, not constrained to representable
machine values. Literals preserve their stored binary rational value. Division
uses total real-field semantics. Shapes are specialized, including empty axes.

Real-only programs keep the existing `Jaxpr.Program` IR. Programs with Boolean
or runtime integer data use `TypedJaxpr.Program`, whose SSA references carry both
dtype and shape. Its real subprogram constructor reuses `Jaxpr.Program`; it does
not reinterpret Booleans or indices as real values. Mixed call graphs retain
function boundaries and use each child's checked certificate.

Other cases fail closed: captured floating array constants, `reduce_prod`
certificates, non-jit calls, scans, random operations, multiple outputs,
float-to-integer conversion, other integer widths, and unsupported combinations
of mixed dtypes. Runtime-index scatter and typed Boolean/integer concatenation
are not covered. Static real scatter updates within mixed typed bodies are covered. The single-body
`certify` API still rejects jit; use `certify_module` for call graphs. Every child
is checked recursively; unsupported children fail the module. Both APIs consume
only Jaxpr and preserve its fixed shapes.

The model's reference sources and deliberate differences are recorded in
[SEMANTICS_REFERENCES.md](SEMANTICS_REFERENCES.md). These references guide the
model; there is no claim of verified conformance to JAX, XLA or StableHLO.

Evidence: [IR semantics](JaxLean/Core/Jaxpr.lean),
[local verification rules](JaxLean/Verification/JaxprRules.lean),
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
[RandintMonteCarloProofs](JaxLean/Examples/RandintMonteCarloProofs.lean).

## Reusable theorem coverage

This is distinct from primitive coverage. A supported primitive does not imply
that every desired property has an automatic proof.

- [Pseudocode](JaxLean/Examples/Pseudocode.lean): arbitrary-size accumulator-loop/finite-sum
  equality and row-major reshape/quotient-remainder equality, including empty
  dimensions. [TensorPuzzleProofs](JaxLean/Examples/TensorPuzzleProofs.lean) applies these
  to the generated functions and composes their boundary contracts.
- [MatrixRules](JaxLean/Stdlib/MatrixRules.lean): arbitrary row/column selection through
  query–key products, permutation of contracted axes, row normalization under
  row selection and column permutation, nonnegative normalized weights and
  row sum one for strictly positive scoring functions on nonempty rows.
  [TransformerProofs](JaxLean/Examples/TransformerProofs.lean) applies these rules to
  the traced two-block transformer; no transformer-specific primitive is added.
- [NormRules](JaxLean/Stdlib/NormRules.lean) and [TensorNorm](JaxLean/Stdlib/TensorNorm.lean):
  explicit L2 norm, coordinatewise domination, scaling, pointwise contraction,
  composition, matrix bounds using the Frobenius norm, symmetric componentwise
  clipping (`radius ≥ 0`) and radial clipping (`radius > 0`). The generic rules
  apply to all finite dimensions; the examples specialize shapes during tracing.
  Evidence: [JAX source](examples/vector_norms.py),
  [generated-program proofs](JaxLean/Examples/NormProofs.lean),
  [norm tests](tests/test_norm_rules.py).
  Vector `jnp.linalg.norm` and symmetric `jnp.clip` use existing lowered
  primitives; this does not claim support for every norm API/order or a new
  norm/clipping Jaxpr primitive. No spectral-norm computation or floating-point
  error bound is provided.
- [Batch](JaxLean/Stdlib/Batch.lean) and [BatchRules](JaxLean/Stdlib/BatchRules.lean): pure
  vmap composition/locality, linear maps commuting with sums, bilinear maps
  expanding one or both reductions, tensor and shared-matrix instances. No
  probability assumptions. Evidence: [batch tests](tests/test_batch_rules.py),
  including generated deterministic functions and a rejected diagonal-only
  bilinear expansion.
- [BatchProbability](JaxLean/Stdlib/BatchProbability.lean): measurable coordinate maps
  preserve an independent family under an arbitrary measure, without `Rand`.
  This preserves independence; it does not establish it from a vmap alone.
- `FiniteLaw.meanLinear`, `FiniteLaw.mean_sum`, `Rand.mean_map_sum`: expectation
  commutes with sums via the generic linear rule, including dependent terms.
- [SelectionRules](JaxLean/Stdlib/SelectionRules.lean): row selection through maps,
  pointwise binary operations, shared matrix multiplication and bias broadcast.
- [FiniteLaw](JaxLean/Stdlib/FiniteLaw.lean) and [Random](JaxLean/Stdlib/Random.lean): exact
  finite-law expectation, variance, covariance, affine transformations and
  independent sums.
- [Random](JaxLean/Stdlib/Random.lean): expression-level propagation through addition,
  scaling, division and constant shifts. Addition preserves a covariance term;
  nonlinear variance exposes second-moment obligations.
- [Independent](JaxLean/Stdlib/Independent.lean): expectation and variance of sums over
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

Complex arithmetic, integer operations outside the signed int32 subset, integer selectors, bfloat16, general nonfinite
constants, general PRNG/bit operations, effects, `while`, `cond`, dynamic shapes
and indexing beyond clipped gather, general scatter beyond the static subset above, convolutions, sorting, general floating exponents,
and unknown primitives are rejected. `jnp.var`'s NaN fallback currently prevents
translation; write the explicit population variance formula instead.

The exact Jaxpr identities `max(-inf, x)` and `min(+inf, x)` (either argument
order, scalar identity, equal output/other-operand shape) are erased in the
ideal finite-real model. This admits JAX's standard softmax lowering without
admitting infinity-valued tensors. Other nonfinite literals remain rejected.

Floating literals preserve their exact stored rational value, but intermediate
rounding, overflow, NaN propagation and signed zeros are not modeled. Division
and transcendental functions use total field/mathlib semantics. The translator
and JAX tracing remain trusted; ordinary translation has no general preservation theorem or
machine-float refinement proof. The optional certified subset above does have
per-program equality proofs relative to the separately imported Lean IR. Lean checks the generated definitions and the
mathematical proofs about them.

## JAX loop specifications

[Tensor puzzles](examples/tensor_puzzles.py) now include `loop_sum`,
`loop_outer`, `loop_flip`, and `loop_flatten`, using ordinary Python `for`
loops and JAX `.at[index].set(value)`. JAX unrolls these loops for the traced
shapes before the Jaxpr reaches the transpiler; this does not add dynamic
`while` or certified `scan` support. Both the loop and vectorized Jaxprs
receive translation certificates. [TensorLoopProofs.lean](JaxLean/Examples/TensorLoopProofs.lean)
proves equality for all real inputs of the displayed shapes and composes the
certificates to equate the two IR evaluations.

## Static scatter extension

[Scatter examples](examples/scatter_updates.py) show contiguous slice replacement,
column replacement, multiple-index replacement, and repeated-index addition.
[ScatterProofs.lean](JaxLean/Examples/ScatterProofs.lean) gives readable function-boundary
specifications and composes them with translation certificates.

The static index decoder follows the window/scatter coordinate mapping in the
[StableHLO scatter specification](https://openxla.org/stablehlo/spec#scatter).
Jaxpr uses the final index-array axis as its index-vector dimension. We check
actual destinations, rather than trusting `unique_indices` or
`indices_are_sorted`. Conflicting replacements are rejected because update
order is unspecified. Addition can repeat destinations: `Tensor.scatterAdd_comm`
proves order independence over a commutative additive monoid. This does not
claim order independence of machine floating-point accumulation.

Runtime indices, explicit operand/index batching dimensions, out-of-bounds
windows (including clip/drop behavior), other scatter combiners, and scatters
larger than 256 update elements remain unsupported. Empty update tensors are
identity operations. Captured floating arrays remain outside the certified
fragment. Static decoding/expansion is part of the trusted Python importer;
Lean certificates verify the expanded IR against the generated function, not
this decoder against JAX. Neither stage reads Python source.

Evidence: [static-update tests](tests/test_static_updates.py) compare decoded
coordinates with JAX, check rejected cases, and verify that corrupted destination
indices and replacement of addition by assignment fail Lean certificates.

## Common primitives and function-boundary proofs

[common_primitives.py](examples/common_primitives.py) contains ordinary JAX
examples for masking, batched query/key products, standard `jax.nn.softmax`,
clipped embedding lookup, wrapped index arithmetic, concatenation/transpose,
and extrema. All seven receive Lean-checked certificates.
[CommonPrimitiveProofs.lean](JaxLean/Examples/CommonPrimitiveProofs.lean) proves the mask
specification, exact embedding lookup (with and without an in-range hypothesis),
and softmax nonnegativity and row sums of one. The softmax proofs reuse the
generic normalization lemmas in `MatrixRules`, rather than adding a softmax
axiom or trusted library-call shortcut.

Shape maps, contraction maps, and clipped-gather dimension-number decoding are
trusted importer work, shared where appropriate with the transpiler. Certificates
prove equality of the generated functions and imported mathematical IR; they do
not independently verify that layout decoding matches JAX. Gather's clipping is
specified explicitly, while `PROMISE_IN_BOUNDS` and `FILL_OR_DROP` are rejected.

Evidence: [common-primitive tests](tests/test_common_primitives.py). Fast checks
trace/generate accepted and rejected cases. Opt-in Lean checks batch certificates
into one module, compare concrete clipped/overflow indexing results with JAX,
and reject a corrupted selection branch. No new routine full-suite Lean subprocess
loop is introduced.

## Periodic flux examples (Noether port)

[Noether's advection and Burgers examples](docs/noether.md) exercise existing
`mul`, `sub`, `integer_pow`, `slice`, `concatenate`, and pure `jit` call support.
No `roll` primitive or trusted stencil shortcut is introduced. Four-cell
function-boundary proofs establish equivariance under every cyclic shift and
sum conservation, using size-generic permutation/flux lemmas in the stdlib.

## Tensor Puzzles coverage

[20 of 21 puzzles](docs/tensor-puzzles.md) have array/imperative JAX pairs and
function-boundary certificates at the documented fixed shapes. Compression (#12)
is deferred and has no active implementation or certificate.

* Real-valued `iota` is now certified. Its IR stores a bounded coordinate map and
  converts that coordinate's natural-number value to a real. This does not certify
  integer `iota` or float rounding of large coordinates.
* Static real `scatter`/`scatter-add` updates now work within mixed Boolean/Int32
  programs as well. The typed importer passes constant index metadata to the
  existing real importer; the same bounds, collision, mode and size restrictions
  apply. Runtime-index scatter remains unsupported.
* The puzzle implementations of bincount and scatter-add use masks and
  reductions, not new trusted primitives or runtime scatter support.
