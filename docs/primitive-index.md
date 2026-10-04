# Jaxpr primitive index

Use this page to locate a definition; use [the support inventory](../SUPPORTED_JAXPR.md)
for accepted parameter combinations. The input is JAX 0.8.0 Jaxpr.

The two semantic dispatch points are [`Jaxpr.Op` / `Op.eval`](../JaxLean/Core/Jaxpr.lean)
and [`TypedJaxpr.Op` / `Op.eval`](../JaxLean/Core/TypedJaxpr.lean). Each evaluator
is adjacent to its syntax definition. `Program.eval` in the same files defines
SSA binding, return, and function calls. Neither file contains application proofs.

The Python dispatch points are [`Emitter.primitive`](../python/jaxlean/translate.py),
[`RealImporter.run`](../python/jaxlean/importers/real.py), and
[`TypedImporter.run`](../python/jaxlean/importers/typed.py).

## Certified real operations

Every constructor below is in [Core/Jaxpr.lean](../JaxLean/Core/Jaxpr.lean).
The emitter's tensor helpers live in [Core/Tensor.lean](../JaxLean/Core/Tensor.lean).

| Source Jaxpr operation | Normalized IR constructor(s) |
| --- | --- |
| `add`, `add_any`, `sub`, `mul`, `div` | `add`, `sub`, `mul`, `div` |
| `neg`, `square`, `integer_pow` | `neg`, `square`, `integerPow` |
| `abs`, `min`, `max` | `abs`, `min`, `max` |
| `exp`, `log`, `sqrt`, `rsqrt`, `sin`, `cos`, `tanh` | `exp`, `log`, `sqrt`, `rsqrt`, `sin`, `cos`, `tanh`; direct mathlib real interpretation |
| `copy`, `stop_gradient`, float-to-float `convert_element_type` | `copy` |
| `broadcast_in_dim` | `broadcast`, `prepend`, `expandFirst`, `appendOne`, `expandLast`, or `reindex` |
| `transpose` | `transpose2`, `reindex`, or identity `copy` |
| `slice`, `squeeze` | `reindex` |
| real `iota` | `iota` with a bounded coordinate map |
| `rev` | `reverseVec` or `reindex` |
| `reshape` | `reshape`; a permutation first emits `reindex` |
| `concatenate` | Binary `concatenate` steps |
| `dot_general` | `matmul`, `vecmat`, `dotVec`, or general `contract` |
| `reduce_sum` | `sumFirst`, `sumLast`, or general `reduceSum` |
| `reduce_max`, `reduce_min` | `reduceMax`, `reduceMin` over nonempty domains |
| `scatter`, `scatter-add` | Scalar `set`, `addAt`; update extraction uses `reindex` |

[Verification/JaxprRules.lean](../JaxLean/Verification/JaxprRules.lean) contains
local evaluator equalities. [Certificate.lean](../JaxLean/Verification/Certificate.lean)
uses those and direct unfolding. Reusable mathematical properties belong in
[Stdlib](../JaxLean/Stdlib.lean), not in the evaluator.

## Certified mixed-type operations

These constructors are in [Core/TypedJaxpr.lean](../JaxLean/Core/TypedJaxpr.lean).

| Source Jaxpr operation | Normalized typed IR constructor(s) |
| --- | --- |
| Real-only equation or subprogram inside a typed body | `real`, embedding the existing real IR |
| `eq`, `ne`, `lt`, `le`, `gt`, `ge` | `compareReal`, `compareInt`; `Comparison.eval` and `intEval` select the comparison |
| Signed int32 `add`, `sub`, `mul` | `intAdd`, `intSub`, `intMul` with wrapping `Int32` arithmetic |
| Boolean `and`, `or`, `xor`, `not` | `boolAnd`, `boolOr`, `boolXor`, `boolNot` |
| Boolean `select_n` | `select` |
| Boolean/int32 shape operations | `reindex` |
| Clipped `gather` | `gather`; the coordinate map uses [Core/Indexing.clipStart](../JaxLean/Core/Indexing.lean) |
| Int32-to-float `convert_element_type` | `intToReal` in the ideal-real model |

Scalar reals use `Atom.realLiteral` (numerator, positive denominator); Boolean
and Int32 constants use `Atom.literal`. Real-only IR uses `Atom.literal` for
rational syntax. `Var` and `Env` carry the relevant dtype/shape information.

## Normalization and boundaries

| Source feature | Where to look |
| --- | --- |
| Static shapes, axes, contractions, clipped-gather dimension numbers | [layout.py](../python/jaxlean/layout.py) |
| Static index scaffolding and expansion of scatter windows | [static_index.py](../python/jaxlean/static_index.py) |
| `max(-inf, x)` / `min(+inf, x)` identities | `layout.identity_extreme`; eliminated before finite-real literal parsing |
| `jit` function boundaries | `certify_module` in [certify.py](../python/jaxlean/certify.py); `Program.call` and `Args` in either core IR |
| Ordinary inlined `jit` / `custom_jvp_call`, bounded `scan` unrolling | `Emitter.program` / `Emitter.scan`; non-jit calls and scan have no certificate rule |
| Floating `iota`, `reduce_prod` | `Emitter.primitive`; no certificate rule |
| Matched `_randint` call and `random_split` | [random_spec.py](../python/jaxlean/random_spec.py), `Emitter.program`; explicit ideal-law specification, not general PRNG semantics |

Coordinate maps are part of the trusted import. A source primitive may produce
several normalized operations, and some transpiled operations have no certified
IR rule. Unsupported cases must fail rather than silently gain a certificate.
