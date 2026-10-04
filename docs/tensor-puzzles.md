# Tensor Puzzles coverage

20 of the 21 puzzles from [srush/Tensor-Puzzles](https://github.com/srush/Tensor-Puzzles/blob/ad5069f7ef48523f733d445a7f94e93427d64f2d/README.md),
revision `ad5069f7ef48523f733d445a7f94e93427d64f2d`, appear in the code-only
`puzzles.html` notebook. Source specifications are MIT licensed; the notice is
preserved in [TENSOR_PUZZLES_LICENSE](../examples/TENSOR_PUZZLES_LICENSE).

Each pair is ordinary JAX: an array implementation and an imperative implementation
using static Python loops and `.at[...].set`. Both are traced separately, imported
from Jaxpr, and certified. The application theorem equates the two functions; its
certificate equates their independently imported Jaxpr evaluations. No stage
inspects raw Python to infer behavior. Python source extraction is only for display.

| # | Puzzle | Traced input shapes | Static size | Proof |
| --- | --- | --- | --- | --- |
| 1 | `ones` | `none` | 3 | [Lean](../JaxLean/Examples/Puzzles/Ones.lean) |
| 2 | `sum` | `(4,)` | — | [Lean](../JaxLean/Examples/TensorLoopProofs.lean) |
| 3 | `outer` | `(2,), (3,)` | — | [Lean](../JaxLean/Examples/TensorLoopProofs.lean) |
| 4 | `diag` | `(3, 3)` | — | [Lean](../JaxLean/Examples/Puzzles/Diag.lean) |
| 5 | `eye` | `none` | 3 | [Lean](../JaxLean/Examples/Puzzles/Eye.lean) |
| 6 | `triu` | `none` | 3 | [Lean](../JaxLean/Examples/Puzzles/Triu.lean) |
| 7 | `cumsum` | `(3,)` | — | [Lean](../JaxLean/Examples/Puzzles/Cumsum.lean) |
| 8 | `diff` | `(3,)` | — | [Lean](../JaxLean/Examples/Puzzles/Diff.lean) |
| 9 | `vstack` | `(3,), (3,)` | — | [Lean](../JaxLean/Examples/Puzzles/Vstack.lean) |
| 10 | `roll` | `(3,)` | — | [Lean](../JaxLean/Examples/Puzzles/Roll.lean) |
| 11 | `flip` | `(4,)` | — | [Lean](../JaxLean/Examples/TensorLoopProofs.lean) |
| 12 | `compress` | — | — | Deferred |
| 13 | `pad_to` | `(3,)` | 5 | [Lean](../JaxLean/Examples/Puzzles/PadTo.lean) |
| 14 | `sequence_mask` | `(2, 3), (2,)` | — | [Lean](../JaxLean/Examples/Puzzles/SequenceMask.lean) |
| 15 | `bincount` | `(3,)` | 3 | [Lean](../JaxLean/Examples/Puzzles/Bincount.lean) |
| 16 | `scatter_add` | `(3,), (3,)` | 3 | [Lean](../JaxLean/Examples/Puzzles/ScatterAdd.lean) |
| 17 | `flatten` | `(2, 3)` | — | [Lean](../JaxLean/Examples/TensorLoopProofs.lean) |
| 18 | `linspace` | `(), ()` | 3 | [Lean](../JaxLean/Examples/Puzzles/Linspace.lean) |
| 19 | `heaviside` | `(3,), (3,)` | — | [Lean](../JaxLean/Examples/Puzzles/Heaviside.lean) |
| 20 | `repeat` | `(3,)` | 2 | [Lean](../JaxLean/Examples/Puzzles/Repeat.lean) |
| 21 | `bucketize` | `(3,), (3,)` | — | [Lean](../JaxLean/Examples/Puzzles/Bucketize.lean) |

## Scope and adaptations

The proofs quantify over all inputs at the listed shapes under the ideal-real
semantic model, with Bool predicates and signed Int32 indices. They are not
shape-polymorphic certificates or IEEE floating-point correctness theorems.
Counts and bucket indices are returned as float tensors; size arguments are static
Python integers. The original one-line/operator competition restrictions are not
imposed here (as with the existing four ports). `sum` returns a scalar instead of
a length-one array; linspace endpoints are scalars; repeat count is static.

`roll` moves left by one. `diff` retains the first input element.
`triu` constructs the binary upper-triangular matrix. `bucketize` places equality
in the next bucket; its max-of-matching-positions implementation also matches the
imperative last-match rule for unsorted boundaries, although upstream assumes sorted.

For bincount and scatter-add, the imperative ports enumerate bounded
destinations instead of writing to a runtime index. They have the same behavior on
the upstream valid-index domains. Out-of-range histogram/scatter links are ignored
by these ports. The comparison casts signed indices to float in JAX; the Lean model
uses exact integer-to-real conversion. Valid indices for these small traced shapes
are represented exactly. This does not add runtime-index scatter support.

Fast tests compare against the original imperative specifications, including
repeated links, equal bucket boundaries, padding
and truncation, and singleton/empty linspace. These extra shape checks are numerical
checks, not additional Lean certificates. Empty boundary arrays for bucketize and
empty input arrays for diff are not claimed.

The core addition is real-valued `iota`: a bounded coordinate map interpreted by
natural-number conversion to a real. The typed importer now reuses the existing
real static-scatter importer inside mixed-type programs, retaining its bounds,
collision and 256-update checks. It does not introduce a new scatter semantic rule.
Certificate references now quote Lean identifiers, so Python names such as `repeat`
remain legal. The support inventory records these changes.

Regenerate with `make generate`. `lake build` checks all 20 application proofs;
`make check` also checks artifact freshness and the fast Python suite. The numbered
registry and coverage test prevent silently dropping a puzzle from the notebook.

Compression (#12) is deferred: its certificate currently requires an impractically
expensive proof check. Its unfinished implementation and certificate are excluded
from the build and notebook. The registry retains all 21 upstream entries and
explicitly records this omission. Row helpers and the masked accumulator are
separately certified JAX functions shown in the notebook.

The notebook ends with five shared library lemmas and short Monte Carlo and
Noether physics examples, all rendered from their checked Lean declarations.
Sampling assumptions remain explicit in the displayed ideal-law statements.
