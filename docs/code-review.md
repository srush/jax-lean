# Implementation review

This review follows the path from a traced Jaxpr through the readable Lean
function, imported IR, translation certificate, and application theorem.

## Findings corrected

- **Generated names could capture source arguments.** Coordinate binders such
  as `i`, elementwise binders such as `a0`, and unqualified callee/theorem
  references could collide with Jaxpr argument names. The emitter now reserves
  its local names and qualifies generated declarations. Adversarial-name
  regression tests compile the resulting certificates.
- **Named SSA could ignore an unused input's type annotation.** The macro now
  ascribes the complete input context to returns, operations, and call
  arguments, even when an input is unused. Negative tests cover all three.
- **Long SSA graphs could generate Lean keywords and too many unused aliases.**
  Names beyond the first alphabet use numbered identifiers. Each operation
  introduces aliases only for variables it references; the full typed context
  and duplicate-name checks remain. A regression compiles an 80-operation graph.
- **Transpose and reversal hid source parameters inside arbitrary functions.**
  Their IR constructors now carry the Jaxpr permutation/dimension lists.
  Lean checks the permutation, result shape, and axis validity and derives the
  coordinate map. Tests change valid parameters to incorrect ones and require
  the certificate to fail; invalid parameters must fail construction.
- **Certificate setup was duplicated between Python and Lean.** All routine
  proof steps now live in `jaxpr_certificate`; Python emits the IR/function
  references and direct callee certificates. Explicit rewrite proofs are
  retained to avoid expensive repeated kernel reduction of nested environments.
- **Importer bookkeeping had redundant paths.** Normal operations and calls
  share environment advancement; primitive rendering returns an expression.
  Duplicate normalization dispatch was removed.

## Representation and proof path

There is one semantic IR, `JaxLean.Jaxpr.Program`, with dtype/shape-indexed
operands and de Bruijn variables. `jaxpr%` is syntax for that IR, not another
evaluator. Named and constructor presentations use the same primitive dispatcher.

There are intentionally two outputs from the input Jaxpr: a readable tensor
function and an independently imported IR. The importer must not obtain its
semantics by reading or invoking the function emitter. Merging those paths would
make matching mistakes harder to detect. Shared layout decoding remains trusted
metadata, rather than independent evidence.

Application theorems state properties of the generated tensor functions.
Translation certificates establish equality with `Program.eval`, for all
inputs of the traced shapes. Child certificates preserve function boundaries.
No certificate relies on computing one sample output.

## Remaining limits to inspect

This is a checked mathematical model of a supported Jaxpr fragment, not a
formalization of all JAX execution:

- Python still imports the original Jaxpr. Lean checks the imported term and
  its equivalence to the generated function, not the Python importer's
  faithfulness to JAX.
- Reshape, squeeze, slice, concatenation, contraction, reductions, gather, and
  static scatter still use decoded coordinate maps or plans. These are the
  remaining places where the IR is less literal than source Jaxpr. They are
  explicitly catalogued in the primitive index; they are not a second IR.
- Floating operations use mathematical reals, including the documented
  handling of casts and extrema identities. There is no IEEE-754 refinement
  theorem.
- The opt-in random model supplies ideal laws; it does not prove JAX PRNG
  uniformity or independence.
- Ordinary translation supports more programs than certification. Unsupported
  certificates fail closed; emitting Lean source alone does not certify it.

No new axioms, admitted proofs, or external proof oracles were introduced.
These limits should remain visible when presenting a theorem as a claim about
the original Python program.

## Review entry points

1. [Primitive inventory and trust boundary](primitive-index.md)
2. [Core IR and evaluator](../JaxLean/Core/Jaxpr.lean)
3. [Named syntax](../JaxLean/Verification/Syntax.lean)
4. [Independent importer](../python/jaxlean/importers/jaxpr.py)
5. [Certificate tactic](../JaxLean/Verification/Certificate.lean)
6. [Regression tests](../tests/test_review_regressions.py)
7. [Function-level transformer proofs](../examples/transformer/proofs/TransformerProofs.lean)

Validation for this review: 137 fast tests and 160 Lean integration tests passed
(including the corrected import-assembly and obsolete-IR assertions). The
library and example targets built successfully; generated artifacts passed
the freshness check.

Reproduce the broad checks with `make check-full`. Corruption tests deliberately
compile invalid candidate certificates and assert their rejection.
