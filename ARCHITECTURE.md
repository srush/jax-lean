# Review map

Start with [the primitive index](docs/primitive-index.md) for a particular Jaxpr
operation, or read the layers below in order. Parameter restrictions remain in
[SUPPORTED_JAXPR.md](SUPPORTED_JAXPR.md).

## Lean layers

| Directory | Responsibility | May depend on project modules in |
| --- | --- | --- |
| [Core](JaxLean/Core.lean) | Tensor values, bounded indices, real operations, normalized IR syntax and evaluators | Core |
| [Verification](JaxLean/Verification.lean) | Local evaluator/translation lemmas and the certificate tactic | Core, Verification |
| [Stdlib](JaxLean/Stdlib.lean) | Reusable tensor, batch, norm, indexing, and probability mathematics | Core tensor/value definitions, Stdlib |
| [Generated](JaxLean/Generated) | Generated functions and imported IR/certificates; ideal sampling artifacts | Core, Verification, relevant Stdlib modules |
| [Examples](JaxLean/Examples.lean) | Application specifications and proofs using generated functions and shared lemmas | All preceding layers |

`import JaxLean` imports the public library, not examples. `lake build` also
checks the separate `JaxLean.Examples` default target. Tensor and IR declaration
names, such as `JaxLean.Tensor` and `JaxLean.Jaxpr.Program`, remain unchanged;
module import paths now identify the layer. The generic accumulation lemma
formerly under `Pseudocode` is now `Imperative.fold_add_eq`.

Core contains construction proofs needed for bounded coordinates and instances.
Reusable mathematical laws belong in Stdlib. Core does not import the
certificate tactic, batch algebra, probability, or application proofs. Stdlib
does not import either Jaxpr evaluator, Verification, or generated examples.
[Architecture checks](tests/test_architecture.py) enforce these import boundaries.

## Where the semantics lives

- [Core/Index.lean](JaxLean/Core/Index.lean): bounded coordinates and row-major indexing.
- [Core/Tensor.lean](JaxLean/Core/Tensor.lean): tensor operations emitted by the transpiler.
- [Core/Indexing.lean](JaxLean/Core/Indexing.lean): signed-index clipping.
- [Core/RealOps.lean](JaxLean/Core/RealOps.lean): the transcendental interface and its real interpretation.
- [Core/Jaxpr.lean](JaxLean/Core/Jaxpr.lean): real-valued SSA syntax (`Op`, `Program`) and direct evaluation.
- [Core/TypedJaxpr.lean](JaxLean/Core/TypedJaxpr.lean): heterogeneous SSA with real, Boolean, and Int32 values. Real subprograms reuse the real IR.
- [Verification/JaxprRules.lean](JaxLean/Verification/JaxprRules.lean): local equalities between IR evaluation and tensor operations.
- [Verification/Certificate.lean](JaxLean/Verification/Certificate.lean): proof automation composing those equalities and unfolding remaining semantic definitions.

This is a **normalized mathematical IR**, not a verbatim encoding of every JAX
primitive. Broadcasts can become coordinate maps; a static scatter can become
several scalar updates. Dtype, shape, and reference correctness are checked by
Lean. The Python importer still determines what the original Jaxpr means in
this IR. A certificate does not prove that normalization/layout decoding is
faithful to JAX, nor that ideal-real operations match machine floating point.

## Python stages

| File | Stage |
| --- | --- |
| [jaxpr.py](python/jaxlean/jaxpr.py) | Shared shape/dtype validation, names from Jaxpr metadata, coordinate/type notation |
| [layout.py](python/jaxlean/layout.py), [static_index.py](python/jaxlean/static_index.py) | Trusted static layout/index decoding, with no dependence on either renderer |
| [translate.py](python/jaxlean/translate.py) | Jaxpr → readable Lean tensor function |
| [importers/real.py](python/jaxlean/importers/real.py) | Jaxpr → real IR syntax; primitive dispatch is `RealImporter.run` |
| [importers/typed.py](python/jaxlean/importers/typed.py) | Jaxpr → typed IR syntax; dispatch is `TypedImporter.run`; real fragments delegate to `RealImporter` |
| [certify.py](python/jaxlean/certify.py) | Runs transpilation/import independently, emits equality obligations, preserves supported function boundaries |
| [random_spec.py](python/jaxlean/random_spec.py), [sampling.py](python/jaxlean/sampling.py) | Explicit ideal sampling specifications; not PRNG implementation proofs |
| [docs.py](python/jaxlean/docs.py) | Displays Python and checked Lean in the notebook; outside the semantic pipeline |

The importers do not use the function emitter or parse its output. Real literals
are encoded as rational syntax; Boolean/integer literals are encoded directly.
Layouts are deliberately shared trusted metadata, not independent validation.
Only the documentation layer reads Python source for display. Transpilation and
IR import consume Jaxpr objects.

## Examples built on shared mathematics

| Application | Shared ingredients |
| --- | --- |
| [Softmax, masking, embeddings](JaxLean/Examples/CommonPrimitiveProofs.lean) | `MatrixRules.normalizeRows_*`, `IndexingRules`, tensor contraction/index laws, child translation certificates |
| [Scatter examples](JaxLean/Examples/ScatterProofs.lean) | Core scalar updates and certificates; concrete specifications use finite cases, while general update laws live in `ScatterRules` |
| [Monte Carlo](JaxLean/Examples/MonteCarloProofs.lean) | Generic expectation/variance, independence, sums, and scalar transformations from Stdlib |
| [Noether flux updates](JaxLean/Examples/NoetherProofs.lean) | `Stdlib/Equivariance`: permutation, pointwise-operation, flux equivariance and conservation rules; generated roll helper certificates |
| [Transformer](JaxLean/Examples/TransformerProofs.lean) | `SelectionRules`, `MatrixRules`, and function-boundary certificates |
| [Norm bounds](JaxLean/Examples/NormProofs.lean) | `NormRules` and `TensorNorm` |
| [20 of 21 tensor puzzles](docs/tensor-puzzles.md) | Array/imperative Jaxpr pairs, shared tensor/index laws and certificates; application proofs in `Examples/Puzzles` |
| [Tensor puzzles](JaxLean/Examples/TensorPuzzleProofs.lean), [JAX loops](JaxLean/Examples/TensorLoopProofs.lean) | Generic folds/updates in `Stdlib/Imperative`, tensor/index laws, example-specific specifications in `Examples/Pseudocode` |

Examples may contain concrete finite-case proofs. They do not introduce trusted
primitive semantics or example-specific shortcuts in the importer. New reusable
mathematics should move to Stdlib; new primitive behavior belongs in Core plus
its Python import/transpilation rule and support inventory.

## Validation

`make check` checks generated-file freshness, builds both library and examples,
and runs fast Python checks (including architectural boundaries). Targeted Lean
integration checks remain opt-in with `pytest -m lean`; they include corrupted
certificates that must fail. The ordinary Python suite does not launch Lean once
per example.
