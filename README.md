# JaxLean

A small **ordinary JAX → Jaxpr → Lean → stdlib proof** workflow. Source programs
use standard JAX. The transpiler emits readable Lean definitions, and separate
theorems refer directly to those definitions. Arithmetic uses an ideal real
model; the opt-in sampling model gives supported random calls finite probability
semantics.

The Lean semantic core defines tensor values and one dtype- and shape-indexed
Jaxpr evaluator. Python performs Jaxpr import and transpilation. Optional
certificates compare the generated function with that independently defined
Lean IR evaluator. A small tactic macro packages routine certificate
rewrites; every resulting proof is checked by Lean. There is no dependency on Aeneas.

For code review, start with [ARCHITECTURE.md](ARCHITECTURE.md) and the
[primitive index](docs/primitive-index.md). They separate Core, Verification,
Stdlib, generated artifacts, and application proofs.

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
lake build JaxLean JaxLeanExamples
lake env lean examples/basics/proofs/Run.lean
python -m pytest -q
```

`lake build` checks only the reusable library. `lake build JaxLeanExamples`
checks the generated example definitions and application proofs, which live
under `examples/<name>/proofs/` and `examples/<name>/generated/`. `make check` checks both. `Run.lean` prints:

```text
[2]                    -- mean of [1, 3]
[1]                    -- population variance of [1, 3]
[1, 4, 2, 5, 3, 6]     -- transpose, flattened in row-major order
[4, 5, 10, 11]         -- matrix product
([6], [1, 3, 6])       -- scan: final carry and prefix sums
```

Read these files in order:

1. [The JAX examples](examples/basics/code.py).
2. [Generated mean](examples/basics/generated/Mean.lean): two equations, two let-bindings.
3. [Tensor semantics](JaxLean/Core/Tensor.lean): bounded indices and coordinate maps.
4. [Proofs](examples/basics/proofs/Proofs.lean): specifications and proofs about generated code.
5. [Translator](python/jaxlean/translate.py) and its
   [sampler boundary checks](python/jaxlean/rewrites/random_spec.py).

## Transformer: proofs at function boundaries

Start with [ordinary JAX](examples/transformer/code.py), then read
[the function-level proofs](examples/transformer/proofs/TransformerProofs.lean), followed by
[the generated functions and certificates](examples/transformer/generated/Transformer.lean).
The example follows the bidirectional block in
[srush/lean-transformer](https://github.com/srush/lean-transformer/blob/main/Transformer.lean):
ReLU projection → Q/K/V projections → attention. Normalization is the article's
`1 + ReLU` divided by its row sum, rather than exponential softmax. The example
has two independently parameterized blocks, three tokens, and two hidden features.
It omits positional features, masks, layer norm, residuals, and multiple heads.

```python
@jax.jit
def normalize(scores):
    weights = 1.0 + jnp.maximum(scores, 0.0)
    return weights / jnp.sum(weights, axis=1, keepdims=True)

@jax.jit
def attention(q, k, v):
    scores = q @ k.T
    return normalize(scores) @ v
```

`jax.jit` leaves call boundaries in the Jaxpr. `certify_module` traverses that
call graph and generates a function, imported IR, and certificate for each body.
Python names and argument labels come from Jaxpr metadata; Python source is
never parsed. Ordinary Python calls that JAX has inlined have no boundary left
to preserve. Repeated calls to the same Jaxpr body share a certificate; distinct
bodies with the same name get distinct generated names.

```python
closed = jax.make_jaxpr(transformer)(x, *weights)
lean_source = certify_module(closed, name="transformer",
                             namespace="JaxLean.TransformerJax")
```

The generated network certificate explicitly names its direct dependency:

```lean
  jaxpr_certificate [transformer_ir, transformer,
    transformer_block_translation_correct]
```

The macro uses the block's equality theorem without expanding the block's
implementation. It contains only ordinary Lean proof tactics, not an axiom or
an external oracle. See [Certificate.lean](JaxLean/Verification/Certificate.lean).
The separate mathematical proof is similarly short:

```lean
theorem transformer_permute (perm : Fin 3 ≃ Fin 3) (x : Matrix 3 2)
    (w0 q0 k0 v0 w1 q1 k1 v1 : Matrix 2 2) :
    transformer (selectRows perm x) w0 q0 k0 v0 w1 q1 k1 v1 =
      selectRows perm (transformer x w0 q0 k0 v0 w1 q1 k1 v1) := by
  simp only [transformer, transformer_block_permute]
```

There are three proof layers:

1. [MatrixRules](JaxLean/Stdlib/MatrixRules.lean) proves general reindexing, contraction,
   and row-normalization laws for arbitrary dimensions and scoring functions.
2. [TransformerProofs](examples/transformer/proofs/TransformerProofs.lean) applies them at the named
   function boundaries. Normalized rows are nonnegative and sum to one;
   attention, the block, and the network commute with token permutations.
3. `certified_transformer_permute` uses the translation certificate to transfer
   the network theorem to the imported Jaxpr evaluator.

Token permutations are bijections. Arbitrary token selection can change the
attention denominator and output; a regression test exhibits this distinction.
The general library laws are dimension-polymorphic; the generated network and
its application proofs are specialized to the traced shapes. The theorem holds
for every real input and weight of those shapes, not just sample tensors.
It retains the documented trusted Python importer and ideal-real arithmetic
boundary, including the absence of a floating-point refinement proof.

```sh
python -m examples.transformer.generate        # regenerate
python -m examples.transformer.generate --check
lake build examples.transformer.proofs.TransformerProofs
python -m pytest tests/test_transformer.py -q -m ""
```

## More examples: randint Monte Carlo

[The JAX source](examples/randint_monte_carlo/code.py) uses `random.randint`, one
`random.split`, `vmap`, and an ordinary estimator function. Its deterministic
kernels have [die](examples/randint_monte_carlo/generated/DieEstimate.lean) and
[grid](examples/randint_monte_carlo/generated/GridEstimate.lean) translation certificates.
Read [RandintMonteCarloProofs](examples/randint_monte_carlo/proofs/RandintMonteCarloProofs.lean) for the
kernel contracts, sampling-boundary equalities, and moment proofs.

| Python function | Quantity estimated | Exact mean | Exact variance |
| --- | --- | --- | --- |
| `mc_die_16` | Average of 16 independent uniform draws from `{1,…,6}` | `7/2` | `35/192` |
| `mc_grid_square_8` | Average of 8 squared draws on `{0,1/4,1/2,3/4}` | `7/32` | `49/8192` |

The second target is a **finite-grid mean**, not the continuous integral of
`x²`. The proof computes the small single-draw moments and propagates them
through the estimator using the existing expectation, variance, division, and
independent-sum rules. It does not enumerate the product sample space.

The stochastic boundary is explicit: `randint` has an ideal uniform law and
the split batch has an iid product law under `random_model="uniform"`. This
specifies sampling, rather than proving JAX PRNG uniformity or independence.
The sampler's pinned Jaxpr is structurally checked by the existing recognizer.
The deterministic kernel certificates verify equality to their imported IR;
`certified_die_variance` and `certified_grid_variance` state the moment results
for that IR evaluator under the supplied sampling laws. All arithmetic remains
in the real model.

## More examples: JAX equals Lean pseudocode

These are adaptations of [Tensor Puzzles](https://github.com/srush/Tensor-Puzzles),
particularly puzzles 2 (sum), 3 (outer), 11 (flip), and 17 (flatten).
They illustrate verification, rather than the original restricted-operator contest.

Read these together:

1. [JAX implementations](examples/tensor_puzzles/code.py): dot with ones, broadcasting,
   reverse slicing, reshape, and a composition of outer product with flatten.
2. [Independent Lean pseudocode](examples/tensor_puzzles/proofs/Pseudocode.lean): an accumulator loop,
   explicit output-cell formulas, and row-major quotient/remainder indexing.
3. [The equality proofs](examples/tensor_puzzles/proofs/TensorPuzzleProofs.lean): each generated function
   equals its specification, then certificates connect the result to Jaxpr.

For example, the sum specification is an actual bounded accumulator loop:

```lean
def sum [Add R] [Zero R] (a : Tensor R [n]) : R :=
  Fin.foldl n (fun total i => total + a (i, ())) 0
```

Its JAX implementation uses `a @ jnp.ones_like(a)`. A generic loop-to-sum lemma
connects the two. The composite example needs only the child contracts:

```lean
theorem outer_flatten_matches_pseudocode (a : Tensor ℝ [2]) (b : Tensor ℝ [3]) :
    outer_flatten a b = Pseudocode.flatten (Pseudocode.outer a b) := by
  simp only [outer_flatten, flatten_matches_pseudocode, outer_matches_pseudocode]
```

Specifications are written independently and do not import generated code.
The loop and row-major indexing lemmas hold for arbitrary sizes, including
empty dimensions; checked-in Jaxpr examples specialize to vectors of length
4 and a `2×3` outer product. Their theorems cover every real input at those
shapes. Tests compare JAX, executable Lean functions, and the pseudocode, and
check that incorrect reversal/flatten order and variance scaling are rejected.

Generate and check both example sets:

```sh
python -m examples.generate
python -m examples.generate --check
lake build examples.randint_monte_carlo.proofs.RandintMonteCarloProofs examples.tensor_puzzles.proofs.TensorPuzzleProofs
python -m pytest tests/test_more_examples.py -q -m ""
```

## Ordinary JAX, transpilation, and a readable theorem

Start with [random-program source](examples/random_program/code.py). It contains only
standard JAX, with no jaxlean imports or distribution wrappers:

```python
import jax
import jax.numpy as jnp

def sample_times_100(key):
    draw = jax.random.randint(key, shape=(), minval=0, maxval=6)
    return draw.astype(jnp.float32) * 100.0
```

The explicit float conversion selects real arithmetic for the later operations.
Unchecked int32 multiplication, including possible overflow, is not modeled as
real multiplication.

Transpilation is a separate build step:

```python
from pathlib import Path
from jaxlean import transpile
from examples.random_program.code import sample_times_100

code = transpile(
    sample_times_100, jax.random.key(0),
    namespace="JaxLean.Generated", random_model="uniform",
)
Path("examples/random_program/generated/SampleTimes100.lean").write_text(code)
```

This runs `jax.make_jaxpr` and the core translator. The lower-level entrypoint
remains available: `translate(closed_jaxpr, name=..., readable=True,
random_model="uniform")`. Nothing patches or rewrites the Python sampler.

The generated function retains the Python name, uses scalar Lean values for
scalar code, and marks the sampling abstraction explicitly. Its body is:

```lean
let sampling := Rand.uniformInt (R := R) (0) 6 (by decide)
Rand.map (fun draw =>
  -- code.py:8 (sample_times_100)
  -- convert_element_type
  let cast_result : R := draw
  -- mul
  let mul_result : R := cast_result * (100 : R)
  mul_result) sampling
```

`Rand.uniformInt 0 6` specifies the ideal uniform law on 0,...,5. `Rand.map`
expresses running the transpiled continuation on the sampled value. The key
is replaced by the explicitly chosen sampling law, not retained as an ignored
argument. Later scalar arithmetic is emitted by the same core used for
ordinary deterministic JAX programs.

Now write the theorem separately, importing `JaxLean.Stdlib` and the generated
module. [RandomProgramProofs.lean](examples/random_program/proofs/RandomProgramProofs.lean) contains:

```lean
theorem sample_times_100_variance :
    (sample_times_100 (R := ℝ)).variance =
      100 ^ 2 * (Rand.uniformInt (R := ℝ) 0 6 (by decide)).variance := by
  simp only [sample_times_100, Rand.variance_map_mul]
```

The statement names the Python function. The proof unfolds the **actual
transpiled definition** and applies a standard-library rule. There is no
handwritten copy of the program and no mean or variance supplied in Python.
The same file proves the parameterized version for `sample_scaled(key, scale)`:

```text
variance(sample_scaled(scale)) = scale² * variance(the original draw)
```

The stdlib rule `Rand.variance_map_mul` works for every finite input law; it
does not require uniformity. A separate numeric corollary for the chosen ideal
uniform law gives variance `87500/3` after multiplication by 100.

### Core interfaces and proof library

- `transpile(fn, *example_args, ...)` traces ordinary JAX, preserves its function
  name and available argument names, and chooses scalar or tensor presentation.
- `translate(jaxpr, ...)` handles an existing Jaxpr directly. Existing tensor
  output remains the default; `scalar=True` requests scalar output.
- `readable=True` emits operation-based local names and source-line comments
  from Jaxpr metadata. Python local assignment names are generally absent from
  Jaxpr and are not claimed to be recovered.
- `import JaxLean.Stdlib` exposes tensor operations, selection rules, finite
  probability, and random-program moment rules. Application proofs remain
  outside generated files.

This naming/presentation support also works for deterministic programs. A
scalar `def affine(value, scale): return value * scale + 1.0` produces a Lean
`affine` with `value` and `scale` scalar arguments. Array programs retain the
named tensor operators used by the reusable selection theorems below.

### Sampling specification and current scope

`random_model="uniform"` is explicit because the result is an ideal sampling
specification. The translator structurally compares the intercepted callee
against the pinned JAX scalar int32 `randint` implementation or its vmapped form. Matching only a
call's `_randint` name is insufficient and is tested against a counterfeit.
This recognition guards the specification boundary; it is not a proof of
JAX's PRNG implementation. JAX itself documents slight modulus bias for some
ranges in [randint](https://docs.jax.dev/en/latest/_autosummary/jax.random.randint.html).
The variance-scaling rule needs no uniformity assumption, while the numeric
uniform-variance corollary uses the declared ideal law.

The supported fragment has one scalar typed key, one scalar int32 `randint`
call (optionally vmapped over one split-key batch), a conversion to floating
point, supported arithmetic, and one real scalar result. The batched form uses
an explicit independent product law. Repeated splitting, parent-key reuse,
multiple sampling calls and dynamic bounds remain rejected. Intermediate
floating-point rounding remains outside the real model. See the canonical
[supported Jaxpr inventory](SUPPORTED_JAXPR.md) for exact boundaries.

```sh
python -m examples.generate
lake build JaxLeanExamples
python -m pytest -q
```

## General vmap and reduction rules

[Batch.lean](JaxLean/Stdlib/Batch.lean) contains pure batch algebra with **no probability
or Rand dependency**. Batches are functions from a finite index type to arbitrary
values, including vector and matrix modules. `vmap` is pointwise application;
`reduceSum` is a finite sum.

For any linear map `L` and bilinear map `B`, the library proves:

```text
L(sum_i x_i)           = sum_i L(x_i)
B(sum_i x_i, y)        = sum_i B(x_i, y)
B(x, sum_j y_j)        = sum_j B(x, y_j)
B(sum_i x_i, sum_j y_j) = sum_i sum_j B(x_i, y_j)
```

The last identity contains all cross terms. It is generally false that it equals
`sum_i B(x_i, y_i)`. A rejection test checks this distinction. The library uses
Lean's `LinearMap` type to carry the linearity proofs, rather than assuming that
an arbitrary JAX function is linear.

[BatchRules.lean](JaxLean/Stdlib/BatchRules.lean) connects these laws to tensor `map`,
row-wise `vmap`, leading-axis reduction and shared matrix multiplication.
The translator now emits `Tensor.sumFirst` for `reduce_sum(axes=(0,))`; other
reductions retain their coordinate expressions. Existing JAX `vmap` calls are
already lowered by JAX, so the translator still only consumes Jaxpr. Tests
apply the laws directly to transpiled scaling and matrix-multiplication code.
There is no reconstruction of a Python `vmap` call.

Three concepts stay separate:

- **Coordinate locality:** a mapped output depends only on its corresponding
  input coordinate. `Batch.Coordinatewise` and its composition rule express this
  without probability. Reductions generally mix coordinates.
- **Algebra:** linearity and bilinearity justify moving functions through sums.
  These rules need no independence.
- **Probabilistic independence:** [BatchProbability.lean](JaxLean/Stdlib/BatchProbability.lean)
  proves that separate measurable maps preserve an already independent family.
  This works over an arbitrary measure and does not depend on `Rand`. A shared
  random parameter or repeated input cannot be silently treated as independent.

[FiniteLaw.meanLinear](JaxLean/Stdlib/FiniteLaw.lean) makes expectation a linear map.
Its `mean_sum` theorem is an application of `Batch.linear_reduceSum`, and
`Rand.mean_map_sum` exposes that result to existing generated random programs.
Expectation commutes with summation even when the summands are dependent.
Variance still requires covariance or independence information; it is not a
linear functional. The finite product-law construction remains the source of
independence in the supported sampling model.

## Vector norms, linear transformations and clipping

[NormRules.lean](JaxLean/Stdlib/NormRules.lean) adds an explicit Euclidean norm
`Batch.l2 x = sqrt(sum_i x[i]^2)`. It deliberately does not use Lean's default
norm on functions, which is a supremum norm. The rules apply to any finite
index type and include empty vectors.

The elementary rules are:

```text
|x_i| ≤ |y_i| for every i  ⇒  ||x||₂ ≤ ||y||₂
||a*x||₂                  =  |a| * ||x||₂
|f(v)| ≤ c*|v|, c ≥ 0     ⇒  ||vmap(f)(x)||₂ ≤ c*||x||₂
||x @ W||₂                ≤  ||W||F * ||x||₂
```

`||W||F` is the Frobenius norm, the square root of the sum of all squared matrix
entries. The matrix bound follows from Cauchy–Schwarz. It is conservative,
not a claim to compute the tighter operator norm. `l2_bound_comp` composes
bounds for arbitrary functions, multiplying their bound constants.

[TensorNorm.lean](JaxLean/Stdlib/TensorNorm.lean) connects these rules to generated
vector tensors. The translator preserves vector–matrix multiplication as
`Tensor.vecmat`; it still consumes only Jaxpr. No new primitive is needed for
the supported vector `jnp.linalg.norm` or symmetric `jnp.clip`: they lower to
existing multiplication, sum, square root, minimum and maximum operations.

[norm example](examples/norms/code.py) contains ordinary JAX:

```python
def linear_clip(x, weights, radius):
    return jnp.clip(x @ weights, -radius, radius)

def radial_clip(x, radius):
    return x * (radius / jnp.maximum(radius, jnp.linalg.norm(x)))
```

[NormProofs.lean](examples/norms/proofs/NormProofs.lean) proves bounds directly about the
transpiled definitions. For the first function, `radius ≥ 0` gives both
`||output||₂ ≤ ||weights||F * ||x||₂` and
`||output||₂ ≤ sqrt(output_dimension) * radius`. The proof composes the matrix
bound with the componentwise clipping rule; there is no stdlib theorem for
this whole program.

For radial clipping, `radius > 0` gives
`||output||₂ ≤ min(radius, ||x||₂)`, including a zero input vector. This positive
radius precondition keeps the JAX formula's denominator nonzero. The same rule
proves the bound for every row of a transpiled `vmap(radial_clip)`.

**Componentwise clipping and radial clipping are different.** Clipping both
coordinates of `(1,1)` to `[-1,1]` leaves norm `sqrt(2)`, not a norm at most 1.
Tests cover this rejected claim, matrices that increase norm, zero vectors,
empty/singleton vectors and changed matrix dimensions.

These remain exact real-model bounds. Floating-point rounding is not covered;
arbitrary norm orders, spectral-norm computation, asymmetric clipping and
radial clipping at a nonpositive radius are not supplied by these proof rules.

## Monte Carlo: attach a general theorem to generated code

[monte_carlo.py](examples/monte_carlo/code.py) is ordinary executable JAX:

```python
def monte_carlo(key, sample, f, n):
    keys = jax.random.split(key, n)
    draws = jax.vmap(sample)(keys)
    values = jax.vmap(f)(draws)
    return jnp.mean(values)

def monte_carlo_4(key):
    return monte_carlo(key, uniform_six, times_100, 4)
```

Here `uniform_six` samples an integer in `[0, 6)` and casts it to float;
`times_100` multiplies by 100. Transpile with the same `random_model="uniform"`
entrypoint. JAX specializes the callables and `n`; the translator sees only
`random_split`, the batched sampler call, conversion, multiplication, sum and
division. It does not read Python source or recognize a function called
`monte_carlo`.

Three system extensions make this work:

1. **Finite product sample spaces.** `Rand` now stores an arbitrary finite
   sample space rather than only `Fin size`. `Rand.iid n x` constructs `n`
   independent copies using product laws.
2. **A checked batched sampling boundary.** The translator tracks one root-key
   split and checks the vmapped sampler's Jaxpr structurally. Generated code
   uses `Rand.iid`; the arithmetic and reduction use the existing tensor
   translation. No variance formula is inserted by the compiler.
3. **Elementary moment propagation.** [Random.lean](JaxLean/Stdlib/Random.lean)
   supplies expectation rules for addition, multiplication by a constant and
   division; variance rules for scaling, division, constant shifts and addition
   with covariance. [Independent.lean](JaxLean/Stdlib/Independent.lean) supplies a sum
   rule for independent coordinates, each with its own function. There is no
   Monte Carlo definition or theorem in the stdlib.

[MonteCarloProofs.lean](examples/monte_carlo/proofs/MonteCarloProofs.lean) unfolds the
[transpiled definition](examples/monte_carlo/generated/MonteCarlo4.lean) and propagates these
rules through its body:

```lean
theorem monte_carlo_4_variance :
    (monte_carlo_4 (R := ℝ)).variance =
      (sample_times_100 (R := ℝ)).variance / 4 := by
  simp only [monte_carlo_4, Tensor.scalar, Tensor.sumFirst, Batch.reduceSum, Tensor.map]
  rw [Rand.variance_map_div, Rand.variance_iid_sum (f := fun _ v => v * 100)]
  simp only [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul,
    sample_times_100]
  ring
```

The first rewrite moves through division: `Var(sum/4) = Var(sum)/4²`.
The second moves through the independent sum: `Var(sum) = ∑ᵢ Var(termᵢ)`.
There are four identical terms, so algebra gives `4 * sigma2 / 4² = sigma2/4`.
No library lemma recognizes the entire estimator. A companion proof propagates
expectation through division and summation to show the mean is preserved.

The same rules handle the eight-sample nonlinear example and a tested weighted
sum `sum(draws * weights) + 7`, whose variance is
`sum(weights[i]² * Var(draw))`. This is ordinary Lean rewriting, not a new
proof-generating compiler pass or an automatic solver for arbitrary programs.

For dependent terms, addition retains covariance. For nonlinear functions,
`variance_map_second_moment` exposes `E[f(X)²] - E[f(X)]²`; mean and variance
alone do not determine all later moments. In particular, variance after squaring
requires a fourth moment. Such obligations remain explicit.

The operation rules are general; the generated programs currently have concrete sizes
and inlined functions. Independence is part of the declared ideal key model,
not a theorem about JAX's deterministic PRNG. A regression test broadcasts one
single draw four times and proves its variance stays unchanged.

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

## Checked translation certificates

The optional `certify` entrypoint emits the ordinary readable function, an
independent encoding of the input Jaxpr, and a theorem connecting them:

```python
from jaxlean import certify

closed = jax.make_jaxpr(lambda x: jnp.mean(x))(jnp.ones(2))
code = certify(closed, name="certified_mean", namespace="JaxLean.Generated")
```

Pass `named_vars=True` to `certify` or `certify_module` for typed, named SSA
syntax. The names follow Jaxpr input/equation order, without reading Python source:

```lean
jaxpr% (a : (.real, [3]), b : (.real, [3])) {
  c : (.real, [3]) := .add a b (t := [3]);
  d : (.real, [3]) := .mul c a (t := [3]);
  return d
}
```

Names keep referring to the same values across bindings. The syntax expands to
the existing `Jaxpr.Program`, so its evaluator and certificates are unchanged.
Lean checks each variable's dtype and shape; new bindings cannot reuse an SSA
name. Calls use `c : resultType := call callee_ir with arguments;` and preserve
function boundaries. The [syntax implementation](JaxLean/Verification/Syntax.lean)
lives outside Core. The blog's [add-and-scale example](examples/blog/generated/AddScale.lean)
uses this option. Positional output remains available with `named_vars=False`.

Write the returned source to a Lean file and check it with `lake env lean`.
**Producing this source is not itself successful certification.** Lean must
accept the emitted theorem. `python -m examples.certificates.generate` generates the checked-in
[CertifiedMean.lean](examples/certificates/generated/CertifiedMean.lean), included in `lake build JaxLeanExamples`.

The IR in that file is small enough to inspect:

```lean
def certified_mean_ir : Jaxpr.Program [(.real, [2])] (.real, []) :=
  .bind (.reduce_sum (s := [2]) (t := []) (k := [2]) (fun i j => (j.1, ())) (.var .here)) <|
  .bind (.div (.var .here) (.literal (2) 1 (by decide)) (t := [])) <|
  .ret (.var .here)
```

`[(.real, [2])]` is the input context (one real vector of length two);
`(.real, [])` is the real scalar output type. `.here` selects the newest environment value. Each binding
pushes its result onto the environment, so these references mean input vector,
then sum, then quotient. `.there` accesses an older binding. Literal `2/1`
records an exact rational value, not a Python floating computation.

The accompanying statement quantifies over every real input:

```lean
theorem certified_mean_translation_correct (x0 : Tensor ℝ [2]) :
    Jaxpr.Program.eval (.cons x0 .nil) certified_mean_ir =
      certified_mean (R := ℝ) x0 := by
  -- Generated proof composing checked equation/environment rules.
  ...
```

[Jaxpr.lean](JaxLean/Core/Jaxpr.lean) defines dtype- and shape-typed variables, environments,
operations and let-bound programs. Its evaluator uses direct coordinate
arithmetic and finite sums, independently of `Tensor.map`, `map₂` and
`sumFirst`. Local lemmas connect these semantics to tensor operations; the
emitted proof composes them. Shapes and variable references are checked by
Lean's type system. No axiom or `sorry` supplies translation correctness.

The **certifiable** fragment includes real arithmetic, transcendental functions,
general fixed-shape broadcasts/transposes/reshapes, concatenation, batched
contractions, arbitrary-axis sums, nonempty max/min reductions, and static
scatter. All programs use the same dtype-and-shape-indexed `Jaxpr.Program`,
including Boolean masks, signed int32 indices, wrapped index arithmetic, and
clipped gather. There is one environment and evaluator; real operations do not
need an embedded subprogram. `certify_module` retains pure single-result jit boundaries
and checks their bodies recursively.

The [single importer](python/jaxlean/importers/jaxpr.py) preserves primitive names
and call boundaries, using typed de Bruijn references for variables. For example,
[eye_ir](examples/tensor_puzzles/generated/PuzzlesEye.lean) contains `iota`, two
`broadcast_in_dim` bindings with their source shape/dimension parameters, `eq`,
and a call to the `where` body. There are no inserted scalar-broadcast bindings
or embedded real-only programs. Static index equations remain in the graph;
each concatenation or static scatter remains one binding. Some layout and
scatter parameters still use bounded coordinate maps or static coordinate plans,
so the encoding is not a verbatim serialization of every Jaxpr parameter.

Generated certificates reuse child certificates and retain explicit simplifier
proof steps to avoid repeatedly reducing large SSA environments in Lean's kernel.

[Ordinary JAX examples](examples/common_primitives/code.py) include standard softmax,
batched query/key products, masking, and embedding lookup with
`jnp.take(table, ids, axis=0, mode="clip")`.
[Function-boundary proofs](examples/common_primitives/proofs/CommonPrimitiveProofs.lean) show softmax row
sums of one, nonnegativity, and exact clipped embedding lookup, and connect
those properties to the imported Jaxpr using translation certificates.

Captured floating arrays, scans, random operations, multiple outputs, general
integer operations, and gather promise/fill modes remain outside certificate
coverage. Runtime scatter is unsupported. `certify` still rejects calls/jit;
use `certify_module` for a call graph. The [inventory](SUPPORTED_JAXPR.md)
lists exact restrictions, including mixed-type combinations. Integer indexing
uses actual signed int32 wraparound, while floating computation remains an
ideal-real model. Layout decoding in the Python importer is still trusted.

### Published semantics references

There are published operational descriptions and reference implementations, but
we have not identified a complete machine-checked semantics of all Jaxpr to
reuse. JAX's [language documentation](https://docs.jax.dev/en/latest/601/jaxpr.html)
and [interpreter tutorial](https://docs.jax.dev/en/latest/notebooks/Writing_custom_interpreters_in_Jax.html)
explain the IR and environment-based execution. For version-specific behavior,
we use the [JAX 0.8.0 implementation](https://github.com/jax-ml/jax/blob/jax-v0.8.0/jax/_src/core.py).

[StableHLO](https://openxla.org/stablehlo/spec) supplies detailed operation
specifications, but is a separate IR. Using it as a reference does not verify
JAX's lowering or our replacement of floating-point arithmetic with reals.
[SEMANTICS_REFERENCES.md](SEMANTICS_REFERENCES.md) maps the supported operations
to those sources and records the semantic gaps, including a test where a JAX
float cast rounds but our real-model identity certificate correctly passes.

There are still explicit trust boundaries:

- The Python [importer](python/jaxlean/certify.py) must faithfully encode the
  original Jaxpr. It constructs the IR from Jaxpr, not from generated Lean, but
  is not formally verified. An importer bug that changes both sides consistently
  is outside the certificate's guarantee.
- The chosen Jaxpr semantics is the exact real-arithmetic abstraction. Relating
  it to JAX's machine arithmetic remains separate work.
- The tensor/index representation and the Lean kernel remain foundational.

The normal transpiler's emitted body is checked **against that imported IR**.
Tests deliberately alter the generated divisor and arithmetic operation while
keeping the IR fixed, and Lean rejects both certificates. Additional tests
cover multiple inputs, variable sequencing, broadcasting, exact float32
literals, and empty/singleton dimensions. This is per-program certification,
not a proof of the Python transpiler for all inputs or an end-to-end JAX theorem.

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

## Supported Jaxpr

[SUPPORTED_JAXPR.md](SUPPORTED_JAXPR.md) lists supported primitives, modeled
sampling calls, restrictions, theorem coverage and test evidence. It is the
canonical inventory to update with every translator extension. Support follows
the emitted Jaxpr, not the name of a Python API.

## Reals, execution, and future precision

Generated algebraic functions are polymorphic over a field. Instantiate them at
`ℝ` for theorems, or `ℚ` for exact `#eval`. This is one generated program with two
interpretations; rational testing does not establish real-valued correctness.
Transcendentals additionally require [RealOps](JaxLean/Core/RealOps.lean). Its real
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

[Proofs.lean](examples/basics/proofs/Proofs.lean) checks, without `sorry` or custom axioms:

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

JAX tracing, the Python Jaxpr importer, and our choice of primitive semantics
remain trusted. Ordinary translation is tested; optional certificates additionally
check generated functions against the imported IR. Type-checking alone prevents
ill-typed shapes/indexes, but does **not** prove that every generated permutation
or primitive implements JAX correctly. Tests
execute the emitted Lean over rationals and compare with JAX on finite examples.
Transcendentals have type-checking tests, not a rational execution comparison.

JAX's [IR documentation](https://docs.jax.dev/en/latest/jaxpr.html) and
[interpreter guide](https://docs.jax.dev/en/latest/notebooks/Writing_custom_interpreters_in_Jax.html)
describe its structure and execution. We do not assume those documents provide
a machine-checked specification. The transpiler calls JAX's structural checker,
then Lean checks the emitted terms. **There is no end-to-end correctness theorem
connecting JAX machine execution to these real-valued proofs.**

Next useful steps: symbolic-size transpilation, broader key dataflow, and
compositional scan certificates. The current typed IR and function-call rules
provide a base for extending certificate coverage. A floating-point extension
needs an explicit relationship to the existing real semantics.

## Reusable proofs applied to transpiled neural networks

The proof components live in [SelectionRules.lean](JaxLean/Stdlib/SelectionRules.lean).
That file imports **no generated programs**. The small application file
[Selection.lean](examples/selection/proofs/Selection.lean) applies the same library to two different
transpiled networks. It follows the selection-equivariance property in
[Lean Verified Transformers](https://srush.github.io/lean-transformer/).

The property is:

```text
network(x[selection], parameters) = network(x, parameters)[selection]
```

For example, `[2, 0, 2]` drops, reorders, and repeats rows. Their output values
must follow the same selection. This is equivariance; it does not say that the
output tensor stays unchanged.

### 1. Preserve operations in the transpiled code

The Python example is an ordinary forward pass:

```python
def mlp(x, w1, b1, w2, b2):
    hidden = jnp.maximum(x @ w1 + b1, 0.0)
    return hidden @ w2 + b2
```

The translator emits calls to named pure functions instead of expanding every
operation into scalar indexing. In [MLP4.lean](examples/selection/generated/MLP4.lean),
`x0, x1, x2, x3, x4` mean `x, w1, b1, w2, b2`:

```lean
let v0 : Tensor R [4, 3] := Tensor.matmul x0 x1
-- v1 is the bias viewed as a tensor with shape [1,3].
let v2 : Tensor R [4, 3] :=
  Tensor.map₂ (fun a0 a1 => a0 + a1) v0 (Tensor.broadcastFirst 4 v1)
let v3 : Tensor R [4, 3] :=
  Tensor.map (fun a0 => max a0 ((Tensor.scalar (0 : R)) ())) v2
```

`map` applies a scalar function to every element. `map₂` combines corresponding
pairs of elements. `broadcastFirst` copies a singleton first axis to the required
batch size. `matmul` is ordinary rank-two matrix multiplication. Their small
implementations are in [Tensor.lean](JaxLean/Core/Tensor.lean), so the output is still
ordinary Lean code, not a separate instruction language.

General contractions and other broadcast patterns still use coordinate
expressions. This change preserves named structure where supported; it does
not claim automatic selection proofs for every supported JAX primitive.

### 2. Prove general facts once

A tensor with shape `n :: s` has a first axis of size `n` and any remaining
shape `s`. Selection changes only that first coordinate:

```lean
def selectRows (selection : Fin m → Fin n) (x : Tensor R (n :: s)) :
    Tensor R (m :: s) :=
  fun i => x (selection i.1, i.2)
```

`Fin n` means a valid index below `n`. Here `i.1` is the row and `i.2` is the
remaining coordinate tuple. No injectivity assumption is needed.

The library proves these rules for arbitrary dimensions and values:

```text
selectRows s (map f x)       = map f (selectRows s x)
selectRows s (map₂ f x y)    = map₂ f (selectRows s x) (selectRows s y)
selectRows s (matmul x w)    = matmul (selectRows s x) w
selectRows s (broadcastFirst n b) = broadcastFirst m b
```

The first rule works for **any** scalar function: ReLU, square, exponential,
and so on. The second handles residual addition and other operations on two
branches, provided both branches follow the same selection. The matmul rule
works for all matrix dimensions, with shared weights.

Each rule's proof compares an arbitrary coordinate (`funext i`) and checks
that the expressions reduce to the same thing (`rfl`). For matmul, both sides
are `sum_k x[s[row], k] * w[k, column]`. There is no batch-row reduction.

The library also defines `SelectionEquivariant` for batch-polymorphic functions
and proves `SelectionEquivariant.comp`: two verified components can be composed
without inspecting either implementation. The simp rules are the convenient
interface for automatically processing the straight-line transpiled code.

### 3. Apply the library directly to a transpiled chunk

The main theorem in `Selection.lean` is about the actual generated functions:

```lean
Generated.mlp3 (selectRows selection x) w1 b1 w2 b2 =
  selectRows selection (Generated.mlp4 x w1 b1 w2 b2)
```

Its proof unfolds only those generated definitions and applies the general rules:

```lean
simp only [Generated.mlp3, Generated.mlp4, Tensor.selectRows_map,
  Tensor.selectRows_map₂, Tensor.selectRows_broadcastFirst, Tensor.selectRows_matmul]
```

Selection is pushed inward through the expression on the right until both
sides match. No handwritten network specification or bridge theorem is needed.
The rules have the `[simp]` attribute, so the shorter form also works:

```lean
simp [Generated.mlp3, Generated.mlp4]
```

A second Python example has a square activation, dimensions `3 → 4 → 3`, and
an input skip connection. Its generated proof uses exactly the same library:

```lean
simp [Generated.residual2, Generated.residual5]
```

Thus a new network assembled from supported operations changes the generated
code and theorem statement, not the mathematical proof components. A new
operation needs an applicable rule, and any necessary hypotheses must be proved.
For example, a batch mean generally destroys this property; a proof cannot
assume that operation is selection equivariant.

### 4. Scope and execution

The general rules quantify over dimensions. The transpiler still emits the
fixed shapes of its Jaxpr input, so a generated theorem compares the two
actual traces. Shape-polymorphic transpilation is a separate extension. The
tests apply the same proof recipe to additional batch sizes, including empty
and singleton batches.

```sh
python -m examples.generate --check
lake build JaxLeanExamples
lake env lean examples/selection/proofs/Selection.lean
python -m pytest -q
```

The concrete rational example prints:

```text
[2, 1, 1, 2, 11, -8, 1, 0]  -- all four output rows, flattened
[11, -8, 2, 1, 11, -8]       -- select [2,0,2], then run
[11, -8, 2, 1, 11, -8]       -- run, then select [2,0,2]
```

These numbers help read the example. The theorem covers every real input and
parameter in the generated mathematical model. Lean checks the assembled
proof; the Python translator is still trusted to model the original Jaxpr.
This is not a proof of bitwise equality between floating-point JAX executions.

## Example layout

Each example family keeps its Python program and Lean artifacts together:

```text
examples/transformer/
  code.py
  generate.py
  proofs/TransformerProofs.lean
  generated/Transformer.lean
```

The same layout is used for basics, selection, norms, certificates, common
primitives, scatter, tensor puzzles, random programs, Monte Carlo, randint
estimators, and Noether updates. `JaxLeanExamples.lean` imports the example
proofs; the core library does not depend on them.

Run `python -m examples.transformer.generate` to regenerate one example, or
add `--check` to check freshness. `make generate` and `make check-generated`
cover every example directory.

## Development

Edit Python examples or the translator, regenerate, and keep proofs separate.

```sh
make check             # freshness checks + incremental Lean build + fast Python tests
make check-lean        # incremental Lean proof checking only
make test              # fast Python tests only (no Lean subprocesses)
make test-lean         # opt-in generated-program integration tests
make check-full        # freshness + Lean build + all Python/integration tests
```

The default `pytest` selection excludes the `lean` marker. It checks Python
validation, unsupported-input rejection, sampling behavior, and documentation
extraction. `make check-lean` kernel-checks all library and example
proofs, reusing unchanged compiled modules. Tests marked `lean` compile extra
temporary programs, compare JAX with Lean execution, and deliberately corrupt
programs to test rejection. Run them when changing the importer, certificate
generator, or semantic model; they are not needed for every documentation edit.

```sh
python -m pytest -q                  # fast default
python -m pytest -q -m lean          # only Lean integration tests
python -m pytest -q -m ""            # all tests
```

Proofs establish theorems about the imported Lean model. They do not verify
the Python importer or establish equivalence to JAX's floating-point/PRNG
implementation, so the integration tests remain useful at that boundary.

The differential suite leaves its emitted Lean in `tests/_generated/` for
inspection. Only project-local `.venv/` and `.lake/` are needed at runtime;
there are no absolute paths to another checkout in the package configuration.

## Python / Lean proof notebook

```sh
python -m pip install -e tools/verso
make docs-serve
# Open http://localhost:8765
# In a second terminal, from the repository root:
make docs-watch
```

The notebook tooling is a separate Python distribution in `tools/verso/`.
It is excluded from the core `jaxlean` installation. `make docs` and
`make docs-serve` run it directly from that directory. To install its CLI:

```sh
pip install -e tools/verso
jaxlean-verso build --root /path/to/jaxlean
jaxlean-verso serve --root /path/to/jaxlean
```

The documentation package uses PyYAML for its manifest and markdown-it-py for
text blocks. Building the notebook also requires the JAX example environment
and Lake/Verso; these are separate from the documentation package dependencies.
`--root` defaults to the current directory and identifies the repository,
independently of where the package is installed.

The small [HTML generator](tools/verso/src/jaxlean_verso/notebook.py) uses **Verso** to render
kernel-checked Lean declarations beside the original Python functions. The
notebook includes tensor puzzles, `randint` Monte Carlo estimators, and the
transformer. Lean names have type/documentation mouseovers; tactic underlines
open proof states. Expand a certificate to see the evaluator connection, or
follow a file link to read the complete Lean module.

Verso is pinned to the project's Lean version. Its first build downloads and
builds additional dependencies; subsequent builds use Lake's cache. Jupytext
would be useful for synchronizing notebooks and Python scripts, but doesn't
provide Lean elaboration or hover information, so it is not needed here.

Edit [docs/examples.yaml](docs/examples.yaml). Each blog section has an ordered
`blocks` list: move entries to reorder the page, and insert a `type: text` entry
between any two snippets. Text supports Markdown, including inline links; an
empty text block displays its `marker`. Python and Lean snippets refer to source
functions and declarations, so code is not copied into the manifest:

```yaml
blocks:
- type: python
  file: examples/blog/code.py
  name: add_then_scale
- type: text
  text: |
  marker: after add_then_scale
- type: lean
  module: examples.blog.generated.AddScale
  name: JaxLean.Blog.add_then_scale_ir
  section: Lean IR
```

Use `detail: Label` on a Lean block to make it expandable. Datatype blocks read
the current declarations from source, including the unified IR's dtype and
shape indices. Python source is inspected only for display; the transpiler
still consumes Jaxpr. `make docs` checks generated-file freshness and builds
Verso's `:literate` artifacts, failing on missing functions or declarations.

`make docs-watch` refreshes prose and style edits using the last checked Lean
output. Code edits trigger a full documentation build; failures preserve the
last successful preview. Keep the HTTP server running alongside the watcher.

`make docs` writes `.lake/build/proof-notebook/`, including local JavaScript/CSS,
full module pages, and a source-hash build record. Serve this directory over
HTTP (opening `index.html` as a local file prevents hover data from loading).
The whole directory can be hosted as a static site; it needs no Lean server at
runtime. A manifest pairing is editorial, not itself an equivalence proof:
the displayed certificate states the checked relationship and its model scope.

The [Noether equivariance port](docs/noether.md) adds ordinary JAX advection and
Burgers examples, reusable permutation lemmas, and function-boundary certificates.
The code-only notebook page is `noether.html`.

[20 of 21 Tensor Puzzles](docs/tensor-puzzles.md) now have JAX implementations,
imperative JAX specifications, and Lean function-boundary certificates.
Compression (#12) is explicitly deferred.
