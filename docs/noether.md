# Noether equivariance port

Source: Philip Mocz's [Noether finite-volume examples](https://github.com/pmocz/noether/blob/c17f37969dcc954634d72661a5c51a6ed52c9000/examples/finite_volume/advection.py),
revision `c17f37969dcc954634d72661a5c51a6ed52c9000`.
The two Python function bodies are adapted from that Apache-2.0 project;
the license is preserved in [examples/NOETHER_LICENSE](../examples/NOETHER_LICENSE).
The cyclic permutation construction also follows Noether's use of `ZMod` and
`Equiv.addRight`, with the sign chosen to match JAX's output-to-input indexing.

Read these in order:

1. [Ordinary JAX](../examples/noether.py): advection and nonlinear Burgers updates.
2. [Shared mathematics](../JaxLean/Stdlib/Equivariance.lean): coordinate permutations,
   cyclic shifts, equivariance of pointwise flux differences, and conservation.
3. [Function-boundary proofs](../JaxLean/Examples/NoetherProofs.lean): identify each
   generated roll helper with a cyclic permutation, identify each Python function
   with a flux update, then apply the shared lemmas.
4. [Generated advection certificate](../JaxLean/Generated/NoetherAdvect.lean) and
   [Burgers certificate](../JaxLean/Generated/NoetherBurgers.lean): imported Jaxpr,
   emitted functions, and their equivalence proofs. The final `*_certificate`
   theorems in NoetherProofs state equivariance directly for imported Jaxpr evaluation.

Both updates commute with **every integer cyclic shift** and preserve the sum of
the input. Burgers demonstrates that these properties do not require linearity.
The shared flux theorem permits any pointwise flux and any commuting coordinate
permutation, over any ring and any finite grid size (including empty grids).
Cyclic shifts are defined for positive sizes.

No core semantic operations or importer special cases were added. JAX lowers
`jnp.roll` to a nested `jit` containing two slices and a concatenation. We certify
that body and preserve its call boundary. The generator consumes only `make_jaxpr`
output; it does not inspect Python source or recognize these function names.

Unlike Noether's symbolic-grid examples, our generated artifacts specialize to
four cells. The theorem covers all real-valued inputs and all integer shifts at
that shape, not merely sampled inputs. The stdlib mathematics is size-generic,
but a new traced shape needs its own certificate and roll-boundary proof.
Arithmetic is ideal real arithmetic; this does not establish bitwise float
conservation. The Python importer and its coordinate decoding remain trusted.
Noether's heat-smoother positivity and time-loop examples are not part of this port.

Regenerate with `.venv/bin/python -m examples.certify_noether`; verify with
`lake build` and `make check-generated`. The code-only notebook page is `noether.html`.
