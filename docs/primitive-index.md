# Jaxpr primitive index

The single semantic dispatch point is [`Jaxpr.Op` / `Op.eval`](../JaxLean/Core/Jaxpr.lean).
`Program`, `Var`, `Atom`, `Args`, and `Env` all carry dtype and shape. There is no
separate mixed-type IR or embedded real subprogram constructor.

The Python dispatch point is [`JaxprImporter`](../python/jaxlean/importers/jaxpr.py).
The importer consumes Jaxpr independently of the tensor-function emitter.

| Source operation | Unified IR |
| --- | --- |
| Arithmetic | `add`, `sub`, `mul`, `div`, `min`, `max`, `neg`, `square`, `integer_pow`, `abs` |
| Transcendentals | `exp`, `log`, `sqrt`, `rsqrt`, `sin`, `cos`, `tanh` |
| Comparisons | `eq`, `ne`, `lt`, `le`, `gt`, `ge` |
| Boolean operations | `and`, `or`, `xor`, `not`, `select_n` |
| Broadcast | `broadcast_in_dim` with shape and dimension list; elementwise broadcasting stays inside its source operation |
| Permutations | `transpose` with a checked source permutation; `rev` with a checked axis list |
| Other layout operations | `squeeze`, `slice`, `reshape` with bounded coordinate maps |
| Initialization | `iota` with shape and axis |
| Products and reductions | `dot_general`, `reduce_sum`, `reduce_max`, `reduce_min` |
| Other tensor operations | `concatenate`, `gather` |
| Identity and casts | `copy`, `stop_gradient`, `convert_element_type` |
| Static scatter | `scatter`, `scatter_add` with a validated list of destination/update coordinates |

Static index scaffolding remains in the graph and is also evaluated as metadata
to validate static scatter positions. Each scatter and variadic concatenation
stays a single binding. These restrictions and the trusted layout decoding are recorded
in [the support inventory](../SUPPORTED_JAXPR.md). Floating values use ideal-real
semantics, not IEEE-754 execution. Unsupported combinations fail closed.

[Verification/JaxprRules.lean](../JaxLean/Verification/JaxprRules.lean) supplies SSA
composition lemmas. [Certificate.lean](../JaxLean/Verification/Certificate.lean)
checks equality with the independently emitted tensor function.
