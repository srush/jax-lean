import JaxLean.Stdlib
import examples.transformer.generated.Transformer
import JaxLean.Stdlib.MatrixRules

/-! Read this file alongside examples/transformer/code.py.

Each proof crosses one function boundary. Only the normalization specification
looks inside scalar arithmetic. Block and network proofs use child theorems.
The final theorem transports the property back to the imported Jaxpr.
-/
namespace JaxLean.TransformerJax
open Tensor
open scoped BigOperators
noncomputable section

abbrev Matrix (n m : Nat) := Tensor ℝ [n, m]

/-- The mathematical meaning of Python's `normalize`, including its denominator. -/
theorem normalize_spec (scores : Matrix 3 3) :
    normalize scores = normalizeRows (fun z => 1 + max z 0) scores := rfl

private theorem score_positive (z : ℝ) : 0 < 1 + max z 0 := by
  have := le_max_right z 0
  linarith

/-- The example's weights really are a probability vector in every row. -/
theorem normalize_nonnegative (scores : Matrix 3 3) (i : Index [3, 3]) :
    0 ≤ normalize scores i := by
  rw [normalize_spec]
  exact normalizeRows_nonneg _ score_positive (by decide) scores i

theorem normalize_sum_one (scores : Matrix 3 3) (row : Fin 3) :
    (∑ col, normalize scores (row, col, ())) = 1 := by
  rw [normalize_spec]
  exact normalizeRows_sum_one _ score_positive (by decide) scores row

/-- Row-local functions permit repeated or dropped rows as well as permutations. -/
theorem project_select (selection : Fin 3 → Fin 3) (x : Matrix 3 2) (w : Matrix 2 2) :
    project (selectRows selection x) w = selectRows selection (project x w) :=
  (selectRows_matmul selection x w).symm

theorem forward_select (selection : Fin 3 → Fin 3) (x : Matrix 3 2) (w : Matrix 2 2) :
    forward (selectRows selection x) w = selectRows selection (forward x w) := by
  simp only [forward, project_select, selectRows_map]

/-- Attention needs a bijection of token positions, since all keys contribute. -/
theorem normalize_permute (perm : Fin 3 ≃ Fin 3) (scores : Matrix 3 3) :
    normalize (reindexMatrix perm perm scores) =
      reindexMatrix perm perm (normalize scores) := by
  simp only [normalize_spec, normalizeRows_reindex]

/-- A readable specification at the attention boundary. -/
theorem attention_spec (q k v : Matrix 3 2) :
    attention q k v = matmul (normalize (matmul q (transposeMatrix k))) v := rfl

theorem attention_permute (perm : Fin 3 ≃ Fin 3) (q k v : Matrix 3 2) :
    attention (selectRows perm q) (selectRows perm k) (selectRows perm v) =
      selectRows perm (attention q k v) := by
  rw [attention_spec, attention_spec, matmul_transpose_select,
    normalize_permute, matmul_reindex_contract]

/-- The block proof uses only its children's contracts. -/
theorem transformer_block_permute (perm : Fin 3 ≃ Fin 3)
    (x : Matrix 3 2) (weight wq wk wv : Matrix 2 2) :
    transformer_block (selectRows perm x) weight wq wk wv =
      selectRows perm (transformer_block x weight wq wk wv) := by
  simp only [transformer_block, forward_select, project_select, attention_permute]

/-- Two separately parameterized blocks: no attention internals appear here. -/
theorem transformer_permute (perm : Fin 3 ≃ Fin 3) (x : Matrix 3 2)
    (w0 q0 k0 v0 w1 q1 k1 v1 : Matrix 2 2) :
    transformer (selectRows perm x) w0 q0 k0 v0 w1 q1 k1 v1 =
      selectRows perm (transformer x w0 q0 k0 v0 w1 q1 k1 v1) := by
  simp only [transformer, transformer_block_permute]

/-- This theorem is about the imported Jaxpr itself. Translation certificates
connect the function-level proof above to its independent IR evaluator. -/
theorem certified_transformer_permute (perm : Fin 3 ≃ Fin 3) (x : Matrix 3 2)
    (w0 q0 k0 v0 w1 q1 k1 v1 : Matrix 2 2) :
    Jaxpr.Program.eval
      (.cons (selectRows perm x) (.cons w0 (.cons q0 (.cons k0 (.cons v0
        (.cons w1 (.cons q1 (.cons k1 (.cons v1 .nil))))))))) transformer_ir =
    selectRows perm (Jaxpr.Program.eval
      (.cons x (.cons w0 (.cons q0 (.cons k0 (.cons v0
        (.cons w1 (.cons q1 (.cons k1 (.cons v1 .nil))))))))) transformer_ir) := by
  rw [transformer_translation_correct, transformer_translation_correct]
  exact transformer_permute perm x w0 q0 k0 v0 w1 q1 k1 v1

end
end JaxLean.TransformerJax
