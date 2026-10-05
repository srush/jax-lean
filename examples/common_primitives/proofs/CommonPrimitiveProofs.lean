import JaxLean.Stdlib
import examples.common_primitives.generated.CommonMasked
import examples.common_primitives.generated.CommonScores
import examples.common_primitives.generated.CommonSoftmax
import examples.common_primitives.generated.CommonEmbeddings
import examples.common_primitives.generated.CommonShifted
import examples.common_primitives.generated.CommonRearrange
import examples.common_primitives.generated.CommonExtrema
import JaxLean.Stdlib.MatrixRules

namespace JaxLean.CommonPrimitiveProofs
open CommonJax
open scoped BigOperators

/-- The predicate and selected values retain their different types. -/
theorem masked_values_spec (x : Tensor ℝ [3]) (mask : Tensor Bool [3]) :
    Masked.masked_values x mask = fun i => if mask i && decide (x i > 0) then x i else 0 := rfl

theorem masked_values_certificate (x : Tensor ℝ [3]) (mask : Tensor Bool [3]) :
    Jaxpr.Program.eval (.cons x (.cons mask .nil)) Masked.masked_values_ir =
      fun i => if mask i && decide (x i > 0) then x i else 0 := by
  rw [Masked.masked_values_translation_correct, masked_values_spec]

/-- Each batch computes the usual query/key inner products. -/
theorem batched_scores_spec (q : Tensor ℝ [2, 3, 4]) (k : Tensor ℝ [2, 5, 4]) :
    Scores.batched_scores q k = fun i =>
      ∑ j : Fin 4, q (i.1, i.2.1, j, ()) * k (i.1, i.2.2.1, j, ()) := by
  funext i
  simp only [Scores.batched_scores, Tensor.contract, Tensor.reindex,
    Index.sum_cons, Index.sum_nil]

/-- Clipping selects a valid embedding row even for negative or oversized indices. -/
theorem embeddings_spec (table : Tensor ℝ [5, 3]) (ids : Tensor Int32 [2]) :
    Embeddings.embeddings table ids = fun i =>
      table (Tensor.clipStart 4 (ids (i.1, ())), i.2) := rfl

theorem embeddings_certificate (table : Tensor ℝ [5, 3]) (ids : Tensor Int32 [2]) :
    Jaxpr.Program.eval (.cons table (.cons ids .nil)) Embeddings.embeddings_ir =
      fun i => table (Tensor.clipStart 4 (ids (i.1, ())), i.2) := by
  rw [Embeddings.embeddings_translation_correct, embeddings_spec]

/-- Under an in-range hypothesis, clipping leaves the requested index unchanged. -/
theorem embeddings_in_bounds (table : Tensor ℝ [5, 3]) (ids : Tensor Int32 [2])
    (row : Fin 2) (column : Fin 3)
    (lo : 0 ≤ (ids (row, ())).toInt) (hi : (ids (row, ())).toInt.toNat < 5) :
    Embeddings.embeddings table ids (row, column, ()) =
      table (⟨(ids (row, ())).toInt.toNat, hi⟩, column, ()) := by
  rw [embeddings_spec]
  have h : Tensor.clipStart 4 (ids (row, ())) = (⟨(ids (row, ())).toInt.toNat, hi⟩ : Fin 5) := by
    apply Fin.ext
    exact Tensor.clipStart_of_inBounds 4 _ lo (by omega)
  exact congrArg (fun r => table (r, column, ())) h

/-- The shift is the row maximum computed by the actual Jaxpr. -/
noncomputable def rowShift (x : Tensor ℝ [2, 3]) : Tensor ℝ [2] :=
  Tensor.reduceMax (s := [2, 3]) (t := [2]) (n := 3) (by decide)
    (fun i j => (i.1, ((Index.equivFin [3]).symm j).1, ())) x

theorem row_softmax_spec (x : Tensor ℝ [2, 3]) :
    Softmax.row_softmax x = Tensor.normalizeRows Real.exp
      (fun i => x i - rowShift x (i.1, ())) := rfl

/-- Apply the existing generic normalization theorem at the function boundary. -/
theorem row_softmax_sum_one (x : Tensor ℝ [2, 3]) (row : Fin 2) :
    (∑ column, Softmax.row_softmax x (row, column, ())) = 1 := by
  rw [row_softmax_spec]
  exact Tensor.normalizeRows_sum_one Real.exp Real.exp_pos (by decide) _ row

theorem row_softmax_nonneg (x : Tensor ℝ [2, 3]) (i : Index [2, 3]) :
    0 ≤ Softmax.row_softmax x i := by
  rw [row_softmax_spec]
  exact Tensor.normalizeRows_nonneg Real.exp Real.exp_pos (by decide) _ i

theorem row_softmax_certificate (x : Tensor ℝ [2, 3]) (row : Fin 2) :
    (∑ column, Jaxpr.Program.eval (.cons x .nil) Softmax.row_softmax_ir (row, column, ())) = 1 := by
  rw [Softmax.row_softmax_translation_correct]
  exact row_softmax_sum_one x row

end JaxLean.CommonPrimitiveProofs
