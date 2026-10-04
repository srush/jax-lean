import JaxLean.Stdlib
import JaxLean.Generated.PuzzlesTriu

set_option maxRecDepth 4096
set_option maxHeartbeats 2000000
namespace JaxLean.Puzzles.Triu

theorem matches_loop  :
    Array.triu (R := ℝ) = Loop.triu (R := ℝ) := by
  funext ⟨i, j, u⟩
  cases u
  fin_cases i <;> fin_cases j <;>
    simp only [Array.fn__where, Array.triu, Loop.triu, Tensor.scalar, Tensor.reindex,
      Tensor.scatterSet_matrix]
  all_goals norm_num [Fin.ext_iff, Bool.cond_decide, decide_eq_true_iff]

theorem certificate  :
    TypedJaxpr.Program.eval .nil Array.triu_ir =
      Jaxpr.Program.eval .nil Loop.triu_ir := by
  rw [Array.triu_translation_correct, Loop.triu_translation_correct]
  exact matches_loop

end JaxLean.Puzzles.Triu
