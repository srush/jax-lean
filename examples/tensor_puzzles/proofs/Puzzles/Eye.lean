import JaxLean.Stdlib
import examples.tensor_puzzles.generated.PuzzlesEye

set_option maxRecDepth 4096
set_option maxHeartbeats 2000000
namespace JaxLean.Puzzles.Eye

theorem matches_loop  :
    Array.eye (R := ℝ) = Loop.eye (R := ℝ) := by
  funext ⟨i, j, u⟩
  cases u
  fin_cases i <;> fin_cases j <;>
    simp only [Array.fn__where, Array.eye, Loop.eye, Tensor.scalar, Tensor.reindex,
      Tensor.scatterSet_matrix]
  all_goals norm_num [Fin.ext_iff, Bool.cond_decide, decide_eq_true_iff]

theorem certificate  :
    Jaxpr.Program.eval .nil Array.eye_ir =
      Jaxpr.Program.eval .nil Loop.eye_ir := by
  rw [Array.eye_translation_correct, Loop.eye_translation_correct]
  exact matches_loop

end JaxLean.Puzzles.Eye
