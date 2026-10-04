import JaxLean.Stdlib
import JaxLean.Generated.PuzzlesDiag

set_option maxRecDepth 4096
set_option maxHeartbeats 2000000
namespace JaxLean.Puzzles.Diag

theorem matches_loop («a» : Tensor ℝ [3, 3]) :
    Array.diag «a» = Loop.diag «a» := by
  funext ⟨i, u⟩
  cases u
  fin_cases i <;>
    simp only [Array.fn__where, Array.diag, Loop.diag, Tensor.map₂, Tensor.scalar,
      Tensor.reindex, Tensor.scatterSet_vector, Fin.sum_univ_succ]
  all_goals norm_num [Fin.ext_iff, Bool.cond_decide, decide_eq_true_iff]
  all_goals simp []

theorem certificate («a» : Tensor ℝ [3, 3]) :
    TypedJaxpr.Program.eval (.cons «a» .nil) Array.diag_ir =
      Jaxpr.Program.eval (.cons «a» .nil) Loop.diag_ir := by
  rw [Array.diag_translation_correct, Loop.diag_translation_correct]
  exact matches_loop «a»

end JaxLean.Puzzles.Diag
