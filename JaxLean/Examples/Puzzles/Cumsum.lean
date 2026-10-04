import JaxLean.Stdlib
import JaxLean.Generated.PuzzlesCumsum

set_option maxRecDepth 4096
set_option maxHeartbeats 2000000
namespace JaxLean.Puzzles.Cumsum

theorem matches_loop («a» : Tensor ℝ [3]) :
    Array.cumsum «a» = Loop.cumsum «a» := by
  funext ⟨i, u⟩
  cases u
  fin_cases i <;>
    simp only [Array.fn__where, Array.puzzle_cumsum, Array.cumsum, Loop.cumsum, Tensor.scalar,
      Tensor.reindex, Tensor.scatterSet_vector, Fin.sum_univ_succ]
  all_goals norm_num [Fin.ext_iff, Bool.cond_decide, decide_eq_true_iff]
  all_goals simp []
  all_goals ring

theorem certificate («a» : Tensor ℝ [3]) :
    TypedJaxpr.Program.eval (.cons «a» .nil) Array.cumsum_ir =
      Jaxpr.Program.eval (.cons «a» .nil) Loop.cumsum_ir := by
  rw [Array.cumsum_translation_correct, Loop.cumsum_translation_correct]
  exact matches_loop «a»

end JaxLean.Puzzles.Cumsum
