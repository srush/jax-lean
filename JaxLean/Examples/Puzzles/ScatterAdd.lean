import JaxLean.Stdlib
import JaxLean.Generated.PuzzlesScatterAdd

set_option maxRecDepth 4096
set_option maxHeartbeats 2000000
namespace JaxLean.Puzzles.ScatterAdd

theorem matches_loop («values» : Tensor ℝ [3]) («link» : Tensor Int32 [3]) :
    Array.scatter_add «values» «link» = Loop.scatter_add «values» «link» := by
  funext ⟨i, u⟩
  cases u
  fin_cases i <;>
    simp only [Array.fn__where, Array.scatter_add, Loop.fn__where, Loop.loop_masked_sum,
      Loop.scatter_add, Tensor.map, Tensor.scalar, Tensor.reindex, Tensor.scatterSet_vector,
      Fin.sum_univ_succ]
  all_goals norm_num [Fin.ext_iff, Bool.cond_decide, decide_eq_true_iff]
  all_goals simp []
  all_goals ring

theorem certificate («values» : Tensor ℝ [3]) («link» : Tensor Int32 [3]) :
    TypedJaxpr.Program.eval (.cons «values» (.cons «link» .nil)) Array.scatter_add_ir =
      TypedJaxpr.Program.eval (.cons «values» (.cons «link» .nil)) Loop.scatter_add_ir := by
  rw [Array.scatter_add_translation_correct, Loop.scatter_add_translation_correct]
  exact matches_loop «values» «link»

end JaxLean.Puzzles.ScatterAdd
