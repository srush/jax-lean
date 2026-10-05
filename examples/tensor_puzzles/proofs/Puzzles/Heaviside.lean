import JaxLean.Stdlib
import examples.tensor_puzzles.generated.PuzzlesHeaviside

set_option maxRecDepth 4096
set_option maxHeartbeats 2000000
namespace JaxLean.Puzzles.Heaviside

theorem matches_loop («a» : Tensor ℝ [3]) («b» : Tensor ℝ [3]) :
    Array.heaviside «a» «b» = Loop.heaviside «a» «b» := by
  funext ⟨i, u⟩
  cases u
  fin_cases i <;>
    simp only [Array.fn__where, Array.fn__where_2, Array.heaviside, Loop.fn__where,
      Loop.fn__where_2, Loop.heaviside, Tensor.map, Tensor.map₂, Tensor.scalar, Tensor.reindex,
      Tensor.scatterSet_vector, Tensor.scatterSet_matrix, Tensor.concatenate, Tensor.sumFirst,
      Tensor.reduceSum, Fin.sum_univ_succ, Bool.cond_decide]
  all_goals norm_num [Fin.ext_iff]
  all_goals exact if_congr decide_eq_true_iff rfl rfl

theorem certificate («a» : Tensor ℝ [3]) («b» : Tensor ℝ [3]) :
    Jaxpr.Program.eval (.cons «a» (.cons «b» .nil)) Array.heaviside_ir =
      Jaxpr.Program.eval (.cons «a» (.cons «b» .nil)) Loop.heaviside_ir := by
  rw [Array.heaviside_translation_correct, Loop.heaviside_translation_correct]
  exact matches_loop «a» «b»

end JaxLean.Puzzles.Heaviside
