import JaxLean.Stdlib
import examples.tensor_puzzles.generated.PuzzlesBincount

set_option maxRecDepth 4096
set_option maxHeartbeats 2000000
namespace JaxLean.Puzzles.Bincount

theorem matches_loop («a» : Tensor Int32 [3]) :
    Array.bincount (R := ℝ) «a» = Loop.bincount (R := ℝ) «a» := by
  funext ⟨i, u⟩
  cases u
  fin_cases i <;>
    simp only [Array.fn__where, Array.bincount, Loop.fn__where, Loop.loop_masked_sum,
      Loop.bincount, Tensor.map, Tensor.scalar, Tensor.reindex, Tensor.scatterSet_vector,
      Fin.sum_univ_succ]
  all_goals norm_num [Fin.ext_iff, Bool.cond_decide, decide_eq_true_iff]
  all_goals simp []
  all_goals ring

theorem certificate («a» : Tensor Int32 [3]) :
    Jaxpr.Program.eval (.cons «a» .nil) Array.bincount_ir =
      Jaxpr.Program.eval (.cons «a» .nil) Loop.bincount_ir := by
  rw [Array.bincount_translation_correct, Loop.bincount_translation_correct]
  exact matches_loop «a»

end JaxLean.Puzzles.Bincount
