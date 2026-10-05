import JaxLean.Stdlib
import examples.tensor_puzzles.generated.PuzzlesLinspace

set_option maxRecDepth 4096
set_option maxHeartbeats 2000000
namespace JaxLean.Puzzles.Linspace

theorem matches_loop («start» : Tensor ℝ []) («stop» : Tensor ℝ []) :
    Array.linspace «start» «stop» = Loop.linspace «start» «stop» := by
  funext ⟨i, u⟩
  cases u
  fin_cases i <;>
    simp only [Array.linspace, Loop.linspace, Tensor.map, Tensor.scalar, Tensor.reindex,
      Tensor.scatterSet_vector]
  all_goals norm_num [Fin.ext_iff, Bool.cond_decide, decide_eq_true_iff]

theorem certificate («start» : Tensor ℝ []) («stop» : Tensor ℝ []) :
    Jaxpr.Program.eval (.cons «start» (.cons «stop» .nil)) Array.linspace_ir =
      Jaxpr.Program.eval (.cons «start» (.cons «stop» .nil)) Loop.linspace_ir := by
  rw [Array.linspace_translation_correct, Loop.linspace_translation_correct]
  exact matches_loop «start» «stop»

end JaxLean.Puzzles.Linspace
