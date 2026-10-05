import JaxLean.Stdlib
import examples.tensor_puzzles.generated.PuzzlesVstack

set_option maxRecDepth 4096
set_option maxHeartbeats 2000000
namespace JaxLean.Puzzles.Vstack

theorem matches_loop («a» : Tensor ℝ [3]) («b» : Tensor ℝ [3]) :
    Array.vstack «a» «b» = Loop.vstack «a» «b» := by
  funext ⟨i, j, u⟩
  cases u
  fin_cases i <;> fin_cases j <;> rfl

theorem certificate («a» : Tensor ℝ [3]) («b» : Tensor ℝ [3]) :
    Jaxpr.Program.eval (.cons «a» (.cons «b» .nil)) Array.vstack_ir =
      Jaxpr.Program.eval (.cons «a» (.cons «b» .nil)) Loop.vstack_ir := by
  rw [Array.vstack_translation_correct, Loop.vstack_translation_correct]
  exact matches_loop «a» «b»

end JaxLean.Puzzles.Vstack
