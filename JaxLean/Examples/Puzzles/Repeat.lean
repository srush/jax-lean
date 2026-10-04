import JaxLean.Stdlib
import JaxLean.Generated.PuzzlesRepeat

set_option maxRecDepth 4096
set_option maxHeartbeats 2000000
namespace JaxLean.Puzzles.Repeat

theorem matches_loop («a» : Tensor ℝ [3]) :
    Array.«repeat» «a» = Loop.«repeat» «a» := by
  funext ⟨i, j, u⟩
  cases u
  fin_cases i <;> fin_cases j <;> rfl

theorem certificate («a» : Tensor ℝ [3]) :
    Jaxpr.Program.eval (.cons «a» .nil) Array.repeat_ir =
      Jaxpr.Program.eval (.cons «a» .nil) Loop.repeat_ir := by
  rw [Array.repeat_translation_correct, Loop.repeat_translation_correct]
  exact matches_loop «a»

end JaxLean.Puzzles.Repeat
