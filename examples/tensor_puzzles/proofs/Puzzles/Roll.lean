import JaxLean.Stdlib
import examples.tensor_puzzles.generated.PuzzlesRoll

set_option maxRecDepth 4096
set_option maxHeartbeats 2000000
namespace JaxLean.Puzzles.Roll

theorem matches_loop («a» : Tensor ℝ [3]) :
    Array.roll «a» = Loop.roll «a» := by
  funext ⟨i, u⟩
  cases u
  fin_cases i <;> rfl

theorem certificate («a» : Tensor ℝ [3]) :
    Jaxpr.Program.eval (.cons «a» .nil) Array.roll_ir =
      Jaxpr.Program.eval (.cons «a» .nil) Loop.roll_ir := by
  rw [Array.roll_translation_correct, Loop.roll_translation_correct]
  exact matches_loop «a»

end JaxLean.Puzzles.Roll
