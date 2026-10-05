import JaxLean.Stdlib
import examples.tensor_puzzles.generated.PuzzlesPadTo

set_option maxRecDepth 4096
set_option maxHeartbeats 2000000
namespace JaxLean.Puzzles.PadTo

theorem matches_loop («a» : Tensor ℝ [3]) :
    Array.pad_to «a» = Loop.pad_to «a» := by
  funext ⟨i, u⟩
  cases u
  fin_cases i <;> rfl

theorem certificate («a» : Tensor ℝ [3]) :
    Jaxpr.Program.eval (.cons «a» .nil) Array.pad_to_ir =
      Jaxpr.Program.eval (.cons «a» .nil) Loop.pad_to_ir := by
  rw [Array.pad_to_translation_correct, Loop.pad_to_translation_correct]
  exact matches_loop «a»

end JaxLean.Puzzles.PadTo
