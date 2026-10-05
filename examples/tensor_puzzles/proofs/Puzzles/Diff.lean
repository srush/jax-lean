import JaxLean.Stdlib
import examples.tensor_puzzles.generated.PuzzlesDiff

set_option maxRecDepth 4096
set_option maxHeartbeats 2000000
namespace JaxLean.Puzzles.Diff

theorem matches_loop («a» : Tensor ℝ [3]) :
    Array.diff «a» = Loop.diff «a» := by
  funext ⟨i, u⟩
  cases u
  fin_cases i <;> rfl

theorem certificate («a» : Tensor ℝ [3]) :
    Jaxpr.Program.eval (.cons «a» .nil) Array.diff_ir =
      Jaxpr.Program.eval (.cons «a» .nil) Loop.diff_ir := by
  rw [Array.diff_translation_correct, Loop.diff_translation_correct]
  exact matches_loop «a»

end JaxLean.Puzzles.Diff
