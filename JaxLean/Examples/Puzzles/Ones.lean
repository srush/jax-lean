import JaxLean.Stdlib
import JaxLean.Generated.PuzzlesOnes

set_option maxRecDepth 4096
set_option maxHeartbeats 2000000
namespace JaxLean.Puzzles.Ones

theorem matches_loop  :
    Array.ones (R := ℝ) = Loop.ones (R := ℝ) := by
  funext ⟨i, u⟩
  cases u
  fin_cases i <;> rfl

theorem certificate  :
    Jaxpr.Program.eval .nil Array.ones_ir =
      Jaxpr.Program.eval .nil Loop.ones_ir := by
  rw [Array.ones_translation_correct, Loop.ones_translation_correct]
  exact matches_loop

end JaxLean.Puzzles.Ones
