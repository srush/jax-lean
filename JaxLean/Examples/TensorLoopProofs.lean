import JaxLean.Stdlib
import JaxLean.Examples.TensorPuzzleProofs
import JaxLean.Generated.LoopSum
import JaxLean.Generated.LoopOuter
import JaxLean.Generated.LoopFlip
import JaxLean.Generated.LoopFlatten

namespace JaxLean.PuzzleJax
open scoped BigOperators

theorem sum_matches_loop (a : Tensor ℝ [4]) : sum a = loop_sum a := by
  funext i
  cases i
  rw [sum_matches_pseudocode, Pseudocode.sum_eq]
  change (∑ i : Fin 4, a (i, ())) =
    (((0 + a (0, ())) + a (1, ())) + a (2, ())) + a (3, ())
  simp [Fin.sum_univ_succ]
  ring_nf

theorem outer_matches_loop (a : Tensor ℝ [2]) (b : Tensor ℝ [3]) :
    puzzle_outer a b = loop_outer a b := by
  funext i
  rcases i with ⟨i, j, ⟨⟩⟩
  fin_cases i <;> fin_cases j <;> rfl

theorem flip_matches_loop (a : Tensor ℝ [4]) : flip a = loop_flip a := by
  funext i
  rcases i with ⟨i, ⟨⟩⟩
  fin_cases i <;> rfl

theorem flatten_matches_loop (a : Tensor ℝ [2, 3]) :
    puzzle_flatten a = loop_flatten a := by
  rw [flatten_matches_pseudocode, Pseudocode.flatten_eq]
  funext i
  rcases i with ⟨i, ⟨⟩⟩
  fin_cases i <;> rfl

theorem certified_sum_loop (a : Tensor ℝ [4]) :
    Jaxpr.Program.eval (.cons a .nil) sum_ir =
      Jaxpr.Program.eval (.cons a .nil) loop_sum_ir := by
  rw [sum_translation_correct, loop_sum_translation_correct, sum_matches_loop]

theorem certified_outer_loop (a : Tensor ℝ [2]) (b : Tensor ℝ [3]) :
    Jaxpr.Program.eval (.cons a (.cons b .nil)) puzzle_outer_ir =
      Jaxpr.Program.eval (.cons a (.cons b .nil)) loop_outer_ir := by
  rw [puzzle_outer_translation_correct, loop_outer_translation_correct, outer_matches_loop]

theorem certified_flip_loop (a : Tensor ℝ [4]) :
    Jaxpr.Program.eval (.cons a .nil) flip_ir =
      Jaxpr.Program.eval (.cons a .nil) loop_flip_ir := by
  rw [flip_translation_correct, loop_flip_translation_correct, flip_matches_loop]

theorem certified_flatten_loop (a : Tensor ℝ [2, 3]) :
    Jaxpr.Program.eval (.cons a .nil) puzzle_flatten_ir =
      Jaxpr.Program.eval (.cons a .nil) loop_flatten_ir := by
  rw [puzzle_flatten_translation_correct, loop_flatten_translation_correct, flatten_matches_loop]

end JaxLean.PuzzleJax
