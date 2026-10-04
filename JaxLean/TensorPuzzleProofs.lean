import JaxLean.Pseudocode
import JaxLean.Generated.PuzzleSum
import JaxLean.Generated.PuzzleFlip
import JaxLean.Generated.PuzzleOuterFlatten

/-! Tensor Puzzles: JAX implementations equal independent Lean pseudocode.
Every theorem quantifies over all real tensor values of the traced shapes.
Read Pseudocode.lean for the specifications; no generated code is used there. -/
namespace JaxLean.PuzzleJax
open scoped BigOperators

/-- Dotting with ones implements the sequential accumulator loop. -/
theorem sum_matches_pseudocode (a : Tensor ℝ [4]) :
    sum a () = Pseudocode.sum a := by
  rw [Pseudocode.sum_eq]
  simp only [sum, puzzle_sum, Tensor.reindex, Tensor.scalar, mul_one]

/-- Broadcasting is just the nested-loop multiplication of individual cells. -/
theorem outer_matches_pseudocode (a : Tensor ℝ [2]) (b : Tensor ℝ [3]) :
    puzzle_outer a b = Pseudocode.outer a b := by
  rw [Pseudocode.outer_eq]
  rfl

/-- Slicing with step -1 implements the explicit n-1-i index. -/
theorem flip_matches_pseudocode (a : Tensor ℝ [4]) :
    flip a = Pseudocode.flip a := by
  rw [Pseudocode.flip_eq]
  rfl

/-- Reshape implements row-major quotient/remainder indexing. -/
theorem flatten_matches_pseudocode (a : Tensor ℝ [2, 3]) :
    puzzle_flatten a = Pseudocode.flatten a :=
  Pseudocode.reshape_flatten a

/-- Composition uses the function contracts; it never unfolds either implementation. -/
theorem outer_flatten_matches_pseudocode (a : Tensor ℝ [2]) (b : Tensor ℝ [3]) :
    outer_flatten a b = Pseudocode.flatten (Pseudocode.outer a b) := by
  simp only [outer_flatten, flatten_matches_pseudocode, outer_matches_pseudocode]

/-- The independent Jaxpr evaluator computes that same pseudocode. -/
theorem certified_outer_flatten (a : Tensor ℝ [2]) (b : Tensor ℝ [3]) :
    Jaxpr.Program.eval (.cons a (.cons b .nil)) outer_flatten_ir =
      Pseudocode.flatten (Pseudocode.outer a b) := by
  rw [outer_flatten_translation_correct, outer_flatten_matches_pseudocode]

theorem certified_sum (a : Tensor ℝ [4]) :
    Jaxpr.Program.eval (.cons a .nil) sum_ir () = Pseudocode.sum a := by
  rw [sum_translation_correct, sum_matches_pseudocode]

theorem certified_flip (a : Tensor ℝ [4]) :
    Jaxpr.Program.eval (.cons a .nil) flip_ir = Pseudocode.flip a := by
  rw [flip_translation_correct, flip_matches_pseudocode]

theorem certified_outer (a : Tensor ℝ [2]) (b : Tensor ℝ [3]) :
    Jaxpr.Program.eval (.cons a (.cons b .nil)) puzzle_outer_ir =
      Pseudocode.outer a b := by
  rw [puzzle_outer_translation_correct, outer_matches_pseudocode]

theorem certified_flatten (a : Tensor ℝ [2, 3]) :
    Jaxpr.Program.eval (.cons a .nil) puzzle_flatten_ir = Pseudocode.flatten a := by
  rw [puzzle_flatten_translation_correct, flatten_matches_pseudocode]

end JaxLean.PuzzleJax
