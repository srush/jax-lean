import JaxLean.Stdlib
import JaxLean.Generated.PuzzlesBucketize

set_option maxRecDepth 4096
set_option maxHeartbeats 2000000
namespace JaxLean.Puzzles.Bucketize

theorem loop_spec (v boundaries : Tensor ℝ [3]) :
    Loop.bucketize v boundaries = fun i =>
      if decide (v i ≥ boundaries (2, ())) then 3
      else if decide (v i ≥ boundaries (1, ())) then 2
      else if decide (v i ≥ boundaries (0, ())) then 1 else 0 := by
  funext ⟨i, u⟩
  cases u
  fin_cases i <;> rfl

theorem matches_loop (v boundaries : Tensor ℝ [3]) :
    Array.bucketize v boundaries = Loop.bucketize v boundaries := by
  rw [loop_spec]
  funext ⟨i, u⟩
  cases u
  let cell (j : Fin 3) :=
    if decide (v (i, ()) ≥ boundaries (j, ())) then (j.val : ℝ) + 1 else 0
  change max (max (max (cell 0) (cell 0)) (cell 1)) (cell 2) = _
  rw [max_self]
  dsimp [cell]
  by_cases h0 : v (i, ()) ≥ boundaries (0, ()) <;>
    by_cases h1 : v (i, ()) ≥ boundaries (1, ()) <;>
    by_cases h2 : v (i, ()) ≥ boundaries (2, ()) <;>
    norm_num [h0, h1, h2]

theorem certificate («v» : Tensor ℝ [3]) («boundaries» : Tensor ℝ [3]) :
    TypedJaxpr.Program.eval (.cons «v» (.cons «boundaries» .nil)) Array.bucketize_ir =
      TypedJaxpr.Program.eval (.cons «v» (.cons «boundaries» .nil)) Loop.bucketize_ir := by
  rw [Array.bucketize_translation_correct, Loop.bucketize_translation_correct]
  exact matches_loop «v» «boundaries»

end JaxLean.Puzzles.Bucketize
