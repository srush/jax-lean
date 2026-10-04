import JaxLean.Generated.Draw
import JaxLean.Generated.AffineDraw
import JaxLean.Generated.SquaredDraw
import JaxLean.Generated.ReusedDraw
import JaxLean.Generated.AverageDraw

/-! The generated files prove numeric moments by exact finite sums.
Here we also apply general rules directly to the transpiled computations,
so subsequent affine steps can reuse earlier results without enumerating again.
-/
namespace JaxLean.SamplingProofs
open Generated

-- This identity unfolds actual generated code, not a handwritten network.
theorem affine_values : affineDrawValue = fun ω => 2 * drawValue ω + 1 := by
  funext ω
  rfl

theorem affine_mean_by_rule : affineDrawLaw.mean affineDrawValue = 7 / 2 := by
  -- Mapping preserves the source law; only the value function changes.
  change drawLaw.mean affineDrawValue = 7 / 2
  rw [affine_values, FiniteLaw.mean_affine, draw_mean]
  norm_num

theorem affine_variance_by_rule : affineDrawLaw.variance affineDrawValue = 27 / 4 := by
  change drawLaw.variance affineDrawValue = 27 / 4
  rw [affine_values, FiniteLaw.variance_affine, draw_variance]
  norm_num

theorem reused_values : reusedDrawValue = fun ω => 2 * squaredDrawValue ω + 0 := by
  funext ω
  change squaredDrawValue ω + squaredDrawValue ω = 2 * squaredDrawValue ω + 0
  ring

theorem reused_variance_by_rule : reusedDrawLaw.variance reusedDrawValue = 432 := by
  change squaredDrawLaw.variance reusedDrawValue = 432
  rw [reused_values, FiniteLaw.variance_affine, squaredDraw_variance]
  norm_num

-- Propagate division, then independent addition on the product sample space.
theorem independent_average_by_rule :
    (squaredDrawLaw.prod squaredDrawLaw).variance
      (fun ω => (squaredDrawValue ω.1 + squaredDrawValue ω.2) / 2) = 54 := by
  rw [FiniteLaw.variance_div, FiniteLaw.variance_independent_add, squaredDraw_variance]
  norm_num

end JaxLean.SamplingProofs
