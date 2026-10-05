import JaxLean.Stdlib
import examples.random_program.generated.SampleTimes100
import examples.random_program.generated.SampleScaled

/-! Read `examples/random_program/code.py`, then the generated definitions.
These statements name the original Python functions. Proofs apply stdlib rules
directly to their transpiled bodies, with no handwritten copy of the program.
-/
namespace JaxLean.RandomProgramProofs
open Generated

/-- The exact property requested: multiplying the draw by 100 scales variance
by 100². No enumeration, sampled estimates, or numeric moment claims in Python. -/
theorem sample_times_100_variance :
    (sample_times_100 (R := ℝ)).variance =
      100 ^ 2 * (Rand.uniformInt (R := ℝ) 0 6 (by decide)).variance := by
  simp only [sample_times_100, Rand.variance_map_mul]

/-- The same stdlib rule proves the result for a runtime scale parameter. -/
theorem sample_scaled_variance (scale : ℝ) :
    (sample_scaled scale).variance =
      scale ^ 2 * (Rand.uniformInt (R := ℝ) 0 6 (by decide)).variance := by
  simp only [sample_scaled, Rand.variance_map_mul]

/-- A numeric corollary, if desired: the ideal draw is uniform on 0,...,5. -/
theorem sample_times_100_variance_value :
    (sample_times_100 (R := ℝ)).variance = 87500 / 3 := by
  rw [sample_times_100_variance]
  dsimp [Rand.variance, Rand.uniformInt, FiniteLaw.uniform,
    FiniteLaw.variance, FiniteLaw.mean]
  norm_num [Fin.sum_univ_succ]

end JaxLean.RandomProgramProofs
