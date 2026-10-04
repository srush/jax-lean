import JaxLean.Stdlib
import JaxLean.Generated.MonteCarlo4
import JaxLean.Generated.MonteCarloSquare8
import JaxLean.Generated.SampleTimes100
import JaxLean.Generated.SampleSquarePlusOne

/-! The theorem statements name the actual Python functions.
Unfold the generated code and propagate elementary moment rules through it.
There is no Monte Carlo definition or theorem in the stdlib.
There is no sampler enumeration or handwritten replacement implementation. -/
namespace JaxLean.MonteCarloProofs
open Generated

theorem monte_carlo_4_variance :
    (monte_carlo_4 (R := ℝ)).variance = (sample_times_100 (R := ℝ)).variance / 4 := by
  simp only [monte_carlo_4, Tensor.scalar, Tensor.sumFirst, Batch.reduceSum, Tensor.map]
  rw [Rand.variance_map_div, Rand.variance_iid_sum (f := fun _ v => v * 100)]
  simp only [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul,
    sample_times_100]
  ring

/-- The same generated program preserves the expectation of one transformed draw. -/
theorem monte_carlo_4_mean :
    (monte_carlo_4 (R := ℝ)).mean = (sample_times_100 (R := ℝ)).mean := by
  simp only [monte_carlo_4, Tensor.scalar, Tensor.sumFirst, Batch.reduceSum, Tensor.map]
  rw [Rand.mean_map_div, Rand.mean_iid_sum (f := fun _ v => v * 100)]
  simp only [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul,
    sample_times_100]
  ring

/-- The requested σ² formulation, about the generated function. -/
theorem monte_carlo_4_of_variance (sigma2 : ℝ)
    (h : (sample_times_100 (R := ℝ)).variance = sigma2) :
    (monte_carlo_4 (R := ℝ)).variance = sigma2 / 4 := by
  rw [monte_carlo_4_variance, h]

/-- Same proof for a nonlinear f and a different sample count. -/
theorem monte_carlo_square_8_variance :
    (monte_carlo_square_8 (R := ℝ)).variance =
      (sample_square_plus_one (R := ℝ)).variance / 8 := by
  simp only [monte_carlo_square_8, Tensor.scalar, Tensor.sumFirst, Batch.reduceSum, Tensor.map, Tensor.map₂]
  rw [Rand.variance_map_div, Rand.variance_iid_sum (f := fun _ v => v * v + 1)]
  simp only [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul,
    sample_square_plus_one]
  ring

end JaxLean.MonteCarloProofs
