import JaxLean.Stdlib
import JaxLean.Generated.Mean
import JaxLean.Generated.Variance
import JaxLean.Generated.TransposeTwice
import JaxLean.Generated.SumTranspose
import Mathlib.Probability.Moments.Variance
import Mathlib.Tactic

/-! Read after Generated/Mean.lean. These are ordinary theorems about the
generated definitions; no special verification framework or admitted lemmas.
The first examples use algebra; the last connects an estimator to probability. -/
namespace JaxLean.Proofs
open Generated
open scoped BigOperators

theorem mean2_spec (x : Tensor ℝ [2]) :
    mean2 x () = (x (0, ()) + x (1, ())) / 2 := by
  simp [mean2, Tensor.scalar, Tensor.sumFirst, Fin.sum_univ_two]

/-- Swapping two samples leaves their average unchanged. -/
theorem mean2_swap (x : Tensor ℝ [2]) :
    mean2 (fun i => x (i.1.rev, ())) = mean2 x := by
  funext i
  cases i
  simp [mean2_spec, add_comm]

/-- Tensor transposition is a coordinate permutation, with no numerical work. -/
theorem transpose_twice (x : Tensor ℝ [2, 3]) : transposeTwice x = x := by
  funext i
  rfl

/-- The sum of entries is invariant under transposition. -/
theorem sum_transpose (x : Tensor ℝ [2, 3]) :
    sumTranspose x () = ∑ i : Fin 2, ∑ j : Fin 3, x (i, j, ()) := by
  simp only [sumTranspose, Tensor.reduceSum, Tensor.reindex, Index.sum_cons, Index.sum_nil]
  exact Finset.sum_comm

theorem variance2_spec (x : Tensor ℝ [2]) :
    variance2 x () = (x (0, ()) - x (1, ())) ^ 2 / 4 := by
  simp [variance2, Tensor.scalar, Tensor.sumFirst, Tensor.map, Tensor.map₂, Fin.sum_univ_two]
  ring

/-- Also demonstrate translation (shift) invariance of population variance. -/
theorem variance2_shift (x : Tensor ℝ [2]) (c : ℝ) :
    variance2 (fun i => x i + c) = variance2 x := by
  funext i
  cases i
  simp [variance2_spec]

theorem variance2_nonneg (x : Tensor ℝ [2]) : 0 ≤ variance2 x () := by
  rw [variance2_spec]
  positivity

open MeasureTheory ProbabilityTheory

/-- Two random samples passed to the *generated* estimator. This definition
does not assert anything about a JAX PRNG or manufacture independence. -/
noncomputable def estimate {Ω : Type*} (X Y : Ω → ℝ) (ω : Ω) : ℝ :=
  mean2 (fun i => if i.1 = 0 then X ω else Y ω) ()

theorem estimate_eq {Ω : Type*} (X Y : Ω → ℝ) :
    estimate X Y = fun ω => (1 / 2 : ℝ) * (X ω + Y ω) := by
  funext ω
  simp [estimate, mean2_spec]
  ring

/-- Independence and finite second moments are explicit obligations. -/
theorem estimate_variance {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ] (X Y : Ω → ℝ)
    (hX : MemLp X 2 μ) (hY : MemLp Y 2 μ) (hInd : IndepFun X Y μ) :
    variance (estimate X Y) μ = (variance X μ + variance Y μ) / 4 := by
  rw [estimate_eq, variance_const_mul, hInd.variance_fun_add hX hY]
  ring

/-- If each sample has variance at most v, their average has variance ≤ v/2. -/
theorem estimate_variance_bound {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ] (X Y : Ω → ℝ) (v : ℝ)
    (hX : MemLp X 2 μ) (hY : MemLp Y 2 μ) (hInd : IndepFun X Y μ)
    (hVX : variance X μ ≤ v) (hVY : variance Y μ ≤ v) :
    variance (estimate X Y) μ ≤ v / 2 := by
  rw [estimate_variance μ X Y hX hY hInd]
  linarith

end JaxLean.Proofs
