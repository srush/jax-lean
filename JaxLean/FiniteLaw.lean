import JaxLean.Batch
import Mathlib.Data.Real.Basic
import Mathlib.Data.Fintype.BigOperators
import Mathlib.Tactic

/-! Exact finite probability, separate from any particular sampler or program.
A random variable is a function on outcomes. Reusing that function preserves
dependence; `prod` explicitly constructs independent sample spaces. -/
namespace JaxLean
open scoped BigOperators

structure FiniteLaw (Ω : Type) [Fintype Ω] where
  mass : Ω → ℝ
  nonneg : ∀ ω, 0 ≤ mass ω
  total : ∑ ω, mass ω = 1

namespace FiniteLaw
variable {Ω Ξ : Type} [Fintype Ω] [Fintype Ξ]

noncomputable def mean (p : FiniteLaw Ω) (X : Ω → ℝ) : ℝ :=
  ∑ ω, p.mass ω * X ω

noncomputable def variance (p : FiniteLaw Ω) (X : Ω → ℝ) : ℝ :=
  p.mean (fun ω => (X ω - p.mean X) ^ 2)

noncomputable def covariance (p : FiniteLaw Ω) (X Y : Ω → ℝ) : ℝ :=
  p.mean (fun ω => (X ω - p.mean X) * (Y ω - p.mean Y))

@[simp] theorem mean_const (p : FiniteLaw Ω) (c : ℝ) :
    p.mean (fun _ => c) = c := by
  simp [mean, ← Finset.sum_mul, p.total]

theorem mean_add (p : FiniteLaw Ω) (X Y : Ω → ℝ) :
    p.mean (fun ω => X ω + Y ω) = p.mean X + p.mean Y := by
  simp [mean, mul_add, Finset.sum_add_distrib]

theorem mean_sub (p : FiniteLaw Ω) (X Y : Ω → ℝ) :
    p.mean (fun ω => X ω - Y ω) = p.mean X - p.mean Y := by
  simp [mean, mul_sub, Finset.sum_sub_distrib]

theorem mean_scale (p : FiniteLaw Ω) (a : ℝ) (X : Ω → ℝ) :
    p.mean (fun ω => a * X ω) = a * p.mean X := by
  simp [mean, mul_left_comm, Finset.mul_sum]

/-- Expectation is just a linear functional on a space of value functions. -/
noncomputable def meanLinear (p : FiniteLaw Ω) : (Ω → ℝ) →ₗ[ℝ] ℝ where
  toFun := p.mean
  map_add' := p.mean_add
  map_smul' := p.mean_scale

/-- Reduction commutation is generic linear algebra, without independence. -/
theorem mean_sum {ι : Type} [Fintype ι] (p : FiniteLaw Ω) (X : ι → Ω → ℝ) :
    p.mean (fun ω => ∑ i, X i ω) = ∑ i, p.mean (X i) := by
  have h : (fun ω => ∑ i, X i ω) = ∑ i, X i := by
    funext ω
    simp only [Finset.sum_apply]
  rw [h]
  exact Batch.linear_reduceSum p.meanLinear X

theorem mean_affine (p : FiniteLaw Ω) (X : Ω → ℝ) (a b : ℝ) :
    p.mean (fun ω => a * X ω + b) = a * p.mean X + b := by
  rw [mean_add, mean_scale, mean_const]

theorem variance_nonneg (p : FiniteLaw Ω) (X : Ω → ℝ) :
    0 ≤ p.variance X := by
  exact Finset.sum_nonneg (fun ω _ => mul_nonneg (p.nonneg ω) (sq_nonneg _))

/-- Useful after a nonlinear operation: retain the second moment, not just the mean. -/
theorem variance_eq_second_moment (p : FiniteLaw Ω) (X : Ω → ℝ) :
    p.variance X = p.mean (fun ω => X ω ^ 2) - p.mean X ^ 2 := by
  unfold variance
  have h : (fun ω => (X ω - p.mean X) ^ 2) =
      (fun ω => X ω ^ 2 - (2 * p.mean X) * X ω + p.mean X ^ 2) := by
    funext ω
    ring
  rw [h, mean_add, mean_sub, mean_scale, mean_const]
  ring

theorem variance_affine (p : FiniteLaw Ω) (X : Ω → ℝ) (a b : ℝ) :
    p.variance (fun ω => a * X ω + b) = a ^ 2 * p.variance X := by
  unfold variance
  rw [mean_affine]
  have h : (fun ω => (a * X ω + b - (a * p.mean X + b)) ^ 2) =
      (fun ω => a ^ 2 * (X ω - p.mean X) ^ 2) := by
    funext ω
    ring
  rw [h, mean_scale]

/-- No independence assumption: reusing a draw contributes covariance. -/
theorem variance_add (p : FiniteLaw Ω) (X Y : Ω → ℝ) :
    p.variance (fun ω => X ω + Y ω) =
      p.variance X + p.variance Y + 2 * p.covariance X Y := by
  unfold variance covariance
  rw [mean_add]
  have h : (fun ω => (X ω + Y ω - (p.mean X + p.mean Y)) ^ 2) =
      (fun ω => (X ω - p.mean X) ^ 2 + (Y ω - p.mean Y) ^ 2 +
        2 * ((X ω - p.mean X) * (Y ω - p.mean Y))) := by
    funext ω
    ring
  rw [h, mean_add, mean_add, mean_scale]

/-- Independent draws use product probabilities. This is an explicit construction. -/
noncomputable def prod (p : FiniteLaw Ω) (q : FiniteLaw Ξ) : FiniteLaw (Ω × Ξ) where
  mass ω := p.mass ω.1 * q.mass ω.2
  nonneg ω := mul_nonneg (p.nonneg _) (q.nonneg _)
  total := by
    simp [Fintype.sum_prod_type, ← Finset.mul_sum, q.total, p.total]

@[simp] theorem mean_prod_left (p : FiniteLaw Ω) (q : FiniteLaw Ξ) (X : Ω → ℝ) :
    (p.prod q).mean (fun ω => X ω.1) = p.mean X := by
  simp [mean, prod, Fintype.sum_prod_type, mul_right_comm,
    ← Finset.mul_sum, q.total]

@[simp] theorem mean_prod_right (p : FiniteLaw Ω) (q : FiniteLaw Ξ) (Y : Ξ → ℝ) :
    (p.prod q).mean (fun ω => Y ω.2) = q.mean Y := by
  simp [mean, prod, Fintype.sum_prod_type, mul_assoc,
    ← Finset.mul_sum, ← Finset.sum_mul, p.total]

theorem mean_prod_mul (p : FiniteLaw Ω) (q : FiniteLaw Ξ)
    (X : Ω → ℝ) (Y : Ξ → ℝ) :
    (p.prod q).mean (fun ω => X ω.1 * Y ω.2) = p.mean X * q.mean Y := by
  simp only [mean, prod, Fintype.sum_prod_type, Finset.sum_mul, Finset.mul_sum]
  rw [Finset.sum_comm]
  apply Finset.sum_congr rfl
  intro ω _
  apply Finset.sum_congr rfl
  intro ξ _
  ring

theorem variance_independent_add (p : FiniteLaw Ω) (q : FiniteLaw Ξ)
    (X : Ω → ℝ) (Y : Ξ → ℝ) :
    (p.prod q).variance (fun ω => X ω.1 + Y ω.2) = p.variance X + q.variance Y := by
  rw [variance_eq_second_moment, mean_add, mean_prod_left, mean_prod_right]
  have h : (fun ω : Ω × Ξ => (X ω.1 + Y ω.2) ^ 2) =
      (fun ω => X ω.1 ^ 2 + Y ω.2 ^ 2 + 2 * (X ω.1 * Y ω.2)) := by
    funext ω
    ring
  rw [h, mean_add, mean_add, mean_scale, mean_prod_mul,
    mean_prod_left p q (fun ω => X ω ^ 2), mean_prod_right p q (fun ω => Y ω ^ 2),
    variance_eq_second_moment, variance_eq_second_moment]
  ring

/-- Constant division scales variance by the squared divisor. -/
theorem variance_div (p : FiniteLaw Ω) (X : Ω → ℝ) (a : ℝ) :
    p.variance (fun ω => X ω / a) = p.variance X / a ^ 2 := by
  have h : (fun ω => X ω / a) = (fun ω => a⁻¹ * X ω + 0) := by
    funext ω
    simp [div_eq_mul_inv, mul_comm]
  rw [h, variance_affine]
  simp [div_eq_mul_inv, mul_comm]

end FiniteLaw
end JaxLean
