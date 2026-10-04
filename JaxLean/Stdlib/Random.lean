import JaxLean.Stdlib.FiniteLaw

/-! Standard-library semantics for finite random computations.
`uniformInt` is an ideal sampling specification, not a verified JAX PRNG.
The map rules hold for every distribution, including nonuniform distributions. -/
namespace JaxLean
open scoped BigOperators

noncomputable def FiniteLaw.uniform (n : Nat) (positive : 0 < n) : FiniteLaw (Fin n) where
  mass _ := 1 / (n : ℝ)
  nonneg _ := by positivity
  total := by
    have hn : (n : ℝ) ≠ 0 := by exact_mod_cast (Nat.ne_of_gt positive)
    simp [Finset.sum_const, nsmul_eq_mul, hn]

structure Rand (α : Type) where
  /-- An explicit finite sample space, allowing products without flattening them. -/
  Ω : Type
  finite : Fintype Ω
  law : @FiniteLaw Ω finite
  value : Ω → α

attribute [instance] Rand.finite

namespace Rand

def map (f : α → β) (x : Rand α) : Rand β :=
  ⟨x.Ω, x.finite, x.law, fun ω => f (x.value ω)⟩

/-- Ideal uniform draw from lo, lo+1, ..., lo+count-1, in real arithmetic. -/
noncomputable def uniformInt {R : Type} [Field R] (lo : Int) (count : Nat)
    (positive : 0 < count) : Rand R :=
  ⟨Fin count, inferInstance, FiniteLaw.uniform count positive, fun i => (lo : R) + (i.val : R)⟩

noncomputable def mean (x : Rand ℝ) : ℝ := x.law.mean x.value
noncomputable def variance (x : Rand ℝ) : ℝ := x.law.variance x.value

@[simp] theorem map_map (f : β → γ) (g : α → β) (x : Rand α) :
    map f (map g x) = map (fun v => f (g v)) x := rfl

@[simp] theorem mean_map_affine (x : Rand ℝ) (a b : ℝ) :
    (map (fun v => a * v + b) x).mean = a * x.mean + b :=
  FiniteLaw.mean_affine x.law x.value a b

@[simp] theorem variance_map_affine (x : Rand ℝ) (a b : ℝ) :
    (map (fun v => a * v + b) x).variance = a ^ 2 * x.variance :=
  FiniteLaw.variance_affine x.law x.value a b

/-- The rule needed by `return draw * scale`; it assumes no particular law. -/
@[simp] theorem variance_map_mul (x : Rand ℝ) (a : ℝ) :
    (map (fun v => v * a) x).variance = a ^ 2 * x.variance := by
  simpa only [add_zero, mul_comm] using variance_map_affine x a 0

/-- Expectation commutes with any finite batch reduction, including dependent terms. -/
theorem mean_map_sum {ι : Type} [Fintype ι] (x : Rand α) (f : ι → α → ℝ) :
    (map (fun v => ∑ i, f i v) x).mean = ∑ i, (map (f i) x).mean :=
  FiniteLaw.mean_sum x.law (fun i ω => f i (x.value ω))

/-- Rules match expressions inside a map, so they compose through emitted let-bindings. -/
theorem mean_map_add (x : Rand α) (f g : α → ℝ) :
    (map (fun v => f v + g v) x).mean = (map f x).mean + (map g x).mean :=
  FiniteLaw.mean_add x.law _ _

theorem mean_map_mul (x : Rand α) (f : α → ℝ) (a : ℝ) :
    (map (fun v => f v * a) x).mean = (map f x).mean * a := by
  simpa only [add_zero, mul_comm, map_map] using mean_map_affine (map f x) a 0

theorem mean_map_div (x : Rand α) (f : α → ℝ) (a : ℝ) :
    (map (fun v => f v / a) x).mean = (map f x).mean / a := by
  simpa only [div_eq_mul_inv] using mean_map_mul x f a⁻¹

theorem variance_map_scale (x : Rand α) (f : α → ℝ) (a : ℝ) :
    (map (fun v => f v * a) x).variance = a ^ 2 * (map f x).variance :=
  variance_map_mul (map f x) a

theorem variance_map_div (x : Rand α) (f : α → ℝ) (a : ℝ) :
    (map (fun v => f v / a) x).variance = (map f x).variance / a ^ 2 := by
  simpa only [div_eq_mul_inv, inv_pow, mul_comm] using variance_map_scale x f a⁻¹

theorem variance_map_add_const (x : Rand α) (f : α → ℝ) (b : ℝ) :
    (map (fun v => f v + b) x).variance = (map f x).variance := by
  simpa only [one_mul, one_pow, map_map] using variance_map_affine (map f x) 1 b

/-- Shared randomness retains covariance; addition alone never implies independence. -/
theorem variance_map_add (x : Rand α) (f g : α → ℝ) :
    (map (fun v => f v + g v) x).variance =
      (map f x).variance + (map g x).variance +
      2 * x.law.covariance (fun ω => f (x.value ω)) (fun ω => g (x.value ω)) :=
  FiniteLaw.variance_add x.law _ _

/-- Nonlinear operations expose the moments they require; two moments are not always enough. -/
theorem variance_map_second_moment (x : Rand α) (f : α → ℝ) :
    (map f x).variance = (map (fun v => f v ^ 2) x).mean - (map f x).mean ^ 2 :=
  FiniteLaw.variance_eq_second_moment x.law _

end Rand
end JaxLean
