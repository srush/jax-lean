import JaxLean.Batch
import Mathlib.Analysis.SpecialFunctions.Sqrt
import Mathlib.Tactic

/-! Explicit Euclidean norms on finite real vectors. We do not use the default
Pi-type norm, which is a different norm. No probability or program imports. -/
namespace JaxLean.Batch
open scoped BigOperators
variable {ι κ : Type} [Fintype ι] [Fintype κ]

noncomputable def l2 (x : ι → ℝ) : ℝ := Real.sqrt (∑ i, x i ^ 2)

theorem l2_nonneg (x : ι → ℝ) : 0 ≤ l2 x := Real.sqrt_nonneg _

theorem l2_mono_abs (x y : ι → ℝ) (h : ∀ i, |x i| ≤ |y i|) : l2 x ≤ l2 y := by
  apply Real.sqrt_le_sqrt
  exact Finset.sum_le_sum (fun i _ => sq_le_sq.mpr (h i))

theorem l2_scale (a : ℝ) (x : ι → ℝ) :
    l2 (fun i => x i * a) = |a| * l2 x := by
  simp only [l2, mul_pow, ← Finset.sum_mul]
  rw [Real.sqrt_mul (Finset.sum_nonneg (fun i _ => sq_nonneg (x i))), Real.sqrt_sq_eq_abs]
  ring

/-- A scalar pointwise contraction is a vector contraction. -/
theorem l2_vmap_le (f : ℝ → ℝ) (x : ι → ℝ) (c : ℝ) (hc : 0 ≤ c)
    (hf : ∀ v, |f v| ≤ c * |v|) : l2 (vmap f x) ≤ c * l2 x := by
  calc
    l2 (vmap f x) ≤ l2 (fun i => x i * c) := l2_mono_abs _ _ (by
      intro i
      simpa [vmap, abs_mul, abs_of_nonneg hc, mul_comm] using hf (x i))
    _ = c * l2 x := by rw [l2_scale, abs_of_nonneg hc]

/-- Concrete, conservative matrix bound: the coefficient is its Frobenius norm. -/
theorem l2_linear_le_frobenius (w : ι → κ → ℝ) (x : ι → ℝ) :
    l2 (fun j => ∑ i, x i * w i j) ≤
      Real.sqrt (∑ j, ∑ i, w i j ^ 2) * l2 x := by
  have h : (∑ j, (∑ i, x i * w i j) ^ 2) ≤
      (∑ j, ∑ i, w i j ^ 2) * (∑ i, x i ^ 2) := by
    calc
      _ ≤ ∑ j, (∑ i, x i ^ 2) * (∑ i, w i j ^ 2) := by
        exact Finset.sum_le_sum (fun j _ => Finset.sum_mul_sq_le_sq_mul_sq Finset.univ x (fun i => w i j))
      _ = _ := by rw [← Finset.mul_sum]; ring
  calc
    _ ≤ Real.sqrt ((∑ j, ∑ i, w i j ^ 2) * (∑ i, x i ^ 2)) := Real.sqrt_le_sqrt h
    _ = _ := by rw [Real.sqrt_mul (by positivity)]; rfl

/-- Bounds compose independently of how each function is implemented. -/
theorem l2_bound_comp {ν : Type} [Fintype ν]
    (f : (ι → ℝ) → κ → ℝ) (g : (κ → ℝ) → ν → ℝ) (a b : ℝ)
    (hb : 0 ≤ b) (hf : ∀ x, l2 (f x) ≤ a * l2 x)
    (hg : ∀ y, l2 (g y) ≤ b * l2 y) (x : ι → ℝ) :
    l2 (g (f x)) ≤ (b * a) * l2 x := by
  calc
    _ ≤ b * l2 (f x) := hg _
    _ ≤ b * (a * l2 x) := mul_le_mul_of_nonneg_left (hf _) hb
    _ = _ := by ring

/-- Symmetric elementwise clipping, matching JAX's min/max lowering. -/
def clip (r v : ℝ) : ℝ := min r (max (-r) v)

theorem abs_clip_le_abs (r v : ℝ) (hr : 0 ≤ r) : |clip r v| ≤ |v| := by
  by_cases hlo : -r ≤ v
  · by_cases hhi : v ≤ r
    · simp [clip, max_eq_right hlo, min_eq_right hhi]
    · have hv : 0 ≤ v := le_trans hr (le_of_not_ge hhi)
      simp only [clip, max_eq_right hlo, min_eq_left (le_of_not_ge hhi),
        abs_of_nonneg hr, abs_of_nonneg hv]
      exact le_of_not_ge hhi
  · have hv : v ≤ 0 := le_trans (le_of_not_ge hlo) (neg_nonpos.mpr hr)
    simp only [clip, max_eq_left (le_of_not_ge hlo), min_eq_right (by linarith : -r ≤ r),
      abs_neg, abs_of_nonneg hr, abs_of_nonpos hv]
    linarith

theorem abs_clip_le (r v : ℝ) (hr : 0 ≤ r) : |clip r v| ≤ r := by
  apply abs_le.mpr
  constructor
  · exact le_min (by linarith) (le_max_left _ _)
  · exact min_le_left _ _

theorem l2_clip_le (r : ℝ) (hr : 0 ≤ r) (x : ι → ℝ) :
    l2 (vmap (clip r) x) ≤ l2 x :=
  l2_mono_abs _ _ (fun i => abs_clip_le_abs r (x i) hr)

/-- Elementwise clipping gives sqrt(dimension)*r, not an r-sized L2 ball. -/
theorem l2_clip_le_radius (r : ℝ) (hr : 0 ≤ r) (x : ι → ℝ) :
    l2 (vmap (clip r) x) ≤ r * Real.sqrt (Fintype.card ι) := by
  calc
    _ ≤ l2 (fun _ : ι => r) := l2_mono_abs _ _ (fun i => by
      simpa [vmap, abs_of_nonneg hr] using abs_clip_le r (x i) hr)
    _ = _ := by
      simp only [l2, Finset.sum_const, Finset.card_univ, nsmul_eq_mul]
      rw [Real.sqrt_mul (Nat.cast_nonneg _), Real.sqrt_sq hr]
      ring

/-- Radial clipping; a positive radius makes the denominator nonzero, even at x=0. -/
noncomputable def clipL2 (r : ℝ) (x : ι → ℝ) : ι → ℝ :=
  fun i => x i * (r / max r (l2 x))

theorem l2_clipL2_le (r : ℝ) (hr : 0 < r) (x : ι → ℝ) :
    l2 (clipL2 r x) ≤ min r (l2 x) := by
  have hd : 0 < max r (l2 x) := lt_of_lt_of_le hr (le_max_left _ _)
  have ha : 0 ≤ r / max r (l2 x) := div_nonneg hr.le hd.le
  change l2 (fun i => x i * (r / max r (l2 x))) ≤ _
  rw [l2_scale, abs_of_nonneg ha]
  apply le_min
  · calc
      _ ≤ (r / max r (l2 x)) * max r (l2 x) :=
        mul_le_mul_of_nonneg_left (le_max_right _ _) ha
      _ = r := div_mul_cancel₀ _ (ne_of_gt hd)
  · calc
      _ ≤ 1 * l2 x := mul_le_mul_of_nonneg_right
        ((div_le_one hd).mpr (le_max_left _ _)) (l2_nonneg x)
      _ = _ := one_mul _

end JaxLean.Batch
