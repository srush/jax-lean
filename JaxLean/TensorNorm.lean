import JaxLean.NormRules
import JaxLean.Tensor

namespace JaxLean.Tensor
open scoped BigOperators

/-- Explicit L2 norm; not the default function-space (supremum) norm. -/
noncomputable def vectorNorm (x : Tensor ℝ [n]) : ℝ := Batch.l2 (fun i => x (i, ()))
noncomputable def frobeniusNorm (w : Tensor ℝ [n, m]) : ℝ :=
  Real.sqrt (∑ j : Fin m, ∑ i : Fin n, w (i, j, ()) ^ 2)

theorem vectorNorm_vecmat_le (x : Tensor ℝ [n]) (w : Tensor ℝ [n, m]) :
    vectorNorm (vecmat x w) ≤ frobeniusNorm w * vectorNorm x :=
  Batch.l2_linear_le_frobenius (fun i j => w (i, j, ())) (fun i => x (i, ()))

theorem vectorNorm_clip_le (r : ℝ) (hr : 0 ≤ r) (x : Tensor ℝ [n]) :
    vectorNorm (map (Batch.clip r) x) ≤ vectorNorm x :=
  Batch.l2_clip_le r hr (fun i => x (i, ()))

theorem vectorNorm_clip_le_radius (r : ℝ) (hr : 0 ≤ r) (x : Tensor ℝ [n]) :
    vectorNorm (map (Batch.clip r) x) ≤ r * Real.sqrt n := by
  have h := Batch.l2_clip_le_radius (ι := Fin n) r hr (fun i => x (i, ()))
  change vectorNorm (map (Batch.clip r) x) ≤ r * Real.sqrt (Fintype.card (Fin n)) at h
  rw [Fintype.card_fin] at h
  exact h

end JaxLean.Tensor
