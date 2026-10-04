import JaxLean.Tensor

/-! Tensor-facing instances of pure batch algebra. No probability assumptions. -/
namespace JaxLean.Tensor
open scoped BigOperators

/-- A linear scalar function commutes with a leading-axis reduction. -/
theorem sumFirst_map_linear [CommSemiring K] [AddCommMonoid R] [AddCommMonoid S]
    [Module K R] [Module K S] (L : R →ₗ[K] S) (x : Tensor R (n :: s)) :
    sumFirst (map L x) = map L (sumFirst x) := by
  funext j
  exact (Batch.linear_reduceSum L (fun i => x (i, j))).symm

/-- A linear per-example function can act on vectors or matrices, not just scalars. -/
theorem sumFirst_vmap_linear [CommSemiring K] [AddCommMonoid R] [AddCommMonoid S]
    [Module K R] [Module K S] (L : Tensor R s →ₗ[K] Tensor S t)
    (x : Tensor R (n :: s)) :
    sumFirst (vmap L x) = L (sumFirst x) := by
  have hx : sumFirst x = ∑ i : Fin n, fun j => x (i, j) := by
    funext j
    simp only [sumFirst, Batch.reduceSum, Finset.sum_apply]
  rw [hx]
  funext j
  change (∑ i : Fin n, L (fun a => x (i, a)) j) = _
  simpa only [Batch.reduceSum, Batch.vmap, Finset.sum_apply] using
    congrFun (Batch.linear_reduceSum L (fun i j => x (i, j))).symm j

/-- Fixed-right-argument bilinear maps are linear in the batch being reduced. -/
theorem sumFirst_map_bilinear_left [CommSemiring K]
    [AddCommMonoid R] [AddCommMonoid S] [AddCommMonoid T]
    [Module K R] [Module K S] [Module K T]
    (B : R →ₗ[K] S →ₗ[K] T) (x : Tensor R (n :: s)) (y : S) :
    sumFirst (map (fun a => B a y) x) = map (fun a => B a y) (sumFirst x) := by
  funext j
  exact (Batch.bilinear_reduceSum_left B (fun i => x (i, j)) y).symm

/-- Shared matrix multiplication commutes with reducing batch rows. -/
theorem sumFirst_matmul [Semiring R] (x : Tensor R [n, k]) (w : Tensor R [k, d]) :
    sumFirst (matmul x w) =
      fun j => ∑ a : Fin k, sumFirst x (a, ()) * w (a, j) := by
  funext j
  simp only [sumFirst, matmul, Batch.reduceSum, Finset.sum_mul]
  exact Finset.sum_comm

end JaxLean.Tensor
