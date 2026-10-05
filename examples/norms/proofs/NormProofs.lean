import JaxLean.Stdlib
import examples.norms.generated.VectorNorm
import examples.norms.generated.LinearClip
import examples.norms.generated.RadialClip
import examples.norms.generated.BatchRadialClip

/-! Application proofs about generated JAX definitions. No new norm laws here. -/
namespace JaxLean.NormProofs
open Generated

/-- JAX's vector norm lowers to the library's explicit Euclidean norm. -/
theorem vector_norm_spec (x : Tensor ℝ [3]) :
    vector_norm x () = Tensor.vectorNorm x := by
  simp only [vector_norm, Tensor.vectorNorm, Batch.l2, Tensor.map, Tensor.map₂,
    Tensor.sumFirst, RealOps.sqrt, pow_two]

/-- Compose a matrix bound with componentwise clipping's nonexpansion. -/
theorem linear_clip_bound (x : Tensor ℝ [3]) (w : Tensor ℝ [3, 2])
    (r : ℝ) (hr : 0 ≤ r) :
    Tensor.vectorNorm (linear_clip x w (Tensor.scalar r)) ≤
      Tensor.frobeniusNorm w * Tensor.vectorNorm x := by
  change Tensor.vectorNorm (Tensor.map (Batch.clip r) (Tensor.vecmat x w)) ≤ _
  exact (Tensor.vectorNorm_clip_le r hr _).trans (Tensor.vectorNorm_vecmat_le x w)

theorem linear_clip_radius_bound (x : Tensor ℝ [3]) (w : Tensor ℝ [3, 2])
    (r : ℝ) (hr : 0 ≤ r) :
    Tensor.vectorNorm (linear_clip x w (Tensor.scalar r)) ≤ r * Real.sqrt 2 := by
  exact Tensor.vectorNorm_clip_le_radius r hr (Tensor.vecmat x w)

/-- The actual radial-clipping program stays inside the L2 ball and never grows the input norm. -/
theorem radial_clip_bound (x : Tensor ℝ [3]) (r : ℝ) (hr : 0 < r) :
    Tensor.vectorNorm (radial_clip x (Tensor.scalar r)) ≤ min r (Tensor.vectorNorm x) := by
  simpa only [radial_clip, Tensor.vectorNorm, Tensor.map, Tensor.map₂, Tensor.scalar,
    Tensor.sumFirst, RealOps.sqrt, Batch.clipL2, Batch.l2, pow_two] using
    Batch.l2_clipL2_le r hr (fun i : Fin 3 => x (i, ()))

/-- Each row of a vmapped program inherits the same bound, without probability. -/
theorem batch_radial_clip_bound (x : Tensor ℝ [4, 3]) (r : ℝ) (hr : 0 < r) (row : Fin 4) :
    Batch.l2 (fun i : Fin 3 => batch_radial_clip x (Tensor.scalar r) (row, i, ())) ≤
      min r (Batch.l2 (fun i : Fin 3 => x (row, i, ()))) := by
  simpa only [batch_radial_clip, Tensor.map, Tensor.map₂, Tensor.scalar, Tensor.reindex,
    RealOps.sqrt, Batch.clipL2, Batch.l2, pow_two] using
    Batch.l2_clipL2_le r hr (fun i : Fin 3 => x (row, i, ()))

end JaxLean.NormProofs
