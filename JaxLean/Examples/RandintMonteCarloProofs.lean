import JaxLean.Stdlib
import JaxLean.Generated.DieEstimate
import JaxLean.Generated.GridEstimate
import JaxLean.Generated.MCDie16
import JaxLean.Generated.MCGridSquare8

/-! Two ordinary-JAX Monte Carlo examples, proved at estimator boundaries.
The uniform/iid laws below are the explicit ideal randint/split specification.
Deterministic estimator kernels have separate checked Jaxpr certificates.
No claim about bitwise PRNG uniformity, seed independence, or floating rounding. -/
namespace JaxLean.MCJax
open scoped BigOperators
noncomputable section

abbrev dieLaw : Rand ℝ := Rand.uniformInt 1 6 (by decide)
abbrev gridIndexLaw : Rand ℝ := Rand.uniformInt 0 4 (by decide)
def gridValueLaw : Rand ℝ := Rand.map (fun v => (v / 4) ^ (2 : Nat)) gridIndexLaw

/-- Contracts for the deterministic Python functions. -/
theorem die_estimate_spec (draws : Tensor ℝ [16]) :
    die_estimate draws () = (∑ i, draws (i, ())) / 16 := rfl

theorem grid_square_spec (indices : Tensor ℝ [8]) :
    grid_square indices = Tensor.map (fun v => (v / 4) ^ (2 : Nat)) indices := by
  funext i
  change (indices i / 4) ^ (2 : Int) = (indices i / 4) ^ (2 : Nat)
  simp only [zpow_ofNat]

theorem grid_estimate_spec (indices : Tensor ℝ [8]) :
    grid_estimate indices () = (∑ i, (indices (i, ()) / 4) ^ (2 : Nat)) / 8 := by
  simp only [grid_estimate, grid_square_spec, Tensor.sumFirst,     Tensor.map, Tensor.scalar]

/-- Bridges from the emitted random programs to their certified estimator functions. -/
theorem mc_die_boundary :
    mc_die_16 (R := ℝ) =
      Rand.map (fun draws => die_estimate (fun i => draws i.1) ()) (Rand.iid 16 dieLaw) := rfl

theorem mc_grid_boundary :
    mc_grid_square_8 (R := ℝ) =
      Rand.map (fun draws => grid_estimate (fun i => draws i.1) ())
        (Rand.iid 8 gridIndexLaw) := rfl

/-- Only the small, single-draw laws are enumerated; never the product sample space. -/
theorem die_mean : dieLaw.mean = 7 / 2 := by
  dsimp [dieLaw, Rand.mean, Rand.uniformInt, FiniteLaw.uniform,
    FiniteLaw.mean]
  norm_num [Fin.sum_univ_succ]

theorem die_variance : dieLaw.variance = 35 / 12 := by
  dsimp [dieLaw, Rand.variance, Rand.uniformInt, FiniteLaw.uniform,
    FiniteLaw.variance, FiniteLaw.mean]
  norm_num [Fin.sum_univ_succ]

theorem grid_value_mean : gridValueLaw.mean = 7 / 32 := by
  dsimp [gridValueLaw, gridIndexLaw, Rand.mean, Rand.map, Rand.uniformInt,
    FiniteLaw.uniform, FiniteLaw.mean]
  norm_num [Fin.sum_univ_succ]

theorem grid_value_variance : gridValueLaw.variance = 49 / 1024 := by
  dsimp [gridValueLaw, gridIndexLaw, Rand.variance, Rand.map, Rand.uniformInt,
    FiniteLaw.uniform, FiniteLaw.variance, FiniteLaw.mean]
  norm_num [Fin.sum_univ_succ]

/-- Sampling and averaging preserve the one-draw mean. -/
theorem mc_die_mean : (mc_die_16 (R := ℝ)).mean = 7 / 2 := by
  rw [mc_die_boundary]
  simp only [die_estimate_spec]
  rw [Rand.mean_map_div, Rand.mean_iid_sum (f := fun _ v => v)]
  change (∑ _ : Fin 16, dieLaw.mean) / 16 = _
  norm_num [die_mean]

/-- The 1/n law is obtained by propagating division and independent-sum rules. -/
theorem mc_die_variance_reduction :
    (mc_die_16 (R := ℝ)).variance = dieLaw.variance / 16 := by
  rw [mc_die_boundary]
  simp only [die_estimate_spec]
  rw [Rand.variance_map_div, Rand.variance_iid_sum (f := fun _ v => v)]
  change (∑ _ : Fin 16, dieLaw.variance) / 16 ^ 2 = _
  simp only [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
  ring

theorem mc_die_variance : (mc_die_16 (R := ℝ)).variance = 35 / 192 := by
  rw [mc_die_variance_reduction, die_variance]
  norm_num

/-- The same argument works after a nonlinear per-sample transformation. -/
theorem mc_grid_mean : (mc_grid_square_8 (R := ℝ)).mean = 7 / 32 := by
  rw [mc_grid_boundary]
  simp only [grid_estimate_spec]
  rw [Rand.mean_map_div, Rand.mean_iid_sum (f := fun _ v => (v / 4) ^ (2 : Nat))]
  change (∑ _ : Fin 8, gridValueLaw.mean) / 8 = _
  norm_num [grid_value_mean]

theorem mc_grid_variance_reduction :
    (mc_grid_square_8 (R := ℝ)).variance = gridValueLaw.variance / 8 := by
  rw [mc_grid_boundary]
  simp only [grid_estimate_spec]
  rw [Rand.variance_map_div, Rand.variance_iid_sum (f := fun _ v => (v / 4) ^ (2 : Nat))]
  change (∑ _ : Fin 8, gridValueLaw.variance) / 8 ^ 2 = _
  simp only [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
  ring

theorem mc_grid_variance : (mc_grid_square_8 (R := ℝ)).variance = 49 / 8192 := by
  rw [mc_grid_variance_reduction, grid_value_variance]
  norm_num

/-- The same numeric result for the independent evaluator of the estimator's Jaxpr.
Only the sampling law is supplied as a specification. -/
theorem certified_die_variance :
    (Rand.map (fun draws =>
      Jaxpr.Program.eval (.cons (fun i => draws i.1) .nil) die_estimate_ir ())
      (Rand.iid 16 dieLaw)).variance = 35 / 192 := by
  simp only [die_estimate_translation_correct]
  rw [← mc_die_boundary]
  exact mc_die_variance

theorem certified_grid_variance :
    (Rand.map (fun draws =>
      Jaxpr.Program.eval (.cons (fun i => draws i.1) .nil) grid_estimate_ir ())
      (Rand.iid 8 gridIndexLaw)).variance = 49 / 8192 := by
  simp only [grid_estimate_translation_correct]
  rw [← mc_grid_boundary]
  exact mc_grid_variance

end
end JaxLean.MCJax
