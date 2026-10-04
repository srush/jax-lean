import JaxLean.Stdlib.Equivariance
import JaxLean.Generated.NoetherAdvect
import JaxLean.Generated.NoetherBurgers

namespace JaxLean.NoetherProofs
open Tensor NoetherJax
open scoped BigOperators

/-- The four-cell periodic neighbor map: 0 ↦ 3, 1 ↦ 0, 2 ↦ 1, 3 ↦ 2. -/
def neighbor := cyclicShift 3 1

/-- Identify the actual slice/concatenate implementation at the roll boundary. -/
theorem advect_roll_spec (u : Tensor ℝ [4]) :
    Advect.fn__roll_static u = permute neighbor u := by
  funext ⟨i, v⟩
  cases v
  fin_cases i <;> rfl

theorem burgers_roll_spec (u : Tensor ℝ [4]) :
    Burgers.fn__roll_static u = permute neighbor u := by
  exact advect_roll_spec u

/-- Readable specification of the original Python function. -/
theorem advect_spec (u : Tensor ℝ [4]) (c : Tensor ℝ []) :
    Advect.advect u c = fluxStep neighbor (fun z => c () * z) u := by
  simp only [Advect.advect, advect_roll_spec]
  rfl

theorem burgers_spec (u : Tensor ℝ [4]) (lam : Tensor ℝ []) :
    Burgers.burgers u lam = fluxStep neighbor (fun z => lam () * (1 / 2 * z ^ 2)) u := by
  simp only [Burgers.burgers, burgers_roll_spec]
  funext i
  simp only [fluxStep, permute, Tensor.map, Tensor.map₂, Tensor.scalar, zpow_ofNat]
  ring

/-- Every cyclic shift, not just the one-cell shift. -/
theorem advect_equivariant (u : Tensor ℝ [4]) (c : Tensor ℝ []) (k : ℤ) :
    Advect.advect (permute (cyclicShift 3 k) u) c =
      permute (cyclicShift 3 k) (Advect.advect u c) := by
  simp only [advect_spec]
  exact fluxStep_equivariant _ _ (cyclicShift_commute 3 1 k) _ _

theorem burgers_equivariant (u : Tensor ℝ [4]) (lam : Tensor ℝ []) (k : ℤ) :
    Burgers.burgers (permute (cyclicShift 3 k) u) lam =
      permute (cyclicShift 3 k) (Burgers.burgers u lam) := by
  simp only [burgers_spec]
  exact fluxStep_equivariant _ _ (cyclicShift_commute 3 1 k) _ _

theorem advect_conserves_sum (u : Tensor ℝ [4]) (c : Tensor ℝ []) :
    (∑ i : Fin 4, Advect.advect u c (i, ())) = ∑ i : Fin 4, u (i, ()) := by
  rw [advect_spec]
  exact fluxStep_sum _ _ _

theorem burgers_conserves_sum (u : Tensor ℝ [4]) (lam : Tensor ℝ []) :
    (∑ i : Fin 4, Burgers.burgers u lam (i, ())) = ∑ i : Fin 4, u (i, ()) := by
  rw [burgers_spec]
  exact fluxStep_sum _ _ _

/-- Equivariance attached directly to the imported Jaxpr semantics. -/
theorem advect_certificate (u : Tensor ℝ [4]) (c : Tensor ℝ []) (k : ℤ) :
    Jaxpr.Program.eval (.cons (permute (cyclicShift 3 k) u) (.cons c .nil))
        Advect.advect_ir =
      permute (cyclicShift 3 k)
        (Jaxpr.Program.eval (.cons u (.cons c .nil)) Advect.advect_ir) := by
  simp only [Advect.advect_translation_correct]
  exact advect_equivariant u c k

theorem burgers_certificate (u : Tensor ℝ [4]) (lam : Tensor ℝ []) (k : ℤ) :
    Jaxpr.Program.eval (.cons (permute (cyclicShift 3 k) u) (.cons lam .nil))
        Burgers.burgers_ir =
      permute (cyclicShift 3 k)
        (Jaxpr.Program.eval (.cons u (.cons lam .nil)) Burgers.burgers_ir) := by
  simp only [Burgers.burgers_translation_correct]
  exact burgers_equivariant u lam k

end JaxLean.NoetherProofs
