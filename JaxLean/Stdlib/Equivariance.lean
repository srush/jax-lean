import JaxLean.Core.Tensor
import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Data.ZMod.Basic
import Mathlib.Tactic

namespace JaxLean.Tensor
open scoped BigOperators

/-- Pull back a vector along a permutation of its coordinates. -/
def permute {R : Type} {n : ℕ} (p : Equiv.Perm (Fin n))
    (x : Tensor R [n]) : Tensor R [n] := fun i => x (p i.1, ())

/-- Pointwise unary operations commute with every coordinate permutation. -/
theorem permute_map {R S : Type} {n : ℕ} (p : Equiv.Perm (Fin n))
    (f : R → S) (x : Tensor R [n]) :
    permute p (map f x) = map f (permute p x) := rfl

/-- Both operands of a pointwise binary operation transform together. -/
theorem permute_map₂ {R S T : Type} {n : ℕ} (p : Equiv.Perm (Fin n))
    (f : R → S → T) (x : Tensor R [n]) (y : Tensor S [n]) :
    permute p (map₂ f x y) = map₂ f (permute p x) (permute p y) := rfl

theorem permute_sum {R : Type} [AddCommMonoid R] {n : ℕ}
    (p : Equiv.Perm (Fin n)) (x : Tensor R [n]) :
    (∑ i : Fin n, permute p x (i, ())) = ∑ i : Fin n, x (i, ()) :=
  Equiv.sum_comp p (fun i => x (i, ()))

/-- Positive shifts follow `jnp.roll`: output i reads input i - k. -/
def cyclicShift (n : ℕ) (k : ℤ) : Equiv.Perm (Fin (n + 1)) :=
  Equiv.addRight (-(k : ZMod (n + 1)))

theorem cyclicShift_commute (n : ℕ) (a b : ℤ) (i : Fin (n + 1)) :
    cyclicShift n a (cyclicShift n b i) = cyclicShift n b (cyclicShift n a i) := by
  exact add_right_comm (G := ZMod (n + 1)) i (-(b : ZMod (n + 1))) (-(a : ZMod (n + 1)))

/-- A local flux difference along any permutation, with any pointwise flux. -/
def fluxStep {R : Type} [Ring R] {n : ℕ} (p : Equiv.Perm (Fin n))
    (flux : R → R) (x : Tensor R [n]) : Tensor R [n] :=
  fun i => x i - (flux (x i) - flux (permute p x i))

/-- A stencil commutes with every permutation that commutes with its neighbor map. -/
theorem fluxStep_equivariant {R : Type} [Ring R] {n : ℕ}
    (p q : Equiv.Perm (Fin n)) (commute : ∀ i, p (q i) = q (p i))
    (flux : R → R) (x : Tensor R [n]) :
    fluxStep p flux (permute q x) = permute q (fluxStep p flux x) := by
  funext ⟨i, u⟩
  cases u
  simp only [fluxStep, permute, commute]

/-- Cancellation of a permuted flux proves conservation without linearity. -/
theorem fluxStep_sum {R : Type} [Ring R] {n : ℕ}
    (p : Equiv.Perm (Fin n)) (flux : R → R) (x : Tensor R [n]) :
    (∑ i : Fin n, fluxStep p flux x (i, ())) = ∑ i : Fin n, x (i, ()) := by
  simp only [fluxStep, permute, Finset.sum_sub_distrib]
  rw [Equiv.sum_comp p (fun i => flux (x (i, ())))]
  simp

end JaxLean.Tensor
