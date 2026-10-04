import Mathlib.Data.Fintype.BigOperators
import Mathlib.Logic.Equiv.Fin.Basic

/-! Bounded coordinates, their row-major equivalence, and structural instances. -/
namespace JaxLean
/-- Unfold for implicit argument checking, while keeping typeclass lookup shape-directed. -/
@[implicit_reducible] def Index : List Nat → Type
  | [] => Unit
  | n :: ns => Fin n × Index ns

instance (s : List Nat) : Fintype (Index s) := by
  induction s with
  | nil => exact inferInstanceAs (Fintype Unit)
  | cons n ns ih => exact inferInstanceAs (Fintype (Fin n × Index ns))

/-- Row-major indexing, including empty dimensions. -/
def Index.equivFin : (s : List Nat) → Index s ≃ Fin s.prod
  | [] => {
      toFun := fun _ => ⟨0, by simp⟩
      invFun := fun _ => ()
      left_inv := fun _ => rfl
      right_inv := fun i => by
        apply Fin.ext
        have := i.isLt
        simp_all }
  | n :: ns => (Equiv.prodCongr (Equiv.refl (Fin n)) (equivFin ns)).trans finProdFinEquiv

instance (s : List Nat) : DecidableEq (Index s) := by
  induction s with
  | nil => exact inferInstanceAs (DecidableEq Unit)
  | cons n ns ih => exact inferInstanceAs (DecidableEq (Fin n × Index ns))

end JaxLean
