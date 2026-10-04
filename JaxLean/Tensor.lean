import Mathlib.Data.Real.Basic
import Mathlib.Data.Fintype.BigOperators
import Mathlib.Logic.Equiv.Fin.Basic

/-! The entire array model: a tensor is a function on bounded coordinates.
There is no default element, unchecked indexing, mutable store, or runtime shape check.
Scalars have shape `[]` and are evaluated at `()`. -/
namespace JaxLean

def Index : List Nat → Type
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

abbrev Tensor (R : Type) (shape : List Nat) := Index shape → R

namespace Tensor

def scalar (x : R) : Tensor R [] := fun _ => x

def map (f : R → S) (x : Tensor R s) : Tensor S s := fun i => f (x i)

def map₂ (f : R → S → T) (x : Tensor R s) (y : Tensor S s) : Tensor T s :=
  fun i => f (x i) (y i)

/-- Transpose, broadcast, slice and squeeze are just coordinate maps. -/
def reindex (f : Index t → Index s) (x : Tensor R s) : Tensor R t := fun i => x (f i)

def reshape (h : s.prod = t.prod) (x : Tensor R s) : Tensor R t :=
  reindex (fun i => (Index.equivFin s).symm (Fin.cast h.symm (Index.equivFin t i))) x

def ofFlat (xs : Fin s.prod → R) : Tensor R s := fun i => xs (Index.equivFin s i)

/-- Array literals avoid the elaboration cost of long nested `![...]` vectors. -/
def ofArray (xs : Array R) (h : xs.size = s.prod) : Tensor R s :=
  fun i => xs[(Index.equivFin s i).val]'(by rw [h]; exact (Index.equivFin s i).isLt)

def toList (x : Tensor R s) : List R :=
  (List.finRange s.prod).map (fun i => x ((Index.equivFin s).symm i))

def stack (xs : Array (Tensor R s)) (h : xs.size = n) : Tensor R (n :: s) :=
  fun i => (xs[i.1.val]'(by rw [h]; exact i.1.isLt)) i.2

@[simp] theorem reindex_apply (f : Index t → Index s) (x : Tensor R s) (i : Index t) :
    reindex f x i = x (f i) := rfl

theorem reindex_reindex (f : Index t → Index s) (g : Index u → Index t) (x : Tensor R s) :
    reindex g (reindex f x) = reindex (f ∘ g) x := rfl

theorem reshape_roundtrip {s t : List Nat} (h : s.prod = t.prod) (x : Tensor R s) :
    reshape h.symm (reshape h x) = x := by
  funext i
  simp [reshape, reindex]

end Tensor
end JaxLean
