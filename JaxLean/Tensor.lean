import JaxLean.Batch
import Mathlib.Data.Real.Basic
import Mathlib.Data.Fintype.BigOperators
import Mathlib.Logic.Equiv.Fin.Basic

/-! The entire array model: a tensor is a function on bounded coordinates.
There is no default element, unchecked indexing, mutable store, or runtime shape check.
Scalars have shape `[]` and are evaluated at `()`. -/
namespace JaxLean
open scoped BigOperators

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

/-- A vector's row-major flat coordinate is just its only coordinate. -/
@[simp] theorem Index.equivFin_vector_val (i : Fin n) :
    (Index.equivFin [n] (i, ())).val = i.val := by
  change 0 + 1 * i.val = i.val
  simp

abbrev Tensor (R : Type) (shape : List Nat) := Index shape → R

namespace Tensor

def scalar (x : R) : Tensor R [] := fun _ => x

def map (f : R → S) (x : Tensor R s) : Tensor S s := fun i => f (x i)

def map₂ (f : R → S → T) (x : Tensor R s) (y : Tensor S s) : Tensor T s :=
  fun i => f (x i) (y i)

/-- Reduction of the leading batch axis; trailing coordinates are retained. -/
def sumFirst [AddCommMonoid R] (x : Tensor R (n :: s)) : Tensor R s :=
  fun j => Batch.reduceSum (fun i => x (i, j))

/-- Map an arbitrary per-example tensor function over the leading axis. -/
def vmap (f : Tensor R s → Tensor S t) (x : Tensor R (n :: s)) : Tensor S (n :: t) :=
  fun i => f (fun j => x (i.1, j)) i.2

/-- Ordinary rank-two matrix multiplication. Keep this operation named in
transpiled code so general theorems can recognize it. -/
def matmul [Semiring R] (x : Tensor R [n, k]) (w : Tensor R [k, d]) :
    Tensor R [n, d] :=
  fun i => ∑ j : Fin k, x (i.1, j, ()) * w (j, i.2)

/-- Row-vector times matrix, preserving the linear transformation in generated code. -/
def vecmat [Semiring R] (x : Tensor R [n]) (w : Tensor R [n, m]) : Tensor R [m] :=
  fun j => ∑ i : Fin n, x (i, ()) * w (i, j)

/-- Expand a singleton first axis; used by elementwise bias broadcasting. -/
def broadcastFirst (n : Nat) (x : Tensor R (1 :: s)) : Tensor R (n :: s) :=
  fun i => x (0, i.2)

@[simp] theorem map_apply (f : R → S) (x : Tensor R s) (i : Index s) :
    map f x i = f (x i) := rfl

@[simp] theorem map₂_apply (f : R → S → T) (x : Tensor R s) (y : Tensor S s)
    (i : Index s) : map₂ f x y i = f (x i) (y i) := rfl

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
