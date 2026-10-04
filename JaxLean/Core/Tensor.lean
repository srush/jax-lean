import JaxLean.Core.Index
import Mathlib.Algebra.Ring.Basic

/-! The entire array model: a tensor is a function on bounded coordinates.
There is no default element, unchecked indexing, mutable store, or runtime shape check.
Scalars have shape `[]` and are evaluated at `()`. -/
namespace JaxLean
open scoped BigOperators

abbrev Tensor (R : Type) (shape : List Nat) := Index shape → R

namespace Tensor

def scatterSet (out : Tensor R s) (position : Index s) (value : R) : Tensor R s :=
  fun i => if i = position then value else out i

def scatterAdd [Add R] (out : Tensor R s) (position : Index s) (value : R) : Tensor R s :=
  fun i => if i = position then out i + value else out i

def scalar (x : R) : Tensor R [] := fun _ => x

def map (f : R → S) (x : Tensor R s) : Tensor S s := fun i => f (x i)

def map₂ (f : R → S → T) (x : Tensor R s) (y : Tensor S s) : Tensor T s :=
  fun i => f (x i) (y i)

/-- Reduction of the leading batch axis; trailing coordinates are retained. -/
def sumFirst [AddCommMonoid R] (x : Tensor R (n :: s)) : Tensor R s :=
  fun j => ∑ i, x (i, j)

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

/-- Transpose, broadcast, slice and squeeze are just coordinate maps. -/
def reindex (f : Index t → Index s) (x : Tensor R s) : Tensor R t := fun i => x (f i)

def reshape (h : s.prod = t.prod) (x : Tensor R s) : Tensor R t :=
  reindex (fun i => (Index.equivFin s).symm (Fin.cast h.symm (Index.equivFin t i))) x

def concatenate (split : Index t → Sum (Index s) (Index u)) (x : Tensor R s) (y : Tensor R u) : Tensor R t :=
  fun i => match split i with
    | .inl j => x j
    | .inr j => y j

/-- A shape-generic contraction with explicit, bounded coordinate maps. -/
def contract [Semiring R] (left : Index t → Index k → Index s)
    (right : Index t → Index k → Index u) (x : Tensor R s) (y : Tensor R u) : Tensor R t :=
  fun i => ∑ j, x (left i j) * y (right i j)

def reduceSum [AddCommMonoid R] (map : Index t → Index k → Index s) (x : Tensor R s) : Tensor R t :=
  fun i => ∑ j, x (map i j)

/-- Nonempty finite extrema; no infinity is introduced into the real model. -/
def reduceMax [LinearOrder R] (positive : 0 < n) (map : Index t → Fin n → Index s)
    (x : Tensor R s) : Tensor R t := fun i =>
  (List.ofFn (fun j => x (map i j))).foldl max (x (map i ⟨0, positive⟩))

def reduceMin [LinearOrder R] (positive : 0 < n) (map : Index t → Fin n → Index s)
    (x : Tensor R s) : Tensor R t := fun i =>
  (List.ofFn (fun j => x (map i j))).foldl min (x (map i ⟨0, positive⟩))

def ofFlat (xs : Fin s.prod → R) : Tensor R s := fun i => xs (Index.equivFin s i)

/-- Array literals avoid the elaboration cost of long nested `![...]` vectors. -/
def ofArray (xs : Array R) (h : xs.size = s.prod) : Tensor R s :=
  fun i => xs[(Index.equivFin s i).val]'(by rw [h]; exact (Index.equivFin s i).isLt)

def toList (x : Tensor R s) : List R :=
  (List.finRange s.prod).map (fun i => x ((Index.equivFin s).symm i))

def stack (xs : Array (Tensor R s)) (h : xs.size = n) : Tensor R (n :: s) :=
  fun i => (xs[i.1.val]'(by rw [h]; exact i.1.isLt)) i.2

end Tensor
end JaxLean
