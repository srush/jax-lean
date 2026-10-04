import JaxLean.Core.Tensor

/-! Reusable tensor and index algebra; no Jaxpr or generated-program imports. -/
namespace JaxLean
open scoped BigOperators

/-- A vector's row-major flat coordinate is just its only coordinate. -/
@[simp] theorem Index.equivFin_vector_val (i : Fin n) :
    (Index.equivFin [n] (i, ())).val = i.val := by
  change 0 + 1 * i.val = i.val
  simp

theorem Index.sum_cons [AddCommMonoid R] (f : Index (n :: s) → R) :
    (∑ i, f i) = ∑ j : Fin n, ∑ k : Index s, f (j, k) :=
  Fintype.sum_prod_type f

theorem Index.sum_nil [AddCommMonoid R] (f : Index [] → R) :
    (∑ i, f i) = f () := by
  change (∑ i : Unit, f i) = f ()
  exact Fintype.sum_unique (ι := Unit) f

namespace Tensor

@[simp] theorem map_apply (f : R → S) (x : Tensor R s) (i : Index s) :
    map f x i = f (x i) := rfl

@[simp] theorem map₂_apply (f : R → S → T) (x : Tensor R s) (y : Tensor S s)
    (i : Index s) : map₂ f x y i = f (x i) (y i) := rfl

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
