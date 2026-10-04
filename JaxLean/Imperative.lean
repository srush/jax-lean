import JaxLean.Tensor
import Mathlib.Tactic

namespace JaxLean

instance (s : List Nat) : DecidableEq (Index s) := by
  induction s with
  | nil => exact inferInstanceAs (DecidableEq Unit)
  | cons n ns ih => exact inferInstanceAs (DecidableEq (Fin n × Index ns))

namespace Tensor

def set [DecidableEq (Index s)] (out : Tensor R s) (i : Index s) (value : R) : Tensor R s :=
  Function.update out i value

instance : GetElem (Tensor R [n]) (Fin n) R (fun _ _ => True) where
  getElem a i _ := a (i, ())

instance : GetElem (Tensor R [n, m]) (Fin n × Fin m) R (fun _ _ => True) where
  getElem a i _ := a (i.1, i.2, ())

@[simp] theorem getElem_vector (a : Tensor R [n]) (i : Fin n) : a[i] = a (i, ()) := rfl

@[simp] theorem getElem_matrix (a : Tensor R [n, m]) (i : Fin n × Fin m) :
    a[i] = a (i.1, i.2, ()) := rfl

end Tensor

namespace Imperative
variable {α β R S : Type}

abbrev range (n : Nat) := List.finRange n

theorem foldl_update [DecidableEq α] (xs : List α) (f initial : α → R) (k : α) :
    (xs.foldl (fun out i => Function.update out i (f i)) initial) k =
      if k ∈ xs then f k else initial k := by
  induction xs generalizing initial with
  | nil => simp
  | cons i xs ih =>
    simp only [List.foldl_cons, ih, List.mem_cons]
    by_cases h : k ∈ xs <;> by_cases e : k = i <;> simp_all

theorem foldl_write (xs : List α) (index : α → β) [DecidableEq β]
    (f initial : β → R) (k : β) :
    (xs.foldl (fun out i => Function.update out (index i) (f (index i))) initial) k =
      if k ∈ xs.map index then f k else initial k := by
  simpa only [List.foldl_map] using foldl_update (xs.map index) f initial k

theorem foldl_nested (xs : List α) (ys : List β) (f : S → α × β → S) (initial : S) :
    xs.foldl (fun out i => ys.foldl (fun out j => f out (i, j)) out) initial =
      (xs.product ys).foldl f initial := by
  induction xs generalizing initial with
  | nil => rfl
  | cons i xs ih => simpa only [List.product, List.flatMap_cons, List.foldl_cons, List.foldl_append, List.foldl_map] using ih (ys.foldl (fun out j => f out (i, j)) initial)

end Imperative
end JaxLean
