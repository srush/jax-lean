import JaxLean.Core.Indexing

namespace JaxLean.Tensor
@[simp] theorem clipStart_of_inBounds (upper : Nat) (x : Int32)
    (_lo : 0 ≤ x.toInt) (hi : x.toInt.toNat ≤ upper) :
    (clipStart upper x).val = x.toInt.toNat := by
  exact Nat.min_eq_left hi

/-- Selection commutes with any fixed coordinate map. -/
theorem reindex_select (map : Index t → Index s) (c : Tensor Bool s) (x y : Tensor R s) :
    reindex map (fun i => if c i then x i else y i) =
      fun i => if reindex map c i then reindex map x i else reindex map y i := rfl
end JaxLean.Tensor
