import JaxLean.Core.Tensor

namespace JaxLean.Tensor

@[simp] theorem scatterSet_same (out : Tensor R s) (p : Index s) (v : R) :
    scatterSet out p v p = v := by simp [scatterSet]

@[simp] theorem scatterSet_other (out : Tensor R s) (p q : Index s) (v : R) (h : q ≠ p) :
    scatterSet out p v q = out q := by simp [scatterSet, h]

@[simp] theorem scatterAdd_same [Add R] (out : Tensor R s) (p : Index s) (v : R) :
    scatterAdd out p v p = out p + v := by simp [scatterAdd]

@[simp] theorem scatterAdd_other [Add R] (out : Tensor R s) (p q : Index s) (v : R) (h : q ≠ p) :
    scatterAdd out p v q = out q := by simp [scatterAdd, h]

/-- Coordinate form for vector updates; keeps Index abstract during simplification. -/
@[simp] theorem scatterSet_vector (out : Tensor R [n]) (p i : Fin n) (v : R) :
    scatterSet out (p, ()) v (i, ()) = if i = p then v else out (i, ()) := by
  by_cases h : i = p
  · subst i
    simpa using scatterSet_same out (p, ()) v
  · have h' : (i, ()) ≠ (p, ()) := fun e => h (congrArg Prod.fst e)
    rw [scatterSet_other out (p, ()) (i, ()) v h', if_neg h]

@[simp] theorem scatterSet_matrix (out : Tensor R [n, m])
    (p i : Fin n) (q j : Fin m) (v : R) :
    scatterSet out (p, q, ()) v (i, j, ()) =
      if i = p ∧ j = q then v else out (i, j, ()) := by
  by_cases h : i = p ∧ j = q
  · rcases h with ⟨rfl, rfl⟩
    simpa using scatterSet_same out (i, j, ()) v
  · have h' : (i, j, ()) ≠ (p, q, ()) := fun e =>
      h ⟨congrArg Prod.fst e, congrArg (fun x => x.2.1) e⟩
    rw [scatterSet_other out (p, q, ()) (i, j, ()) v h', if_neg h]

/-- Additive collisions are order-independent in this algebraic model. -/
theorem scatterAdd_comm [AddCommMonoid R] (out : Tensor R s) (p q : Index s) (a b : R) :
    scatterAdd (scatterAdd out p a) q b = scatterAdd (scatterAdd out q b) p a := by
  funext i
  simp only [scatterAdd]
  split_ifs <;> simp_all [add_right_comm]

end JaxLean.Tensor
