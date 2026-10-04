import JaxLean.Stdlib.Random

/-!
Independent repeated sampling is defined by a product law, not by an assumption
about arbitrary inputs. The proof is by induction on the number of draws; it
never enumerates the exponentially growing collection of possible worlds.
-/
namespace JaxLean.Rand
open scoped BigOperators

/-- n independent copies of any finite random computation. -/
noncomputable def iid (n : Nat) (x : Rand α) : Rand (Fin n → α) :=
  match n with
  | 0 => ⟨Unit, inferInstance,
      { mass := fun _ => 1, nonneg := by simp, total := by simp },
      fun _ => Fin.elim0⟩
  | n + 1 =>
      let rest := iid n x
      ⟨x.Ω × rest.Ω, inferInstance, x.law.prod rest.law,
        fun ω => Fin.cons (x.value ω.1) (rest.value ω.2)⟩

/-- Expectation propagates through a coordinatewise sum; each coordinate may use a different function. -/
theorem mean_iid_sum (n : Nat) (x : Rand α) (f : Fin n → α → ℝ) :
    (map (fun xs => ∑ i, f i (xs i)) (iid n x)).mean = ∑ i, (map (f i) x).mean := by
  induction n with
  | zero => simp [iid, map, mean, FiniteLaw.mean]
  | succ n ih =>
    change (x.law.prod (iid n x).law).mean
      (fun ω : x.Ω × (iid n x).Ω => ∑ i : Fin (n + 1),
        f i (Fin.cons (α := fun _ => α) (x.value ω.1) ((iid n x).value ω.2) i)) = _
    simp only [Fin.sum_univ_succ, Fin.cons_zero, Fin.cons_succ]
    rw [FiniteLaw.mean_add,
      FiniteLaw.mean_prod_left x.law (iid n x).law (fun ω => f 0 (x.value ω)),
      FiniteLaw.mean_prod_right x.law (iid n x).law
        (fun ω => ∑ i, f i.succ ((iid n x).value ω i))]
    change (map (f 0) x).mean + (map (fun xs => ∑ i, f i.succ (xs i)) (iid n x)).mean = _
    rw [ih]

/-- Variances add because iid builds a product distribution. -/
theorem variance_iid_sum (n : Nat) (x : Rand α) (f : Fin n → α → ℝ) :
    (map (fun xs => ∑ i, f i (xs i)) (iid n x)).variance =
      ∑ i, (map (f i) x).variance := by
  induction n with
  | zero => simp [iid, map, variance, FiniteLaw.variance, FiniteLaw.mean]
  | succ n ih =>
    change (x.law.prod (iid n x).law).variance
      (fun ω : x.Ω × (iid n x).Ω => ∑ i : Fin (n + 1),
        f i (Fin.cons (α := fun _ => α) (x.value ω.1) ((iid n x).value ω.2) i)) = _
    simp only [Fin.sum_univ_succ, Fin.cons_zero, Fin.cons_succ]
    rw [FiniteLaw.variance_independent_add x.law (iid n x).law
      (fun ω => f 0 (x.value ω)) (fun ω => ∑ i, f i.succ ((iid n x).value ω i))]
    change (map (f 0) x).variance + (map (fun xs => ∑ i, f i.succ (xs i)) (iid n x)).variance = _
    rw [ih]


end JaxLean.Rand
