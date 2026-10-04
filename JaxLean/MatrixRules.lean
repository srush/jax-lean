import JaxLean.SelectionRules
import Mathlib.Algebra.BigOperators.Field
import Mathlib.Tactic.Positivity

/-! Reindexing and normalization rules for arbitrary matrix sizes.
These rules know nothing about transformers or generated programs. -/
namespace JaxLean.Tensor
open scoped BigOperators

/-- Select rows and columns independently; selections may repeat or drop entries. -/
def reindexMatrix (rows : Fin a → Fin n) (cols : Fin b → Fin m)
    (x : Tensor R [n, m]) : Tensor R [a, b] :=
  fun i => x (rows i.1, cols i.2.1, ())

def transposeMatrix (x : Tensor R [n, m]) : Tensor R [m, n] :=
  fun i => x (i.2.1, i.1, ())

/-- Normalize each row using any scalar scoring function. -/
def normalizeRows [Field R] (score : R → R) (x : Tensor R [n, m]) :
    Tensor R [n, m] :=
  fun i => score (x i) / ∑ j, score (x (i.1, j, ()))

/-- Selecting query and key rows selects the corresponding entries of their product. -/
theorem matmul_transpose_select [Semiring R]
    (rows : Fin a → Fin n) (cols : Fin b → Fin m)
    (q : Tensor R [n, d]) (k : Tensor R [m, d]) :
    matmul (selectRows rows q) (transposeMatrix (selectRows cols k)) =
      reindexMatrix rows cols (matmul q (transposeMatrix k)) := rfl

/-- A contraction is unchanged by the same permutation of both contracted axes.
The free row axis can independently be selected, repeated, or dropped. -/
theorem matmul_reindex_contract [Semiring R]
    (rows : Fin a → Fin n) (perm : Fin m ≃ Fin m)
    (x : Tensor R [n, m]) (y : Tensor R [m, d]) :
    matmul (reindexMatrix rows perm x) (selectRows perm y) =
      selectRows rows (matmul x y) := by
  funext i
  exact Equiv.sum_comp perm (fun j => x (rows i.1, j, ()) * y (j, i.2))

/-- Row normalization permits arbitrary row selection but only a permutation
of columns: dropping keys would change the denominator. -/
theorem normalizeRows_reindex [Field R] (score : R → R)
    (rows : Fin a → Fin n) (perm : Fin m ≃ Fin m) (x : Tensor R [n, m]) :
    normalizeRows score (reindexMatrix rows perm x) =
      reindexMatrix rows perm (normalizeRows score x) := by
  funext i
  rcases i with ⟨row, col, ⟨⟩⟩
  dsimp [normalizeRows, reindexMatrix]
  rw [Equiv.sum_comp perm (fun j => score (x (rows row, j, ())))]

/-- A positive score function gives a strictly positive denominator on nonempty rows. -/
theorem normalization_total_pos (score : ℝ → ℝ) (positive : ∀ z, 0 < score z)
    (nonempty : 0 < m) (x : Tensor ℝ [n, m]) (row : Fin n) :
    0 < ∑ j, score (x (row, j, ())) := by
  apply Finset.sum_pos
  · intro j _
    exact positive _
  · exact ⟨⟨0, nonempty⟩, Finset.mem_univ _⟩

theorem normalizeRows_nonneg (score : ℝ → ℝ) (positive : ∀ z, 0 < score z)
    (nonempty : 0 < m) (x : Tensor ℝ [n, m]) (i : Index [n, m]) :
    0 ≤ normalizeRows score x i :=
  le_of_lt (div_pos (positive _) (normalization_total_pos score positive nonempty x i.1))

theorem normalizeRows_sum_one (score : ℝ → ℝ) (positive : ∀ z, 0 < score z)
    (nonempty : 0 < m) (x : Tensor ℝ [n, m]) (row : Fin n) :
    (∑ j, normalizeRows score x (row, j, ())) = 1 := by
  simp only [normalizeRows, ← Finset.sum_div]
  exact div_self (ne_of_gt (normalization_total_pos score positive nonempty x row))

end JaxLean.Tensor
