import JaxLean.Core.Tensor

/-!
# Reusable selection rules

This file does not import any generated program. Its theorems apply to all
batch sizes, all feature shapes, and arbitrary scalar functions.

The simp rules push selection INTO an expression. After unfolding a transpiled
function, `simp` uses them to compare selecting its input with its output.
The operation definitions themselves do not need to be unfolded in that proof.
-/
namespace JaxLean.Tensor

/-- Reorder, drop, or duplicate the first-axis entries. The remaining axes
may describe a vector, a matrix, or any other per-example tensor. -/
def selectRows (selection : Fin m → Fin n) (x : Tensor R (n :: s)) :
    Tensor R (m :: s) :=
  fun i => x (selection i.1, i.2)

@[simp] theorem broadcastFirst_one (x : Tensor R (1 :: s)) :
    broadcastFirst 1 x = x := by
  funext i
  rcases i with ⟨row, rest⟩
  change x (0, rest) = x (row, rest)
  rw [Subsingleton.elim row (0 : Fin 1)]

-- First push selection through named operations; normalize singleton inputs last.
@[simp 100] theorem selectRows_singleton (selection : Fin m → Fin 1)
    (x : Tensor R (1 :: s)) :
    selectRows selection x = broadcastFirst m x := by
  funext i
  change x (selection i.1, i.2) = x (0, i.2)
  rw [Subsingleton.elim (selection i.1) (0 : Fin 1)]

/-- Any scalar function commutes with row selection: ReLU, square, exp, etc. -/
@[simp] theorem selectRows_map (selection : Fin m → Fin n)
    (f : R → S) (x : Tensor R (n :: s)) :
    selectRows selection (map f x) = map f (selectRows selection x) := by
  funext i
  rfl

/-- Both branches of a binary operation must follow the SAME selection.
This handles residual addition, multiplication/gating, and other binary maps. -/
@[simp] theorem selectRows_map₂ (selection : Fin m → Fin n)
    (f : R → S → T) (x : Tensor R (n :: s)) (y : Tensor S (n :: s)) :
    selectRows selection (map₂ f x y) =
      map₂ f (selectRows selection x) (selectRows selection y) := by
  funext i
  rfl

/-- Selecting a broadcast value just changes how many copies are requested. -/
@[simp] theorem selectRows_broadcastFirst (selection : Fin m → Fin n)
    (x : Tensor R (1 :: s)) :
    selectRows selection (broadcastFirst n x) = broadcastFirst m x := by
  funext i
  rfl

/-- Multiply arbitrary selected rows by any shared weight matrix.
The theorem is independent of the row count and both feature dimensions. -/
@[simp] theorem selectRows_matmul [Semiring R] (selection : Fin m → Fin n)
    (x : Tensor R [n, k]) (w : Tensor R [k, d]) :
    selectRows selection (matmul x w) = matmul (selectRows selection x) w := by
  funext i
  rfl

/-- A property of a batch-polymorphic operation, available for library authors. -/
def SelectionEquivariant
    (op : {n : Nat} → Tensor R (n :: s) → Tensor S (n :: t)) : Prop :=
  ∀ {n m} (selection : Fin m → Fin n) (x : Tensor R (n :: s)),
    op (selectRows selection x) = selectRows selection (op x)

/-- Once two components have been verified, their composition is verified too.
No inspection of their implementations is necessary. -/
theorem SelectionEquivariant.comp
    {f : {n : Nat} → Tensor R (n :: s) → Tensor S (n :: t)}
    {g : {n : Nat} → Tensor S (n :: t) → Tensor T (n :: u)}
    (hf : SelectionEquivariant f) (hg : SelectionEquivariant g) :
    SelectionEquivariant (fun x => g (f x)) := by
  intro n m selection x
  exact (congrArg (@g m) (hf selection x)).trans (hg selection (f x))

end JaxLean.Tensor
