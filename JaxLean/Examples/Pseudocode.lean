import JaxLean.Stdlib
import JaxLean.Stdlib.Imperative
import Mathlib.Tactic

namespace JaxLean.Pseudocode
open scoped BigOperators
open Imperative

def sum [Add R] [Zero R] (a : Tensor R [n]) : R := Id.run do
  let mut total := 0
  for i in range n do
    total := total + a[i]
  return total

def outer [Mul R] [Zero R] (a : Tensor R [n]) (b : Tensor R [m]) : Tensor R [n, m] := Id.run do
  let mut out : Tensor R [n, m] := fun _ => 0
  for i in range n do
    for j in range m do
      out := out.set (i, j, ()) (a[i] * b[j])
  return out

def flip [Zero R] (a : Tensor R [n]) : Tensor R [n] := Id.run do
  let mut out : Tensor R [n] := fun _ => 0
  for i in range n do
    out := out.set (i, ()) a[i.rev]
  return out

def flatten [Zero R] (a : Tensor R [n, m]) : Tensor R [n * m] := Id.run do
  let mut out : Tensor R [n * m] := fun _ => 0
  for k in range (n * m) do
    out := out.set (k, ()) a[(k.divNat, k.modNat)]
  return out

theorem sum_eq [AddCommMonoid R] (a : Tensor R [n]) :
    sum a = ∑ i, a (i, ()) := by
  simp [sum, range, ← Fin.foldl_eq_finRange_foldl, fold_add_eq]

theorem outer_eq [Mul R] [Zero R] (a : Tensor R [n]) (b : Tensor R [m]) :
    outer a b = fun i => a (i.1, ()) * b (i.2.1, ()) := by
  simp only [outer, range, Tensor.set, List.forIn_pure_yield_eq_foldl, pure_bind]
  funext i
  rcases i with ⟨i, j, ⟨⟩⟩
  change ((List.finRange n).foldl (fun out i => (List.finRange m).foldl
    (fun out j => Function.update out (i, j, ()) (a (i, ()) * b (j, ()))) out)
    (fun _ => (0 : R))) (i, j, ()) = _
  erw [foldl_nested (f := fun (out : Fin n × Fin m × Unit → R) (ij : Fin n × Fin m) =>
    Function.update out (ij.1, ij.2, ()) (a (ij.1, ()) * b (ij.2, ())))]
  erw [foldl_write (index := fun ij : Fin n × Fin m => (ij.1, ij.2, ()))
    (f := fun ij : Fin n × Fin m × Unit => a (ij.1, ()) * b (ij.2.1, ()))]
  simp

theorem flip_eq [Zero R] (a : Tensor R [n]) :
    flip a = fun i => a (i.1.rev, ()) := by
  simp only [flip, range, Tensor.set]
  simp only [List.forIn_pure_yield_eq_foldl]
  funext i
  rcases i with ⟨i, ⟨⟩⟩
  change ((List.finRange n).foldl (fun out j => Function.update out (j, ()) (a (j.rev, ()))) (fun _ => (0 : R))) (i, ()) = _
  erw [foldl_write (index := fun j : Fin n => (j, ())) (f := fun j : Fin n × Unit => a (j.1.rev, ()))]
  simp

theorem flatten_eq [Zero R] (a : Tensor R [n, m]) :
    flatten a = fun i => a (i.1.divNat, i.1.modNat, ()) := by
  simp only [flatten, range, Tensor.set]
  simp only [List.forIn_pure_yield_eq_foldl]
  funext i
  rcases i with ⟨i, ⟨⟩⟩
  change ((List.finRange (n*m)).foldl (fun out j => Function.update out (j, ()) (a (j.divNat, j.modNat, ()))) (fun _ => (0 : R))) (i, ()) = _
  erw [foldl_write (index := fun j : Fin (n*m) => (j, ())) (f := fun j : Fin (n*m) × Unit => a (j.1.divNat, j.1.modNat, ()))]
  simp

/-- The row-major tensor reshape agrees with independent quotient/remainder indexing. -/
theorem reshape_flatten [Zero R] (a : Tensor R [n, m]) :
    Tensor.reshape (t := [n * m]) (by simp) a = flatten a := by
  rw [flatten_eq]
  funext i
  rcases i with ⟨i, ⟨⟩⟩
  apply congrArg a
  apply Prod.ext
  · apply Fin.ext
    change (0 + 1 * i.val) / (m * 1) = i.val / m
    simp
  · apply Prod.ext
    · apply Fin.ext
      change ((0 + 1 * i.val) % (m * 1)) / 1 = i.val % m
      simp
    · rfl

end JaxLean.Pseudocode
