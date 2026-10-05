import JaxLean.Stdlib
import JaxLean.Stdlib.SelectionRules
import examples.selection.generated.MLP4
import examples.selection.generated.MLP3
import examples.selection.generated.Residual5
import examples.selection.generated.Residual2

/-!
# Apply reusable theorems directly to transpiled programs

Read `SelectionRules.lean` for the general components, then `Generated/MLP4.lean`.
There is no hand-written copy of either network and no per-network bridge proof.
Only the generated definitions are unfolded; simp applies the library's rules
for matmul, arbitrary unary/binary maps, and first-axis broadcasting.
-/
namespace JaxLean.Selection
open Tensor (selectRows)

/-- Select any three rows from four, before or after the transpiled MLP.
The inputs and parameters are arbitrary real tensors. -/
theorem generated_selection (selection : Fin 3 → Fin 4)
    (x : Tensor ℝ [4, 2])
    (w1 : Tensor ℝ [2, 3]) (b1 : Tensor ℝ [3])
    (w2 : Tensor ℝ [3, 2]) (b2 : Tensor ℝ [2]) :
    Generated.mlp3 (selectRows selection x) w1 b1 w2 b2 =
      selectRows selection (Generated.mlp4 x w1 b1 w2 b2) := by
  -- Expose the generated operations, then apply the general selection rules.
  simp only [Generated.mlp3, Generated.mlp4, Tensor.selectRows_map,
    Tensor.selectRows_map₂, Tensor.selectRows_broadcastFirst, Tensor.selectRows_matmul]

/-- A different graph, activation, and dimensions: the SAME library applies.
The residual connection needs map₂'s rule, which selects both of its branches. -/
theorem residual_selection (selection : Fin 2 → Fin 5)
    (x : Tensor ℝ [5, 3])
    (w1 : Tensor ℝ [3, 4]) (b1 : Tensor ℝ [4])
    (w2 : Tensor ℝ [4, 3]) (b2 : Tensor ℝ [3]) :
    Generated.residual2 (selectRows selection x) w1 b1 w2 b2 =
      selectRows selection (Generated.residual5 x w1 b1 w2 b2) := by
  -- The selection rules are also registered with simp, giving this short form.
  simp [Generated.residual2, Generated.residual5]

-- A concrete selection: [row 2, row 0, row 2], using zero-based indexing.
-- It both changes order and repeats a row, so it is more than a permutation.
def pick : Fin 3 → Fin 4 := ![2, 0, 2]

-- Exact rational execution helps us read the result. It is not the proof above.
def sampleX : Tensor ℚ [4, 2] :=
  Tensor.ofArray #[1, 2, -1, 1, 3, -2, 0, 0] (by decide)
def sampleW1 : Tensor ℚ [2, 3] :=
  Tensor.ofArray #[1, -1, 2, 0, 1, -1] (by decide)
def sampleB1 : Tensor ℚ [3] := Tensor.ofArray #[0, 1, -1] (by decide)
def sampleW2 : Tensor ℚ [3, 2] :=
  Tensor.ofArray #[1, 0, 0, 1, 1, -1] (by decide)
def sampleB2 : Tensor ℚ [2] := Tensor.ofArray #[1, -1] (by decide)

-- All rows: [[2,1], [1,2], [11,-8], [1,0]].
#eval Tensor.toList (Generated.mlp4 sampleX sampleW1 sampleB1 sampleW2 sampleB2)
-- Select first, then run: [[11,-8], [2,1], [11,-8]].
#eval Tensor.toList
  (Generated.mlp3 (selectRows pick sampleX) sampleW1 sampleB1 sampleW2 sampleB2)
-- Run first, then select: the same values.
#eval Tensor.toList
  (selectRows pick (Generated.mlp4 sampleX sampleW1 sampleB1 sampleW2 sampleB2))

end JaxLean.Selection
