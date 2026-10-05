import examples.tensor_puzzles.generated.PuzzlesEye
open JaxLean

#eval Tensor.toList (Puzzles.Eye.Array.eye (R := ℚ))

example : Tensor.toList (Puzzles.Eye.Array.eye (R := ℚ)) =
    [1, 0, 0, 0, 1, 0, 0, 0, 1] := by decide
