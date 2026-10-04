import JaxLean.Generated.Mean
import JaxLean.Generated.Variance
import JaxLean.Generated.Transpose
import JaxLean.Generated.Matmul
import JaxLean.Generated.Scan

/-! Exact execution of the same definitions used in the real proofs.
Run `lake env lean JaxLean/Run.lean`. Transcendental programs require RealOps
and intentionally cannot be run over rationals. -/
open JaxLean JaxLean.Generated

def samples : Tensor ℚ [2] := Tensor.ofFlat ![1, 3]

#eval Tensor.toList (mean2 samples)       -- [2]
#eval Tensor.toList (variance2 samples)   -- [1]
#eval Tensor.toList (transpose23 (Tensor.ofFlat ![1, 2, 3, 4, 5, 6] : Tensor ℚ [2, 3]))
-- [1, 4, 2, 5, 3, 6]

#eval Tensor.toList (matmul
  (Tensor.ofFlat ![1, 2, 3, 4, 5, 6] : Tensor ℚ [2, 3])
  (Tensor.ofFlat ![1, 0, 0, 1, 1, 1] : Tensor ℚ [3, 2]))
-- [4, 5, 10, 11]

#eval let (total, prefixes) := prefixSum (Tensor.ofFlat ![1, 2, 3] : Tensor ℚ [3])
      (Tensor.toList total, Tensor.toList prefixes)
-- ([6], [1, 3, 6])
