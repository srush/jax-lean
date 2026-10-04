import JaxLean.Tensor
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Analysis.SpecialFunctions.Sqrt

namespace JaxLean

/-- Only required by programs using transcendental primitives. There is deliberately
no rational instance: algebraic execution must not fake transcendental operations. -/
class RealOps (R : Type) where
  exp : R → R
  log : R → R
  sqrt : R → R
  sin : R → R
  cos : R → R
  tanh : R → R

noncomputable instance : RealOps ℝ where
  exp := Real.exp
  log := Real.log
  sqrt := Real.sqrt
  sin := Real.sin
  cos := Real.cos
  tanh := fun x => (Real.exp x - Real.exp (-x)) / (Real.exp x + Real.exp (-x))

end JaxLean
