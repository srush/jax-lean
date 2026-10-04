import JaxLean.Core.Tensor

namespace JaxLean.Tensor
/-- Clip a signed slice start to the valid interval [0, upper]. -/
def clipStart (upper : Nat) (x : Int32) : Fin (upper + 1) :=
  ⟨min x.toInt.toNat upper, Nat.lt_succ_of_le (Nat.min_le_right _ _)⟩

end JaxLean.Tensor
