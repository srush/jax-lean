import JaxLean.Stdlib.Batch
import Mathlib.Probability.Independence.Basic

/-! Probability bridges for generic batches, independent of Rand and finite laws. -/
namespace JaxLean.Batch
open MeasureTheory ProbabilityTheory

/-- Separate measurable maps preserve an independent family. A vmap does not
create independence if it was absent from the input family. -/
theorem independent_vmap {Ω ι α β : Type*} [MeasurableSpace Ω]
    [MeasurableSpace α] [MeasurableSpace β] {μ : Measure Ω}
    {X : ι → Ω → α} (hX : iIndepFun X μ) (f : ι → α → β)
    (hf : ∀ i, Measurable (f i)) :
    iIndepFun (fun i ω => f i (X i ω)) μ := hX.comp f hf

end JaxLean.Batch
