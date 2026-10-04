import Mathlib.LinearAlgebra.Pi

/-! Pure batch algebra. No probability, sampler, or generated-program imports.
A batch may contain scalars, vectors, matrices, or other modules. -/
namespace JaxLean.Batch
open scoped BigOperators

def vmap (f : α → β) (x : ι → α) : ι → β := fun i => f (x i)
def vmap₂ (f : α → β → γ) (x : ι → α) (y : ι → β) : ι → γ :=
  fun i => f (x i) (y i)
def reduceSum [Fintype ι] [AddCommMonoid M] (x : ι → M) : M := ∑ i, x i

/-- An output coordinate depends only on the corresponding input coordinate.
This is structural locality, not probabilistic independence. -/
def Coordinatewise (F : (ι → α) → (ι → β)) : Prop :=
  ∀ i x y, x i = y i → F x i = F y i

theorem coordinatewise_vmap (f : α → β) : Coordinatewise (vmap (ι := ι) f) := by
  intro i x y h
  exact congrArg f h

theorem Coordinatewise.comp {F : (ι → α) → (ι → β)} {G : (ι → β) → (ι → γ)}
    (hF : Coordinatewise F) (hG : Coordinatewise G) : Coordinatewise (G ∘ F) := by
  intro i x y h
  exact hG i (F x) (F y) (hF i x y h)

@[simp] theorem vmap_comp (g : β → γ) (f : α → β) (x : ι → α) :
    vmap g (vmap f x) = vmap (g ∘ f) x := rfl

@[simp] theorem vmap₂_vmap (b : β → δ → ε) (f : α → β) (g : γ → δ)
    (x : ι → α) (y : ι → γ) :
    vmap₂ b (vmap f x) (vmap g y) = vmap₂ (fun a c => b (f a) (g c)) x y := rfl

section Linear
variable [CommSemiring R] [AddCommMonoid M] [AddCommMonoid N] [AddCommMonoid P]
variable [Module R M] [Module R N] [Module R P]

/-- Any linear operation commutes with summing a batch. -/
theorem linear_reduceSum [Fintype ι] (L : M →ₗ[R] N) (x : ι → M) :
    L (reduceSum x) = reduceSum (vmap L x) := by
  simp [reduceSum, vmap]

/-- Bilinear operations commute with reduction in either argument separately. -/
theorem bilinear_reduceSum_left [Fintype ι] (B : M →ₗ[R] N →ₗ[R] P)
    (x : ι → M) (y : N) :
    B (reduceSum x) y = reduceSum (vmap (fun a => B a y) x) := by
  simp [reduceSum, vmap]

theorem bilinear_reduceSum_right [Fintype ι] (B : M →ₗ[R] N →ₗ[R] P)
    (x : M) (y : ι → N) :
    B x (reduceSum y) = reduceSum (vmap (B x) y) := by
  simp [reduceSum, vmap]

/-- Reducing both arguments gives ALL pairs, not just matching coordinates. -/
theorem bilinear_reduceSum_both [Fintype ι] [Fintype κ]
    (B : M →ₗ[R] N →ₗ[R] P) (x : ι → M) (y : κ → N) :
    B (reduceSum x) (reduceSum y) =
      reduceSum (fun i => reduceSum (fun j => B (x i) (y j))) := by
  simp only [reduceSum, map_sum, LinearMap.sum_apply]
  exact Finset.sum_comm
end Linear
end JaxLean.Batch
