import examples.autodiff.generated.Quadratic
import examples.autodiff.generated.JvpTangent
import examples.autodiff.generated.Gradient
import Mathlib.Analysis.Calculus.Deriv.Mul
import Mathlib.Analysis.Calculus.Deriv.Add

namespace JaxLean.Autodiff

 theorem quadratic_hasDerivAt (x : ℝ) :
    HasDerivAt (fun y : ℝ => quadratic (Tensor.scalar y) ()) (2 * x + 3) x := by
  have h : HasDerivAt (fun y : ℝ => y * y + 3 * y)
      (1 * x + x * 1 + 3 * 1) x :=
    ((hasDerivAt_id x).mul (hasDerivAt_id x)).add
      ((hasDerivAt_id x).const_mul 3)
  convert h using 1 <;> simp [quadratic, Tensor.scalar] <;> ring


 theorem gradient_correct (x : ℝ) :
    gradient (Tensor.scalar x) () =
      deriv (fun y : ℝ => quadratic (Tensor.scalar y) ()) x := by
  rw [(quadratic_hasDerivAt x).deriv]
  simp [gradient, Tensor.scalar]
  ring

 theorem jvp_correct (x v : ℝ) :
    jvp_tangent (Tensor.scalar x) (Tensor.scalar v) () =
      deriv (fun y : ℝ => quadratic (Tensor.scalar y) ()) x * v := by
  rw [(quadratic_hasDerivAt x).deriv]
  simp [jvp_tangent, Tensor.scalar]
  ring

 theorem certified_gradient_correct (x : ℝ) :
    Jaxpr.Program.eval (.cons (Tensor.scalar x) .nil) gradient_ir () =
      deriv (fun y : ℝ =>
        Jaxpr.Program.eval (.cons (Tensor.scalar y) .nil) quadratic_ir ()) x := by
  simp only [gradient_translation_correct, quadratic_translation_correct]
  exact gradient_correct x

 theorem certified_jvp_correct (x v : ℝ) :
    Jaxpr.Program.eval (.cons (Tensor.scalar x) (.cons (Tensor.scalar v) .nil))
        jvp_tangent_ir () =
      deriv (fun y : ℝ =>
        Jaxpr.Program.eval (.cons (Tensor.scalar y) .nil) quadratic_ir ()) x * v := by
  simp only [jvp_tangent_translation_correct, quadratic_translation_correct]
  exact jvp_correct x v

end JaxLean.Autodiff
