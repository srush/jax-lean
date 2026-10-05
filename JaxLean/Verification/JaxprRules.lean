import JaxLean.Core.Jaxpr

namespace JaxLean.Jaxpr
@[simp] theorem eval_ret (env : Env ctx) (a : Atom ctx t) :
    (Program.ret a).eval env = a.eval env := rfl

@[simp] theorem eval_bind (env : Env ctx) (op : Op ctx s) (rest : Program (s :: ctx) t) :
    (Program.bind op rest).eval env = rest.eval (.cons (op.eval env) env) := rfl

@[simp] theorem eval_call (env : Env ctx) (callee : Program args s)
    (actual : Args ctx args) (rest : Program (s :: ctx) t) :
    (Program.call callee actual rest).eval env =
      rest.eval (.cons (callee.eval (actual.eval env)) env) := rfl
open scoped BigOperators

theorem sum_index_cons (f : Index (n :: s) → ℝ) :
    (∑ i, f i) = ∑ j : Fin n, ∑ k : Index s, f (j, k) :=
  Fintype.sum_prod_type f

theorem sum_index_nil (f : Index [] → ℝ) : (∑ i, f i) = f () := by
  change (∑ i : Unit, f i) = f ()
  exact Fintype.sum_unique (fun i : Unit => f i)

end JaxLean.Jaxpr
