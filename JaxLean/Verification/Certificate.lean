import JaxLean.Verification.JaxprRules
import JaxLean.Core.TypedJaxpr
import Mathlib.Tactic

/-! Routine certificate plumbing. The arguments identify the current function,
its imported IR, and its immediate callees' certificates. Callee definitions
are deliberately absent: their already-checked theorems discharge each call. -/
macro "jaxpr_certificate" "[" rules:term,* "]" : tactic => do
  let rs ← rules.getElems.mapM fun r => `(Lean.Parser.Tactic.simpLemma| $r:term)
  `(tactic| norm_num only [$rs,*,
    JaxLean.Jaxpr.eval_bind, JaxLean.Jaxpr.eval_ret,
    JaxLean.Jaxpr.eval_call, JaxLean.Jaxpr.Args.eval,
    JaxLean.Jaxpr.eval_add, JaxLean.Jaxpr.eval_mul, JaxLean.Jaxpr.eval_div,
    JaxLean.Jaxpr.eval_broadcast, JaxLean.Jaxpr.eval_sumLast,
    JaxLean.Jaxpr.eval_appendOne, JaxLean.Jaxpr.eval_expandLast,
    JaxLean.Jaxpr.eval_sumFirst, JaxLean.Jaxpr.eval_sub, JaxLean.Jaxpr.eval_neg,
    JaxLean.Jaxpr.eval_square, JaxLean.Jaxpr.eval_integerPow,
    JaxLean.Jaxpr.eval_abs, JaxLean.Jaxpr.eval_min, JaxLean.Jaxpr.eval_max,
    JaxLean.Jaxpr.eval_copy, JaxLean.Jaxpr.eval_prepend,
    JaxLean.Jaxpr.eval_expandFirst, JaxLean.Jaxpr.eval_transpose2,
    JaxLean.Jaxpr.eval_matmul, JaxLean.Jaxpr.eval_vecmat,
    JaxLean.Jaxpr.Op.eval, JaxLean.Tensor.scatterSet, JaxLean.Tensor.scatterAdd, JaxLean.Tensor.reshape,
    JaxLean.Jaxpr.eval_reindex, JaxLean.Jaxpr.eval_set,
    JaxLean.Jaxpr.eval_dotVec, JaxLean.Jaxpr.eval_reverseVec, JaxLean.Jaxpr.eval_reshape,
    JaxLean.Jaxpr.Atom.eval, JaxLean.Jaxpr.Env.get,
    JaxLean.Tensor.concatenate, JaxLean.Tensor.contract, JaxLean.Tensor.reduceSum,
    JaxLean.Tensor.reduceMax, JaxLean.Tensor.reduceMin,
    JaxLean.RealOps.exp, JaxLean.RealOps.log, JaxLean.RealOps.sqrt,
    JaxLean.RealOps.sin, JaxLean.RealOps.cos, JaxLean.RealOps.tanh,
    JaxLean.Tensor.map, JaxLean.Tensor.map₂, JaxLean.Tensor.scalar,
    JaxLean.Tensor.reindex, JaxLean.Tensor.sumFirst,
    JaxLean.Tensor.broadcastFirst, JaxLean.Tensor.matmul,
    JaxLean.Tensor.vecmat, Fin.fin_one_eq_zero] <;> try rfl)
