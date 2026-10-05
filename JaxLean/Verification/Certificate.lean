import JaxLean.Verification.JaxprRules
import Mathlib.Tactic

/-! Routine certificate plumbing. The arguments identify the current function,
its imported IR, and its immediate callees' certificates. Callee definitions
are deliberately absent: their already-checked theorems discharge each call. -/
macro "jaxpr_certificate" "[" rules:term,* "]" : tactic => do
  let rs ← rules.getElems.mapM fun r => `(Lean.Parser.Tactic.simpLemma| $r:term)
  let simplify ← `(tactic| norm_num only [$rs,*, *,
    JaxLean.Jaxpr.eval_bind, JaxLean.Jaxpr.eval_ret,
    JaxLean.Jaxpr.eval_call, JaxLean.Jaxpr.Args.eval,
    JaxLean.Jaxpr.Op.eval, JaxLean.Jaxpr.Conversion.eval,
    JaxLean.Jaxpr.DType.add, JaxLean.Jaxpr.DType.sub, JaxLean.Jaxpr.DType.mul, JaxLean.Jaxpr.broadcastIndex, JaxLean.Jaxpr.coordinate,
    dif_pos, dif_neg, List.length_cons, List.length_nil, List.range, List.range.loop, List.foldl, JaxLean.Jaxpr.sum_index_cons, JaxLean.Jaxpr.sum_index_nil, Fin.sum_univ_zero, JaxLean.Jaxpr.Comparison.scalar,
    JaxLean.Jaxpr.Comparison.eval, JaxLean.Jaxpr.Comparison.intEval, JaxLean.Tensor.scatterSet, JaxLean.Tensor.scatterAdd, JaxLean.Tensor.reshape,
    JaxLean.Jaxpr.Atom.eval, JaxLean.Jaxpr.Env.get, JaxLean.Jaxpr.Env.read_dite, JaxLean.Jaxpr.Env.read,
    JaxLean.Tensor.concatenate, JaxLean.Tensor.contract, JaxLean.Tensor.reduceSum,
    JaxLean.Tensor.reduceMax, JaxLean.Tensor.reduceMin,
    JaxLean.RealOps.exp, JaxLean.RealOps.log, JaxLean.RealOps.sqrt,
    JaxLean.RealOps.sin, JaxLean.RealOps.cos, JaxLean.RealOps.tanh,
    JaxLean.Tensor.map, JaxLean.Tensor.map₂, JaxLean.Tensor.scalar,
    JaxLean.Tensor.reindex, JaxLean.Tensor.sumFirst,
    JaxLean.Tensor.broadcastFirst, JaxLean.Tensor.matmul,
    JaxLean.Tensor.vecmat, Fin.fin_one_eq_zero] <;> try rfl)

  `(tactic| ($simplify:tactic) <;> ((try split_ifs) <;> $simplify:tactic <;> try simp_all [JaxLean.Jaxpr.Env.get] <;> try omega))
