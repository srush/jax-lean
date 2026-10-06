import JaxLean.Verification.JaxprRules
import Mathlib.Tactic

/-! Routine certificate plumbing. The first two arguments identify the imported
IR and readable function; the rest are immediate callees' certificates. Callee definitions
are deliberately absent: their already-checked theorems discharge each call. -/
macro "jaxpr_certificate" "[" rules:term,* "]" : tactic => do
  let rs ← rules.getElems.mapM fun r => `(Lean.Parser.Tactic.simpLemma| $r:term)
  let definitions := rs.extract 0 2
  let calls ← (rules.getElems.extract 2 rules.getElems.size).mapM fun r =>
    `(Lean.Parser.Tactic.simpLemma| ↓$r:term)
  let prepareRules := definitions ++ calls
  -- Keep rewrite proofs explicit: repeatedly reducing nested SSA environments
  -- in the kernel can otherwise take exponential time.
  let prepare ← `(tactic| simp (config := { implicitDefEqProofs := false }) only
    [$prepareRules,*, ↓JaxLean.Jaxpr.eval_bind, ↓JaxLean.Jaxpr.eval_ret,
     ↓JaxLean.Jaxpr.eval_call, ↓JaxLean.Jaxpr.Args.eval, ↓JaxLean.Jaxpr.Op.eval,
     ↓JaxLean.Jaxpr.Atom.eval, ↓JaxLean.Jaxpr.Env.get,
     JaxLean.Jaxpr.transposeDimensions, JaxLean.Jaxpr.reverseIndex,
     List.length_cons, List.length_nil, List.map, List.idxOf_cons, List.idxOf_nil,
     Nat.reduceBEq, Bool.cond_true, Bool.cond_false, List.range, List.range.loop,
     JaxLean.Jaxpr.broadcastIndex, JaxLean.Jaxpr.coordinate,
     JaxLean.Jaxpr.DType.add, JaxLean.Jaxpr.DType.sub, JaxLean.Jaxpr.DType.mul,
     JaxLean.Jaxpr.Conversion.eval])
  -- Calls can leave dependent Decidable instances; normalize those before
  -- arithmetic simplification to avoid repeated definitional-equality work.
  let normalize ← if calls.isEmpty then `(tactic| skip) else
    `(tactic| try dsimp (config := { instances := true }) only
      [List.foldl, JaxLean.Jaxpr.Comparison.scalar, JaxLean.Jaxpr.Comparison.eval,
       JaxLean.Tensor.scatterSet, JaxLean.Tensor.reindex, JaxLean.Tensor.scalar])
  let simplify ← `(tactic| norm_num only [$rs,*, *,
    JaxLean.Jaxpr.eval_bind, JaxLean.Jaxpr.eval_ret,
    JaxLean.Jaxpr.eval_call, JaxLean.Jaxpr.Args.eval,
    JaxLean.Jaxpr.Op.eval, JaxLean.Jaxpr.Conversion.eval,
    JaxLean.Jaxpr.DType.add, JaxLean.Jaxpr.DType.sub, JaxLean.Jaxpr.DType.mul, JaxLean.Jaxpr.transposeDimensions, JaxLean.Jaxpr.reverseIndex,
     List.length_cons, List.length_nil, List.map, List.idxOf_cons, List.idxOf_nil,
     Nat.reduceBEq, Bool.cond_true, Bool.cond_false, List.range, List.range.loop,
     JaxLean.Jaxpr.broadcastIndex, JaxLean.Jaxpr.coordinate,
    dif_pos, dif_neg, List.foldl, JaxLean.Jaxpr.sum_index_cons, JaxLean.Jaxpr.sum_index_nil, Fin.sum_univ_zero, JaxLean.Jaxpr.Comparison.scalar,
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

  `(tactic| ($prepare:tactic) <;>
    (try simp (config := { implicitDefEqProofs := false }) only [JaxLean.Jaxpr.Env.read_dite]) <;>
    ($normalize:tactic) <;> ($simplify:tactic) <;> ((try split_ifs) <;> $simplify:tactic <;> try simp_all [JaxLean.Jaxpr.Env.get] <;> try omega))
