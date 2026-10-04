import JaxLean.Core.Jaxpr

/-! Local soundness lemmas connecting IR evaluation to tensor operations. -/
namespace JaxLean.Jaxpr
open scoped BigOperators

@[simp] theorem eval_reindex (env : Env ctx) (map : Index t → Index s) (x : Atom ctx s) :
    (Op.reindex map x).eval env = Tensor.reindex map (x.eval env) := rfl

@[simp] theorem eval_set (env : Env ctx) (position : Index s) (x : Atom ctx s) (v : Atom ctx []) :
    (Op.set position x v).eval env = Tensor.scatterSet (x.eval env) position (v.eval env ()) := rfl

@[simp] theorem eval_addAt (env : Env ctx) (position : Index s) (x : Atom ctx s) (v : Atom ctx []) :
    (Op.addAt position x v).eval env = Tensor.scatterAdd (x.eval env) position (v.eval env ()) := rfl

@[simp] theorem eval_ret (env : Env ctx) (a : Atom ctx s) :
    (Program.ret a).eval env = a.eval env := rfl

/-- Local translation rules. Certificates compose these with environment lookup. -/
@[simp] theorem eval_add (env : Env ctx) (x y : Atom ctx s) :
    (Op.add x y).eval env = Tensor.map₂ (· + ·) (x.eval env) (y.eval env) := rfl
@[simp] theorem eval_mul (env : Env ctx) (x y : Atom ctx s) :
    (Op.mul x y).eval env = Tensor.map₂ (· * ·) (x.eval env) (y.eval env) := rfl
@[simp] theorem eval_div (env : Env ctx) (x y : Atom ctx s) :
    (Op.div x y).eval env = Tensor.map₂ (· / ·) (x.eval env) (y.eval env) := rfl
@[simp] theorem eval_broadcast (env : Env ctx) (x : Atom ctx []) (s : Shape) :
    (Op.broadcast s x).eval env = Tensor.reindex (s := []) (t := s) (fun _ => ()) (x.eval env) := rfl
@[simp] theorem eval_sumFirst (env : Env ctx) (x : Atom ctx (n :: s)) :
    (Op.sumFirst x).eval env = Tensor.sumFirst (x.eval env) := rfl

@[simp] theorem eval_sub (env : Env ctx) (x y : Atom ctx s) :
    (Op.sub x y).eval env = Tensor.map₂ (· - ·) (x.eval env) (y.eval env) := rfl
@[simp] theorem eval_min (env : Env ctx) (x y : Atom ctx s) :
    (Op.min x y).eval env = Tensor.map₂ min (x.eval env) (y.eval env) := rfl
@[simp] theorem eval_max (env : Env ctx) (x y : Atom ctx s) :
    (Op.max x y).eval env = Tensor.map₂ max (x.eval env) (y.eval env) := rfl
@[simp] theorem eval_neg (env : Env ctx) (x : Atom ctx s) :
    (Op.neg x).eval env = Tensor.map (fun x => -x) (x.eval env) := rfl
@[simp] theorem eval_square (env : Env ctx) (x : Atom ctx s) :
    (Op.square x).eval env = Tensor.map (fun x => x ^ (2 : Nat)) (x.eval env) := rfl
@[simp] theorem eval_abs (env : Env ctx) (x : Atom ctx s) :
    (Op.abs x).eval env = Tensor.map abs (x.eval env) := rfl
@[simp] theorem eval_copy (env : Env ctx) (x : Atom ctx s) :
    (Op.copy x).eval env = x.eval env := rfl
@[simp] theorem eval_integerPow (env : Env ctx) (p : Int) (x : Atom ctx s) :
    (Op.integerPow p x).eval env = Tensor.map (fun x => x ^ p) (x.eval env) := rfl
@[simp] theorem eval_prepend (env : Env ctx) (n : Nat) (x : Atom ctx s) :
    (Op.prepend n x).eval env = Tensor.reindex (s := s) (t := n :: s) (fun i => i.2) (x.eval env) := rfl
@[simp] theorem eval_expandFirst (env : Env ctx) (n : Nat) (x : Atom ctx (1 :: s)) :
    (Op.expandFirst n x).eval env = Tensor.broadcastFirst n (x.eval env) := rfl
@[simp] theorem eval_transpose2 (env : Env ctx) (x : Atom ctx [n, m]) :
    (Op.transpose2 x).eval env =
      Tensor.reindex (s := [n, m]) (t := [m, n]) (fun i => (i.2.1, i.1, ())) (x.eval env) := rfl
@[simp] theorem eval_matmul (env : Env ctx) (x : Atom ctx [n, k]) (y : Atom ctx [k, m]) :
    (Op.matmul x y).eval env = Tensor.matmul (x.eval env) (y.eval env) := rfl
@[simp] theorem eval_vecmat (env : Env ctx) (x : Atom ctx [k]) (y : Atom ctx [k, m]) :
    (Op.vecmat x y).eval env = Tensor.vecmat (x.eval env) (y.eval env) := rfl

/-- Sequential composition introduces exactly the value computed by the IR operation. -/
@[simp] theorem eval_bind (env : Env ctx) (op : Op ctx s) (rest : Program (s :: ctx) t) :
    (Program.bind op rest).eval env = rest.eval (.cons (op.eval env) env) := rfl

@[simp] theorem eval_call (env : Env ctx) (callee : Program args s)
    (actual : Args ctx args) (rest : Program (s :: ctx) t) :
    (Program.call callee actual rest).eval env =
      rest.eval (.cons (callee.eval (actual.eval env)) env) := rfl

@[simp] theorem eval_appendOne (env : Env ctx) (x : Atom ctx [n]) :
    (Op.appendOne x).eval env = (fun i => x.eval env (i.1, ())) := rfl
@[simp] theorem eval_expandLast (env : Env ctx) (m : Nat) (x : Atom ctx [n, 1]) :
    (Op.expandLast m x).eval env = (fun i => x.eval env (i.1, 0, ())) := rfl
@[simp] theorem eval_sumLast (env : Env ctx) (x : Atom ctx [n, m]) :
    (Op.sumLast x).eval env = (fun i => ∑ j, x.eval env (i.1, j, ())) := rfl

@[simp] theorem eval_dotVec (env : Env ctx) (x y : Atom ctx [n]) :
    (Op.dotVec x y).eval env = (fun _ => ∑ j, x.eval env (j, ()) * y.eval env (j, ())) := rfl
@[simp] theorem eval_reverseVec (env : Env ctx) (x : Atom ctx [n]) :
    (Op.reverseVec x).eval env = (fun i => x.eval env (i.1.rev, ())) := rfl
@[simp] theorem eval_reshape (env : Env ctx) (target : Shape)
    (sameSize : s.prod = target.prod) (x : Atom ctx s) :
    (Op.reshape target sameSize x).eval env = Tensor.reshape sameSize (x.eval env) := rfl

end JaxLean.Jaxpr
