import JaxLean.Tensor

/-! A small, intrinsically shape-typed Jaxpr model, independent of the Python
transpiler. Its evaluator uses direct coordinate semantics. The importer is
trusted to encode the original Jaxpr faithfully; this is not IEEE semantics. -/
namespace JaxLean.Jaxpr
open scoped BigOperators
abbrev Shape := List Nat

/-- A reference into the environment, carrying its tensor shape. -/
inductive Var : List Shape → Shape → Type
  | here : Var (s :: ctx) s
  | there : Var ctx s → Var (t :: ctx) s

inductive Env : List Shape → Type
  | nil : Env []
  | cons : Tensor ℝ s → Env ctx → Env (s :: ctx)

def Env.get : Env ctx → Var ctx s → Tensor ℝ s
  | .cons x _, .here => x
  | .cons _ env, .there v => env.get v

inductive Atom (ctx : List Shape) : Shape → Type
  | var : Var ctx s → Atom ctx s
  /-- Exact rational literal, storing its numerator and positive denominator. -/
  | literal (numerator : Int) (denominator : Nat) (positive : 0 < denominator) : Atom ctx []

noncomputable def Atom.eval (env : Env ctx) : Atom ctx s → Tensor ℝ s
  | .var v => env.get v
  | .literal n d _ => fun _ => (n : ℝ) / (d : ℝ)

/-- The first certified fragment. Unsupported primitives cannot be encoded. -/
inductive Op (ctx : List Shape) : Shape → Type
  | copy : Atom ctx s → Op ctx s
  | add : Atom ctx s → Atom ctx s → Op ctx s
  | sub : Atom ctx s → Atom ctx s → Op ctx s
  | neg : Atom ctx s → Op ctx s
  | square : Atom ctx s → Op ctx s
  | integerPow (power : Int) : Atom ctx s → Op ctx s
  | abs : Atom ctx s → Op ctx s
  | min : Atom ctx s → Atom ctx s → Op ctx s
  | max : Atom ctx s → Atom ctx s → Op ctx s
  | mul : Atom ctx s → Atom ctx s → Op ctx s
  | div : Atom ctx s → Atom ctx s → Op ctx s
  | broadcast (shape : Shape) : Atom ctx [] → Op ctx shape
  | prepend (n : Nat) : Atom ctx s → Op ctx (n :: s)
  | expandFirst (n : Nat) : Atom ctx (1 :: s) → Op ctx (n :: s)
  | transpose2 : Atom ctx [n, m] → Op ctx [m, n]
  | matmul : Atom ctx [n, k] → Atom ctx [k, m] → Op ctx [n, m]
  | vecmat : Atom ctx [k] → Atom ctx [k, m] → Op ctx [m]
  | appendOne : Atom ctx [n] → Op ctx [n, 1]
  | expandLast (m : Nat) : Atom ctx [n, 1] → Op ctx [n, m]
  | sumLast : Atom ctx [n, m] → Op ctx [n]
  | dotVec : Atom ctx [n] → Atom ctx [n] → Op ctx []
  | reverseVec : Atom ctx [n] → Op ctx [n]
  | reshape (target : Shape) (sameSize : s.prod = target.prod) : Atom ctx s → Op ctx target
  | sumFirst : Atom ctx (n :: s) → Op ctx s

/-- Direct semantics, without calling Tensor.map/map₂/sumFirst. -/
noncomputable def Op.eval (env : Env ctx) : Op ctx s → Tensor ℝ s
  | .copy x => x.eval env
  | .add x y => fun i => x.eval env i + y.eval env i
  | .sub x y => fun i => x.eval env i - y.eval env i
  | .neg x => fun i => -(x.eval env i)
  | .square x => fun i => x.eval env i ^ (2 : Nat)
  | .integerPow p x => fun i => x.eval env i ^ p
  | .abs x => fun i => |x.eval env i|
  | .min x y => fun i => Min.min (x.eval env i) (y.eval env i)
  | .max x y => fun i => Max.max (x.eval env i) (y.eval env i)
  | .mul x y => fun i => x.eval env i * y.eval env i
  | .div x y => fun i => x.eval env i / y.eval env i
  | .broadcast _ x => fun _ => x.eval env ()
  | .prepend _ x => fun i => x.eval env i.2
  | .expandFirst _ x => fun i => x.eval env (0, i.2)
  | .transpose2 x => fun i => x.eval env (i.2.1, i.1, ())
  | .matmul x y => fun i => ∑ j, x.eval env (i.1, j, ()) * y.eval env (j, i.2)
  | .vecmat x y => fun i => ∑ j, x.eval env (j, ()) * y.eval env (j, i)
  | .appendOne x => fun i => x.eval env (i.1, ())
  | .expandLast _ x => fun i => x.eval env (i.1, 0, ())
  | .sumLast x => fun i => ∑ j, x.eval env (i.1, j, ())
  | .dotVec x y => fun _ => ∑ j, x.eval env (j, ()) * y.eval env (j, ())
  | .reverseVec x => fun i => x.eval env (i.1.rev, ())
  | .reshape target sameSize x => fun i =>
      x.eval env ((Index.equivFin _).symm (Fin.cast sameSize.symm (Index.equivFin target i)))
  | .sumFirst x => fun j => ∑ i, x.eval env (i, j)

/-- Typed actual arguments at a function boundary. -/
inductive Args (ctx : List Shape) : List Shape → Type
  | nil : Args ctx []
  | cons : Atom ctx s → Args ctx ss → Args ctx (s :: ss)

noncomputable def Args.eval (env : Env ctx) : Args ctx ss → Env ss
  | .nil => .nil
  | .cons x xs => .cons (x.eval env) (xs.eval env)

/-- Explicit SSA let-bindings followed by one tensor result. -/
inductive Program : List Shape → Shape → Type
  | ret : Atom ctx s → Program ctx s
  | bind : Op ctx s → Program (s :: ctx) t → Program ctx t
  | call : Program args s → Args ctx args → Program (s :: ctx) t → Program ctx t

noncomputable def Program.eval (env : Env ctx) : Program ctx s → Tensor ℝ s
  | .ret a => a.eval env
  | .bind op rest => rest.eval (.cons (op.eval env) env)
  | .call callee args rest => rest.eval (.cons (callee.eval (args.eval env)) env)

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
