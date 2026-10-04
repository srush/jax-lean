import JaxLean.Core.Jaxpr
import JaxLean.Core.Indexing

/-! A heterogeneous SSA layer. Real subprograms reuse the existing real IR;
Boolean and Int32 values cannot be mistaken for real tensors. -/
namespace JaxLean.TypedJaxpr

inductive DType where
  | real | bool | int

abbrev DType.denote : DType → Type
  | .real => ℝ
  | .bool => Bool
  | .int => Int32

abbrev Ty := DType × List Nat
abbrev Value (t : Ty) := Tensor t.1.denote t.2

inductive Var : List Ty → Ty → Type
  | here : Var (t :: ctx) t
  | there : Var ctx t → Var (u :: ctx) t

inductive Env : List Ty → Type
  | nil : Env []
  | cons : Value t → Env ctx → Env (t :: ctx)

def Env.get : Env ctx → Var ctx t → Value t
  | .cons x _, .here => x
  | .cons _ xs, .there v => xs.get v

inductive Atom (ctx : List Ty) : Ty → Type
  | var : Var ctx t → Atom ctx t
  | literal : Value t → Atom ctx t
  | realLiteral (numerator : Int) (denominator : Nat) (positive : 0 < denominator) : Atom ctx (.real, [])

noncomputable def Atom.eval (env : Env ctx) : Atom ctx t → Value t
  | .var v => env.get v
  | .literal x => x
  | .realLiteral n d _ => fun _ => (n : ℝ) / (d : ℝ)

inductive RealArgs (ctx : List Ty) : List (List Nat) → Type
  | nil : RealArgs ctx []
  | cons : Atom ctx (.real, s) → RealArgs ctx ss → RealArgs ctx (s :: ss)

noncomputable def RealArgs.eval (env : Env ctx) : RealArgs ctx ss → Jaxpr.Env ss
  | .nil => .nil
  | .cons x xs => .cons (x.eval env) (xs.eval env)

inductive Comparison where
  | eq | ne | lt | le | gt | ge

def Comparison.eval [LinearOrder α] (c : Comparison) (a b : α) : Bool :=
  match c with
  | .eq => decide (a = b)
  | .ne => decide (a ≠ b)
  | .lt => decide (a < b)
  | .le => decide (a ≤ b)
  | .gt => decide (a > b)
  | .ge => decide (a ≥ b)

/-- Comparisons on signed machine indices use their signed mathematical values. -/
def Comparison.intEval (c : Comparison) (a b : Int32) : Bool := c.eval a.toInt b.toInt

inductive Op (ctx : List Ty) : Ty → Type
  | real (body : Jaxpr.Program args s) : RealArgs ctx args → Op ctx (.real, s)
  | compareReal (c : Comparison) : Atom ctx (.real, s) → Atom ctx (.real, s) → Op ctx (.bool, s)
  | compareInt (c : Comparison) : Atom ctx (.int, s) → Atom ctx (.int, s) → Op ctx (.bool, s)
  | intAdd : Atom ctx (.int, s) → Atom ctx (.int, s) → Op ctx (.int, s)
  | intSub : Atom ctx (.int, s) → Atom ctx (.int, s) → Op ctx (.int, s)
  | intMul : Atom ctx (.int, s) → Atom ctx (.int, s) → Op ctx (.int, s)
  | boolAnd : Atom ctx (.bool, s) → Atom ctx (.bool, s) → Op ctx (.bool, s)
  | boolOr : Atom ctx (.bool, s) → Atom ctx (.bool, s) → Op ctx (.bool, s)
  | boolXor : Atom ctx (.bool, s) → Atom ctx (.bool, s) → Op ctx (.bool, s)
  | boolNot : Atom ctx (.bool, s) → Op ctx (.bool, s)
  | select : Atom ctx (.bool, s) → Atom ctx (d, s) → Atom ctx (d, s) → Op ctx (d, s)
  | reindex (map : Index t → Index s) : Atom ctx (d, s) → Op ctx (d, t)
  | gather (map : Tensor Int32 u → Index t → Index s) : Atom ctx (.real, s) → Atom ctx (.int, u) → Op ctx (.real, t)
  | intToReal : Atom ctx (.int, s) → Op ctx (.real, s)

noncomputable def Op.eval (env : Env ctx) : Op ctx t → Value t
  | .real body args => body.eval (args.eval env)
  | .compareReal c x y => fun i => c.eval (x.eval env i) (y.eval env i)
  | .compareInt c x y => fun i => c.intEval (x.eval env i) (y.eval env i)
  | .intAdd x y => fun i => x.eval env i + y.eval env i
  | .intSub x y => fun i => x.eval env i - y.eval env i
  | .intMul x y => fun i => x.eval env i * y.eval env i
  | .boolAnd x y => fun i => (x.eval env i) && (y.eval env i)
  | .boolOr x y => fun i => (x.eval env i) || (y.eval env i)
  | .boolXor x y => fun i => Bool.xor (x.eval env i) (y.eval env i)
  | .boolNot x => fun i => !(x.eval env i)
  | .select c no yes => fun i => if c.eval env i then yes.eval env i else no.eval env i
  | @Op.reindex _ target source d map x => fun i => (Atom.eval (t := (d, source)) env x) (map i)
  | @Op.gather _ _u target source map x indices => fun i =>
      (Atom.eval (t := (.real, source)) env x) (map (indices.eval env) i)
  | .intToReal x => fun i => ((x.eval env i).toInt : ℝ)

inductive Args (ctx : List Ty) : List Ty → Type
  | nil : Args ctx []
  | cons : Atom ctx t → Args ctx ts → Args ctx (t :: ts)

noncomputable def Args.eval (env : Env ctx) : Args ctx ts → Env ts
  | .nil => .nil
  | .cons x xs => .cons (x.eval env) (xs.eval env)

inductive Program : List Ty → Ty → Type
  | ret : Atom ctx t → Program ctx t
  | bind : Op ctx t → Program (t :: ctx) u → Program ctx u
  | call : Program args t → Args ctx args → Program (t :: ctx) u → Program ctx u

noncomputable def Program.eval (env : Env ctx) : Program ctx t → Value t
  | .ret x => x.eval env
  | .bind op rest => rest.eval (.cons (op.eval env) env)
  | .call body args rest => rest.eval (.cons (body.eval (args.eval env)) env)

end JaxLean.TypedJaxpr
