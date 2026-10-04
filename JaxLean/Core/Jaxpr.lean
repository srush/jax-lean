import JaxLean.Core.RealOps

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

/-- Operations of the normalized real IR. The importer restricts source Jaxpr forms. -/
inductive Op (ctx : List Shape) : Shape → Type
  | iota (coordinate : Index s → Fin n) : Op ctx s
  | exp : Atom ctx s → Op ctx s
  | log : Atom ctx s → Op ctx s
  | sqrt : Atom ctx s → Op ctx s
  | rsqrt : Atom ctx s → Op ctx s
  | sin : Atom ctx s → Op ctx s
  | cos : Atom ctx s → Op ctx s
  | tanh : Atom ctx s → Op ctx s
  | contract (left : Index t → Index k → Index s) (right : Index t → Index k → Index u) :
      Atom ctx s → Atom ctx u → Op ctx t
  | reduceSum (map : Index t → Index k → Index s) : Atom ctx s → Op ctx t
  | reduceMax (positive : 0 < n) (map : Index t → Fin n → Index s) : Atom ctx s → Op ctx t
  | reduceMin (positive : 0 < n) (map : Index t → Fin n → Index s) : Atom ctx s → Op ctx t
  | concatenate (split : Index t → Sum (Index s) (Index u)) : Atom ctx s → Atom ctx u → Op ctx t
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
  | reindex (map : Index t → Index s) : Atom ctx s → Op ctx t
  | addAt (position : Index s) : Atom ctx s → Atom ctx [] → Op ctx s
  | set (position : Index s) : Atom ctx s → Atom ctx [] → Op ctx s

/-- Direct semantics, without calling Tensor.map/map₂/sumFirst. -/
noncomputable def Op.eval (env : Env ctx) : Op ctx s → Tensor ℝ s
  | .iota coordinate => fun i => ((coordinate i).val : ℝ)
  | .exp x => fun i => Real.exp (x.eval env i)
  | .log x => fun i => Real.log (x.eval env i)
  | .sqrt x => fun i => Real.sqrt (x.eval env i)
  | .rsqrt x => fun i => (Real.sqrt (x.eval env i))⁻¹
  | .sin x => fun i => Real.sin (x.eval env i)
  | .cos x => fun i => Real.cos (x.eval env i)
  | .tanh x => fun i => (Real.exp (x.eval env i) - Real.exp (-x.eval env i)) /
      (Real.exp (x.eval env i) + Real.exp (-x.eval env i))
  | .contract left right x y => fun i => ∑ j, x.eval env (left i j) * y.eval env (right i j)
  | .reduceSum map x => fun i => ∑ j, x.eval env (map i j)
  | .reduceMax positive map x => fun i =>
      (List.ofFn (fun j => x.eval env (map i j))).foldl Max.max (x.eval env (map i ⟨0, positive⟩))
  | .reduceMin positive map x => fun i =>
      (List.ofFn (fun j => x.eval env (map i j))).foldl Min.min (x.eval env (map i ⟨0, positive⟩))
  | .concatenate split x y => fun i => match split i with
      | .inl j => x.eval env j
      | .inr j => y.eval env j
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
  | @Op.reshape _ source target sameSize x => fun i =>
      x.eval env ((Index.equivFin source).symm (Fin.cast sameSize.symm (Index.equivFin target i)))
  | .sumFirst x => fun j => ∑ i, x.eval env (i, j)
  | .reindex map x => fun i => x.eval env (map i)
  | .addAt position x value => fun i => if i = position then x.eval env i + value.eval env () else x.eval env i
  | .set position x value => fun i => if i = position then value.eval env () else x.eval env i

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

end JaxLean.Jaxpr
