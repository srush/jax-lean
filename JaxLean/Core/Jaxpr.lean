import JaxLean.Core.RealOps
import JaxLean.Core.Indexing

/-! One dtype- and shape-indexed Jaxpr. Floating arithmetic is interpreted over
mathematical reals; integer indices use Int32 and predicates use Bool. -/
namespace JaxLean.Jaxpr
open scoped BigOperators
abbrev Shape := List Nat

def coordinate : (s : Shape) → Index s → (axis : Nat) → Fin (s[axis]?.getD 1)
  | [], _, 0 => ⟨0, by simp⟩
  | [], _, _ + 1 => ⟨0, by simp⟩
  | _ :: _, i, 0 => i.1
  | _ :: s, i, n + 1 => coordinate s i.2 n

def broadcastValid : Shape → Shape → List Nat → Prop
  | [], _, dims => dims = []
  | n :: ns, t, axis :: axes => axis < t.length ∧ (n = 1 ∨ n = t[axis]?.getD 1) ∧ broadcastValid ns t axes
  | _ :: _, _, [] => False

instance (s t : Shape) (dims : List Nat) : Decidable (broadcastValid s t dims) := by
  induction s generalizing dims with
  | nil => exact inferInstanceAs (Decidable (dims = []))
  | cons n ns ih =>
    cases dims with
    | nil => exact isFalse id
    | cons a axes => unfold broadcastValid; exact instDecidableAnd

def broadcastIndex : (s t : Shape) → (dims : List Nat) → broadcastValid s t dims → Index t → Index s
  | [], _, _, _, _ => ()
  | n :: ns, t, axis :: axes, h, i =>
      (if hn : n = 1 then ⟨0, by omega⟩ else
        ⟨(coordinate t i axis).val, by have := (coordinate t i axis).isLt; have := h.2.1.resolve_left hn; omega⟩,
       broadcastIndex ns t axes h.2.2 i)
  | _ :: _, _, [], h, _ => False.elim h

inductive DType where
  | real | bool | int
  deriving DecidableEq

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
  | tensorLiteral : Value t → Atom ctx t
  | literal (numerator : Int) (denominator : Nat) (positive : 0 < denominator) : Atom ctx (.real, [])

noncomputable def Atom.eval (env : Env ctx) : Atom ctx t → Value t
  | .var v => env.get v
  | .tensorLiteral x => x
  | .literal n d _ => fun _ => (n : ℝ) / (d : ℝ)

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

noncomputable def Comparison.scalar (c : Comparison) : (d : DType) → d.denote → d.denote → Bool
  | .real => c.eval
  | .int => c.intEval
  | .bool => fun a b => c.eval a.toNat b.toNat

inductive Conversion : DType → DType → Type
  | identity : Conversion d d
  | intToReal : Conversion .int .real

noncomputable def Conversion.eval : Conversion a b → a.denote → b.denote
  | .identity => id
  | .intToReal => fun x => (x.toInt : ℝ)

noncomputable def DType.add : (d : DType) → d.denote → d.denote → d.denote
  | .real => fun a b => a + b
  | .int => fun a b => a + b
  | .bool => fun _ _ => false -- excluded by the primitive's numeric witness

noncomputable def DType.sub : (d : DType) → d.denote → d.denote → d.denote
  | .real => fun a b => a - b
  | .int => fun a b => a - b
  | .bool => fun _ _ => false -- excluded by the primitive's numeric witness

noncomputable def DType.mul : (d : DType) → d.denote → d.denote → d.denote
  | .real => fun a b => a * b
  | .int => fun a b => a * b
  | .bool => fun _ _ => false -- excluded by the primitive's numeric witness

inductive Args (ctx : List Ty) : List Ty → Type
  | nil : Args ctx []
  | cons : Atom ctx t → Args ctx ts → Args ctx (t :: ts)

noncomputable def Args.eval (env : Env ctx) : Args ctx ts → Env ts
  | .nil => .nil
  | .cons x xs => .cons (x.eval env) (xs.eval env)

/-- Read a variadic operand at the selected bounded coordinate. -/
def Env.read (env : Env types) : ((s : Shape) × Var types (d, s) × Index s) → d.denote
  | ⟨_, v, j⟩ =>
      let value := env.get v
      value j

theorem Env.read_dite (env : Env types) (p : Prop) [Decidable p]
    (yes : p → (s : Shape) × Var types (d, s) × Index s)
    (no : ¬p → (s : Shape) × Var types (d, s) × Index s) :
    env.read (if h : p then yes h else no h) =
      if h : p then env.read (yes h) else env.read (no h) := by
  split <;> rfl

/-- Jaxpr primitives. Layout maps are bounded; scatter plans are validated static metadata. -/
inductive Op (ctx : List Ty) : Ty → Type
  | iota (shape : Shape) (dimension : Nat) (valid : dimension < shape.length := by decide) : Op ctx (.real, shape)
  | exp : Atom ctx (.real, s) → Op ctx (.real, s)
  | log : Atom ctx (.real, s) → Op ctx (.real, s)
  | sqrt : Atom ctx (.real, s) → Op ctx (.real, s)
  | rsqrt : Atom ctx (.real, s) → Op ctx (.real, s)
  | sin : Atom ctx (.real, s) → Op ctx (.real, s)
  | cos : Atom ctx (.real, s) → Op ctx (.real, s)
  | tanh : Atom ctx (.real, s) → Op ctx (.real, s)
  | dot_general (left : Index t → Index k → Index s) (right : Index t → Index k → Index u) :
      Atom ctx (.real, s) → Atom ctx (.real, u) → Op ctx (.real, t)
  | reduce_sum (map : Index t → Index k → Index s) : Atom ctx (.real, s) → Op ctx (.real, t)
  | reduce_max (positive : 0 < n) (map : Index t → Fin n → Index s) : Atom ctx (.real, s) → Op ctx (.real, t)
  | reduce_min (positive : 0 < n) (map : Index t → Fin n → Index s) : Atom ctx (.real, s) → Op ctx (.real, t)
  | concatenate (args : Args ctx types)
      (select : Index t → (s : Shape) × Var types (d, s) × Index s) : Op ctx (d, t)
  | copy : Atom ctx (d, s) → Op ctx (d, s)
  | stop_gradient : Atom ctx (d, s) → Op ctx (d, s)
  | add (x : Atom ctx (d, s)) (y : Atom ctx (d, u))
      (left : broadcastValid s t (List.range s.length) := by decide)
      (right : broadcastValid u t (List.range u.length) := by decide)
      (numeric : d ≠ .bool := by decide) : Op ctx (d, t)
  | sub (x : Atom ctx (d, s)) (y : Atom ctx (d, u))
      (left : broadcastValid s t (List.range s.length) := by decide)
      (right : broadcastValid u t (List.range u.length) := by decide)
      (numeric : d ≠ .bool := by decide) : Op ctx (d, t)
  | neg : Atom ctx (.real, s) → Op ctx (.real, s)
  | square : Atom ctx (.real, s) → Op ctx (.real, s)
  | integer_pow (power : Int) : Atom ctx (.real, s) → Op ctx (.real, s)
  | abs : Atom ctx (.real, s) → Op ctx (.real, s)
  | min (x : Atom ctx (.real, s)) (y : Atom ctx (.real, u))
      (left : broadcastValid s t (List.range s.length) := by decide)
      (right : broadcastValid u t (List.range u.length) := by decide) : Op ctx (.real, t)
  | max (x : Atom ctx (.real, s)) (y : Atom ctx (.real, u))
      (left : broadcastValid s t (List.range s.length) := by decide)
      (right : broadcastValid u t (List.range u.length) := by decide) : Op ctx (.real, t)
  | mul (x : Atom ctx (d, s)) (y : Atom ctx (d, u))
      (left : broadcastValid s t (List.range s.length) := by decide)
      (right : broadcastValid u t (List.range u.length) := by decide)
      (numeric : d ≠ .bool := by decide) : Op ctx (d, t)
  | div (x : Atom ctx (.real, s)) (y : Atom ctx (.real, u))
      (left : broadcastValid s t (List.range s.length) := by decide)
      (right : broadcastValid u t (List.range u.length) := by decide) : Op ctx (.real, t)

  | broadcast_in_dim (shape : Shape) (broadcast_dimensions : List Nat) (x : Atom ctx (d, s))
      (valid : broadcastValid s shape broadcast_dimensions := by decide) : Op ctx (d, shape)
  | transpose (map : Index t → Index s) : Atom ctx (d, s) → Op ctx (d, t)
  | squeeze (map : Index t → Index s) : Atom ctx (d, s) → Op ctx (d, t)
  | slice (map : Index t → Index s) : Atom ctx (d, s) → Op ctx (d, t)
  | rev (map : Index t → Index s) : Atom ctx (d, s) → Op ctx (d, t)
  | reshape (map : Index t → Index s) : Atom ctx (d, s) → Op ctx (d, t)

  | scatter_add (plan : List (Index s × Index u)) : Atom ctx (.real, s) → Atom ctx (.real, u) → Op ctx (.real, s)
  | scatter (plan : List (Index s × Index u)) : Atom ctx (.real, s) → Atom ctx (.real, u) → Op ctx (.real, s)
  | eq (x : Atom ctx (d, s)) (y : Atom ctx (d, u))
      (left : broadcastValid s t (List.range s.length) := by decide)
      (right : broadcastValid u t (List.range u.length) := by decide) : Op ctx (.bool, t)
  | ne (x : Atom ctx (d, s)) (y : Atom ctx (d, u))
      (left : broadcastValid s t (List.range s.length) := by decide)
      (right : broadcastValid u t (List.range u.length) := by decide) : Op ctx (.bool, t)
  | lt (x : Atom ctx (d, s)) (y : Atom ctx (d, u))
      (left : broadcastValid s t (List.range s.length) := by decide)
      (right : broadcastValid u t (List.range u.length) := by decide) : Op ctx (.bool, t)
  | le (x : Atom ctx (d, s)) (y : Atom ctx (d, u))
      (left : broadcastValid s t (List.range s.length) := by decide)
      (right : broadcastValid u t (List.range u.length) := by decide) : Op ctx (.bool, t)
  | gt (x : Atom ctx (d, s)) (y : Atom ctx (d, u))
      (left : broadcastValid s t (List.range s.length) := by decide)
      (right : broadcastValid u t (List.range u.length) := by decide) : Op ctx (.bool, t)
  | ge (x : Atom ctx (d, s)) (y : Atom ctx (d, u))
      (left : broadcastValid s t (List.range s.length) := by decide)
      (right : broadcastValid u t (List.range u.length) := by decide) : Op ctx (.bool, t)

  | and (x : Atom ctx (.bool, s)) (y : Atom ctx (.bool, u))
      (left : broadcastValid s t (List.range s.length) := by decide)
      (right : broadcastValid u t (List.range u.length) := by decide) : Op ctx (.bool, t)
  | or (x : Atom ctx (.bool, s)) (y : Atom ctx (.bool, u))
      (left : broadcastValid s t (List.range s.length) := by decide)
      (right : broadcastValid u t (List.range u.length) := by decide) : Op ctx (.bool, t)
  | xor (x : Atom ctx (.bool, s)) (y : Atom ctx (.bool, u))
      (left : broadcastValid s t (List.range s.length) := by decide)
      (right : broadcastValid u t (List.range u.length) := by decide) : Op ctx (.bool, t)
  | not : Atom ctx (.bool, s) → Op ctx (.bool, s)
  | select_n (condition : Atom ctx (.bool, c)) (no : Atom ctx (d, s)) (yes : Atom ctx (d, u))
      (predicate : broadcastValid c t (List.range c.length) := by decide)
      (left : broadcastValid s t (List.range s.length) := by decide)
      (right : broadcastValid u t (List.range u.length) := by decide) : Op ctx (d, t)
  | gather (map : Tensor Int32 u → Index t → Index s) : Atom ctx (.real, s) → Atom ctx (.int, u) → Op ctx (.real, t)
  | convert_element_type (new_dtype : DType) (x : Atom ctx (d, s))
      (conversion : Conversion d new_dtype := by first | exact .identity | exact .intToReal) : Op ctx (new_dtype, s)

noncomputable def Op.eval (env : Env ctx) : Op ctx t → Value t
  | .iota shape dimension _ => fun i => ((coordinate shape i dimension).val : ℝ)
  | .exp x =>
      let xv := x.eval env
      fun i => Real.exp (xv i)
  | .log x =>
      let xv := x.eval env
      fun i => Real.log (xv i)
  | .sqrt x =>
      let xv := x.eval env
      fun i => Real.sqrt (xv i)
  | .rsqrt x =>
      let xv := x.eval env
      fun i => (Real.sqrt (xv i))⁻¹
  | .sin x =>
      let xv := x.eval env
      fun i => Real.sin (xv i)
  | .cos x =>
      let xv := x.eval env
      fun i => Real.cos (xv i)
  | .tanh x =>
      let xv := x.eval env
      fun i => (Real.exp (xv i) - Real.exp (-xv i)) /
      (Real.exp (xv i) + Real.exp (-xv i))
  | .dot_general left right x y =>
      let yv := y.eval env
      let xv := x.eval env
      fun i => ∑ j, xv (left i j) * yv (right i j)
  | .reduce_sum map x =>
      let xv := x.eval env
      fun i => ∑ j, xv (map i j)
  | .reduce_max positive map x =>
      let xv := x.eval env
      fun i =>
      (List.ofFn (fun j => xv (map i j))).foldl Max.max (xv (map i ⟨0, positive⟩))
  | .reduce_min positive map x =>
      let xv := x.eval env
      fun i =>
      (List.ofFn (fun j => xv (map i j))).foldl Min.min (xv (map i ⟨0, positive⟩))
  | .concatenate args select => fun i => (args.eval env).read (select i)
  | .stop_gradient x =>
      let xv := x.eval env
      xv
  | .copy x =>
      let xv := x.eval env
      xv
  | .add x y left right _ =>
      let xv := x.eval env
      let yv := y.eval env
      fun i => DType.add _ (xv (broadcastIndex _ _ _ left i)) (yv (broadcastIndex _ _ _ right i))
  | .sub x y left right _ =>
      let xv := x.eval env
      let yv := y.eval env
      fun i => DType.sub _ (xv (broadcastIndex _ _ _ left i)) (yv (broadcastIndex _ _ _ right i))
  | .neg x =>
      let xv := x.eval env
      fun i => -(xv i)
  | .square x =>
      let xv := x.eval env
      fun i => xv i ^ (2 : Nat)
  | .integer_pow p x =>
      let xv := x.eval env
      fun i => xv i ^ p
  | .abs x =>
      let xv := x.eval env
      fun i => |xv i|
  | .min x y left right =>
      let yv := y.eval env
      let xv := x.eval env
      fun i => Min.min (xv (broadcastIndex _ _ _ left i)) (yv (broadcastIndex _ _ _ right i))
  | .max x y left right =>
      let yv := y.eval env
      let xv := x.eval env
      fun i => Max.max (xv (broadcastIndex _ _ _ left i)) (yv (broadcastIndex _ _ _ right i))
  | .mul x y left right _ =>
      let xv := x.eval env
      let yv := y.eval env
      fun i => DType.mul _ (xv (broadcastIndex _ _ _ left i)) (yv (broadcastIndex _ _ _ right i))
  | .div x y left right =>
      let yv := y.eval env
      let xv := x.eval env
      fun i => xv (broadcastIndex _ _ _ left i) / yv (broadcastIndex _ _ _ right i)
  | @Op.broadcast_in_dim _ d source shape dims x h =>
      let xv := x.eval env
      fun i => xv (broadcastIndex source shape dims h i)
  | @Op.transpose _ target source d map x => fun i => (Atom.eval (t := (d, source)) env x) (map i)
  | @Op.squeeze _ target source d map x => fun i => (Atom.eval (t := (d, source)) env x) (map i)
  | @Op.slice _ target source d map x => fun i => (Atom.eval (t := (d, source)) env x) (map i)
  | @Op.rev _ target source d map x => fun i => (Atom.eval (t := (d, source)) env x) (map i)
  | @Op.reshape _ target source d map x => fun i => (Atom.eval (t := (d, source)) env x) (map i)

  | .scatter_add plan x update =>
      let xv := x.eval env
      let uv := update.eval env
      plan.foldl (fun tensor (position, source) => fun i => if i = position then tensor i + uv source else tensor i) xv
  | .scatter plan x update =>
      let xv := x.eval env
      let uv := update.eval env
      plan.foldl (fun tensor (position, source) => fun i => if i = position then uv source else tensor i) xv
  | .eq x y left right =>
      let xv := x.eval env
      let yv := y.eval env
      fun i => Comparison.scalar .eq _ (xv (broadcastIndex _ _ _ left i)) (yv (broadcastIndex _ _ _ right i))
  | .ne x y left right =>
      let xv := x.eval env
      let yv := y.eval env
      fun i => Comparison.scalar .ne _ (xv (broadcastIndex _ _ _ left i)) (yv (broadcastIndex _ _ _ right i))
  | .lt x y left right =>
      let xv := x.eval env
      let yv := y.eval env
      fun i => Comparison.scalar .lt _ (xv (broadcastIndex _ _ _ left i)) (yv (broadcastIndex _ _ _ right i))
  | .le x y left right =>
      let xv := x.eval env
      let yv := y.eval env
      fun i => Comparison.scalar .le _ (xv (broadcastIndex _ _ _ left i)) (yv (broadcastIndex _ _ _ right i))
  | .gt x y left right =>
      let xv := x.eval env
      let yv := y.eval env
      fun i => Comparison.scalar .gt _ (xv (broadcastIndex _ _ _ left i)) (yv (broadcastIndex _ _ _ right i))
  | .ge x y left right =>
      let xv := x.eval env
      let yv := y.eval env
      fun i => Comparison.scalar .ge _ (xv (broadcastIndex _ _ _ left i)) (yv (broadcastIndex _ _ _ right i))

  | .and x y left right =>
      let yv := y.eval env
      let xv := x.eval env
      fun i => (xv (broadcastIndex _ _ _ left i)) && (yv (broadcastIndex _ _ _ right i))
  | .or x y left right =>
      let yv := y.eval env
      let xv := x.eval env
      fun i => (xv (broadcastIndex _ _ _ left i)) || (yv (broadcastIndex _ _ _ right i))
  | .xor x y left right =>
      let yv := y.eval env
      let xv := x.eval env
      fun i => Bool.xor (xv (broadcastIndex _ _ _ left i)) (yv (broadcastIndex _ _ _ right i))
  | .not x =>
      let xv := x.eval env
      fun i => !(xv i)
  | .select_n c no yes predicate left right =>
      let cv := c.eval env
      let nv := no.eval env
      let yv := yes.eval env
      fun i => if cv (broadcastIndex _ _ _ predicate i)
        then yv (broadcastIndex _ _ _ right i) else nv (broadcastIndex _ _ _ left i)
  | @Op.gather _ _u target source map x indices => fun i =>
      (Atom.eval (t := (.real, source)) env x) (map (indices.eval env) i)
  | .convert_element_type _ x conversion =>
      let xv := x.eval env
      fun i => conversion.eval (xv i)

inductive Program : List Ty → Ty → Type
  | ret : Atom ctx t → Program ctx t
  | bind : Op ctx t → Program (t :: ctx) u → Program ctx u
  | call : Program args t → Args ctx args → Program (t :: ctx) u → Program ctx u

noncomputable def Program.eval (env : Env ctx) : Program ctx t → Value t
  | .ret x => x.eval env
  | .bind op rest =>
      let value := op.eval env
      let next := Env.cons value env
      rest.eval next
  | .call body args rest =>
      let value := body.eval (args.eval env)
      let next := Env.cons value env
      rest.eval next

end JaxLean.Jaxpr
