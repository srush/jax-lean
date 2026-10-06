import JaxLean.Core.Jaxpr
import Lean

/-! Named SSA syntax, elaborated into the existing typed Jaxpr. -/
namespace JaxLean.Jaxpr
open Lean

declare_syntax_cat jaxprBinder
syntax ident " : " term : jaxprBinder
declare_syntax_cat jaxprStep
syntax jaxprBinder " := " term ";" : jaxprStep
syntax jaxprBinder " := " "call " term " with " term ";" : jaxprStep
syntax "jaxpr% " "(" jaxprBinder,* ")" "{" jaxprStep* "return " term "}" : term

private def binderName (b : TSyntax `jaxprBinder) : MacroM Ident := do
  let `(jaxprBinder| $name:ident : $_) := b | Macro.throwErrorAt b "expected typed variable"
  return name

private def binderContext (binders : Array (TSyntax `jaxprBinder)) : MacroM Term := do
  let types ← binders.mapM fun b => do
    let `(jaxprBinder| $_:ident : $t) := b | Macro.throwErrorAt b "expected typed variable"
    pure t
  `([$types,*])

private partial def references (name : Name) : Syntax → Bool
  | .ident _ _ id _ => id == name
  | .node _ _ args => args.any (references name)
  | _ => false

private def bindNames (binders : Array (TSyntax `jaxprBinder)) (body : Term) : MacroM Term := do
  let mut seen : Array Name := #[]
  for b in binders do
    let name ← binderName b
    if seen.contains name.getId then
      Macro.throwErrorAt name "duplicate SSA variable"
    seen := seen.push name.getId
  let types ← binders.mapM fun b => do
    let `(jaxprBinder| $_:ident : $t) := b | Macro.throwErrorAt b "expected typed variable"
    pure t
  let ctx ← `([$types,*])
  let mut result := body
  for i in (List.range binders.size).reverse do
    let name ← binderName binders[i]!
    if !references name.getId body.raw then continue
    let mut ref ← `(JaxLean.Jaxpr.Var.here)
    for _ in [:i] do
      ref ← `(JaxLean.Jaxpr.Var.there $ref)
    let t := types[i]!
    result ← `(let $name : JaxLean.Jaxpr.Atom $ctx $t := .var $ref; $result)
  return result

macro_rules
  | `(jaxpr% ($binders:jaxprBinder,*) { return $result }) => do
    let ctx ← binderContext binders.getElems
    bindNames binders.getElems (← `((JaxLean.Jaxpr.Program.ret $result : JaxLean.Jaxpr.Program $ctx _)))
  | `(jaxpr% ($binders:jaxprBinder,*) { $b:jaxprBinder := $op; $rest:jaxprStep* return $result }) => do
    let name ← binderName b
    for previous in binders.getElems do
      if (← binderName previous).getId == name.getId then
        Macro.throwErrorAt name "duplicate SSA variable"
    let `(jaxprBinder| $_:ident : $t) := b | Macro.throwErrorAt b "expected typed variable"
    let ctx ← binderContext binders.getElems
    let operation ← bindNames binders.getElems (← `(($op : JaxLean.Jaxpr.Op $ctx $t)))
    `(JaxLean.Jaxpr.Program.bind $operation
      (jaxpr% ($b, $binders,*) { $rest* return $result }))
  | `(jaxpr% ($binders:jaxprBinder,*) { $b:jaxprBinder := call $callee with $args; $rest:jaxprStep* return $result }) => do
    let name ← binderName b
    for previous in binders.getElems do
      if (← binderName previous).getId == name.getId then
        Macro.throwErrorAt name "duplicate SSA variable"
    let `(jaxprBinder| $_:ident : $t) := b | Macro.throwErrorAt b "expected typed variable"
    let ctx ← binderContext binders.getElems
    let arguments ← bindNames binders.getElems (← `(($args : JaxLean.Jaxpr.Args $ctx _)))
    `(JaxLean.Jaxpr.Program.call (t := $t) $callee $arguments
      (jaxpr% ($b, $binders,*) { $rest* return $result }))

end JaxLean.Jaxpr
