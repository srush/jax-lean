"""Per-program certificates relative to an independently defined Lean IR evaluator.

The importers in `importers/` are still trusted to faithfully record the input
Jaxpr. They never read the generated function or compute an expected answer. Lean checks equality of the independent IR and the normal translation.
"""
import re

from jax.extend import core

from .translate import translate
from .jaxpr import TranslationError, shape, lean_shape, argument_labels
from .importers.real import RealImporter
from .importers.typed import needs_typed, TypedImporter, ty, binder_type


def certify(closed_jaxpr, *, name="program", namespace="Generated", readable=False, _calls=None):
    """Emit the normal tensor translation, imported IR and a Lean proof obligation.

    The returned source is certified only after Lean successfully checks it.
    No fallback: operations outside the certificate fragment raise an error even
    if the ordinary transpiler supports them. No Python source is inspected.
    """
    if not isinstance(closed_jaxpr, core.ClosedJaxpr):
        raise TranslationError("certify requires a ClosedJaxpr")
    typed = needs_typed(closed_jaxpr)
    importer = (TypedImporter if typed else RealImporter)(closed_jaxpr, _calls)
    # Run the ordinary translator (including its version and Jaxpr checks).
    target = translate(closed_jaxpr, name=name, namespace=namespace, readable=readable,
                       _certified_calls=_calls)
    ir_body = importer.run()
    inputs = closed_jaxpr.jaxpr.invars
    ir_type = ty if typed else lambda aval: lean_shape(shape(aval))
    ir_namespace = "TypedJaxpr" if typed else "Jaxpr"
    ctx = "[" + ", ".join(ir_type(v.aval) for v in inputs) + "]"
    out = ir_type(closed_jaxpr.jaxpr.outvars[0].aval)
    labels = argument_labels(closed_jaxpr.jaxpr, readable)
    binders = " ".join(f"({labels[i]} : {binder_type(v.aval)})" for i, v in enumerate(inputs))
    env = ".nil"
    for i in reversed(range(len(inputs))):
        env = f"(.cons {labels[i]} {env})"
    args = " ".join(labels)
    ns = ".".join(f"«{p}»" for p in namespace.split("."))
    pattern = "⟨⟩"
    for i in reversed(range(len(shape(closed_jaxpr.jaxpr.outvars[0].aval)))):
        pattern = f"⟨j{i}, {pattern}⟩"
    call_rules = "".join(f", {n}_translation_correct" for n in dict.fromkeys((_calls or {}).values()))
    if typed:
        call_rules += ", TypedJaxpr.Program.eval, TypedJaxpr.Args.eval, TypedJaxpr.RealArgs.eval, TypedJaxpr.Op.eval, TypedJaxpr.Atom.eval, TypedJaxpr.Env.get, TypedJaxpr.Comparison.eval, TypedJaxpr.Comparison.intEval"
    reference = f"«{name}»"
    # Reduce SSA bookkeeping before rewriting arithmetic. This avoids repeated
    # simplifier traversal of the growing mixed-type environment at every bind.
    prepare = (f"  dsimp only [{name}_ir, {reference}, TypedJaxpr.Program.eval, "
               "TypedJaxpr.Args.eval, TypedJaxpr.RealArgs.eval, TypedJaxpr.Op.eval, "
               "TypedJaxpr.Atom.eval, TypedJaxpr.Env.get, Jaxpr.Program.eval, "
               "Jaxpr.Args.eval, Jaxpr.Op.eval, Jaxpr.Atom.eval, Jaxpr.Env.get]\n"
               if typed else "")
    proof = prepare + f"  jaxpr_certificate [{name}_ir, {reference}{call_rules}]"
    certificate = f"""
-- IMPORTED IR: the Python importer is trusted to encode the original Jaxpr.
namespace {ns}
def {name}_ir : {ir_namespace}.Program {ctx} {out} :=
{ir_body}

-- Relative to Jaxpr.Program.eval's real-arithmetic semantics, for every input.
set_option linter.unusedSimpArgs false in
theorem {name}_translation_correct {binders} :
    {ir_namespace}.Program.eval {env} {name}_ir = {reference} (R := ℝ) {args} := by
  funext i
  rcases i with {pattern}
{proof}

end {ns}
"""
    return "import JaxLean.Verification.Certificate\n" + target + certificate


def certify_module(closed_jaxpr, *, name="program", namespace="Generated"):
    """Certify a call graph, retaining pure single-result jit boundaries.

    Names and argument labels come only from Jaxpr metadata, never Python source.
    Children are certified before parents. Each parent uses child certificates,
    without unfolding child definitions. All shapes remain trace-specialized.
    """
    calls, used, pieces = {}, {name, name + "_ir", name + "_translation_correct"}, []

    def visit(closed, label):
        if not isinstance(closed, core.ClosedJaxpr):
            raise TranslationError("certify_module requires ClosedJaxpr call bodies")
        for eq in closed.jaxpr.eqns:
            if eq.primitive.name != "jit":
                continue
            child = eq.params["jaxpr"]
            if id(child) in calls:
                continue
            stem = re.sub(r"[^A-Za-z0-9_]", "_", eq.params.get("name", "call"))
            if not stem or not stem[0].isalpha():
                stem = "fn_" + stem
            child_name, suffix = stem, 2
            # Reserve generated symbols as well as user labels.
            while any(x in used for x in (child_name, child_name + "_ir", child_name + "_translation_correct")):
                child_name = f"{stem}_{suffix}"
                suffix += 1
            used.update((child_name, child_name + "_ir", child_name + "_translation_correct"))
            visit(child, child_name)
            calls[id(child)] = child_name
        direct = {id(eq.params["jaxpr"]): calls[id(eq.params["jaxpr"])]
                  for eq in closed.jaxpr.eqns if eq.primitive.name == "jit"}
        pieces.append(certify(closed, name=label, namespace=namespace,
                              readable=True, _calls=direct))

    visit(closed_jaxpr, name)
    imports = []
    bodies = []
    for piece in pieces:
        lines = []
        for line in piece.splitlines():
            if line.startswith("import "):
                if line not in imports:
                    imports.append(line)
            else:
                lines.append(line)
        bodies.append("\n".join(lines))
    return "\n".join(imports) + "\n\n" + "\n\n".join(bodies) + "\n"
