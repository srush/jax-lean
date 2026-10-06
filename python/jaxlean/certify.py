"""Per-program certificates relative to an independently defined Lean IR evaluator.

The importer in `importers/jaxpr.py` is trusted to faithfully record the input
Jaxpr. It never reads the generated function or computes an expected answer. Lean checks equality of the independent IR and the normal translation.
"""
import re

from jax.extend import core

from .translate import translate
from .jaxpr import TranslationError, shape, argument_labels
from .importers.jaxpr import JaxprImporter, ty, binder_type


def certify(closed_jaxpr, *, name="program", namespace="Generated", readable=False, _calls=None, named_vars=False):
    """Emit the normal tensor translation, imported IR and a Lean proof obligation.

    named_vars emits typed, named SSA syntax that expands to the same imported IR.
    The returned source is certified only after Lean successfully checks it.
    No fallback: operations outside the certificate fragment raise an error even
    if the ordinary transpiler supports them. No Python source is inspected.
    """
    if not isinstance(closed_jaxpr, core.ClosedJaxpr):
        raise TranslationError("certify requires a ClosedJaxpr")
    importer = JaxprImporter(closed_jaxpr, _calls, named_vars=named_vars)
    # Run the ordinary translator (including its version and Jaxpr checks).
    target = translate(closed_jaxpr, name=name, namespace=namespace, readable=readable,
                       _certified_calls=_calls)
    ir_body = importer.run()
    inputs = closed_jaxpr.jaxpr.invars
    ctx = "[" + ", ".join(ty(v.aval) for v in inputs) + "]"
    out = ty(closed_jaxpr.jaxpr.outvars[0].aval)
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
    # Qualify declarations so source argument names cannot shadow definitions.
    qualified = f"_root_.{ns}"
    reference = f"{qualified}.«{name}»"
    ir_reference = f"{qualified}.{name}_ir"
    call_rules = "".join(f", {qualified}.{n}_translation_correct"
                         for n in dict.fromkeys((_calls or {}).values()))
    proof = f"  jaxpr_certificate [{ir_reference}, {reference}{call_rules}]"
    depth_option = "set_option maxRecDepth 4096 in\n" if len(closed_jaxpr.jaxpr.eqns) >= 32 else ""
    linter_option = "set_option linter.unusedSimpArgs false in\n"
    certificate = f"""
-- IMPORTED IR: the Python importer is trusted to encode the original Jaxpr.
namespace {ns}
def {name}_ir : Jaxpr.Program {ctx} {out} :=
{ir_body}

-- Relative to Jaxpr.Program.eval's real-arithmetic semantics, for every input.
{depth_option}{linter_option}theorem {name}_translation_correct {binders} :
    Jaxpr.Program.eval {env} {ir_reference} = {reference} (R := ℝ) {args} := by
  funext i
  rcases i with {pattern}
{proof}

end {ns}
"""
    syntax_import = "import JaxLean.Verification.Syntax\n" if named_vars else ""
    return syntax_import + "import JaxLean.Verification.Certificate\n" + target + certificate


def certify_module(closed_jaxpr, *, name="program", namespace="Generated", named_vars=False):
    """Certify a call graph, retaining pure single-result jit boundaries.

    named_vars is propagated to every child certificate.
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
                              readable=True, _calls=direct, named_vars=named_vars))

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
