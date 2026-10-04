"""Per-program certificates relative to an independently defined Lean IR evaluator.

The importer below is still trusted to faithfully record the input Jaxpr. It
never reads the generated function to construct its IR or computes an expected
answer. Lean checks equality of the independent IR and the normal translation.
"""
from fractions import Fraction
import re

from jax.extend import core
import numpy as np

from .translate import TranslationError, translate, shape, kind, lean_shape, argument_labels


class Importer:
    def __init__(self, closed, calls=None):
        self.calls = calls or {}
        if not isinstance(closed, core.ClosedJaxpr):
            raise TranslationError("certify requires a ClosedJaxpr")
        self.jp = closed.jaxpr
        if closed.consts or self.jp.constvars:
            raise TranslationError("certified fragment does not yet support captured array constants")
        if self.jp.effects or len(self.jp.outvars) != 1:
            raise TranslationError("certified fragment requires one output and no effects")
        self.env = {v: i for i, v in enumerate(self.jp.invars)}
        self.lines = []
        for v in self.jp.invars:
            self.check_real(v)

    @staticmethod
    def check_real(v):
        if kind(v.aval) != "real":
            raise TranslationError("certified fragment supports floating inputs, literals and outputs only")
        shape(v.aval)

    def atom(self, v):
        if isinstance(v, core.Literal):
            self.check_real(v)
            if shape(v.aval) or not np.isfinite(v.val):
                raise TranslationError("certified literals must be finite scalars")
            q = Fraction(float(v.val))
            return f"(.literal ({q.numerator}) {q.denominator} (by decide))"
        idx = self.env[v]
        ref = ".here"
        for _ in range(idx):
            ref = f"(.there {ref})"
        return f"(.var {ref})"

    def emit(self, op, out):
        self.lines.append(f"  .bind ({op}) <|")
        self.env = {v: i + 1 for v, i in self.env.items()}
        self.env[out] = 0

    def promote(self, v, target):
        s = shape(v.aval)
        if s == target:
            return v
        if not s:
            op = f".broadcast {lean_shape(target)}"
        elif len(s) == len(target) and s[0] == 1 and s[1:] == target[1:]:
            op = f".expandFirst {target[0]}"
        elif len(s) == 2 and s[1] == 1 and len(target) == 2 and s[0] == target[0]:
            op = f".expandLast {target[1]}"
        else:
            raise TranslationError("certified implicit broadcasting supports scalars, a singleton leading axis, or a rank-two singleton last axis")
        temp = object()
        self.emit(f"{op} {self.atom(v)}", temp)
        return temp

    def run(self):
        for i, eq in enumerate(self.jp.eqns):
            op = eq.primitive.name
            if eq.effects or len(eq.outvars) != 1:
                raise TranslationError(f"certified equation {i}: expected one pure result")
            out = eq.outvars[0]
            self.check_real(out)
            for v in eq.invars:
                self.check_real(v)
            s = shape(out.aval)
            if op == "jit" and id(eq.params["jaxpr"]) in self.calls:
                args = ".nil"
                for v in reversed(eq.invars):
                    args = f"(.cons {self.atom(v)} {args})"
                callee = self.calls[id(eq.params["jaxpr"])]
                self.lines.append(f"  .call {callee}_ir {args} <|")
                self.env = {v: i + 1 for v, i in self.env.items()}
                self.env[out] = 0
                continue
            if op in ("add", "add_any", "sub", "mul", "div", "min", "max"):
                a, b = (self.promote(v, s) for v in eq.invars)
                expr = f".{'add' if op == 'add_any' else op} {self.atom(a)} {self.atom(b)}"
            elif op in ("neg", "square", "abs"):
                expr = f".{op} {self.atom(eq.invars[0])}"
            elif op == "integer_pow":
                expr = f".integerPow ({int(eq.params['y'])}) {self.atom(eq.invars[0])}"
            elif op in ("copy", "stop_gradient", "convert_element_type"):
                # check_real above restricts this to the real model's float-to-float identity.
                expr = f".copy {self.atom(eq.invars[0])}"
            elif op == "broadcast_in_dim":
                src = shape(eq.invars[0].aval)
                dims = tuple(eq.params["broadcast_dimensions"])
                if not src and not dims:
                    expr = f".broadcast {lean_shape(s)} {self.atom(eq.invars[0])}"
                elif s == src and dims == tuple(range(len(src))):
                    expr = f".copy {self.atom(eq.invars[0])}"
                elif s[1:] == src and dims == tuple(range(1, len(s))):
                    expr = f".prepend {s[0]} {self.atom(eq.invars[0])}"
                elif (len(s) == len(src) and src[0] == 1 and s[1:] == src[1:]
                      and dims == tuple(range(len(src)))):
                    expr = f".expandFirst {s[0]} {self.atom(eq.invars[0])}"
                elif len(src) == 1 and s == (src[0], 1) and dims == (0,):
                    expr = f".appendOne {self.atom(eq.invars[0])}"
                elif len(src) == len(s) == 2 and src[1] == 1 and src[0] == s[0] and dims == (0, 1):
                    expr = f".expandLast {s[1]} {self.atom(eq.invars[0])}"
                else:
                    raise TranslationError("no certificate rule for this broadcast layout")
            elif op == "reshape":
                if eq.params.get("dimensions") is not None:
                    raise TranslationError("certified reshape does not yet support dimension permutations")
                expr = f".reshape {lean_shape(s)} (by decide) {self.atom(eq.invars[0])}"
            elif op == "rev":
                if len(s) != 1 or tuple(eq.params["dimensions"]) != (0,):
                    raise TranslationError("certified rev supports one vector axis only")
                expr = f".reverseVec {self.atom(eq.invars[0])}"
            elif op == "transpose":
                perm = tuple(eq.params["permutation"])
                if perm == tuple(range(len(s))):
                    expr = f".copy {self.atom(eq.invars[0])}"
                elif perm == (1, 0):
                    expr = f".transpose2 {self.atom(eq.invars[0])}"
                else:
                    raise TranslationError("certified transpose supports rank-two swap or identity")
            elif op == "dot_general":
                (lc, rc), (lb, rb) = eq.params["dimension_numbers"]
                a, b = eq.invars
                if lb or rb:
                    raise TranslationError("batched dot_general has no certificate rule")
                if len(shape(a.aval)) == len(shape(b.aval)) == 2 and tuple(lc) == (1,) and tuple(rc) == (0,):
                    expr = f".matmul {self.atom(a)} {self.atom(b)}"
                elif len(shape(a.aval)) == 1 and len(shape(b.aval)) == 2 and tuple(lc) == (0,) and tuple(rc) == (0,):
                    expr = f".vecmat {self.atom(a)} {self.atom(b)}"
                elif len(shape(a.aval)) == len(shape(b.aval)) == 1 and tuple(lc) == tuple(rc) == (0,):
                    expr = f".dotVec {self.atom(a)} {self.atom(b)}"
                else:
                    raise TranslationError("no certificate rule for this contraction")
            elif op == "reduce_sum":
                axes = tuple(eq.params["axes"])
                if axes == (0,):
                    expr = f".sumFirst {self.atom(eq.invars[0])}"
                elif axes == (1,) and len(shape(eq.invars[0].aval)) == 2:
                    expr = f".sumLast {self.atom(eq.invars[0])}"
                else:
                    raise TranslationError("certified reduction supports the leading axis or rank-two trailing axis")
            else:
                raise TranslationError(f"primitive {op!r} has no translation certificate rule")
            self.emit(expr, out)
        self.check_real(self.jp.outvars[0])
        return "\n".join([*self.lines, f"  .ret {self.atom(self.jp.outvars[0])}"])


def certify(closed_jaxpr, *, name="program", namespace="Generated", readable=False, _calls=None):
    """Emit the normal tensor translation, imported IR and a Lean proof obligation.

    The returned source is certified only after Lean successfully checks it.
    No fallback: operations outside the certificate fragment raise an error even
    if the ordinary transpiler supports them. No Python source is inspected.
    """
    importer = Importer(closed_jaxpr, _calls)
    # Run the ordinary translator (including its version and Jaxpr checks).
    target = translate(closed_jaxpr, name=name, namespace=namespace, readable=readable,
                       _certified_calls=_calls)
    ir_body = importer.run()
    inputs = closed_jaxpr.jaxpr.invars
    ctx = "[" + ", ".join(lean_shape(shape(v.aval)) for v in inputs) + "]"
    out = lean_shape(shape(closed_jaxpr.jaxpr.outvars[0].aval))
    labels = argument_labels(closed_jaxpr.jaxpr, readable)
    binders = " ".join(f"({labels[i]} : Tensor ℝ {lean_shape(shape(v.aval))})" for i, v in enumerate(inputs))
    env = ".nil"
    for i in reversed(range(len(inputs))):
        env = f"(.cons {labels[i]} {env})"
    args = " ".join(labels)
    ns = ".".join(f"«{p}»" for p in namespace.split("."))
    pattern = "⟨⟩"
    for i in reversed(range(len(shape(closed_jaxpr.jaxpr.outvars[0].aval)))):
        pattern = f"⟨j{i}, {pattern}⟩"
    call_rules = "".join(f", {n}_translation_correct" for n in dict.fromkeys((_calls or {}).values()))
    certificate = f"""
-- IMPORTED IR: the Python importer is trusted to encode the original Jaxpr.
namespace {ns}
def {name}_ir : Jaxpr.Program {ctx} {out} :=
{ir_body}

-- Relative to Jaxpr.Program.eval's real-arithmetic semantics, for every input.
set_option linter.unusedSimpArgs false in
theorem {name}_translation_correct {binders} :
    Jaxpr.Program.eval {env} {name}_ir = {name} (R := ℝ) {args} := by
  funext i
  rcases i with {pattern}
  jaxpr_certificate [{name}_ir, {name}{call_rules}]

end {ns}
"""
    return "import JaxLean.Certificate\n" + target + certificate


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
