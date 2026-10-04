"""A shallow embedding: one Jaxpr equation becomes one typed Lean let-binding.

Python handles static shapes; Lean checks the resulting coordinate maps. No JAX
primitive is executed by this translator. See README.md for the trust boundary.
"""
from dataclasses import dataclass
from fractions import Fraction
import re

import jax
from jax.extend import core
import numpy as np


class TranslationError(ValueError):
    """The input is outside the explicitly supported real-valued fragment."""


def shape(aval):
    if not hasattr(aval, "shape"):
        raise TranslationError(f"unsupported abstract value {aval}")
    s = tuple(aval.shape)
    if any(not isinstance(n, int) or n < 0 for n in s):
        raise TranslationError(f"static, nonnegative dimensions required: {s}")
    return s


def kind(aval):
    try:
        dtype = np.dtype(aval.dtype)
    except TypeError as e:
        raise TranslationError(f"unsupported dtype {aval.dtype}") from e
    if dtype.kind == "b":
        return "bool"
    if dtype.kind == "f" and dtype.itemsize in (2, 4, 8):
        return "real"
    if dtype.kind in "iu":
        return "int"  # Only compile-time constants passed to a real conversion.
    raise TranslationError(f"unsupported dtype {dtype}; expected real floating values or bool")


def lean_shape(s):
    return "[" + ", ".join(map(str, s)) + "]"


def tensor_type(s, k):
    return f"Tensor {'Bool' if k == 'bool' else 'R'} {lean_shape(s)}"


def coords(s, var="i"):
    return [var + ".2" * k + ".1" for k in range(len(s))]


def index(xs):
    return "(" + ", ".join([*xs, "()"]) + ")" if xs else "()"


def number(x):
    if not np.isfinite(x):
        raise TranslationError("NaN and infinity have no value in the real semantics")
    q = Fraction(float(x))  # Exact represented binary float, not its printed decimal.
    return f"({q.numerator} : R)" if q.denominator == 1 else f"({q.numerator} / {q.denominator} : R)"


@dataclass(frozen=True)
class Value:
    expr: str
    shape: tuple
    kind: str

    def at(self, xs):
        return f"({self.expr} {index(xs)})"


class Emitter:
    def __init__(self, max_scan_length):
        self.lines = []
        self.counter = 0
        self.ordered = False
        self.transcendental = False
        self.max_scan_length = max_scan_length

    def let(self, expr, aval, comment):
        s, k = shape(aval), kind(aval)
        if k == "int":
            raise TranslationError("runtime integer arithmetic is not real arithmetic")
        name = f"v{self.counter}"
        self.counter += 1
        self.lines += [f"  -- {comment}", f"  let {name} : {tensor_type(s, k)} := {expr}"]
        return Value(name, s, k)

    def constant(self, value, aval):
        s, k = shape(aval), kind(aval)
        a = np.asarray(value)
        if a.shape != s:
            raise TranslationError(f"constant shape {a.shape} differs from its abstract shape {s}")
        def scalar(x):
            if k == "bool":
                return "true" if x else "false"
            if k == "int":
                return f"({int(x)} : R)"
            return number(x)
        if not s:
            return Value(f"(Tensor.scalar {scalar(a.item())})", s, k)
        xs = ", ".join(scalar(x) for x in a.flat)
        return Value(f"(Tensor.ofArray (s := {lean_shape(s)}) #[{xs}] (by rfl))", s, k)

    def program(self, closed, args, consts=None):
        jp = closed.jaxpr if isinstance(closed, core.ClosedJaxpr) else closed
        cs = closed.consts if isinstance(closed, core.ClosedJaxpr) else (() if consts is None else consts)
        if jp.effects:
            raise TranslationError(f"effects are unsupported: {jp.effects}")
        if len(cs) != len(jp.constvars) or len(args) != len(jp.invars):
            raise TranslationError("Jaxpr argument/constant arity mismatch")
        env = dict(zip(jp.invars, args, strict=True))
        env.update((v, self.constant(c, v.aval)) for v, c in zip(jp.constvars, cs, strict=True))

        def read(atom):
            return self.constant(atom.val, atom.aval) if isinstance(atom, core.Literal) else env[atom]

        for n, eq in enumerate(jp.eqns):
            op, p = eq.primitive.name, eq.params
            try:
                if eq.effects:
                    raise TranslationError(f"effects are unsupported: {eq.effects}")
                vs = [read(a) for a in eq.invars]
                if op in ("jit", "custom_jvp_call"):
                    nested = p["jaxpr" if op == "jit" else "call_jaxpr"]
                    self.lines.append(f"  -- begin {op} (inlined)")
                    outs = self.program(nested, vs)
                elif op == "scan":
                    outs = self.scan(p, vs, eq.outvars)
                else:
                    if len(eq.outvars) != 1:
                        raise TranslationError("only call/scan may have multiple results")
                    expr = self.primitive(op, p, vs, eq.outvars[0].aval)
                    outs = [self.let(expr, eq.outvars[0].aval, op)]
                if len(outs) != len(eq.outvars):
                    raise TranslationError("result arity mismatch")
                env.update(zip(eq.outvars, outs, strict=True))
            except (TranslationError, KeyError, IndexError) as e:
                raise TranslationError(f"equation {n} ({op}): {e}") from e
        return [read(v) for v in jp.outvars]

    def element(self, v, out_shape):
        # JAX elementwise primitives permit equal rank singleton broadcasting,
        # or scalars. NumPy rank promotion has already inserted broadcast_in_dim.
        if not v.shape:
            return v.at([])
        if len(v.shape) != len(out_shape):
            raise TranslationError("unexpected implicit rank promotion")
        if any(a != b and a != 1 for a, b in zip(v.shape, out_shape)):
            raise TranslationError("incompatible elementwise shapes")
        return v.at(["0" if n == 1 else c for n, c in zip(v.shape, coords(out_shape))])

    def primitive(self, op, p, vs, aval):
        s, k = shape(aval), kind(aval)
        if any(v.kind == "int" for v in vs) and op != "convert_element_type":
            raise TranslationError("integer values may only be constants converted to real values")
        if op in ("convert_element_type", "copy", "stop_gradient"):
            if op == "convert_element_type" and (k != "real" or vs[0].kind == "bool"):
                raise TranslationError("only float-to-float or constant integer-to-float conversions are supported")
            if s != vs[0].shape:
                raise TranslationError("conversion changed shape")
            return vs[0].expr

        binary = {"add": "+", "add_any": "+", "sub": "-", "mul": "*", "div": "/"}
        unary = {"neg": "fun x => -x", "square": "fun x => x ^ (2 : Nat)"}
        comparisons = {"eq": "=", "ne": "≠", "lt": "<", "le": "≤", "gt": ">", "ge": "≥"}
        real_ops = {"exp", "log", "sqrt", "sin", "cos", "tanh"}
        if op in binary:
            return f"fun i => {self.element(vs[0], s)} {binary[op]} {self.element(vs[1], s)}"
        if op in unary:
            return f"Tensor.map ({unary[op]}) {vs[0].expr}"
        if op == "integer_pow":
            return f"Tensor.map (fun x => x ^ ({int(p['y'])} : Int)) {vs[0].expr}"
        if op in real_ops or op == "rsqrt":
            self.transcendental = True
            f = "fun x => (RealOps.sqrt x)⁻¹" if op == "rsqrt" else f"RealOps.{op}"
            return f"Tensor.map ({f}) {vs[0].expr}"
        if op in comparisons:
            self.ordered = True
            return f"fun i => decide ({self.element(vs[0], s)} {comparisons[op]} {self.element(vs[1], s)})"
        if op in ("min", "max", "abs"):
            self.ordered = True
            return f"fun i => {op} " + " ".join(self.element(v, s) for v in vs)
        if op == "select_n":
            if vs[0].kind != "bool" or len(vs) != 3:
                raise TranslationError("select_n requires a boolean selector and two cases")
            c, no, yes = (self.element(v, s) for v in vs)
            return f"fun i => if {c} then {yes} else {no}"
        if op in ("and", "or", "xor", "not"):
            if any(v.kind != "bool" for v in vs):
                raise TranslationError("bitwise integer operations are unsupported")
            return "fun i => Bool." + op + " " + " ".join(self.element(v, s) for v in vs)

        if op == "reshape":
            x = vs[0].expr
            if p.get("dimensions") is not None:
                x = self.transpose(vs[0], tuple(p["dimensions"]))
            return f"Tensor.reshape (t := {lean_shape(s)}) (by decide) ({x})"
        if op == "transpose":
            return self.transpose(vs[0], tuple(p["permutation"]))
        if op == "broadcast_in_dim":
            cs = coords(s)
            axes = p["broadcast_dimensions"]
            ix = ["0" if n == 1 else cs[d] for n, d in zip(vs[0].shape, axes, strict=True)]
            return f"Tensor.reindex (s := {lean_shape(vs[0].shape)}) (fun i => {index(ix)}) {vs[0].expr}"
        if op == "squeeze":
            it = iter(coords(s))
            ix = ["0" if d in p["dimensions"] else next(it) for d in range(len(vs[0].shape))]
            return f"Tensor.reindex (s := {lean_shape(vs[0].shape)}) (fun i => {index(ix)}) {vs[0].expr}"
        if op == "slice":
            strides = p["strides"] or (1,) * len(s)
            ix = [f"⟨{start} + {stride} * {c}.val, by omega⟩"
                  for c, start, stride in zip(coords(s), p["start_indices"], strides, strict=True)]
            return f"Tensor.reindex (s := {lean_shape(vs[0].shape)}) (fun i => {index(ix)}) {vs[0].expr}"
        if op == "rev":
            ix = [f"{c}.rev" if d in p["dimensions"] else c for d, c in enumerate(coords(s))]
            return f"Tensor.reindex (s := {lean_shape(vs[0].shape)}) (fun i => {index(ix)}) {vs[0].expr}"
        if op == "concatenate":
            axis, start = p["dimension"], 0
            cs = coords(s)
            expr = ""
            for n, v in enumerate(vs):
                end = start + v.shape[axis]
                ix = cs.copy()
                ix[axis] = f"⟨{cs[axis]}.val - {start}, by omega⟩"
                term = v.at(ix)
                expr += (f"if h{n} : {cs[axis]}.val < {end} then {term} else "
                         if n + 1 < len(vs) else term)
                start = end
            return "fun i => " + expr
        if op in ("reduce_sum", "reduce_prod"):
            axes = tuple(p["axes"])
            remaining = iter(coords(s))
            ix = [f"k{axes.index(d)}" if d in axes else next(remaining)
                  for d in range(len(vs[0].shape))]
            body = vs[0].at(ix)
            symbol = "∑" if op == "reduce_sum" else "∏"
            for j, d in reversed(list(enumerate(axes))):
                body = f"{symbol} k{j} : Fin {vs[0].shape[d]}, {body}"
            return "fun i => " + body
        if op == "dot_general":
            return self.dot(p, vs, s)
        if op == "iota" and k == "real":
            return f"fun i => ({coords(s)[p['dimension']]}.val : R)"
        raise TranslationError(f"unsupported primitive {op!r}")

    def transpose(self, v, perm):
        cs = coords(perm)
        ix = [cs[perm.index(d)] for d in range(len(perm))]
        return f"Tensor.reindex (s := {lean_shape(v.shape)}) (t := {lean_shape(tuple(v.shape[d] for d in perm))}) (fun i => {index(ix)}) {v.expr}"

    def dot(self, p, vs, out_shape):
        (lc, rc), (lb, rb) = p["dimension_numbers"]
        lhs, rhs = vs
        cs = iter(coords(out_shape))
        li, ri = [None] * len(lhs.shape), [None] * len(rhs.shape)
        for l, r in zip(lb, rb, strict=True):
            li[l] = ri[r] = next(cs)
        for j, (l, r) in enumerate(zip(lc, rc, strict=True)):
            li[l] = ri[r] = f"k{j}"
        for ix in (li, ri):
            for d in range(len(ix)):
                if ix[d] is None:
                    ix[d] = next(cs)
        body = f"{lhs.at(li)} * {rhs.at(ri)}"
        for j, l in reversed(list(enumerate(lc))):
            body = f"∑ k{j} : Fin {lhs.shape[l]}, {body}"
        return "fun i => " + body

    def scan(self, p, vs, outvars):
        length, nc, nk = p["length"], p["num_consts"], p["num_carry"]
        if length > self.max_scan_length:
            raise TranslationError(f"scan length {length} exceeds unrolling limit {self.max_scan_length}")
        constants, carry, xs = vs[:nc], vs[nc:nc + nk], vs[nc + nk:]
        ys = [[None] * length for _ in outvars[nk:]]
        steps = reversed(range(length)) if p["reverse"] else range(length)
        for step in steps:
            self.lines.append(f"  -- scan step {step}")
            slices = [Value(f"(fun i => {v.expr} ({step}, i))", v.shape[1:], v.kind) for v in xs]
            out = self.program(p["jaxpr"], [*constants, *carry, *slices])
            carry = out[:nk]
            for col, y in zip(ys, out[nk:], strict=True):
                col[step] = y.expr
        return carry + [self.let("Tensor.stack #[" + ", ".join(col) + "] (by rfl)", v.aval, "scan stack")
                        for col, v in zip(ys, outvars[nk:], strict=True)]


def translate(jaxpr, *, name="program", namespace="Generated", consts=None, max_scan_length=32):
    """Return a standalone Lean module for a ClosedJaxpr (or Jaxpr + consts).

    Inputs/outputs follow Jaxpr's flattened order; pytrees are not reconstructed.
    An integer input, unknown primitive, effect or nonfinite literal fails closed.
    The pinned JAX version is intentional: this is a compiler boundary, not a
    best-effort pretty-printer for arbitrary versions of an evolving IR.
    """
    if jax.__version__ != "0.8.0":
        raise TranslationError(f"expected JAX 0.8.0, found {jax.__version__}")
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", name):
        raise TranslationError("name must be an ASCII Lean identifier")
    if not all(re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", n) for n in namespace.split(".")):
        raise TranslationError("namespace must be a dotted ASCII Lean identifier")
    if not isinstance(jaxpr, (core.ClosedJaxpr, core.Jaxpr)):
        raise TranslationError("expected a Jaxpr object, not its printed text")
    if isinstance(jaxpr, core.ClosedJaxpr) and consts is not None:
        raise TranslationError("ClosedJaxpr already contains its constants")
    jp = jaxpr.jaxpr if isinstance(jaxpr, core.ClosedJaxpr) else jaxpr
    # This is JAX's own structural/type checker. It is not a correctness proof.
    from jax._src.core import check_jaxpr
    check_jaxpr(jp)
    e = Emitter(max_scan_length)
    args = [Value(f"x{n}", shape(v.aval), kind(v.aval)) for n, v in enumerate(jp.invars)]
    if any(v.kind == "int" for v in args):
        raise TranslationError("integer inputs are unsupported; trace with floating inputs")
    outs = e.program(jaxpr, args, consts)
    if any(v.kind == "int" for v in outs):
        raise TranslationError("integer outputs are unsupported")
    classes = "[Field R]" + (" [LinearOrder R]" if e.ordered else "")
    classes += " [RealOps R]" if e.transcendental else ""
    binders = " ".join(f"({v.expr} : {tensor_type(v.shape, v.kind)})" for v in args)
    result_type = " × ".join(tensor_type(v.shape, v.kind) for v in outs) or "Unit"
    result = outs[0].expr if len(outs) == 1 else "(" + ", ".join(v.expr for v in outs) + ")"
    ns = ".".join(f"«{n}»" for n in namespace.split("."))
    return "\n".join([
        "-- Generated by jaxlean from JAX 0.8.0. Edit the source, not this file.",
        "-- Real arithmetic abstraction; no claim of IEEE-754 equivalence.",
        "import JaxLean.RealOps", "", "open JaxLean", "open scoped BigOperators",
        "set_option linter.unusedVariables false", f"namespace {ns}", "",
        f"def «{name}» {{R : Type}} {classes} {binders} : {result_type} :=",
        *e.lines, "  " + result, "", f"end {ns}", "",
    ])
