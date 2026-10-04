"""A shallow embedding: one Jaxpr equation becomes one typed Lean let-binding.

Python handles static shapes; Lean checks the resulting coordinate maps. No JAX
primitive is executed by this translator. See README.md for the trust boundary.
"""
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
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
    is_scalar: bool = False

    def at(self, xs):
        if self.is_scalar:
            return f"({self.expr})"
        return f"({self.expr} {index(xs)})"


_ARGUMENT_RESERVED = {"R", "draw", "draws", "sampling", "split_keys"}


def argument_labels(jp, readable=False):
    """Stable labels from Jaxpr debug metadata; no source inspection."""
    arg_names = getattr(jp.debug_info, "arg_names", None) or ()
    used = set(_ARGUMENT_RESERVED)
    labels = []
    for n, _ in enumerate(jp.invars):
        label = arg_names[n] if readable and n < len(arg_names) else f"x{n}"
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", label) or label in used:
            label = f"arg{n}"
        while label in used:
            label += "_"
        used.add(label)
        labels.append(f"«{label}»" if readable else label)
    return labels


class Emitter:
    def __init__(self, max_scan_length, *, scalar=False, readable=False, random_model=None, source_file=None, certified_calls=None):
        self.certified_calls = certified_calls or {}
        self.lines = []
        self.counter = 0
        self.ordered = False
        self.transcendental = False
        self.max_scan_length = max_scan_length
        self.scalar = scalar
        self.readable = readable
        self.random_model = random_model
        self.random_source = None
        self.split_count = None
        self.source_file = source_file
        self.used_names = set(_ARGUMENT_RESERVED)

    def let(self, expr, aval, comment):
        s, k = shape(aval), kind(aval)
        if k == "int":
            raise TranslationError("runtime integer arithmetic is not real arithmetic")
        if self.scalar and s:
            raise TranslationError("scalar presentation requires scalar intermediate values")
        name = f"v{self.counter}"
        if self.readable:
            base = {"convert_element_type": "cast", "dot_general": "matmul"}.get(comment, comment)
            base = re.sub(r"[^A-Za-z0-9_]", "_", base) + "_result"
            name = base
            suffix = 2
            while name in self.used_names:
                name = f"{base}_{suffix}"
                suffix += 1
        self.used_names.add(name)
        self.counter += 1
        ty = ("Bool" if k == "bool" else "R") if self.scalar else tensor_type(s, k)
        self.lines += [f"  -- {comment}", f"  let {name} : {ty} := {expr}"]
        return Value(name, s, k, self.scalar)

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
            if self.scalar:
                return Value(scalar(a.item()), s, k, True)
            return Value(f"(Tensor.scalar {scalar(a.item())})", s, k)
        if self.scalar:
            raise TranslationError("scalar presentation does not support array constants")
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
                if self.readable and self.source_file and eq.source_info.traceback:
                    frames = eq.source_info.traceback.frames
                    frame = next((f for f in frames if f.file_name == self.source_file), None)
                    if frame:
                        self.lines.append(f"  -- {Path(frame.file_name).name}:{frame.line_num} ({frame.function_name})")
                if self.random_model and op == "random_split":
                    if self.split_count is not None or self.random_source is not None:
                        raise TranslationError("one root-key split is supported; repeated splitting/key reuse is rejected")
                    split_shape = tuple(p["shape"])
                    if (len(vs) != 1 or vs[0].kind != "key" or vs[0].shape
                            or len(split_shape) != 1 or split_shape[0] <= 0):
                        raise TranslationError("random_split requires the scalar root key and one positive static count")
                    self.split_count = split_shape[0]
                    self.lines.append("  -- random_split: distinct children are independent under the selected product-law model")
                    outs = [Value("split_keys", split_shape, "split_key")]
                elif self.random_model and op == "jit" and p.get("name") == "_randint":
                    from .random_spec import uniform_bounds
                    if self.random_source is not None:
                        raise TranslationError("uniform model supports one draw call (scalar or vmapped); key reuse is rejected")
                    if len(vs) != 3 or vs[0].kind not in ("key", "split_key"):
                        raise TranslationError("randint requires the supplied typed key")
                    self.random_source = uniform_bounds(eq)
                    _, _, batch = self.random_source
                    if batch is None:
                        if vs[0].kind != "key" or self.split_count is not None:
                            raise TranslationError("cannot use the parent key after splitting")
                        self.lines.append("  -- randint: draw is supplied by the uniform sampling specification")
                        outs = [Value("draw" if self.scalar else "(Tensor.scalar draw)", (), "int", self.scalar)]
                    else:
                        if vs[0].kind != "split_key" or self.split_count != batch:
                            raise TranslationError("batched randint must consume the tracked split-key batch")
                        self.lines.append("  -- vmapped randint: draws come from Rand.iid, an explicit product law")
                        outs = [Value("(fun i => draws i.1)", (batch,), "int")]
                elif op == "jit" and id(p["jaxpr"]) in self.certified_calls:
                    callee = self.certified_calls[id(p["jaxpr"])]
                    self.ordered = True
                    if len(eq.outvars) != 1:
                        raise TranslationError("certified calls require one result")
                    expr = f"«{callee}» (R := R) " + " ".join(f"({v.expr})" for v in vs)
                    outs = [self.let(expr, eq.outvars[0].aval, f"call {callee}")]
                elif op in ("jit", "custom_jvp_call"):
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

    def elementwise(self, fn, vs, out_shape):
        """Preserve map/map₂ structure for reusable Lean proofs.

        Scalar operands become captured scalar values in a map. A singleton
        leading axis uses a named broadcast. Other broadcasting retains the
        existing coordinate semantics; no new claim of equivariance is made.
        """
        tensors = []
        arguments = []
        for v in vs:
            if not v.shape:
                arguments.append(v.at([]))
                continue
            if v.shape == out_shape:
                expr = v.expr
            elif (len(v.shape) == len(out_shape) and v.shape[0] == 1
                  and v.shape[1:] == out_shape[1:]):
                expr = f"(Tensor.broadcastFirst {out_shape[0]} {v.expr})"
            else:
                expr = f"(fun i => {self.element(v, out_shape)})"
            arguments.append(f"a{len(tensors)}")
            tensors.append(expr)
        body = fn(*arguments)
        if not tensors:
            return body if self.scalar else f"Tensor.scalar ({body})"
        binders = " ".join(f"a{i}" for i in range(len(tensors)))
        combinator = "Tensor.map" if len(tensors) == 1 else "Tensor.map₂"
        return f"{combinator} (fun {binders} => {body}) " + " ".join(tensors)

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
            return self.elementwise(lambda a, b: f"{a} {binary[op]} {b}", vs, s)
        if op in unary:
            if self.scalar:
                return f"({unary[op]}) {vs[0].at([])}"
            return f"Tensor.map ({unary[op]}) {vs[0].expr}"
        if op == "integer_pow":
            if self.scalar:
                return f"{vs[0].at([])} ^ ({int(p['y'])} : Int)"
            return f"Tensor.map (fun x => x ^ ({int(p['y'])} : Int)) {vs[0].expr}"
        if op in real_ops or op == "rsqrt":
            self.transcendental = True
            f = "fun x => (RealOps.sqrt x)⁻¹" if op == "rsqrt" else f"RealOps.{op}"
            if self.scalar:
                return f"({f}) {vs[0].at([])}"
            return f"Tensor.map ({f}) {vs[0].expr}"
        if op in comparisons:
            self.ordered = True
            return ("" if self.scalar else "fun i => ") + f"decide ({self.element(vs[0], s)} {comparisons[op]} {self.element(vs[1], s)})"
        if op in ("min", "max", "abs"):
            self.ordered = True
            return self.elementwise(lambda *args: op + " " + " ".join(args), vs, s)
        if op == "select_n":
            if vs[0].kind != "bool" or len(vs) != 3:
                raise TranslationError("select_n requires a boolean selector and two cases")
            c, no, yes = (self.element(v, s) for v in vs)
            return ("" if self.scalar else "fun i => ") + f"if {c} then {yes} else {no}"
        if op in ("and", "or", "xor", "not"):
            if any(v.kind != "bool" for v in vs):
                raise TranslationError("bitwise integer operations are unsupported")
            return ("" if self.scalar else "fun i => ") + "Bool." + op + " " + " ".join(self.element(v, s) for v in vs)

        if self.scalar:
            raise TranslationError(f"unsupported scalar primitive {op!r}; use tensor presentation for array operations")

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
            if op == "reduce_sum" and axes == (0,):
                return f"Tensor.sumFirst {vs[0].expr}"
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
        if (len(lhs.shape) == len(rhs.shape) == 2 and tuple(lc) == (1,)
                and tuple(rc) == (0,) and not lb and not rb):
            return f"Tensor.matmul {lhs.expr} {rhs.expr}"
        if (len(lhs.shape) == 1 and len(rhs.shape) == 2 and tuple(lc) == (0,)
                and tuple(rc) == (0,) and not lb and not rb):
            return f"Tensor.vecmat {lhs.expr} {rhs.expr}"
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


def translate(jaxpr, *, name="program", namespace="Generated", consts=None, max_scan_length=32,
              scalar=False, readable=False, random_model=None, _certified_calls=None):
    """Return a standalone Lean module for a ClosedJaxpr (or Jaxpr + consts).

    Inputs/outputs follow Jaxpr's flattened order; pytrees are not reconstructed.
    An integer input, unknown primitive, effect or nonfinite literal fails closed.
    The pinned JAX version is intentional: this is a compiler boundary, not a
    best-effort pretty-printer for arbitrary versions of an evolving IR.

    scalar=True emits R instead of Tensor R [] for entirely scalar programs.
    readable=True uses Python argument names and source locations when available.
    random_model='uniform' opts into a checked-call, ideal scalar/vmapped randint spec;
    it does not claim to verify JAX's PRNG or machine floating-point behavior.
    """
    if jax.__version__ != "0.8.0":
        raise TranslationError(f"expected JAX 0.8.0, found {jax.__version__}")
    if random_model not in (None, "uniform"):
        raise TranslationError("random_model must be None or 'uniform'")
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", name):
        raise TranslationError("name must be an ASCII Lean identifier")
    if not all(re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", n) for n in namespace.split(".")):
        raise TranslationError("namespace must be a dotted ASCII Lean identifier")
    if not isinstance(jaxpr, (core.ClosedJaxpr, core.Jaxpr)):
        raise TranslationError("expected a Jaxpr object, not its printed text")
    if isinstance(jaxpr, core.ClosedJaxpr) and consts is not None:
        raise TranslationError("ClosedJaxpr already contains its constants")
    jp = jaxpr.jaxpr if isinstance(jaxpr, core.ClosedJaxpr) else jaxpr
    if random_model:
        from .random_spec import has_user_arrays
        scalar = not has_user_arrays(jp)
    # This is JAX's own structural/type checker. It is not a correctness proof.
    from jax._src.core import check_jaxpr
    check_jaxpr(jp)
    debug = jp.debug_info
    source = getattr(debug, "func_src_info", "") or ""
    source_file = source.rsplit(" at ", 1)[-1].rsplit(":", 1)[0] if " at " in source else None
    e = Emitter(max_scan_length, scalar=scalar, readable=readable,
                random_model=random_model, source_file=source_file, certified_calls=_certified_calls)
    labels = argument_labels(jp, readable)
    args = []
    for n, v in enumerate(jp.invars):
        is_key = jax.dtypes.issubdtype(v.aval.dtype, jax.dtypes.prng_key)
        k = "key" if is_key and random_model else kind(v.aval)
        s = shape(v.aval)
        if k == "key" and s:
            raise TranslationError("uniform model requires a scalar root key, not an input batch of keys")
        if scalar and s:
            raise TranslationError("scalar presentation requires scalar inputs")
        label = labels[n]
        e.used_names.add(label.strip("«»"))
        args.append(Value(label, s, k, scalar))
    if any(v.kind == "int" for v in args):
        raise TranslationError("integer inputs are unsupported; trace with floating inputs")
    outs = e.program(jaxpr, args, consts)
    if any(v.kind in ("int", "key", "split_key") for v in outs):
        raise TranslationError("integer/key outputs are unsupported; cast randint to float before real arithmetic")
    classes = "[Field R]" + (" [LinearOrder R]" if e.ordered else "")
    classes += " [RealOps R]" if e.transcendental else ""
    def ty(v):
        return ("Bool" if v.kind == "bool" else "R") if scalar else tensor_type(v.shape, v.kind)
    binders = " ".join(f"({v.expr} : {ty(v)})" for v in args if v.kind != "key")
    result_type = " × ".join(ty(v) for v in outs) or "Unit"
    result = outs[0].expr if len(outs) == 1 else "(" + ", ".join(v.expr for v in outs) + ")"
    ns = ".".join(f"«{n}»" for n in namespace.split("."))
    body = [*e.lines, "  " + result]
    declaration = "def"
    extra = []
    if random_model:
        if sum(v.kind == "key" for v in args) != 1 or e.random_source is None:
            raise TranslationError("uniform model requires one typed key input and one scalar or vmapped randint call")
        if len(outs) != 1 or outs[0].kind != "real" or outs[0].shape:
            raise TranslationError("uniform model currently requires one real scalar result")
        lo, hi, batch = e.random_source
        declaration = "noncomputable def"
        result_type = "Rand R"
        result = outs[0].at([])
        extra = ["import JaxLean.Independent" if batch is not None else "import JaxLean.Random",
                 "-- SAMPLER SPECIFICATION: scalar randint is modeled as an ideal uniform draw.",
                 "-- The Python key is abstracted into this law; no PRNG/IEEE-754 equivalence is proved."]
        sampler = f"Rand.uniformInt (R := R) ({lo}) {hi - lo} (by decide)"
        if batch is not None:
            sampler = f"Rand.iid {batch} ({sampler})"
            extra.append("-- KEY SPECIFICATION: one split produces independent children, represented by a product law.")
        body = [f"  let sampling := {sampler}",
                f"  Rand.map (fun {'draws' if batch is not None else 'draw'} =>", *["  " + line for line in body[:-1]],
                "    " + result + ") sampling"]
    return "\n".join([
        "-- Generated by jaxlean from JAX 0.8.0. Edit the source, not this file.",
        "-- Real arithmetic abstraction; no claim of IEEE-754 equivalence.",
        "import JaxLean.RealOps", *extra, "", "open JaxLean", "open scoped BigOperators",
        "set_option linter.unusedVariables false", f"namespace {ns}", "",
        f"{declaration} «{name}» {{R : Type}} {classes} {binders} : {result_type} :=",
        *body, "", f"end {ns}", "",
    ])


def transpile(fn, *example_args, name=None, namespace="Generated", **options):
    """Trace ordinary JAX and translate it, keeping the Python function/argument names.

    Scalar functions get scalar Lean types; arrays keep the tensor representation.
    random_model='uniform' explicitly opts into the ideal randint specification.
    The source function itself needs no jaxlean imports, decorators, or wrappers.
    """
    closed = jax.make_jaxpr(fn)(*example_args)
    if name is None:
        candidate = getattr(fn, "__name__", "program")
        name = candidate if re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", candidate) else "program"
    options.setdefault("readable", True)
    def entirely_scalar(jp):
        values = [*jp.constvars, *jp.invars, *jp.outvars,
                  *(v for eq in jp.eqns for v in eq.outvars)]
        return all(not shape(v.aval) for v in values) and all(
            entirely_scalar(v.jaxpr) for eq in jp.eqns for v in eq.params.values()
            if isinstance(v, core.ClosedJaxpr))
    options.setdefault("scalar", entirely_scalar(closed.jaxpr))
    return translate(closed, name=name, namespace=namespace, **options)
