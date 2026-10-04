"""Import real-valued Jaxpr equations into Core.Jaxpr syntax."""
from fractions import Fraction
import numpy as np
from jax.extend import core

from ..jaxpr import TranslationError, shape, kind, lean_shape, coords, index
from ..static_index import integer_equation, scatter_plan


class RealImporter:
    def __init__(self, closed, calls=None):
        self.calls = calls or {}
        if not isinstance(closed, core.ClosedJaxpr):
            raise TranslationError("certify requires a ClosedJaxpr")
        self.jp = closed.jaxpr
        self.integers = {}
        for v, c in zip(self.jp.constvars, closed.consts, strict=True):
            if np.asarray(c).dtype.kind not in 'iub':
                raise TranslationError("certified captured constants must be integer index metadata")
            self.integers[v] = np.asarray(c)
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
        if v not in self.env:
            raise TranslationError("static index metadata cannot be used as a real tensor")
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
            if len(s) != len(target) or any(a != b and a != 1 for a, b in zip(s, target)):
                raise TranslationError("incompatible elementwise broadcast")
            ix = ["0" if n == 1 else c for n, c in zip(s, coords(target))]
            op = f".reindex (s := {lean_shape(s)}) (t := {lean_shape(target)}) (fun i => {index(ix)})"
        temp = object()
        self.emit(f"{op} {self.atom(v)}", temp)
        return temp

    def run(self):
        integers = dict(self.integers)
        for i, eq in enumerate(self.jp.eqns):
            op = eq.primitive.name
            if eq.effects or len(eq.outvars) != 1:
                raise TranslationError(f"certified equation {i}: expected one pure result")
            out = eq.outvars[0]
            static = integer_equation(eq, integers)
            if static is not None:
                integers[out] = static
                continue
            if op == "iota" and kind(out.aval) == "real":
                s = shape(out.aval)
                axis = eq.params['dimension']
                self.emit(f".iota (s := {lean_shape(s)}) (n := {s[axis]}) (fun i => {coords(s)[axis]})", out)
                continue
            if op in ("scatter", "scatter-add"):
                plan = scatter_plan(eq, integers)
                result, _, update = eq.invars
                for at, source in plan:
                    value = update
                    if shape(update.aval):
                        value = object()
                        self.emit(f".reindex (s := {lean_shape(shape(update.aval))}) (t := []) (fun _ => {index([str(i) for i in source])}) {self.atom(update)}", value)
                    target = object()
                    rule = "addAt" if op == "scatter-add" else "set"
                    self.emit(f".{rule} (s := {lean_shape(shape(eq.invars[0].aval))}) {index([str(i) for i in at])} {self.atom(result)} {self.atom(value)}", target)
                    result = target
                self.env[out] = self.env[result]
                continue
            from ..layout import identity_extreme
            finite = identity_extreme(eq)
            if finite is not None:
                self.env[out] = self.env[finite]
                continue
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
            elif op in ("neg", "square", "abs", "exp", "log", "sqrt", "rsqrt", "sin", "cos", "tanh"):
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
                    ix = ["0" if n == 1 else coords(s)[d] for n, d in zip(src, dims, strict=True)]
                    expr = f".reindex (s := {lean_shape(src)}) (t := {lean_shape(s)}) (fun i => {index(ix)}) {self.atom(eq.invars[0])}"
            elif op in ("slice", "squeeze"):
                src = shape(eq.invars[0].aval)
                if op == "slice":
                    starts, limits = eq.params["start_indices"], eq.params["limit_indices"]
                    strides = eq.params["strides"] or (1,) * len(src)
                    if any(not (0 <= lo <= hi <= n) or step <= 0 for lo, hi, n, step in zip(starts, limits, src, strides, strict=True)):
                        raise TranslationError("certified slice requires static in-bounds positive strides")
                    ix = [f"⟨{lo} + {step} * {c}.val, by omega⟩" for c, lo, step in zip(coords(s), starts, strides, strict=True)]
                else:
                    dims = eq.params["dimensions"]
                    if any(src[d] != 1 for d in dims):
                        raise TranslationError("squeeze must remove singleton axes")
                    it = iter(coords(s))
                    ix = ["0" if d in dims else next(it) for d in range(len(src))]
                expr = f".reindex (s := {lean_shape(src)}) (t := {lean_shape(s)}) (fun i => {index(ix)}) {self.atom(eq.invars[0])}"
            elif op == "concatenate":
                from ..layout import concatenate_map
                result, src = eq.invars[0], shape(eq.invars[0].aval)
                for v in eq.invars[1:]:
                    dst, mapping = concatenate_map(src, shape(v.aval), eq.params["dimension"])
                    target = object()
                    self.emit(f".concatenate (s := {lean_shape(src)}) (u := {lean_shape(shape(v.aval))}) (t := {lean_shape(dst)}) {mapping} {self.atom(result)} {self.atom(v)}", target)
                    result, src = target, dst
                self.env[out] = self.env[result]
                continue
            elif op == "reshape":
                value = eq.invars[0]
                perm = eq.params.get("dimensions")
                if perm is not None:
                    src = shape(value.aval)
                    dst = tuple(src[d] for d in perm)
                    ix = [coords(dst)[tuple(perm).index(d)] for d in range(len(src))]
                    temp = object()
                    self.emit(f".reindex (s := {lean_shape(src)}) (t := {lean_shape(dst)}) (fun i => {index(ix)}) {self.atom(value)}", temp)
                    value = temp
                expr = f".reshape {lean_shape(s)} (by decide) {self.atom(value)}"
            elif op == "rev":
                if len(s) == 1 and tuple(eq.params["dimensions"]) == (0,):
                    expr = f".reverseVec {self.atom(eq.invars[0])}"
                else:
                    ix = [f"{c}.rev" if d in eq.params["dimensions"] else c for d, c in enumerate(coords(s))]
                    expr = f".reindex (s := {lean_shape(s)}) (t := {lean_shape(s)}) (fun i => {index(ix)}) {self.atom(eq.invars[0])}"
            elif op == "transpose":
                perm = tuple(eq.params["permutation"])
                if perm == tuple(range(len(s))):
                    expr = f".copy {self.atom(eq.invars[0])}"
                elif perm == (1, 0):
                    expr = f".transpose2 {self.atom(eq.invars[0])}"
                else:
                    src = shape(eq.invars[0].aval)
                    ix = [coords(s)[perm.index(d)] for d in range(len(src))]
                    expr = f".reindex (s := {lean_shape(src)}) (t := {lean_shape(s)}) (fun i => {index(ix)}) {self.atom(eq.invars[0])}"
            elif op == "dot_general":
                (lc, rc), (lb, rb) = eq.params["dimension_numbers"]
                a, b = eq.invars
                if not lb and not rb and len(shape(a.aval)) == len(shape(b.aval)) == 2 and tuple(lc) == (1,) and tuple(rc) == (0,):
                    expr = f".matmul {self.atom(a)} {self.atom(b)}"
                elif len(shape(a.aval)) == 1 and len(shape(b.aval)) == 2 and tuple(lc) == (0,) and tuple(rc) == (0,):
                    expr = f".vecmat {self.atom(a)} {self.atom(b)}"
                elif len(shape(a.aval)) == len(shape(b.aval)) == 1 and tuple(lc) == tuple(rc) == (0,):
                    expr = f".dotVec {self.atom(a)} {self.atom(b)}"
                else:
                    from ..layout import contraction
                    reduced, left, right = contraction(shape(a.aval), shape(b.aval), eq.params["dimension_numbers"])
                    expr = f".contract (s := {lean_shape(shape(a.aval))}) (u := {lean_shape(shape(b.aval))}) (t := {lean_shape(s)}) (k := {lean_shape(reduced)}) {left} {right} {self.atom(a)} {self.atom(b)}"
            elif op in ("reduce_max", "reduce_min"):
                from ..layout import reduction
                _, n, mapping = reduction(shape(eq.invars[0].aval), eq.params["axes"], flat=True)
                if not n:
                    raise TranslationError("empty extrema require infinity, outside real semantics")
                expr = f".{'reduceMax' if op == 'reduce_max' else 'reduceMin'} (s := {lean_shape(shape(eq.invars[0].aval))}) (t := {lean_shape(s)}) (n := {n}) (by decide) {mapping} {self.atom(eq.invars[0])}"
            elif op == "reduce_sum":
                axes = tuple(eq.params["axes"])
                if axes == (0,):
                    expr = f".sumFirst {self.atom(eq.invars[0])}"
                elif axes == (1,) and len(shape(eq.invars[0].aval)) == 2:
                    expr = f".sumLast {self.atom(eq.invars[0])}"
                else:
                    from ..layout import reduction
                    reduced, _, mapping = reduction(shape(eq.invars[0].aval), axes)
                    expr = f".reduceSum (s := {lean_shape(shape(eq.invars[0].aval))}) (t := {lean_shape(s)}) (k := {lean_shape(reduced)}) {mapping} {self.atom(eq.invars[0])}"
            else:
                raise TranslationError(f"primitive {op!r} has no translation certificate rule")
            self.emit(expr, out)
        self.check_real(self.jp.outvars[0])
        return "\n".join([*self.lines, f"  .ret {self.atom(self.jp.outvars[0])}"])
