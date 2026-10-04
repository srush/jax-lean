"""Typed Jaxpr importer: Boolean/Int32 SSA plus existing real subprograms."""
from fractions import Fraction

import numpy as np
from jax.extend import core
from ..jaxpr import TranslationError, shape, kind, lean_shape, coords, index
from ..static_index import integer_equation


def needs_typed(closed):
    jp = closed.jaxpr
    metadata = dict(zip(jp.constvars, map(np.asarray, closed.consts)))
    if any(kind(v.aval) != 'real' for v in (*jp.invars, *jp.outvars)):
        return True
    for eq in jp.eqns:
        static = integer_equation(eq, metadata)
        if static is not None:
            metadata[eq.outvars[0]] = static
            continue
        if any(kind(v.aval) != 'real' for v in eq.outvars):
            return True
        if eq.primitive.name == 'gather':
            return True
        if eq.primitive.name == 'jit' and needs_typed(eq.params['jaxpr']):
            return True
    return False


def dtype(aval):
    k = kind(aval)
    if k == 'int' and np.dtype(aval.dtype) != np.dtype('int32'):
        raise TranslationError('typed index semantics currently requires signed int32')
    return k


def ty(aval):
    return f"(.{dtype(aval)}, {lean_shape(shape(aval))})"


def binder_type(aval):
    scalar = {'real': 'ℝ', 'bool': 'Bool', 'int': 'Int32'}[dtype(aval)]
    return f"Tensor {scalar} {lean_shape(shape(aval))}"


class TypedImporter:
    def __init__(self, closed, calls=None):
        self.jp = closed.jaxpr
        if self.jp.effects or len(self.jp.outvars) != 1:
            raise TranslationError('typed certificates require one pure output')
        self.calls = calls or {}
        self.env = {v: i for i, v in enumerate(self.jp.invars)}
        for v in self.jp.invars:
            dtype(v.aval)
        self.constants = dict(zip(self.jp.constvars, closed.consts, strict=True))
        if any(np.asarray(c).dtype.kind not in 'iub' for c in closed.consts):
            raise TranslationError('captured floating arrays are not yet certified')
        self.lines = []

    def atom(self, v):
        if isinstance(v, core.Literal) or v in self.constants:
            value = v.val if isinstance(v, core.Literal) else self.constants[v]
            # Encode Jaxpr literals directly, independently of the function emitter.
            data = np.asarray(value)
            k = dtype(v.aval)
            if data.shape != shape(v.aval):
                raise TranslationError('literal shape mismatch')
            if k == 'real':
                if data.shape or not np.isfinite(data.item()):
                    raise TranslationError('typed real literals must be finite scalars')
                q = Fraction(float(data.item()))
                return f'(.realLiteral ({q.numerator}) {q.denominator} (by decide))'
            def scalar(x):
                if k == 'bool':
                    return 'true' if x else 'false'
                if k == 'int':
                    return f'(Int32.ofInt ({int(x)}))'
                raise TranslationError('unsupported literal dtype')
            if data.shape:
                items = ', '.join(scalar(x) for x in data.flat)
                expr = f'(Tensor.ofArray (s := {lean_shape(data.shape)}) #[{items}] (by rfl))'
            else:
                expr = f'(Tensor.scalar {scalar(data.item())})'
            return f"(.literal {expr})"
        ref = '.here'
        for _ in range(self.env[v]):
            ref = f'(.there {ref})'
        return f'(.var {ref})'

    def emit(self, expr, out):
        self.lines.append(f'  .bind ({expr}) <|')
        self.env = {v: i + 1 for v, i in self.env.items()}
        self.env[out] = 0

    def promote(self, v, target):
        src = shape(v.aval)
        if src == target:
            return v
        if src and (len(src) != len(target) or any(a != b and a != 1 for a, b in zip(src, target))):
            raise TranslationError('incompatible typed broadcasting')
        ix = ["0" if n == 1 else c for n, c in zip(src, coords(target))] if src else []
        out = object()
        self.emit(f'.reindex (s := {lean_shape(src)}) (t := {lean_shape(target)}) (fun i => {index(ix)}) {self.atom(v)}', out)
        return out

    def run(self):
        from .real import RealImporter
        from ..layout import gather_map, identity_extreme
        for eq in self.jp.eqns:
            op, p = eq.primitive.name, eq.params
            if eq.effects or len(eq.outvars) != 1:
                raise TranslationError('typed certificates require pure single-result equations')
            out = eq.outvars[0]
            s, k = shape(out.aval), dtype(out.aval)
            static = integer_equation(eq, self.constants)
            if static is not None:
                self.constants[out] = static
                continue
            finite = identity_extreme(eq)
            if finite is not None:
                self.env[out] = self.env[finite]
                continue
            if op == 'jit' and id(p['jaxpr']) in self.calls:
                callee = self.calls[id(p['jaxpr'])]
                args = '.nil'
                for v in reversed(eq.invars):
                    args = f'(.cons {self.atom(v)} {args})'
                if needs_typed(p['jaxpr']):
                    self.lines.append(f'  .call {callee}_ir {args} <|')
                    self.env = {v: i+1 for v, i in self.env.items()}
                    self.env[out] = 0
                else:
                    self.emit(f'.real {callee}_ir {args}', out)
                continue
            if op in ('eq', 'ne', 'lt', 'le', 'gt', 'ge'):
                d = dtype(eq.invars[0].aval)
                if d not in ('real', 'int'):
                    raise TranslationError('comparison requires real or int32 operands')
                a, b = (self.promote(v, s) for v in eq.invars)
                expr = f".{'compareReal' if d == 'real' else 'compareInt'} .{op} {self.atom(a)} {self.atom(b)}"
            elif op in ('add', 'sub', 'mul') and k == 'int':
                a, b = (self.promote(v, s) for v in eq.invars)
                expr = f'.int{op.title()} {self.atom(a)} {self.atom(b)}'
            elif op in ('and', 'or', 'xor', 'not') and k == 'bool':
                args = [self.promote(v, s) for v in eq.invars]
                expr = f'.bool{op.title()} ' + ' '.join(self.atom(v) for v in args)
            elif op == 'select_n':
                if len(eq.invars) != 3 or dtype(eq.invars[0].aval) != 'bool':
                    raise TranslationError('select_n requires boolean predicate and two cases')
                args = [self.promote(v, s) for v in eq.invars]
                expr = '.select ' + ' '.join(self.atom(v) for v in args)
            elif op == 'gather':
                mapping = gather_map(eq)
                expr = f'.gather (s := {lean_shape(shape(eq.invars[0].aval))}) (t := {lean_shape(s)}) (u := {lean_shape(shape(eq.invars[1].aval))}) {mapping} {self.atom(eq.invars[0])} {self.atom(eq.invars[1])}'
            elif op == 'convert_element_type' and dtype(eq.invars[0].aval) == 'int' and k == 'real':
                expr = f'.intToReal {self.atom(eq.invars[0])}'
            elif k != 'real' and op in ('broadcast_in_dim', 'reshape', 'transpose', 'squeeze', 'slice', 'copy', 'stop_gradient', 'convert_element_type'):
                src = shape(eq.invars[0].aval)
                if op == 'broadcast_in_dim':
                    ix = ['0' if n == 1 else coords(s)[d] for n, d in zip(src, p['broadcast_dimensions'], strict=True)]
                elif op == 'transpose':
                    ix = [coords(s)[tuple(p['permutation']).index(d)] for d in range(len(src))]
                elif op == 'squeeze':
                    remaining = iter(coords(s))
                    ix = ['0' if d in p['dimensions'] else next(remaining) for d in range(len(src))]
                elif op == 'slice':
                    strides = p['strides'] or (1,) * len(src)
                    ix = [f'⟨{lo} + {step} * {c}.val, by omega⟩' for c, lo, step in zip(coords(s), p['start_indices'], strides, strict=True)]
                elif op == 'reshape':
                    if p.get('dimensions') is not None:
                        raise TranslationError('typed reshape permutation not yet supported')
                    mapping = f'(fun i => (Index.equivFin {lean_shape(src)}).symm (Fin.cast (by decide) (Index.equivFin {lean_shape(s)} i)))'
                    ix = None
                else:
                    if dtype(eq.invars[0].aval) != k or src != s:
                        raise TranslationError('typed copy/conversion must preserve dtype and shape')
                    ix = coords(s)
                if ix is not None:
                    mapping = f'(fun i => {index(ix)})'
                expr = f'.reindex (s := {lean_shape(src)}) (t := {lean_shape(s)}) {mapping} {self.atom(eq.invars[0])}'
            elif k == 'real' and (all(dtype(v.aval) == 'real' for v in eq.invars) or op in ('scatter', 'scatter-add')):
                # Reuse the independent real importer, not the generated function.
                variables = list(dict.fromkeys(v for v in eq.invars if not isinstance(v, core.Literal)))
                constants = [v for v in variables if v in self.constants]
                args = [v for v in variables if v not in self.constants]
                if any(dtype(v.aval) != 'real' for v in args):
                    raise TranslationError('real subprogram requires real arguments and static index metadata')
                fragment = core.ClosedJaxpr(core.Jaxpr(constants, args, [out], [eq], debug_info=self.jp.debug_info._replace(arg_names=None, result_paths=None)), [self.constants[v] for v in constants])
                body = RealImporter(fragment).run()
                actual = '.nil'
                for v in reversed(args):
                    actual = f'(.cons {self.atom(v)} {actual})'
                expr = f'.real (\n{body}\n  ) {actual}'
            else:
                raise TranslationError(f'no typed certificate rule for {op}')
            self.emit(expr, out)
        return '\n'.join([*self.lines, f'  .ret {self.atom(self.jp.outvars[0])}'])
