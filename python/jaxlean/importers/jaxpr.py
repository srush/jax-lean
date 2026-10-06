"""Import source Jaxpr into the single dtype- and shape-indexed Lean Jaxpr."""
from fractions import Fraction

import numpy as np
from jax.extend import core
from ..jaxpr import TranslationError, shape, kind, lean_shape, coords, index
from ..static_index import integer_equation


def dtype(aval):
    k = kind(aval)
    if k == 'int' and np.dtype(aval.dtype) != np.dtype('int32'):
        raise TranslationError('certified index semantics currently requires signed int32')
    return k


def ty(aval):
    return f"(.{dtype(aval)}, {lean_shape(shape(aval))})"


def binder_type(aval):
    scalar = {'real': 'ℝ', 'bool': 'Bool', 'int': 'Int32'}[dtype(aval)]
    return f"Tensor {scalar} {lean_shape(shape(aval))}"


class JaxprImporter:
    def __init__(self, closed, calls=None, *, named_vars=False):
        self.jp = closed.jaxpr
        if self.jp.effects or len(self.jp.outvars) != 1:
            raise TranslationError('certificates require one pure output')
        self.calls = calls or {}
        self.env = {v: i for i, v in enumerate(self.jp.invars)}
        for v in self.jp.invars:
            dtype(v.aval)
        self.constants = dict(zip(self.jp.constvars, closed.consts, strict=True))
        if any(np.asarray(c).dtype.kind not in 'iub' for c in closed.consts):
            raise TranslationError('captured floating arrays are not yet certified')
        self.lines = []
        self.named_vars = named_vars
        # Stable SSA names, independent of Python source and operand traversal order.
        variables = [*self.jp.invars, *(v for eq in self.jp.eqns for v in eq.outvars)]
        self.names = {v: self.variable_name(i) for i, v in enumerate(variables)}

    @staticmethod
    def variable_name(i):
        # Single letters match small Jaxprs. Numbered names avoid Lean keywords
        # such as `by`, `do`, `if`, and `in` in larger graphs.
        return chr(ord('a') + i) if i < 26 else f'v{i}'

    def atom(self, v):
        if isinstance(v, core.Literal) or (v in self.constants and v not in self.env):
            value = v.val if isinstance(v, core.Literal) else self.constants[v]
            # Encode Jaxpr literals directly, independently of the function emitter.
            data = np.asarray(value)
            k = dtype(v.aval)
            if data.shape != shape(v.aval):
                raise TranslationError('literal shape mismatch')
            if k == 'real':
                if data.shape or not np.isfinite(data.item()):
                    raise TranslationError('certified real literals must be finite scalars')
                q = Fraction(float(data.item()))
                return f'(.literal ({q.numerator}) {q.denominator} (by decide))'
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
            return f"(.tensorLiteral (t := {ty(v.aval)}) {expr})"
        ref = '.here'
        for _ in range(self.env[v]):
            ref = f'(.there {ref})'
        if self.named_vars:
            return self.names[v]
        return f'(.var {ref})'

    def emit(self, expr, out, *, args=None):
        if args is not None:
            line = (f'    {self.names[out]} : {ty(out.aval)} := call {expr} with {args};'
                    if self.named_vars else f'  .call {expr} {args} <|')
            self.lines.append(line)
        elif self.named_vars:
            self.lines.append(f'    {self.names[out]} : {ty(out.aval)} := {expr};')
        else:
            self.lines.append(f'  .bind ({expr}) <|')
        self.env = {v: i + 1 for v, i in self.env.items()}
        self.env[out] = 0

    def broadcast_shape(self, variables, target):
        for v in variables:
            src = shape(v.aval)
            if src and (len(src) != len(target) or any(a != b and a != 1 for a, b in zip(src, target))):
                raise TranslationError('incompatible elementwise broadcasting')
        return f' (t := {lean_shape(target)})'

    def run(self):
        from ..layout import gather_map, identity_extreme
        for eq in self.jp.eqns:
            op, p = eq.primitive.name, eq.params
            if eq.effects or len(eq.outvars) != 1:
                raise TranslationError('certificates require pure single-result equations')
            out = eq.outvars[0]
            s, k = shape(out.aval), dtype(out.aval)
            static = integer_equation(eq, self.constants)
            if static is not None:
                self.constants[out] = static
                # Retain the source equation; metadata is only used to validate static scatter.
            finite = identity_extreme(eq)
            if finite is not None:
                self.env[out] = self.env[finite]
                self.names[out] = self.names[finite]
                continue
            if op == 'jit' and id(p['jaxpr']) in self.calls:
                callee = self.calls[id(p['jaxpr'])]
                args = '.nil'
                for v in reversed(eq.invars):
                    args = f'(.cons {self.atom(v)} {args})'
                self.emit(f'{callee}_ir', out, args=args)
                continue
            if op in ('eq', 'ne', 'lt', 'le', 'gt', 'ge'):
                d = dtype(eq.invars[0].aval)
                if d not in ('real', 'int'):
                    raise TranslationError('comparison requires real or int32 operands')
                expr = f'.{op} ' + ' '.join(self.atom(v) for v in eq.invars) + self.broadcast_shape(eq.invars, s)
            elif op in ('add', 'sub', 'mul') and k == 'int':
                expr = f'.{op} ' + ' '.join(self.atom(v) for v in eq.invars) + self.broadcast_shape(eq.invars, s)
            elif op in ('and', 'or', 'xor', 'not') and k == 'bool':
                expr = f'.{op} ' + ' '.join(self.atom(v) for v in eq.invars)
                if op != 'not':
                    expr += self.broadcast_shape(eq.invars, s)
            elif op == 'select_n':
                if len(eq.invars) != 3 or dtype(eq.invars[0].aval) != 'bool':
                    raise TranslationError('select_n requires boolean predicate and two cases')
                expr = '.select_n ' + ' '.join(self.atom(v) for v in eq.invars) + self.broadcast_shape(eq.invars, s)
            elif op == 'gather':
                mapping = gather_map(eq)
                expr = f'.gather (s := {lean_shape(shape(eq.invars[0].aval))}) (t := {lean_shape(s)}) (u := {lean_shape(shape(eq.invars[1].aval))}) {mapping} {self.atom(eq.invars[0])} {self.atom(eq.invars[1])}'
            elif op == 'convert_element_type':
                before = dtype(eq.invars[0].aval)
                if before != k and (before, k) != ('int', 'real'):
                    raise TranslationError('unsupported dtype conversion')
                expr = f'.convert_element_type .{k} {self.atom(eq.invars[0])}'
            elif op in ('copy', 'stop_gradient'):
                expr = f'.{op} {self.atom(eq.invars[0])}'
            elif op == 'broadcast_in_dim':
                expr = f'.broadcast_in_dim {lean_shape(s)} {lean_shape(p["broadcast_dimensions"])} {self.atom(eq.invars[0])}'
            elif op == 'transpose':
                expr = f'.transpose {lean_shape(p["permutation"])} {self.atom(eq.invars[0])} (t := {lean_shape(s)})'
            elif op == 'rev':
                expr = f'.rev {lean_shape(p["dimensions"])} {self.atom(eq.invars[0])}'
            elif op in ('reshape', 'squeeze', 'slice'):
                src = shape(eq.invars[0].aval)
                if op == 'squeeze':
                    remaining = iter(coords(s))
                    ix = ['0' if d in p['dimensions'] else next(remaining) for d in range(len(src))]
                elif op == 'slice':
                    strides = p['strides'] or (1,) * len(src)
                    ix = [f'⟨{lo} + {step} * {c}.val, by omega⟩' for c, lo, step in zip(coords(s), p['start_indices'], strides, strict=True)]
                elif op == 'reshape':
                    perm = tuple(p.get('dimensions') or range(len(src)))
                    intermediate = tuple(src[d] for d in perm)
                    flat = f'((Index.equivFin {lean_shape(intermediate)}).symm (Fin.cast (by decide) (Index.equivFin {lean_shape(s)} i)))'
                    ix = [coords(intermediate, flat)[perm.index(d)] for d in range(len(src))]
                    mapping = f'(fun i => {index(ix)})'
                    ix = None
                if ix is not None:
                    mapping = f'(fun i => {index(ix)})'
                expr = f'.{op} (s := {lean_shape(src)}) (t := {lean_shape(s)}) {mapping} {self.atom(eq.invars[0])}'
            elif op == 'concatenate' or (k == 'real' and (
                    all(dtype(v.aval) == 'real' for v in eq.invars)
                    or op in ('scatter', 'scatter-add'))):
                expr = self.numeric_operation(eq)
            else:
                raise TranslationError(f'no certificate rule for {op}')
            self.emit(expr, out)
        result = self.atom(self.jp.outvars[0])
        if self.named_vars:
            inputs = ', '.join(f'{self.names[v]} : {ty(v.aval)}' for v in self.jp.invars)
            return '\n'.join([f'  jaxpr% ({inputs}) {{', *self.lines, f'    return {result}', '  }'])
        return '\n'.join([*self.lines, f'  .ret {result}'])

    def numeric_operation(self, eq):
        from ..static_index import scatter_plan
        op, out, integers = eq.primitive.name, eq.outvars[0], self.constants
        if op == "iota" and kind(out.aval) == "real":
            s = shape(out.aval)
            axis = eq.params['dimension']
            return f".iota {lean_shape(s)} {axis}"
        if op in ("scatter", "scatter-add"):
            plan = scatter_plan(eq, integers)
            result, _, update = eq.invars
            positions = ', '.join(f'({index([str(i) for i in at])}, {index([str(i) for i in source])})' for at, source in plan)
            rule = 'scatter_add' if op == 'scatter-add' else 'scatter'
            return f'.{rule} (s := {lean_shape(shape(result.aval))}) (u := {lean_shape(shape(update.aval))}) [{positions}] {self.atom(result)} {self.atom(update)}'
        s = shape(out.aval)
        if op in ("add", "add_any", "sub", "mul", "div", "min", "max"):
            expr = f".{'add' if op == 'add_any' else op} " + " ".join(self.atom(v) for v in eq.invars) + self.broadcast_shape(eq.invars, s)
        elif op in ("neg", "square", "abs", "exp", "log", "sqrt", "rsqrt", "sin", "cos", "tanh"):
            expr = f".{op} {self.atom(eq.invars[0])}"
        elif op == "integer_pow":
            expr = f".integer_pow ({int(eq.params['y'])}) {self.atom(eq.invars[0])}"
        elif op == "concatenate":
            args = '.nil'
            for v in reversed(eq.invars):
                args = f'(.cons {self.atom(v)} {args})'
            axis = eq.params['dimension']
            coordinate = coords(s)[axis]
            choices, offset = [], 0
            for j, v in enumerate(eq.invars):
                src = shape(v.aval)
                end = offset + src[axis]
                ix = coords(s)
                ix[axis] = f'⟨{coordinate}.val - {offset}, by omega⟩'
                ref = '.here'
                for _ in range(j):
                    ref = f'(.there {ref})'
                choice = f'⟨{lean_shape(src)}, {ref}, {index(ix)}⟩'
                if j < len(eq.invars) - 1:
                    choices.append(f'if h{j} : {coordinate}.val < {end} then {choice} else ')
                else:
                    choices.append(choice)
                offset = end
            mapping = '(fun i => ' + ''.join(choices) + ')'
            expr = f'.concatenate (t := {lean_shape(s)}) {args} {mapping}'
        elif op == "dot_general":
            a, b = eq.invars
            from ..layout import contraction
            reduced, left, right = contraction(shape(a.aval), shape(b.aval), eq.params["dimension_numbers"])
            expr = f".dot_general (s := {lean_shape(shape(a.aval))}) (u := {lean_shape(shape(b.aval))}) (t := {lean_shape(s)}) (k := {lean_shape(reduced)}) {left} {right} {self.atom(a)} {self.atom(b)}"
        elif op in ("reduce_max", "reduce_min"):
            from ..layout import reduction
            _, n, mapping = reduction(shape(eq.invars[0].aval), eq.params["axes"], flat=True)
            if not n:
                raise TranslationError("empty extrema require infinity, outside real semantics")
            expr = f".{'reduce_max' if op == 'reduce_max' else 'reduce_min'} (s := {lean_shape(shape(eq.invars[0].aval))}) (t := {lean_shape(s)}) (n := {n}) (by decide) {mapping} {self.atom(eq.invars[0])}"
        elif op == "reduce_sum":
            from ..layout import reduction
            reduced, _, mapping = reduction(shape(eq.invars[0].aval), eq.params["axes"])
            expr = f".reduce_sum (s := {lean_shape(shape(eq.invars[0].aval))}) (t := {lean_shape(s)}) (k := {lean_shape(reduced)}) {mapping} {self.atom(eq.invars[0])}"
        else:
            raise TranslationError(f"primitive {op!r} has no translation certificate rule")
        return expr
