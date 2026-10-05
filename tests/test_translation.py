"""Differential tests execute emitted Lean, not a Python copy of its semantics."""
from fractions import Fraction
import json
from pathlib import Path
import subprocess
import sys

import jax
import jax.numpy as jnp
from jax import lax
import numpy as np
import pytest

from jaxlean import translate, TranslationError
from examples.selection.code import mlp, mlp_inputs, residual

ROOT = Path(__file__).resolve().parents[1]


def arr(shape, offset=0):
    return jnp.asarray(np.arange(np.prod(shape)).reshape(shape) + offset, dtype=jnp.float32)


M = arr((2, 3), 1)
V = arr((3,), 1)
MLP_X = jnp.array([[1., 2.], [-1., 1.], [3., -2.], [0., 0.]])
MLP_PARAMS = (jnp.array([[1., -1., 2.], [0., 1., -1.]]),
              jnp.array([0., 1., -1.]),
              jnp.array([[1., 0.], [0., 1.], [1., -1.]]),
              jnp.array([1., -1.]))
CASES = [
    ("residual5", residual, (arr((5, 3), -6), arr((3, 4), -5), arr((4,), -2),
                              arr((4, 3), -4), arr((3,), -1))),
    ("residual2", residual, (arr((2, 3), -3), arr((3, 4), -5), arr((4,), -2),
                              arr((4, 3), -4), arr((3,), -1))),
    ("mlp_empty", mlp, (jnp.empty((0, 2)), *MLP_PARAMS)),
    ("mlp_singleton", mlp, (MLP_X[:1], *MLP_PARAMS)),
    ("mlp4", mlp, (MLP_X, *MLP_PARAMS)),
    ("mlp3_selected", mlp, (MLP_X[jnp.array([2, 0, 2])], *MLP_PARAMS)),
    ("arithmetic", lambda x: -(x + 2.0) * x - x / 4.0, (M,)),
    ("powers", lambda x: (x ** 3, x ** -2, lax.square(x)), (V,)),
    ("exact_literal", lambda x: x + jnp.float32(0.1), (V,)),
    ("closed_constant", lambda x: x + M, (M,)),
    ("constant_result", lambda x: jnp.array([[1.0, 2.0], [3.0, 4.0]]), (V,)),
    ("literal_result", lambda x: jnp.array(7.0), (V,)),
    ("identity", lambda x: x, (M,)),
    ("zero_results", lambda x: (), (V,)),
    ("transpose", lambda x: x.T, (M,)),
    ("permute", lambda x: x.transpose(2, 0, 1), (arr((2, 3, 4)),)),
    ("reshape", lambda x: x.reshape(3, 2), (M,)),
    ("reshape_permute", lambda x: lax.reshape(x, (3, 2), dimensions=(1, 0)), (M,)),
    ("scalar_reshape", lambda x: x.reshape(()), (jnp.ones((1, 1)),)),
    ("broadcast", lambda x: jnp.broadcast_to(x, (4, 2, 3)), (M,)),
    ("singleton_broadcast", lambda x, y: x + y, (arr((2, 1)), arr((1, 3)))),
    ("squeeze", lambda x: jnp.squeeze(x, (0, 2)), (arr((1, 3, 1)),)),
    ("slice", lambda x: lax.slice(x, (0, 1), (2, 3)), (M,)),
    ("strided_slice", lambda x: lax.slice(x, (1, 0), (5, 4), (2, 2)), (arr((5, 4)),)),
    ("reverse", lambda x: lax.rev(x, (0, 1)), (M,)),
    ("concatenate", lambda x: jnp.concatenate((x, x + 10.0, x), axis=1), (M,)),
    ("sum", lambda x: (jnp.sum(x), jnp.sum(x, axis=0), jnp.sum(x, axis=1)), (M,)),
    ("sum_keepdims", lambda x: jnp.sum(x, axis=1, keepdims=True), (M,)),
    ("product", lambda x: jnp.prod(x, axis=0), (M,)),
    ("mean", lambda x: jnp.mean(x, axis=1), (M,)),
    ("inner", lambda x: x @ x, (V,)),
    ("matmul", lambda x: x @ x.T, (M,)),
    ("outer", lambda x: lax.dot_general(x, x, (((), ()), ((), ()))), (V,)),
    ("batched_dot", lambda x, y: x @ y, (arr((2, 3, 4)), arr((2, 4, 5)))),
    ("multi_contract", lambda x, y: lax.dot_general(x, y, (((2, 0), (0, 2)), ((), ()))),
     (arr((2, 3, 4)), arr((4, 5, 2)))),
    ("where", lambda x: jnp.where(x > 2.0, x, 0), (M,)),
    ("comparisons", lambda x: (x < 2.0, x <= 2.0, x == 2.0, x != 2.0, x >= 2.0), (V,)),
    ("bools", lambda x: ((x > 1.0) & (x < 3.0), ~(x > 1.0),
                         (x > 1.0) | (x < 3.0), (x > 1.0) ^ (x < 3.0)), (V,)),
    ("bool_input", lambda x, y: jnp.where(x, y, -y), (jnp.array([True, False, True]), V)),
    ("min_max_abs", lambda x: (jnp.minimum(x, 2.0), jnp.maximum(x, 2.0), jnp.abs(-x)), (V,)),
    ("relu", jax.nn.relu, (arr((3,), -1),)),
    ("jit", jax.jit(lambda x: jax.jit(lambda y: y * y)(x) + 1.0), (M,)),
    ("grad", jax.grad(lambda x: jnp.sum(x * x)), (V,)),
    ("vmap", jax.vmap(lambda x: jnp.sum(x * x)), (M,)),
    ("stop_gradient", lambda x: lax.stop_gradient(x) + x, (V,)),
    ("float_cast", lambda x: x.astype(jnp.float16).astype(jnp.float32), (V,)),
    ("iota", lambda x: lax.iota(jnp.float32, 4), (V,)),
    ("scan", lambda x: lax.scan(lambda c, a: (c + a, c * a), jnp.array(1.0), x), (V,)),
    ("reverse_scan", lambda x: lax.scan(lambda c, a: (c + a, c * a), jnp.array(1.0), x, reverse=True), (V,)),
    ("scan_constants", lambda x, y: lax.scan(lambda c, a: (c + a * y, c), 0.0, x), (V, jnp.array(2.0))),
    ("scan_no_xs", lambda x: lax.scan(lambda c, _: (c * 2.0, c), x, None, length=3), (jnp.array(1.0),)),
    ("scan_no_carry", lambda x: lax.scan(lambda c, a: (c, a * a), (), x), (V,)),
    ("empty_scan", lambda x: lax.scan(lambda c, a: (c + a, a), jnp.array(2.0), x), (arr((0,)),)),
    ("empty_sum", lambda x: jnp.sum(x, axis=1), (arr((2, 0)),)),
    ("empty_product", lambda x: jnp.prod(x, axis=1), (arr((2, 0)),)),
    ("empty_reshape", lambda x: x.reshape(0, 2), (arr((2, 0)),)),
    ("empty_slice", lambda x: lax.slice(x, (0, 1), (0, 3)), (M,)),
    ("empty_concat", lambda x: jnp.concatenate((jnp.empty((2, 0)), x), axis=1), (M,)),
]


def lean_input(a):
    a = np.asarray(a)
    if a.dtype.kind == "b":
        values = ["true" if x else "false" for x in a.flat]
        ty = "Bool"
    else:
        qs = [Fraction(float(x)) for x in a.flat]
        values = [f"({q.numerator} / {q.denominator} : ℚ)" for q in qs]
        ty = "ℚ"
    return f"(Tensor.ofArray #[{', '.join(values)}] (by decide) : Tensor {ty} {list(a.shape)})"


@pytest.fixture(scope="module")
def executed():
    modules = ["import JaxLean.Core.RealOps\n"]
    expected = {}
    for n, (label, fn, args) in enumerate(CASES):
        jp = jax.make_jaxpr(fn)(*args)
        source = translate(jp, namespace=f"Case{n}")
        modules.append(source.replace("import JaxLean.Core.RealOps", ""))
        outputs = jax.tree.leaves(fn(*args))
        call = f"Case{n}.program (R := ℚ) " + " ".join(lean_input(a) for a in args)
        for k, a in enumerate(outputs):
            expr = f"({call})"
            if len(outputs) > 1:
                expr += ".2" * k + (".1" if k + 1 < len(outputs) else "")
            fmt = "fun x => if x then \"true\" else \"false\"" if np.asarray(a).dtype.kind == "b" else 'fun x => s!"{x.num}/{x.den}"'
            modules.append(f'#eval IO.println ("RESULT:{label}:{k}:" ++ Lean.Json.compress (Lean.toJson ((Tensor.toList {expr}).map ({fmt}))))\n')
            expected[label, k] = np.asarray(a).flatten()
    target = ROOT / "tests/_generated/Differential.lean"
    target.parent.mkdir(exist_ok=True)
    target.write_text("\n".join(modules))
    proc = subprocess.run(["lake", "env", "lean", "-j", "1", str(target)], cwd=ROOT,
                          capture_output=True, text=True, timeout=180)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    actual = {}
    for line in proc.stdout.splitlines():
        if line.startswith("RESULT:"):
            _, label, k, data = line.split(":", 3)
            values = json.loads(data)
            actual[label, int(k)] = [v == "true" if v in ("true", "false") else float(Fraction(v)) for v in values]
    assert actual.keys() == expected.keys()
    return actual, expected


@pytest.mark.lean
@pytest.mark.parametrize("label", [c[0] for c in CASES])
def test_lean_matches_jax(executed, label):
    actual, expected = executed
    for key in expected:
        if key[0] == label:
            np.testing.assert_allclose(actual[key], expected[key], rtol=2e-6, atol=1e-6)


@pytest.mark.parametrize("fn,args,match", [
    (lambda x: jnp.floor(x), (V,), "floor"),
    (lambda x: jnp.sort(x), (V,), "sort"),
    (lambda x: x + 1, (jnp.array([1, 2], dtype=jnp.uint32),), "int32"),
    (lambda x: x.astype(jnp.int32), (V,), "conversions"),
    (lambda x: x + jnp.inf, (V,), "infinity"),
    (lambda x: jnp.var(x), (V,), "NaN"),
    (lambda x: jax.random.normal(jax.random.key(0), (3,)) + x, (V,), "dtype|random|prng"),
    (lambda x: lax.while_loop(lambda y: y < 4, lambda y: y + 1, x), (jnp.array(1.0),), "while"),
    (lambda x: jax.debug.print("{x}", x=x), (V,), "effects"),
    (lambda x: lax.scan(lambda c, _: (c + 1.0, c), x, None, length=33), (V,), "unrolling limit"),
    (lambda x: x + 1j, (jnp.array(1j),), "dtype"),
])
def test_rejects_unsupported(fn, args, match):
    with pytest.raises(TranslationError, match=match):
        translate(jax.make_jaxpr(fn)(*args))


def test_raw_jaxpr_and_constants():
    closed = jax.make_jaxpr(lambda x: x + M)(M)
    assert translate(closed) == translate(closed.jaxpr, consts=closed.consts)
    with pytest.raises(TranslationError, match="arity"):
        translate(closed.jaxpr)
    with pytest.raises(TranslationError, match="already contains"):
        translate(closed, consts=[])


def test_reproducible_examples():
    subprocess.run([sys.executable, "-m", "examples.generate", "--check"], cwd=ROOT, check=True)


@pytest.mark.lean
def test_selection_library_applies_to_new_traces():
    """Apply one proof recipe to new batch sizes, including singleton/empty axes."""
    parts = ["import JaxLean.Core.RealOps\nimport JaxLean.Stdlib.SelectionRules\n"]
    for n in (0, 1, 2, 3, 5):
        source = translate(jax.make_jaxpr(mlp)(*mlp_inputs(n)),
                           name=f"mlp{n}", namespace="SelectionTest")
        parts.append(source.replace("import JaxLean.Core.RealOps", ""))
    for n, m in ((5, 2), (1, 3), (3, 1), (0, 0), (1, 0)):
        parts.append(f"""
example (selection : Fin {m} → Fin {n}) (x : Tensor ℝ [{n}, 2])
    (w1 : Tensor ℝ [2, 3]) (b1 : Tensor ℝ [3])
    (w2 : Tensor ℝ [3, 2]) (b2 : Tensor ℝ [2]) :
    SelectionTest.mlp{m} (Tensor.selectRows selection x) w1 b1 w2 b2 =
      Tensor.selectRows selection (SelectionTest.mlp{n} x w1 b1 w2 b2) := by
  simp [SelectionTest.mlp{m}, SelectionTest.mlp{n}]
""")
    path = ROOT / "tests/_generated/SelectionLibrary.lean"
    path.parent.mkdir(exist_ok=True)
    path.write_text("\n".join(parts))
    proc = subprocess.run(["lake", "env", "lean", str(path)], cwd=ROOT,
                          capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stdout + proc.stderr


@pytest.mark.lean
def test_transcendentals_typecheck():
    fn = lambda x: (jnp.exp(x), jnp.log(x), jnp.sqrt(x), jnp.sin(x), jnp.cos(x), jnp.tanh(x), lax.rsqrt(x))
    text = translate(jax.make_jaxpr(fn)(V), namespace="Transcendentals")
    text += "\nnoncomputable def realExample := Transcendentals.program (R := ℝ) (Tensor.ofFlat ![1, 2, 3])\n"
    path = ROOT / "tests/_generated/Transcendentals.lean"
    path.parent.mkdir(exist_ok=True)
    path.write_text(text)
    p = subprocess.run(["lake", "env", "lean", str(path)], cwd=ROOT, text=True, capture_output=True, timeout=120)
    assert p.returncode == 0, p.stdout + p.stderr


def test_api_errors():
    jp = jax.make_jaxpr(lambda x: x)(V)
    for kwargs in ({"name": "bad name"}, {"namespace": "Bad;end"}):
        with pytest.raises(TranslationError, match="identifier"):
            translate(jp, **kwargs)
    with pytest.raises(TranslationError, match="printed text"):
        translate(str(jp))
