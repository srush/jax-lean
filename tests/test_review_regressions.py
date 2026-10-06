"""Adversarial names and source parameters at the compiler/proof boundary."""
import jax
import jax.numpy as jnp
import pytest

from jaxlean import certify_module
from test_random_translation import lean


def shadow_index(i):
    return i.T


def shadow_elementwise(x, a0):
    return x + a0


@jax.jit
def helper(x):
    return x + 1.


helper_alias = helper


def shadow_callee(helper):
    return helper_alias(helper)


def shadow_theorem(program, program_ir, program_translation_correct):
    return program + program_ir * program_translation_correct


@pytest.mark.lean
@pytest.mark.parametrize('fn,args', [
    (shadow_index, (jnp.ones((2, 3)),)),
    (shadow_elementwise, (jnp.ones(3), jnp.float32(2))),
    (shadow_callee, (jnp.ones(3),)),
    (shadow_theorem, (jnp.ones(3), jnp.ones(3), jnp.ones(3))),
])
def test_source_names_cannot_capture_generated_binders_or_declarations(fn, args):
    source = certify_module(jax.make_jaxpr(fn)(*args), named_vars=True)
    result = lean(source, f'Names_{fn.__name__}.lean')
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'sorryAx' not in result.stdout


@pytest.mark.lean
def test_named_graph_crosses_the_old_lean_keyword_boundary():
    def long_graph(x):
        for _ in range(80):
            x = x + 1.
        return x
    source = certify_module(jax.make_jaxpr(long_graph)(jnp.float32(1)), named_vars=True)
    result = lean(source, 'LongNamedGraph.lean')
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.lean
@pytest.mark.parametrize('fn,old,new', [
    (lambda x: x.T, '.transpose [1, 0]', '.transpose [0, 1]'),
    (lambda x: jax.lax.rev(x, (0,)), '.rev [0]', '.rev [1]'),
])
def test_layout_parameter_mutations_fail_certificate(fn, old, new):
    source = certify_module(jax.make_jaxpr(fn)(jnp.ones((2, 2))))
    assert old in source
    good = lean(source, 'SourceLayout.lean')
    assert good.returncode == 0, good.stdout + good.stderr
    # Both parameters are well typed for a square matrix; only one is correct.
    bad = lean(source.replace(old, new), 'CorruptSourceLayout.lean')
    assert bad.returncode != 0
    assert 'unsolved goals' in bad.stdout


@pytest.mark.lean
@pytest.mark.parametrize('operation', [
    '.transpose [0, 0] (.var .here) (t := [2, 2])',
    '.rev [2] (.var .here)',
    '.rev [0, 0] (.var .here)',
])
def test_ir_rejects_invalid_layout_parameters(operation):
    source = '''import JaxLean.Core.Jaxpr
open JaxLean
example : Jaxpr.Program [(.real, [2, 2])] (.real, [2, 2]) :=
  .bind (''' + operation + ''') <| .ret (.var .here)
'''
    result = lean(source, 'InvalidLayoutParameters.lean')
    assert result.returncode != 0
    assert 'decide' in result.stdout


@pytest.mark.lean
@pytest.mark.parametrize('body', [
    'return (.literal 0 1 (by decide))',
    'b : (.real, []) := .neg (.literal 0 1 (by decide)); return b',
    'b : (.real, []) := call empty with .nil; return b',
])
def test_unused_named_input_must_match_the_program_context(body):
    source = '''import JaxLean.Verification.Syntax
open JaxLean
def empty : Jaxpr.Program [] (.real, []) := .ret (.literal 0 1 (by decide))
example : Jaxpr.Program [(.real, [3])] (.real, []) :=
  jaxpr% (unused : (.bool, [3])) { ''' + body + ' }\n'
    result = lean(source, 'WrongUnusedInput.lean')
    assert result.returncode != 0
    assert 'mismatch' in result.stdout.lower()
