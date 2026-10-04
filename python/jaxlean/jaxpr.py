"""Jaxpr metadata validation and Lean type/coordinate notation.

No emitter, importer, source inspection, or primitive evaluation lives here.
"""
import re
import numpy as np

class TranslationError(ValueError):
    """The input is outside the explicitly supported Jaxpr fragment."""


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
        return "int"  # Static metadata, modeled samples, or runtime signed int32 indices.
    raise TranslationError(f"unsupported dtype {dtype}; expected real floating values or bool")


def lean_shape(s):
    return "[" + ", ".join(map(str, s)) + "]"


def tensor_type(s, k):
    return f"Tensor {'Bool' if k == 'bool' else 'Int32' if k == 'index' else 'R'} {lean_shape(s)}"


def coords(s, var="i"):
    return [var + ".2" * k + ".1" for k in range(len(s))]


def index(xs):
    return "(" + ", ".join([*xs, "()"]) + ")" if xs else "()"


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
