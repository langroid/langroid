import pandas as pd
import pytest

from langroid.utils.pandas_utils import (
    UnsafeCommandError,
    safe_eval_globals,
    sanitize_command,
)

SAFE = [
    "df.groupby('state')['income'].mean()",
    "df['a'] + df['b'] * 2",
    "df.pivot_table(index='year', columns='state', values='sales', aggfunc='sum')",
    "df.sort_values('income').head(10)",
    "(df['x'] - df['y']).abs().mean()",
    "df.sample(n=5)",
    "df.nsmallest(3, 'income')['income']",
    "df.where(df['income'] > 50000)['state'].value_counts()",
    "df.describe()",
    "df.loc[0:100, 'income'].sum()",
    "df.head(5)['income'].mean()",
    "df.select_dtypes(include=['number']).sum().sum()",
    "df.rank(method='average')['score']",
    "df.groupby('state', sort=True)['income'].median()",
    "df.sample(frac=0.1, random_state=42)",
]

DEEP_EXPR = "df" + "[0]" * 26  # depth bomb (26 > MAX_DEPTH)

BLOCK_WITH_MSG = [
    ("df.eval('2+2')", r"method 'eval' not permitted"),
    ("df.sample(n=5, regex=True)", r"kwarg 'regex' is blocked"),
    ("df['b'] * 12345678901", r"numeric constant exceeds limit"),
    ("df['a'] ** 8", r"operator not allowed"),
    (
        "df.head().tail().sort_values('a').groupby('state').sum().mean().std()",
        r"method-chain too long",
    ),
    ("df.sample(n=10, inplace=True)", r"kwarg 'inplace' is blocked"),
    ("sales.sum()", r"unexpected variable 'sales'"),
    ("df2.head()", r"unexpected variable 'df2'"),
    ("df[other_var]", r"subscript must be literal"),
    (
        "df.where(df['income'] > other_var)['income']",
        r"unexpected variable 'other_var'",
    ),
    (DEEP_EXPR, r"AST nesting too deep"),
    # CVE-2025-46724 bypass tests - dunder attribute access
    ("df.__init__", r"dunder attribute '__init__' not allowed"),
    ("df.__class__", r"dunder attribute '__class__' not allowed"),
    ("df.__globals__", r"dunder attribute '__globals__' not allowed"),
    ("df.__builtins__", r"dunder attribute '__builtins__' not allowed"),
    # CVE-2025-46724 bypass tests - private attribute access
    ("df._private", r"private attribute '_private' not allowed"),
    ("df._internal_method()", r"method '_internal_method' not permitted"),
    # CVE-2025-46724 bypass tests - dunder access via kwargs (the actual bypass vector)
    (
        "df.groupby(by=df.__init__)",
        r"dunder attribute '__init__' not allowed",
    ),
    (
        "df.groupby(by=df.__class__.__bases__)",
        r"dunder attribute '__.+__' not allowed",
    ),
    # Full PoC exploit payload - blocks on dunder attribute access
    (
        "df.add_prefix(\"__import__('os').system('ls')#\").T.groupby("
        "by=df.__init__.__globals__['__builtins__']['eval'])",
        r"dunder attribute '__.+__' not allowed",
    ),
]


@pytest.mark.parametrize("expr", SAFE)
def test_safe(expr):
    """All SAFE expressions must pass without exception."""
    assert sanitize_command(expr) == expr


@pytest.mark.parametrize("expr,msg", BLOCK_WITH_MSG)
def test_block(expr, msg):
    """All BLOCK expressions must raise UnsafeCommandError with the right message."""
    with pytest.raises(UnsafeCommandError, match=msg):
        sanitize_command(expr)


# ---------------------------------------------------------------------------
# safe_eval_globals -- builtin-injection guard (GHSA-q9p7-wqxg-mrhc).
#
# The TableChatAgent.pandas_eval and VectorStoreBase.compute_from_docs methods
# call eval(code, vars, {}). The empty `locals={}` looked sandboxed but
# Python implicitly injects all builtins into `vars` -- so even with
# full_eval=True, an expression like __import__('os').system('...') would
# execute and yield RCE. These tests confirm the fix: building globals via
# `safe_eval_globals(vars)` restricts __builtins__ to a curated safe set, so
# the direct RCE primitive is closed.
# ---------------------------------------------------------------------------

_DF = pd.DataFrame({"name": ["Alice", "Bob"], "age": [25, 30]})


def _eval_via_safe_globals(expr):
    """Evaluate `expr` exactly as the agent call sites do."""
    code = compile(expr, "<calc>", "eval")
    return eval(code, safe_eval_globals({"df": _DF}), {})


def test_safe_eval_globals_restricts_builtins():
    g = safe_eval_globals({"df": _DF})
    assert g["df"] is _DF
    assert isinstance(g["__builtins__"], dict)
    # Dangerous primitives must not be in the curated set.
    for forbidden in (
        "__import__",
        "eval",
        "exec",
        "compile",
        "open",
        "globals",
        "vars",
        "getattr",
        "setattr",
        "delattr",
        "input",
        "dir",
    ):
        assert forbidden not in g["__builtins__"], forbidden
    # Common safe ones must remain available.
    for keep in ("len", "min", "max", "sum", "abs", "round", "list", "range"):
        assert keep in g["__builtins__"], keep


def test_safe_eval_globals_does_not_leak_caller_builtins_key():
    """If a caller's vars dict already contains a __builtins__ key (unusual),
    the helper must still install the restricted set, never the caller's."""
    g = safe_eval_globals({"df": _DF, "__builtins__": {"__import__": __import__}})
    assert "__import__" not in g["__builtins__"]


@pytest.mark.parametrize(
    "payload",
    [
        # The exact GHSA-q9p7-wqxg-mrhc PoC payload.
        "__import__('os').system('touch /tmp/rce_success_table')",
        # Equivalent dangerous primitives.
        "open('/etc/passwd').read()",
        "exec('import os; os.system(\"id\")')",
        "compile('os.system(\"id\")', '<x>', 'exec')",
        "globals()",
        "vars()",
        "getattr({}, 'pop')()",
        # Lookup of __import__ through builtins itself, in case the dict leaks.
        "__builtins__['__import__']('os')",
    ],
)
def test_dangerous_payloads_blocked_via_safe_eval_globals(payload):
    """All these must fail with NameError (or KeyError on __builtins__ lookup),
    not actually execute."""
    with pytest.raises((NameError, KeyError, TypeError)):
        _eval_via_safe_globals(payload)


@pytest.mark.parametrize(
    "expr,expected",
    [
        ("df.shape", (2, 2)),
        ("df['age'].mean()", 27.5),
        ("len(df)", 2),
        ("sum(df['age'])", 55),
        ("min(df['age'])", 25),
        ("max(df['age'])", 30),
        ("round(df['age'].mean(), 2)", 27.5),
        ("list(df['name'])", ["Alice", "Bob"]),
        ("sorted(df['age'].tolist())", [25, 30]),
    ],
)
def test_benign_expressions_still_work_via_safe_eval_globals(expr, expected):
    """The restricted builtins must not break legitimate pandas expressions."""
    assert _eval_via_safe_globals(expr) == expected
