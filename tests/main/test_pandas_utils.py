import pandas as pd
import pytest

from langroid.utils.pandas_utils import (
    UnsafeCommandError,
    safe_eval_globals,
    sanitize_command,
    stringify,
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
    # Non-literal arguments that evaluate to *data* stay allowed: the
    # GHSA-jccq-v27v-rhjv restriction is on passing a callable (a bare
    # attribute reference), not on every non-literal argument.
    "df.groupby(df['state'])['income'].mean()",
    "df.add_prefix(df.columns[0])",
    "df.where(df['income'] > df['income'].mean())['state']",
    "df.align(df)",
    # Data attributes (SAFE_DF_DATA_ATTRS) stay usable as arguments.
    "df.filter(items=df.columns)",
    "df.groupby(df.index)['income'].sum()",
    "df.align(df.T)",
    "df.head(df.shape[0])",
    # Indexer subscripts and accessor data members, which main accepted.
    "df.align(df.iloc[:1])",
    "df.groupby(df['date'].dt.month)['sales'].sum()",
    "df.groupby(df['date'].dt.year)['sales'].max()",
    "df.groupby(df['elapsed'].dt.days).sum()",
    # Data attributes of a pandas receiver reached through a chain, and
    # operator expressions as receivers -- all accepted by main.
    "df.filter(items=df.dtypes.index)",
    "df.groupby(df.index.year).sum()",
    "df.sort_values(df['amount'].name)",
    "df.filter(items=df.columns.values)",
    "df.align((df + 1).iloc[:1])",
    "df.head((df['amount'] > 0).size)",
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
    # ---------------------------------------------------------------
    # GHSA-jccq-v27v-rhjv: a method referenced *without* parentheses is
    # never seen by the WHITELISTED_DF_METHODS check in visit_Call, and
    # pandas calls some arguments internally (``groupby(by=<callable>)``
    # applies the callable to every index label). So passing any bare
    # attribute reference as an argument must be rejected.
    # ---------------------------------------------------------------
    # The shape of the reported chain: the allowlisted add_prefix/.T hops
    # are irrelevant; it is the callable in `by=` that must be refused.
    (
        'df.add_prefix("p").T.groupby(by=df.eval)',
        r"kwarg 'by' of 'groupby' must be a literal or safe df expression",
    ),
    # Each POTENTIALLY_DANGEROUS_DF_METHODS entry, as a value not a call.
    ("df.groupby(by=df.eval)", r"kwarg 'by' of 'groupby' must be a literal"),
    ("df.groupby(by=df.query)", r"kwarg 'by' of 'groupby' must be a literal"),
    ("df.groupby(by=df.apply)", r"kwarg 'by' of 'groupby' must be a literal"),
    ("df.groupby(by=df.pipe)", r"kwarg 'by' of 'groupby' must be a literal"),
    ("df.groupby(by=df.agg)", r"kwarg 'by' of 'groupby' must be a literal"),
    ("df.groupby(by=df.transform)", r"kwarg 'by' of 'groupby' must be a literal"),
    # A method outside the allowlist entirely is smuggled the same way,
    # which is why the rule is on the node type, not on a name list.
    ("df.groupby(by=df.to_csv)", r"kwarg 'by' of 'groupby' must be a literal"),
    ("df.groupby(by=df.to_pickle)", r"kwarg 'by' of 'groupby' must be a literal"),
    # Positional, not just keyword.
    ("df.groupby(df.eval)", r"argument of 'groupby' must be a literal"),
    # Hidden inside a sequence argument.
    ("df.groupby(by=[df.eval])", r"kwarg 'by' of 'groupby' must be a literal"),
    ("df.groupby(by=(df.eval,))", r"kwarg 'by' of 'groupby' must be a literal"),
    ("df.groupby(by=['state', df.eval])", r"kwarg 'by' of 'groupby' must be a literal"),
    # Reached through an allowlisted hop rather than off `df` directly.
    ("df.groupby(by=df.T.eval)", r"kwarg 'by' of 'groupby' must be a literal"),
    ("df.groupby(by=df.head(2).eval)", r"kwarg 'by' of 'groupby' must be a literal"),
    # Smuggled through a container or operator node: each of these evaluates
    # to the bound method even though no individual node looks dangerous, so
    # _arg_ok has to recurse through subscripts and operators.
    ("df.groupby(by=[df.eval][0])", r"kwarg 'by' of 'groupby' must be a literal"),
    (
        "df.groupby(by=([df.eval] + [])[0])",
        r"kwarg 'by' of 'groupby' must be a literal",
    ),
    ("df.groupby(by=(df.eval, 'x')[0])", r"kwarg 'by' of 'groupby' must be a literal"),
    ("df.groupby(by=[[df.eval]][0][0])", r"kwarg 'by' of 'groupby' must be a literal"),
    # An allowlisted trailing name must not launder what sits under it, and a
    # data attribute is only a *pandas* attribute when the receiver is a
    # DataFrame/Series: df.shape.index is tuple.index, a callable.
    ("df.groupby(by=df.eval.index)", r"kwarg 'by' of 'groupby' must be a literal"),
    ("df.groupby(by=df.shape.index)", r"kwarg 'by' of 'groupby' must be a literal"),
    ("df.groupby(by=df.axes.index)", r"kwarg 'by' of 'groupby' must be a literal"),
    # A receiver must actually be pandas-valued: an operator expression over
    # plain literals is not, and ['a'].index / ('a',).index are callables
    # that pandas will happily invoke on every index label.
    (
        "df.groupby((['a'] + ['b']).index)['v'].sum()",
        r"argument of 'groupby' must be a literal",
    ),
    ("df.groupby(['a'].index)", r"argument of 'groupby' must be a literal"),
    ("df.groupby((-1).real.index)", r"argument of 'groupby' must be a literal"),
    # Chaining through a pandas receiver must not re-admit its methods.
    ("df.groupby(by=df.index.map)", r"kwarg 'by' of 'groupby' must be a literal"),
    ("df.groupby(by=df.columns.map)", r"kwarg 'by' of 'groupby' must be a literal"),
    (
        "df.groupby(by=df.index.values.tolist)",
        r"kwarg 'by' of 'groupby' must be a literal",
    ),
    # Indexers are callable objects, and calling one returns another.
    ("df.groupby(by=df.iloc)", r"kwarg 'by' of 'groupby' must be a literal"),
    ("df.groupby(by=df.loc)", r"kwarg 'by' of 'groupby' must be a literal"),
    ("df.groupby(by=df.loc(axis=0))", r"kwarg 'by' of 'groupby' must be a literal"),
    # Accessor *methods* are callables; only data members are allowed.
    (
        "df.groupby(by=df['x'].str.lower)",
        r"kwarg 'by' of 'groupby' must be a literal",
    ),
    (
        "df.groupby(by=df['d'].dt.to_period)",
        r"kwarg 'by' of 'groupby' must be a literal",
    ),
    # Attribute-style column access as an argument is refused along with it:
    # pandas resolves `df.<name>` to a method when the name collides with
    # one, so it cannot be told apart from the attack. Use df['state'].
    ("df.groupby(df.state)", r"argument of 'groupby' must be a literal"),
    # Other sinks that call an argument.
    ("df.filter(df.eval)", r"argument of 'filter' must be a literal"),
    ("df.combine(df, df.eval)", r"argument of 'combine' must be a literal"),
    ("df.sort_values('a', key=df.eval)", r"kwarg 'key' of 'sort_values' must be"),
    # The numeric limit must still be enforced through the new arg check.
    ("df.head(12345678901)", r"numeric constant exceeds limit"),
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
# GHSA-jccq-v27v-rhjv -- the two sinks, not just the validator helper.
#
# Both TableChatAgent.pandas_eval and VectorStore.compute_from_docs run
# sanitize_command() on their default (full_eval=False) path. Before the fix,
# an expression ending in ``.groupby(by=df.eval)`` passed validation and
# pandas evaluated it, returning a DataFrameGroupBy; the callable was invoked
# inside pandas, where langroid's restricted __builtins__ no longer applies.
# These tests assert the expression is now refused *at the call sites* and
# never evaluated -- the result is the validator's error, not a GroupBy.
# ---------------------------------------------------------------------------

# A method reference passed where pandas will call it. No payload: the point
# of the test is that the expression is refused before pandas ever runs it.
_CALLABLE_ARG_EXPR = 'df.add_prefix("p").T.groupby(by=df.eval)'
_REFUSED = "must be a literal or safe df expression"


def test_pandas_eval_refuses_callable_argument():
    """TableChatAgent.pandas_eval must refuse a callable passed as an arg."""
    from types import SimpleNamespace

    from langroid.agent.special.table_chat_agent import (
        PandasEvalTool,
        TableChatAgent,
        TableChatAgentConfig,
    )

    # Bound to a stand-in so the test needs no LLM and no API key;
    # pandas_eval only reads self.df and self.config.full_eval.
    agent = SimpleNamespace(
        sent_expression=False,
        df=_DF,
        config=TableChatAgentConfig(data=_DF),
    )
    assert agent.config.full_eval is False, "test must exercise the default path"

    result = TableChatAgent.pandas_eval(
        agent, PandasEvalTool(expression=_CALLABLE_ARG_EXPR)
    )

    assert "UnsafeCommandError" in result, result
    assert _REFUSED in result, result
    # The expression must not have been evaluated at all.
    assert "GroupBy" not in result, result


def test_compute_from_docs_refuses_callable_argument():
    """VectorStore.compute_from_docs must refuse a callable passed as an arg."""
    from types import SimpleNamespace

    from langroid.mytypes import DocMetaData, Document
    from langroid.vector_store.base import VectorStore, VectorStoreConfig

    store = SimpleNamespace(config=VectorStoreConfig())
    assert store.config.full_eval is False, "test must exercise the default path"
    docs = [Document(content="x", metadata=DocMetaData(source="test"))]

    result = VectorStore.compute_from_docs(store, docs, _CALLABLE_ARG_EXPR)

    assert _REFUSED in result, result
    assert "GroupBy" not in result, result


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


@pytest.mark.parametrize(
    "values, expected",
    [
        ([["long" * 251, "right"]], "long" * 250 + "..."),
        ([["left", 7]], "left"),
    ],
    ids=["truncates_duplicate_text_columns", "preserves_duplicate_mixed_columns"],
)
def test_stringify_handles_duplicate_column_names(values, expected):
    """Stringify each column by position so duplicate labels stay valid."""
    frame = pd.DataFrame(values, columns=["duplicate", "duplicate"])

    result = stringify(frame)

    assert expected in result


@pytest.mark.parametrize("copy_on_write", [False, True])
@pytest.mark.parametrize("as_series", [False, True])
def test_stringify_preserves_tabular_input(
    copy_on_write: bool, as_series: bool
) -> None:
    """Display truncation must preserve every row of the caller's data."""
    values = ["x" * 1001, None, 7] + [f"row {i}" for i in range(3, 11)]
    values.append("y" * 1001)  # The final row is not displayed.
    series = pd.Series(values, name="text", dtype=object)
    original = series if as_series else series.to_frame()
    before = original.copy(deep=True)
    expected = pd.DataFrame({"text": ["x" * 1000 + "..."] + values[1:10]})

    with pd.option_context("mode.copy_on_write", copy_on_write):
        rendered = stringify(original)

    if isinstance(original, pd.Series):
        pd.testing.assert_series_equal(original, before)
    else:
        pd.testing.assert_frame_equal(original, before)
    assert rendered == expected.to_string(index=False)


@pytest.mark.parametrize("value", ["plain text", 42, [1, 2]])
def test_stringify_non_tabular_input(value: object) -> None:
    assert stringify(value) == str(value)
