import pytest

from langroid.utils.pandas_utils import UnsafeCommandError, sanitize_command

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
