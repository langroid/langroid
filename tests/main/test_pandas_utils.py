import pytest
from langroid.utils.pandas_utils import sanitize_command, UnsafeCommandError


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

DEEP_EXPR = "df" + "[0]" * 26               # depth bomb (26 > MAX_DEPTH)

BLOCK_WITH_MSG = [
    ("df.eval('2+2')",
     r"method 'eval' not permitted"),

    ("df.sample(n=5, regex=True)",
     r"kwarg 'regex' is blocked"),

    ("df['b'] * 12345678901",
     r"numeric constant exceeds limit"),

    ("df['a'] ** 8",
     r"operator not allowed"),

    ("df.head().tail().sort_values('a').groupby('state').sum().mean().std()",
     r"method-chain too long"),

    ("df.sample(n=10, inplace=True)",
     r"kwarg 'inplace' is blocked"),

    ("sales.sum()",
     r"unexpected variable 'sales'"),

    ("df2.head()",
     r"unexpected variable 'df2'"),

    ("df[other_var]",
     r"subscript must be literal"),

    ("df.where(df['income'] > other_var)['income']",
     r"unexpected variable 'other_var'"),

    (DEEP_EXPR,
     r"AST nesting too deep"),
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
