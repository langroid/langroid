import ast
from typing import Any

import pandas as pd

COMMON_USE_DF_METHODS = {
    "T",
    "abs",
    "add",
    "add_prefix",
    "add_suffix",
    "agg",
    "aggregate",
    "align",
    "all",
    "any",
    "apply",
    "applymap",
    "assign",
    "at",
    "at_time",
    "between_time",
    "bfill",
    "clip",
    "combine",
    "combine_first",
    "convert_dtypes",
    "corr",
    "corrwith",
    "count",
    "cov",
    "cummax",
    "cummin",
    "cumprod",
    "cumsum",
    "describe",
    "diff",
    "dot",
    "drop_duplicates",
    "duplicated",
    "eq",
    "eval",
    "ewm",
    "expanding",
    "explode",
    "filter",
    "first",
    "groupby",
    "head",
    "idxmax",
    "idxmin",
    "infer_objects",
    "interpolate",
    "isin",
    "kurt",
    "kurtosis",
    "last",
    "le",
    "loc",
    "lt",
    "gt",
    "ge",
    "iloc",
    "mask",
    "max",
    "mean",
    "median",
    "melt",
    "min",
    "mode",
    "mul",
    "nlargest",
    "nsmallest",
    "notna",
    "notnull",
    "nunique",
    "pct_change",
    "pipe",
    "pivot",
    "pivot_table",
    "prod",
    "product",
    "quantile",
    "query",
    "rank",
    "replace",
    "resample",
    "rolling",
    "round",
    "sample",
    "select_dtypes",
    "sem",
    "shift",
    "skew",
    "sort_index",
    "sort_values",
    "squeeze",
    "stack",
    "std",
    "sum",
    "tail",
    "transform",
    "transpose",
    "unstack",
    "value_counts",
    "var",
    "where",
    "xs",
}

POTENTIALLY_DANGEROUS_DF_METHODS = {
    "eval",
    "query",
    "apply",
    "applymap",
    "pipe",
    "agg",
    "aggregate",
    "transform",
    "rolling",
    "expanding",
    "resample",
}

WHITELISTED_DF_METHODS = COMMON_USE_DF_METHODS - POTENTIALLY_DANGEROUS_DF_METHODS


BLOCKED_KW = {
    "engine",
    "parser",
    "inplace",
    "regex",
    "dtype",
    "converters",
    "eval",
}
MAX_CHAIN = 6
MAX_DEPTH = 25
NUMERIC_LIMIT = 1_000_000_000


class UnsafeCommandError(ValueError):
    """Raised when a command string violates security policy."""

    pass


def _literal_ok(node: ast.AST) -> bool:
    """Return True if *node* is a safe literal (and within numeric limit)."""
    if isinstance(node, ast.Constant):
        if (
            isinstance(node.value, (int, float, complex))
            and abs(node.value) > NUMERIC_LIMIT
        ):
            raise UnsafeCommandError("numeric constant exceeds limit")
        return True
    if isinstance(node, (ast.Tuple, ast.List)):
        return all(_literal_ok(elt) for elt in node.elts)
    if isinstance(node, ast.Slice):
        return all(
            sub is None or _literal_ok(sub)
            for sub in (node.lower, node.upper, node.step)
        )
    return False


class CommandValidator(ast.NodeVisitor):
    """AST walker that enforces the security policy."""

    # Comparison operators we allow
    ALLOWED_CMPOP = (ast.Gt, ast.GtE, ast.Lt, ast.LtE, ast.Eq, ast.NotEq)

    # Arithmetic operators we allow (power ** intentionally omitted)
    ALLOWED_BINOP = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod)
    ALLOWED_UNARY = (ast.UAdd, ast.USub)

    # Node whitelist
    ALLOWED_NODES = (
        ast.Expression,
        ast.Attribute,
        ast.Name,
        ast.Load,
        ast.Call,
        ast.Subscript,
        ast.Constant,
        ast.Tuple,
        ast.List,
        ast.Slice,
        ast.keyword,
        ast.BinOp,
        ast.UnaryOp,
        ast.Compare,
        *ALLOWED_BINOP,
        *ALLOWED_UNARY,
        *ALLOWED_CMPOP,
    )

    def __init__(self, df_name: str = "df"):
        self.df_name = df_name
        self.depth = 0
        self.chain = 0

    # Depth guard
    def generic_visit(self, node: ast.AST) -> None:
        self.depth += 1
        if self.depth > MAX_DEPTH:
            raise UnsafeCommandError("AST nesting too deep")
        super().generic_visit(node)
        self.depth -= 1

    # Literal validation
    def visit_Constant(self, node: ast.Constant) -> None:
        _literal_ok(node)

    # Arithmetic
    def visit_BinOp(self, node: ast.BinOp) -> None:
        if not isinstance(node.op, self.ALLOWED_BINOP):
            raise UnsafeCommandError("operator not allowed")
        self.generic_visit(node)

    def visit_UnaryOp(self, node: ast.UnaryOp) -> None:
        if not isinstance(node.op, self.ALLOWED_UNARY):
            raise UnsafeCommandError("unary operator not allowed")
        self.generic_visit(node)

    # Comparisons
    def visit_Compare(self, node: ast.Compare) -> None:
        if not all(isinstance(op, self.ALLOWED_CMPOP) for op in node.ops):
            raise UnsafeCommandError("comparison operator not allowed")
        for comp in node.comparators:
            _literal_ok(comp)
        self.generic_visit(node)

    # Subscripts
    def visit_Subscript(self, node: ast.Subscript) -> None:
        if not _literal_ok(node.slice):
            raise UnsafeCommandError("subscript must be literal")
        self.generic_visit(node)

    # Method calls
    def visit_Call(self, node: ast.Call) -> None:
        if not isinstance(node.func, ast.Attribute):
            raise UnsafeCommandError("only DataFrame method calls allowed")

        method = node.func.attr
        self.chain += 1
        if self.chain > MAX_CHAIN:
            raise UnsafeCommandError("method-chain too long")
        if method not in WHITELISTED_DF_METHODS:
            raise UnsafeCommandError(f"method '{method}' not permitted")

        # kwarg / arg checks
        for kw in node.keywords:
            if kw.arg in BLOCKED_KW:
                raise UnsafeCommandError(f"kwarg '{kw.arg}' is blocked")
            _literal_ok(kw.value)
        for arg in node.args:
            _literal_ok(arg)

        try:
            self.generic_visit(node)
        finally:
            self.chain -= 1

    # Names
    def visit_Name(self, node: ast.Name) -> None:
        if node.id != self.df_name:
            raise UnsafeCommandError(f"unexpected variable '{node.id}'")

    # Top-level gate
    def visit(self, node: ast.AST) -> None:
        if not isinstance(node, self.ALLOWED_NODES):
            raise UnsafeCommandError(f"disallowed node {type(node).__name__}")
        super().visit(node)


def sanitize_command(expr: str, df_name: str = "df") -> str:
    """
    Validate *expr*; return it unchanged if it passes all rules,
    else raise UnsafeCommandError with the first violation encountered.
    """
    tree = ast.parse(expr, mode="eval")
    CommandValidator(df_name).visit(tree)
    return expr


def stringify(x: Any) -> str:
    # Convert x to DataFrame if it is not one already
    if isinstance(x, pd.Series):
        df = x.to_frame()
    elif not isinstance(x, pd.DataFrame):
        return str(x)
    else:
        df = x

    # Truncate long text columns to 1000 characters
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].apply(
                lambda item: (
                    (item[:1000] + "...")
                    if isinstance(item, str) and len(item) > 1000
                    else item
                )
            )

    # Limit to 10 rows
    df = df.head(10)

    # Convert to string
    return df.to_string(index=False)  # type: ignore
