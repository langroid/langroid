import ast
from typing import Any, Dict

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


# DataFrame attributes that hold *data* rather than a callable, so they stay
# usable as call arguments (e.g. ``df.filter(items=df.columns)``). These are
# allowlists on purpose: any other attribute reference as an argument is
# refused, since a name that collides with a method resolves to that method
# (GHSA-jccq-v27v-rhjv). See :func:`_arg_ok`.
SAFE_DF_DATA_ATTRS = {
    "T",
    "axes",
    "columns",
    "dtype",
    "dtypes",
    "empty",
    "index",
    "name",
    "names",
    "ndim",
    "shape",
    "size",
    "values",
}

# Attributes that evaluate to a pandas object (DataFrame, Series or Index), so
# attribute access may continue through them -- ``df.dtypes.index`` and
# ``df.index.year`` are pandas data. Anything else in SAFE_DF_DATA_ATTRS
# yields a tuple, ndarray or scalar whose own attributes are NOT pandas
# attributes: ``df.shape.index`` is ``tuple.index``, a callable.
# NOTE: ``axes`` is deliberately absent -- it returns a plain list, so
# ``df.axes.index`` would be ``list.index``, a callable.
CHAINABLE_DF_ATTRS = {"T", "columns", "dtypes", "index"}

# Index/position accessors. Usable as the receiver of a subscript
# (``df.iloc[:1]``), never as a value of their own: the indexer object is
# itself callable, and calling it returns another callable indexer.
INDEXER_ATTRS = {"loc", "iloc", "at", "iat"}

# Namespace accessors (``df['d'].dt.month``, ``df['s'].str.len()``).
ACCESSOR_ATTRS = {"cat", "dt", "sparse", "str"}

# Members of those accessors that hold data rather than a callable. Accessor
# *methods* (``.str.lower``) stay refused as references; calling them
# (``.str.lower()``) is a Call and unaffected.
ACCESSOR_DATA_MEMBERS = {
    "categories",
    "codes",
    "components",
    "date",
    "day",
    "days",
    "day_of_week",
    "day_of_year",
    "end_time",
    "dayofweek",
    "dayofyear",
    "days_in_month",
    "daysinmonth",
    "hour",
    "is_leap_year",
    "is_month_end",
    "is_month_start",
    "is_quarter_end",
    "is_quarter_start",
    "is_year_end",
    "is_year_start",
    "microsecond",
    "minute",
    "month",
    "nanosecond",
    "ordered",
    "quarter",
    "second",
    "seconds",
    "start_time",
    "time",
    "tz",
    "week",
    "weekday",
    "weekofyear",
    "year",
}


# Terminal attribute names that hold data, on any pandas receiver. The two
# sets are merged because the same names are data on a DataFrame, a Series
# and an Index: ``df.index.year`` and ``df['d'].dt.year`` are both data.
_DATA_ATTRS = SAFE_DF_DATA_ATTRS | ACCESSOR_DATA_MEMBERS


def _pandas_valued(node: ast.AST) -> bool:
    """Return True if *node* evaluates to a pandas object.

    Attribute access is only meaningful as a *pandas* attribute when the
    receiver is a DataFrame, Series or Index; otherwise an allowlisted name
    belongs to some other object (``df.shape.index`` is ``tuple.index``, a
    callable).

    Args:
        node: The AST node of the receiver.

    Returns:
        True if the node evaluates to a pandas object.
    """
    if isinstance(node, (ast.Name, ast.Call)):
        # NOTE: treating every allowlisted call as pandas-valued is a bet that
        # no whitelisted method returns an object carrying a *dangerous*
        # callable under one of the data-attribute names. The cases found so
        # far are benign -- e.g. GroupBy.size, dict.values, str.index,
        # tuple.index, Timestamp.date -- in that they search, size or convert,
        # and none interprets a string handed to it as Python code or as a
        # module name; the list is not exhaustive. Tightening this would mean
        # mapping each method to its result type (open-ended, and a
        # conservative "unknown" would reject many legitimate expressions);
        # the complete alternative is a runtime check at the two sinks that
        # rejects a callable argument, including one nested in a sequence
        # argument, since pandas accepts a list of groupers.
        return True
    if isinstance(node, ast.BinOp):
        # Arithmetic involving a pandas object yields a pandas object, e.g.
        # ``(df + 1).iloc[:1]``. At least one operand must be pandas-valued:
        # ``(['a'] + ['b']).index`` is ``list.index``, a callable.
        return _pandas_valued(node.left) or _pandas_valued(node.right)
    if isinstance(node, ast.UnaryOp):
        return _pandas_valued(node.operand)
    if isinstance(node, ast.Compare):
        # e.g. ``(df['a'] > 0).size``
        return _pandas_valued(node.left) or any(
            _pandas_valued(c) for c in node.comparators
        )
    if isinstance(node, ast.Subscript):
        value = node.value
        if isinstance(value, ast.Attribute) and value.attr in INDEXER_ATTRS:
            return _pandas_valued(value.value)
        return _pandas_valued(value)
    if isinstance(node, ast.Attribute):
        return node.attr in CHAINABLE_DF_ATTRS and _pandas_valued(node.value)
    return False


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


def _arg_ok(node: ast.AST) -> bool:
    """Return True if *node* may be passed as a call argument.

    Several pandas APIs *call* an argument they are given --
    ``df.groupby(by=<callable>)`` applies it to every index label, for
    instance. :meth:`CommandValidator.visit_Call` only checks the name of a
    method that is being *called*, so an unparenthesized reference such as
    ``df.eval`` is a callable in disguise that would otherwise slip past
    :data:`WHITELISTED_DF_METHODS` and be invoked inside pandas, where
    langroid's restricted ``__builtins__`` no longer applies
    (GHSA-jccq-v27v-rhjv). Methods outside the allowlist entirely
    (``df.to_csv`` and friends) are smuggled the same way, which is why the
    rule is structural rather than a list of method names.

    An argument must therefore evaluate to *data*: a safe literal, a method
    call (itself allowlisted), the DataFrame name, a data attribute from
    :data:`SAFE_DF_DATA_ATTRS`, or a subscript / sequence / arithmetic /
    comparison expression built from those. Every container and operator node
    is recursed into, because a callable is otherwise easily smuggled through
    one -- ``[df.eval][0]`` and ``([df.eval] + [])[0]`` both evaluate to the
    bound method while every individual node looks innocuous.

    Note that attribute-style *column* access (``df.state``) is refused as an
    argument: pandas resolves attribute access to the method whenever the name
    collides with one, so the two cases are indistinguishable here. Use
    ``df['state']`` instead -- the bracket form the agent's system-message
    examples use. Outside argument position ``df.state.mean()`` still works.

    Args:
        node: The AST node of a positional argument or keyword value.

    Returns:
        True if the node is an acceptable argument, False otherwise.

    Raises:
        UnsafeCommandError: If a numeric literal exceeds ``NUMERIC_LIMIT``.
    """
    if _literal_ok(node):
        return True
    if isinstance(node, (ast.Tuple, ast.List)):
        return all(_arg_ok(elt) for elt in node.elts)
    if isinstance(node, ast.Slice):
        return all(
            sub is None or _arg_ok(sub) for sub in (node.lower, node.upper, node.step)
        )
    if isinstance(node, ast.Subscript):
        # ``visit_Subscript`` already requires a literal index; the *indexed*
        # value still has to be data, or ``[df.eval][0]`` gets through.
        value = node.value
        if isinstance(value, ast.Attribute) and value.attr in INDEXER_ATTRS:
            # ``df.iloc[:1]`` -- indexing an indexer yields data.
            return _pandas_valued(value.value)
        return _arg_ok(value)
    if isinstance(node, ast.BinOp):
        return _arg_ok(node.left) and _arg_ok(node.right)
    if isinstance(node, ast.UnaryOp):
        return _arg_ok(node.operand)
    if isinstance(node, ast.Compare):
        return _arg_ok(node.left) and all(_arg_ok(c) for c in node.comparators)
    if isinstance(node, ast.Attribute):
        if node.attr in INDEXER_ATTRS:
            # The bare indexer object is callable.
            return False
        receiver = node.value
        if isinstance(receiver, ast.Attribute) and receiver.attr in ACCESSOR_ATTRS:
            # ``df['d'].dt.month`` -- a data member of a namespace accessor.
            return node.attr in ACCESSOR_DATA_MEMBERS and _pandas_valued(receiver.value)
        # A data attribute, and only off something pandas-valued, so that
        # ``df.index.year`` works while ``df.shape.index`` does not.
        return node.attr in _DATA_ATTRS and _pandas_valued(receiver)
    if isinstance(node, ast.Call):
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr in INDEXER_ATTRS:
            # ``df.loc(axis=0)`` returns another callable indexer.
            return False
        # Any other allowlisted method call returns data, never a callable.
        return True
    return isinstance(node, ast.Name)


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

    # Attribute access
    def visit_Attribute(self, node: ast.Attribute) -> None:
        # Block dunder attributes to prevent access to __init__, __globals__, etc.
        if node.attr.startswith("__") and node.attr.endswith("__"):
            raise UnsafeCommandError(f"dunder attribute '{node.attr}' not allowed")
        # Block single underscore private attributes as well for defense in depth
        if node.attr.startswith("_") and node.attr not in WHITELISTED_DF_METHODS:
            raise UnsafeCommandError(f"private attribute '{node.attr}' not allowed")
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

        # Blocked kwargs are rejected by name regardless of their value.
        for kw in node.keywords:
            if kw.arg in BLOCKED_KW:
                raise UnsafeCommandError(f"kwarg '{kw.arg}' is blocked")

        # Walk the subtree first, so that the more specific policies (dunder
        # and private attributes, unexpected variables, numeric limits) report
        # their own error rather than the generic argument-shape one below.
        try:
            self.generic_visit(node)
        finally:
            self.chain -= 1

        # Argument shape: an argument must be data, never a callable
        # (GHSA-jccq-v27v-rhjv). See :func:`_arg_ok`.
        for kw in node.keywords:
            if not _arg_ok(kw.value):
                raise UnsafeCommandError(
                    f"kwarg '{kw.arg}' of '{method}' must be a literal "
                    "or safe df expression"
                )
        for arg in node.args:
            if not _arg_ok(arg):
                raise UnsafeCommandError(
                    f"argument of '{method}' must be a literal " "or safe df expression"
                )

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


# Curated ``__builtins__`` dict for ``eval()`` calls on LLM-generated pandas
# expressions (GHSA-q9p7-wqxg-mrhc). Without this, Python implicitly injects
# the full builtins into eval's globals -- so even with an empty ``locals={}``
# and ``full_eval=True``, an expression like
# ``__import__('os').system('...')`` would execute and yield RCE. Restricting
# builtins to this curated read-only set closes the direct primitive while
# leaving enough functionality for typical pandas expressions.
_SAFE_BUILTINS: Dict[str, Any] = {
    # Constants
    "True": True,
    "False": False,
    "None": None,
    # Safe builtins occasionally used in pandas-style expressions
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "dict": dict,
    "enumerate": enumerate,
    "float": float,
    "int": int,
    "len": len,
    "list": list,
    "max": max,
    "min": min,
    "range": range,
    "reversed": reversed,
    "round": round,
    "set": set,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "zip": zip,
}


def safe_eval_globals(local_vars: Dict[str, Any]) -> Dict[str, Any]:
    """Return a globals dict for :func:`eval` with ``__builtins__`` restricted
    to a safe read-only set, so that bare names such as ``__import__``,
    ``eval``, ``exec`` and ``open`` do not resolve via Python's implicit
    builtin injection.

    This is defence in depth for expressions that have already passed
    :func:`sanitize_command`. It is not a sandbox on its own and does not make
    an unvalidated (``full_eval=True``) expression safe; see SECURITY.md.

    Args:
        local_vars: User-provided variables to expose to the eval'd
            expression (e.g. ``{"df": <DataFrame>}``).

    Returns:
        A new dict suitable as the ``globals`` argument of ``eval()``.
    """
    return {**local_vars, "__builtins__": _SAFE_BUILTINS}


def stringify(x: Any) -> str:
    # Convert x to DataFrame if it is not one already
    if isinstance(x, pd.Series):
        df = x.head(10).to_frame().copy()
    elif not isinstance(x, pd.DataFrame):
        return str(x)
    else:
        df = x.head(10).copy()

    # Truncate long text columns to 1000 characters
    for position in range(df.shape[1]):
        column = df.iloc[:, position]
        if column.dtype == object:
            df.isetitem(
                position,
                column.apply(
                    lambda item: (
                        (item[:1000] + "...")
                        if isinstance(item, str) and len(item) > 1000
                        else item
                    )
                ),
            )

    # Convert to string
    return df.to_string(index=False)  # type: ignore
