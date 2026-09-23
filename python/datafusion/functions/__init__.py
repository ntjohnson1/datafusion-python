# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""Scalar, aggregate, and window functions for :py:class:`~datafusion.expr.Expr`.

Each function returns an :py:class:`~datafusion.expr.Expr` that can be combined
with other expressions and passed to
:py:class:`~datafusion.dataframe.DataFrame` methods such as
:py:meth:`~datafusion.dataframe.DataFrame.select`,
:py:meth:`~datafusion.dataframe.DataFrame.filter`,
:py:meth:`~datafusion.dataframe.DataFrame.aggregate`, and
:py:meth:`~datafusion.dataframe.DataFrame.window`. The module is conventionally
imported as ``F`` so calls read like ``F.sum("price")``.

Arguments that hold the data being operated on also accept a plain string, which
is read as a column name: ``F.sum("price")`` is the same as
``F.sum(datafusion.col("price"))``. Arguments that hold a fixed scalar instead
read a plain string as a literal, so ``F.array_to_string("tags", ",")`` joins the
values of column ``tags`` with a comma. Arguments holding a value compared
*against* the data -- ``element`` in :py:func:`array_append`, ``key`` in
:py:func:`map_extract` -- are ambiguous either way, so they still require an
explicit :py:func:`~datafusion.col` or :py:func:`~datafusion.lit`.

Examples:
    >>> from datafusion import functions as F
    >>> ctx = dfn.SessionContext()
    >>> df = ctx.from_pydict({"a": [1, 2, 3, 4]})
    >>> df.aggregate([], [F.sum("a").alias("total")]).to_pydict()
    {'total': [10]}

See :ref:`aggregation` and :ref:`window_functions` in the online documentation
for categorized catalogs of aggregate and window functions.
"""

from __future__ import annotations

import inspect
import warnings
from typing import TYPE_CHECKING, Any

import pyarrow as pa

if TYPE_CHECKING:
    from collections.abc import Callable

from datafusion._internal import functions as f
from datafusion.common import NullTreatment
from datafusion.expr import (
    CaseBuilder,
    Expr,
    SortExpr,
    SortKey,
    coerce_to_column,
    coerce_to_column_list,
    coerce_to_column_or_none,
    coerce_to_literal,
    coerce_to_literal_list,
    coerce_to_literal_or_none,
    ensure_expr,
    ensure_expr_or_none,
    expr_list_to_raw_expr_list,
    sort_list_to_raw_sort_list,
    sort_or_default,
)
from datafusion.functions import spark


def _warn_if_expr_for_literal_arg(
    value: Any, function_name: str, arg_name: str
) -> None:
    if isinstance(value, Expr):
        warnings.warn(
            f"Passing Expr for {function_name}() argument {arg_name!r} is deprecated; "
            "pass a Python literal instead.",
            DeprecationWarning,
            stacklevel=3,
        )


__all__ = [
    "abs",
    "acos",
    "acosh",
    "alias",
    "any_match",
    "approx_distinct",
    "approx_median",
    "approx_percentile_cont",
    "approx_percentile_cont_with_weight",
    "array",
    "array_agg",
    "array_any_match",
    "array_any_value",
    "array_append",
    "array_cat",
    "array_compact",
    "array_concat",
    "array_contains",
    "array_dims",
    "array_distance",
    "array_distinct",
    "array_element",
    "array_empty",
    "array_except",
    "array_extract",
    "array_filter",
    "array_has",
    "array_has_all",
    "array_has_any",
    "array_indexof",
    "array_intersect",
    "array_join",
    "array_length",
    "array_max",
    "array_min",
    "array_ndims",
    "array_normalize",
    "array_pop_back",
    "array_pop_front",
    "array_position",
    "array_positions",
    "array_prepend",
    "array_push_back",
    "array_push_front",
    "array_remove",
    "array_remove_all",
    "array_remove_n",
    "array_repeat",
    "array_replace",
    "array_replace_all",
    "array_replace_n",
    "array_resize",
    "array_reverse",
    "array_slice",
    "array_sort",
    "array_to_string",
    "array_transform",
    "array_union",
    "arrays_overlap",
    "arrays_zip",
    "arrow_cast",
    "arrow_field",
    "arrow_metadata",
    "arrow_try_cast",
    "arrow_typeof",
    "ascii",
    "asin",
    "asinh",
    "atan",
    "atan2",
    "atanh",
    "avg",
    "bit_and",
    "bit_length",
    "bit_or",
    "bit_xor",
    "bool_and",
    "bool_or",
    "btrim",
    "cardinality",
    "case",
    "cast_to_type",
    "cbrt",
    "ceil",
    "char_length",
    "character_length",
    "chr",
    "coalesce",
    "col",
    "concat",
    "concat_ws",
    "contains",
    "corr",
    "cos",
    "cosh",
    "cosine_distance",
    "cot",
    "count",
    "count_star",
    "covar",
    "covar_pop",
    "covar_samp",
    "cume_dist",
    "current_date",
    "current_time",
    "current_timestamp",
    "date_bin",
    "date_format",
    "date_part",
    "date_trunc",
    "datepart",
    "datetrunc",
    "decode",
    "degrees",
    "dense_rank",
    "digest",
    "dot_product",
    "element_at",
    "empty",
    "encode",
    "ends_with",
    "exp",
    "extract",
    "factorial",
    "find_in_set",
    "first_value",
    "flatten",
    "floor",
    "from_unixtime",
    "gcd",
    "gen_series",
    "generate_series",
    "get_field",
    "greatest",
    "grouping",
    "ifnull",
    "in_list",
    "initcap",
    "inner_product",
    "instr",
    "is_nan",
    "isnan",
    "iszero",
    "lag",
    "lambda_",
    "lambda_var",
    "last_value",
    "lcm",
    "lead",
    "least",
    "left",
    "length",
    "levenshtein",
    "list_any_match",
    "list_any_value",
    "list_append",
    "list_cat",
    "list_compact",
    "list_concat",
    "list_contains",
    "list_dims",
    "list_distance",
    "list_distinct",
    "list_element",
    "list_empty",
    "list_except",
    "list_extract",
    "list_filter",
    "list_has",
    "list_has_all",
    "list_has_any",
    "list_indexof",
    "list_intersect",
    "list_join",
    "list_length",
    "list_max",
    "list_min",
    "list_ndims",
    "list_normalize",
    "list_overlap",
    "list_pop_back",
    "list_pop_front",
    "list_position",
    "list_positions",
    "list_prepend",
    "list_push_back",
    "list_push_front",
    "list_remove",
    "list_remove_all",
    "list_remove_n",
    "list_repeat",
    "list_replace",
    "list_replace_all",
    "list_replace_n",
    "list_resize",
    "list_reverse",
    "list_slice",
    "list_sort",
    "list_to_string",
    "list_transform",
    "list_union",
    "list_zip",
    "ln",
    "log",
    "log2",
    "log10",
    "lower",
    "lpad",
    "ltrim",
    "make_array",
    "make_date",
    "make_list",
    "make_map",
    "make_time",
    "map_entries",
    "map_extract",
    "map_keys",
    "map_values",
    "max",
    "md5",
    "mean",
    "median",
    "min",
    "named_struct",
    "nanvl",
    "now",
    "nth_value",
    "ntile",
    "nullif",
    "nvl",
    "nvl2",
    "octet_length",
    "order_by",
    "overlay",
    "percent_rank",
    "percentile_cont",
    "pi",
    "position",
    "pow",
    "power",
    "quantile_cont",
    "radians",
    "random",
    "range",
    "rank",
    "regexp_count",
    "regexp_instr",
    "regexp_like",
    "regexp_match",
    "regexp_replace",
    "regr_avgx",
    "regr_avgy",
    "regr_count",
    "regr_intercept",
    "regr_r2",
    "regr_slope",
    "regr_sxx",
    "regr_sxy",
    "regr_syy",
    "repeat",
    "replace",
    "reverse",
    "right",
    "round",
    "row",
    "row_number",
    "rpad",
    "rtrim",
    "sha224",
    "sha256",
    "sha384",
    "sha512",
    "signum",
    "sin",
    "sinh",
    "spark",
    "split_part",
    "sqrt",
    "starts_with",
    "stddev",
    "stddev_pop",
    "stddev_samp",
    "string_agg",
    "string_to_array",
    "string_to_list",
    "strpos",
    "struct",
    "substr",
    "substr_index",
    "substring",
    "sum",
    "tan",
    "tanh",
    "to_char",
    "to_date",
    "to_hex",
    "to_local_time",
    "to_time",
    "to_timestamp",
    "to_timestamp_micros",
    "to_timestamp_millis",
    "to_timestamp_nanos",
    "to_timestamp_seconds",
    "to_unixtime",
    "today",
    "translate",
    "trim",
    "trunc",
    "try_cast_to_type",
    "union_extract",
    "union_tag",
    "upper",
    "uuid",
    "var",
    "var_pop",
    "var_population",
    "var_samp",
    "var_sample",
    "version",
    "when",
    "with_metadata",
]


def isnan(expr: Expr | str) -> Expr:
    """Returns true if a given number is +NaN or -NaN otherwise returns false.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"a": [1.0, np.nan]})
        >>> result = df.select(dfn.functions.isnan("a").alias("isnan"))
        >>> result.collect_column("isnan")[1].as_py()
        True
    """
    return Expr(f.isnan(coerce_to_column(expr)))


def is_nan(expr: Expr | str) -> Expr:
    """Alias for :func:`isnan`."""
    return isnan(expr)


def nullif(expr1: Expr | str, expr2: Expr | str) -> Expr:
    """Returns NULL if expr1 equals expr2; otherwise it returns expr1.

    This can be used to perform the inverse operation of the COALESCE expression.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"a": [1, 2], "b": [1, 3]})
        >>> result = df.select(dfn.functions.nullif("a", "b").alias("nullif"))
        >>> result.collect_column("nullif").to_pylist()
        [None, 2]
    """
    return Expr(f.nullif(coerce_to_column(expr1), coerce_to_column(expr2)))


def encode(expr: Expr | str, encoding: Expr | str) -> Expr:
    """Encode the ``input``, using the ``encoding``. encoding can be base64 or hex.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"a": ["hello"]})
        >>> result = df.select(dfn.functions.encode("a", "base64").alias("enc"))
        >>> result.collect_column("enc")[0].as_py()
        'aGVsbG8'
    """
    _warn_if_expr_for_literal_arg(encoding, "encode", "encoding")
    encoding = coerce_to_literal(encoding)
    return Expr(f.encode(coerce_to_column(expr), encoding))


def decode(expr: Expr | str, encoding: Expr | str) -> Expr:
    """Decode the ``input``, using the ``encoding``. encoding can be base64 or hex.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"a": ["aGVsbG8="]})
        >>> result = df.select(dfn.functions.decode("a", "base64").alias("dec"))
        >>> result.collect_column("dec")[0].as_py()
        b'hello'
    """
    _warn_if_expr_for_literal_arg(encoding, "decode", "encoding")
    encoding = coerce_to_literal(encoding)
    return Expr(f.decode(coerce_to_column(expr), encoding))


def array_to_string(expr: Expr | str, delimiter: Expr | str) -> Expr:
    """Converts each element to its text representation.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"a": [[1, 2, 3]]})
        >>> result = df.select(dfn.functions.array_to_string("a", ",").alias("s"))
        >>> result.collect_column("s")[0].as_py()
        '1,2,3'
    """
    delimiter = coerce_to_literal(delimiter)
    return Expr(f.array_to_string(coerce_to_column(expr), delimiter.cast(pa.string())))


def array_join(expr: Expr | str, delimiter: Expr | str) -> Expr:
    """Converts each element to its text representation.

    See Also:
        This is an alias for :py:func:`array_to_string`.
    """
    return array_to_string(expr, delimiter)


def list_to_string(expr: Expr | str, delimiter: Expr | str) -> Expr:
    """Converts each element to its text representation.

    See Also:
        This is an alias for :py:func:`array_to_string`.
    """
    return array_to_string(expr, delimiter)


def list_join(expr: Expr | str, delimiter: Expr | str) -> Expr:
    """Converts each element to its text representation.

    See Also:
        This is an alias for :py:func:`array_to_string`.
    """
    return array_to_string(expr, delimiter)


def lambda_var(name: str) -> Expr:
    """Create an unresolved reference to a lambda parameter by ``name``.

    Use this inside the body passed to :py:func:`lambda_` to refer to one of the
    lambda's parameters. The owning higher-order function (such as
    :py:func:`array_transform`) binds the variable to a concrete element type
    during query planning.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"a": [[1, 2, 3]]})
        >>> double_fn = F.lambda_(["v"], F.lambda_var("v") * lit(2))
        >>> df.select(
        ...     F.array_transform("a", double_fn).alias("d")
        ... ).collect_column("d")[0].as_py()
        [2, 4, 6]

    See Also:
        :py:func:`lambda_`, :py:func:`array_transform`, :py:func:`array_any_match`.
    """
    return Expr(f.lambda_var(name))


def lambda_(params: list[str], body: Expr | str) -> Expr:
    """Create a lambda expression from parameter names and a body expression.

    This is the explicit form of building a lambda. Most callers can instead
    pass a Python callable directly to a higher-order function such as
    :py:func:`array_transform`, which builds the lambda automatically. Reach for
    ``lambda_`` when you want explicit control over the parameter names.

    Args:
        params: Ordered lambda parameter names.
        body: Body expression that references the parameters via
            :py:func:`lambda_var`.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"a": [[1, 2, 3]]})
        >>> double_fn = F.lambda_(["v"], F.lambda_var("v") * lit(2))
        >>> df.select(
        ...     F.array_transform("a", double_fn).alias("d")
        ... ).collect_column("d")[0].as_py()
        [2, 4, 6]

    See Also:
        :py:func:`lambda_var`, :py:func:`array_transform`, :py:func:`array_any_match`.
    """
    return Expr(f.lambda_(params, coerce_to_column(body)))


def _to_lambda(fn: Expr | Callable[..., Any]) -> Expr:
    """Coerce ``fn`` to a lambda ``Expr``.

    Accepts either an ``Expr`` produced by :py:func:`lambda_` (returned
    unchanged) or a Python callable. A callable is introspected for its
    parameter names; those names become :py:func:`lambda_var` references passed
    positionally into the callable, and its return value (coerced to an
    ``Expr``) becomes the lambda body.
    """
    if isinstance(fn, Expr):
        return fn
    if not callable(fn):
        msg = f"expected an Expr or callable, got {type(fn).__name__}"
        raise TypeError(msg)
    params = list(inspect.signature(fn).parameters)
    if not params:
        msg = "lambda callable must accept at least one parameter"
        raise ValueError(msg)
    body = fn(*[lambda_var(p) for p in params])
    return lambda_(params, body if isinstance(body, Expr) else Expr.literal(body))


def array_transform(array: Expr | str, transform: Expr | Callable[..., Any]) -> Expr:
    """Transform each element of ``array`` with a lambda.

    ``transform`` may be a Python callable, which is converted to a lambda
    automatically (its parameter names become the lambda parameters), or an
    explicit lambda built with :py:func:`lambda_`.

    Examples:
        Using a Python callable:

        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"a": [[1, 2, 3]]})
        >>> df.select(
        ...     F.array_transform("a", lambda v: v * 2).alias("d")
        ... ).collect_column("d")[0].as_py()
        [2, 4, 6]

        Using an explicit lambda built with :py:func:`lambda_`:

        >>> double_fn = F.lambda_(["v"], F.lambda_var("v") * lit(2))
        >>> df.select(
        ...     F.array_transform("a", double_fn).alias("d")
        ... ).collect_column("d")[0].as_py()
        [2, 4, 6]

    See Also:
        :py:func:`array_any_match`, :py:func:`lambda_`.
    """
    return Expr(f.array_transform(coerce_to_column(array), _to_lambda(transform).expr))


def list_transform(array: Expr | str, transform: Expr | Callable[..., Any]) -> Expr:
    """Transform each element of a list with a lambda.

    See Also:
        This is an alias for :py:func:`array_transform`.
    """
    return array_transform(array, transform)


def array_any_match(array: Expr | str, predicate: Expr | Callable[..., Any]) -> Expr:
    """Return ``True`` if any element of ``array`` satisfies ``predicate``.

    ``predicate`` may be a Python callable, converted to a lambda
    automatically, or an explicit lambda built with :py:func:`lambda_`. It must
    return a boolean expression.

    Examples:
        Using a Python callable:

        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"a": [[1, 2, 3]]})
        >>> df.select(
        ...     F.array_any_match("a", lambda v: v > 2).alias("m")
        ... ).collect_column("m")[0].as_py()
        True

        Using an explicit lambda built with :py:func:`lambda_`:

        >>> predicate = F.lambda_(["v"], F.lambda_var("v") > lit(2))
        >>> df.select(
        ...     F.array_any_match("a", predicate).alias("m")
        ... ).collect_column("m")[0].as_py()
        True

    See Also:
        :py:func:`array_transform`, :py:func:`lambda_`.
    """
    return Expr(f.array_any_match(coerce_to_column(array), _to_lambda(predicate).expr))


def any_match(array: Expr | str, predicate: Expr | Callable[..., Any]) -> Expr:
    """Return ``True`` if any element of an array satisfies a predicate.

    See Also:
        This is an alias for :py:func:`array_any_match`.
    """
    return array_any_match(array, predicate)


def list_any_match(array: Expr | str, predicate: Expr | Callable[..., Any]) -> Expr:
    """Return ``True`` if any element of a list satisfies a predicate.

    See Also:
        This is an alias for :py:func:`array_any_match`.
    """
    return array_any_match(array, predicate)


def array_filter(array: Expr | str, predicate: Expr | Callable[..., Any]) -> Expr:
    """Keep the elements of ``array`` for which ``predicate`` is ``True``.

    ``predicate`` may be a Python callable, converted to a lambda
    automatically, or an explicit lambda built with :py:func:`lambda_`. It must
    return a boolean expression. The result is a new array containing only the
    matching elements.

    Examples:
        Using a Python callable:

        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"a": [[1, 2, 3, 4, 5]]})
        >>> df.select(
        ...     F.array_filter("a", lambda v: v > 2).alias("f")
        ... ).collect_column("f")[0].as_py()
        [3, 4, 5]

        Using an explicit lambda built with :py:func:`lambda_`:

        >>> predicate = F.lambda_(["v"], F.lambda_var("v") > lit(2))
        >>> df.select(
        ...     F.array_filter("a", predicate).alias("f")
        ... ).collect_column("f")[0].as_py()
        [3, 4, 5]

    See Also:
        :py:func:`array_transform`, :py:func:`array_any_match`, :py:func:`lambda_`.
    """
    return Expr(f.array_filter(coerce_to_column(array), _to_lambda(predicate).expr))


def list_filter(array: Expr | str, predicate: Expr | Callable[..., Any]) -> Expr:
    """Keep the elements of a list for which a predicate is ``True``.

    See Also:
        This is an alias for :py:func:`array_filter`.
    """
    return array_filter(array, predicate)


def in_list(arg: Expr | str, values: list[Expr], negated: bool = False) -> Expr:
    """Returns whether the argument is contained within the list ``values``.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"a": [1, 2, 3]})
        >>> result = df.select(
        ...     dfn.functions.in_list(
        ...         "a", [dfn.lit(1), dfn.lit(3)]
        ...     ).alias("in")
        ... )
        >>> result.collect_column("in").to_pylist()
        [True, False, True]

        >>> result = df.select(
        ...     dfn.functions.in_list(
        ...         "a", [dfn.lit(1), dfn.lit(3)],
        ...         negated=True,
        ...     ).alias("not_in")
        ... )
        >>> result.collect_column("not_in").to_pylist()
        [False, True, False]
    """
    values = [ensure_expr(v) for v in values]
    return Expr(f.in_list(coerce_to_column(arg), values, negated))


def digest(value: Expr | str, method: Expr | str) -> Expr:
    """Computes the binary hash of an expression using the specified algorithm.

    Standard algorithms are md5, sha224, sha256, sha384, sha512, blake2s,
    blake2b, and blake3.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"a": ["hello"]})
        >>> result = df.select(dfn.functions.digest("a", "md5").alias("d"))
        >>> len(result.collect_column("d")[0].as_py()) > 0
        True
    """
    _warn_if_expr_for_literal_arg(method, "digest", "method")
    method = coerce_to_literal(method)
    return Expr(f.digest(coerce_to_column(value), method))


def contains(string: Expr | str, search_str: Expr | str) -> Expr:
    """Returns true if ``search_str`` is found within ``string`` (case-sensitive).

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"a": ["the quick brown fox"]})
        >>> result = df.select(dfn.functions.contains("a", "brown").alias("c"))
        >>> result.collect_column("c")[0].as_py()
        True
    """
    search_str = coerce_to_literal(search_str)
    return Expr(f.contains(coerce_to_column(string), search_str))


def concat(*args: Expr | str) -> Expr:
    """Concatenates the text representations of all the arguments.

    NULL arguments are ignored.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"a": ["hello"], "b": [" world"]})
        >>> result = df.select(dfn.functions.concat("a", "b").alias("c"))
        >>> result.collect_column("c")[0].as_py()
        'hello world'
    """
    args = coerce_to_column_list(args)
    return Expr(f.concat(args))


def concat_ws(separator: str, *args: Expr | str) -> Expr:
    """Concatenates the list ``args`` with the separator.

    ``NULL`` arguments are ignored. ``separator`` should not be ``NULL``.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"a": ["hello"], "b": ["world"]})
        >>> result = df.select(dfn.functions.concat_ws("-", "a", "b").alias("c"))
        >>> result.collect_column("c")[0].as_py()
        'hello-world'
    """
    args = coerce_to_column_list(args)
    return Expr(f.concat_ws(separator, args))


def order_by(
    expr: Expr | str, ascending: bool = True, nulls_first: bool = True
) -> SortExpr:
    """Creates a new sort expression.

    Examples:
        >>> sort_expr = dfn.functions.order_by("a", ascending=False)
        >>> sort_expr.ascending()
        False

        >>> sort_expr = dfn.functions.order_by("a", ascending=True, nulls_first=False)
        >>> sort_expr.nulls_first()
        False
    """
    return SortExpr(expr, ascending=ascending, nulls_first=nulls_first)


def alias(expr: Expr | str, name: str, metadata: dict[str, str] | None = None) -> Expr:
    """Creates an alias expression with an optional metadata dictionary.

    Args:
        expr: The expression to alias
        name: The alias name
        metadata: Optional metadata to attach to the column

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"a": [1, 2]})
        >>> result = df.select(dfn.functions.alias("a", "b"))
        >>> result.collect_column("b")[0].as_py()
        1

        >>> result = df.select(dfn.functions.alias("a", "b", metadata={"info": "test"}))
        >>> result.schema()
        b: int64
          -- field metadata --
          info: 'test'
    """
    return Expr(f.alias(coerce_to_column(expr), name, metadata))


def col(name: str) -> Expr:
    """Creates a column reference expression.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"a": [1, 2, 3]})
        >>> df.select(dfn.functions.col("a")).collect_column("a")[0].as_py()
        1
    """
    return Expr(f.col(name))


def count_star(filter: Expr | str | None = None) -> Expr:
    """Create a COUNT(1) aggregate expression.

    This aggregate function will count all of the rows in the partition.

    If using the builder functions described in ref:`_aggregation` this function ignores
    the options ``order_by``, ``distinct``, and ``null_treatment``.

    Args:
        filter: If provided, only count rows for which the filter is True

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"a": [1, 2, 3]})
        >>> result = df.aggregate(
        ...     [], [dfn.functions.count_star(
        ...     ).alias("cnt")])
        >>> result.collect_column("cnt")[0].as_py()
        3

        >>> result = df.aggregate(
        ...     [], [dfn.functions.count_star(
        ...         filter=dfn.col("a") > dfn.lit(1)
        ...     ).alias("cnt")])
        >>> result.collect_column("cnt")[0].as_py()
        2
    """
    return count(Expr.literal(1), filter=filter)


def case(expr: Expr | str) -> CaseBuilder:
    """Create a case expression.

    Create a :py:class:`~datafusion.expr.CaseBuilder` to match cases for the
    expression ``expr``. See :py:class:`~datafusion.expr.CaseBuilder` for
    detailed usage.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"a": [1, 2, 3]})
        >>> result = df.select(
        ...     dfn.functions.case("a").when(dfn.lit(1),
        ...     dfn.lit("one")).otherwise(dfn.lit("other")).alias("c"))
        >>> result.collect_column("c")[0].as_py()
        'one'
    """
    return CaseBuilder(f.case(coerce_to_column(expr)))


def when(when: Expr | str, then: Expr) -> CaseBuilder:
    """Create a case expression that has no base expression.

    Create a :py:class:`~datafusion.expr.CaseBuilder` to match cases for the
    expression ``expr``. See :py:class:`~datafusion.expr.CaseBuilder` for
    detailed usage.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"a": [1, 2, 3]})
        >>> result = df.select(
        ...     dfn.functions.when(dfn.col("a") > dfn.lit(2),
        ...     dfn.lit("big")).otherwise(dfn.lit("small")).alias("c"))
        >>> result.collect_column("c")[2].as_py()
        'big'
    """
    return CaseBuilder(f.when(coerce_to_column(when), ensure_expr(then)))


# scalar functions
def abs(arg: Expr | str) -> Expr:
    """Return the absolute value of a given number.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"a": [-1, 0, 1]})
        >>> result = df.select(dfn.functions.abs("a").alias("abs"))
        >>> result.collect_column("abs")[0].as_py()
        1
    """
    return Expr(f.abs(coerce_to_column(arg)))


def acos(arg: Expr | str) -> Expr:
    """Returns the arc cosine or inverse cosine of a number.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"a": [1.0]})
        >>> result = df.select(dfn.functions.acos("a").alias("acos"))
        >>> result.collect_column("acos")[0].as_py()
        0.0
    """
    return Expr(f.acos(coerce_to_column(arg)))


def acosh(arg: Expr | str) -> Expr:
    """Returns inverse hyperbolic cosine.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"a": [1.0]})
        >>> result = df.select(dfn.functions.acosh("a").alias("acosh"))
        >>> result.collect_column("acosh")[0].as_py()
        0.0
    """
    return Expr(f.acosh(coerce_to_column(arg)))


def ascii(arg: Expr | str) -> Expr:
    """Returns the numeric code of the first character of the argument.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"a": ["a","b","c"]})
        >>> ascii_df = df.select(dfn.functions.ascii("a").alias("ascii"))
        >>> ascii_df.collect_column("ascii")[0].as_py()
        97
    """
    return Expr(f.ascii(coerce_to_column(arg)))


def asin(arg: Expr | str) -> Expr:
    """Returns the arc sine or inverse sine of a number.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"a": [0.0]})
        >>> result = df.select(dfn.functions.asin("a").alias("asin"))
        >>> result.collect_column("asin")[0].as_py()
        0.0
    """
    return Expr(f.asin(coerce_to_column(arg)))


def asinh(arg: Expr | str) -> Expr:
    """Returns inverse hyperbolic sine.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"a": [0.0]})
        >>> result = df.select(dfn.functions.asinh("a").alias("asinh"))
        >>> result.collect_column("asinh")[0].as_py()
        0.0
    """
    return Expr(f.asinh(coerce_to_column(arg)))


def atan(arg: Expr | str) -> Expr:
    """Returns inverse tangent of a number.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"a": [0.0]})
        >>> result = df.select(dfn.functions.atan("a").alias("atan"))
        >>> result.collect_column("atan")[0].as_py()
        0.0
    """
    return Expr(f.atan(coerce_to_column(arg)))


def atanh(arg: Expr | str) -> Expr:
    """Returns inverse hyperbolic tangent.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"a": [0.0]})
        >>> result = df.select(dfn.functions.atanh("a").alias("atanh"))
        >>> result.collect_column("atanh")[0].as_py()
        0.0
    """
    return Expr(f.atanh(coerce_to_column(arg)))


def atan2(y: Expr | str, x: Expr | str) -> Expr:
    """Returns inverse tangent of a division given in the argument.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"y": [0.0], "x": [1.0]})
        >>> result = df.select(dfn.functions.atan2("y", "x").alias("atan2"))
        >>> result.collect_column("atan2")[0].as_py()
        0.0
    """
    return Expr(f.atan2(coerce_to_column(y), coerce_to_column(x)))


def bit_length(arg: Expr | str) -> Expr:
    """Returns the number of bits in the string argument.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"a": ["a","b","c"]})
        >>> bit_df = df.select(dfn.functions.bit_length("a").alias("bit_len"))
        >>> bit_df.collect_column("bit_len")[0].as_py()
        8
    """
    return Expr(f.bit_length(coerce_to_column(arg)))


def btrim(arg: Expr | str) -> Expr:
    """Removes all characters, spaces by default, from both sides of a string.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"a": [" a  "]})
        >>> trim_df = df.select(dfn.functions.btrim("a").alias("trimmed"))
        >>> trim_df.collect_column("trimmed")[0].as_py()
        'a'
    """
    return Expr(f.btrim(coerce_to_column(arg)))


def cbrt(arg: Expr | str) -> Expr:
    """Returns the cube root of a number.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"a": [27]})
        >>> cbrt_df = df.select(dfn.functions.cbrt("a").alias("cbrt"))
        >>> cbrt_df.collect_column("cbrt")[0].as_py()
        3.0
    """
    return Expr(f.cbrt(coerce_to_column(arg)))


def ceil(arg: Expr | str) -> Expr:
    """Returns the nearest integer greater than or equal to argument.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"a": [1.9]})
        >>> ceil_df = df.select(dfn.functions.ceil("a").alias("ceil"))
        >>> ceil_df.collect_column("ceil")[0].as_py()
        2.0
    """
    return Expr(f.ceil(coerce_to_column(arg)))


def character_length(arg: Expr | str) -> Expr:
    """Returns the number of characters in the argument.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"a": ["abc","b","c"]})
        >>> char_len_df = df.select(
        ...     dfn.functions.character_length("a").alias("char_len"))
        >>> char_len_df.collect_column("char_len")[0].as_py()
        3
    """
    return Expr(f.character_length(coerce_to_column(arg)))


def length(string: Expr | str) -> Expr:
    """The number of characters in the ``string``.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"a": ["hello"]})
        >>> result = df.select(dfn.functions.length("a").alias("len"))
        >>> result.collect_column("len")[0].as_py()
        5
    """
    return Expr(f.length(coerce_to_column(string)))


def char_length(string: Expr | str) -> Expr:
    """The number of characters in the ``string``.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"a": ["hello"]})
        >>> result = df.select(dfn.functions.char_length("a").alias("len"))
        >>> result.collect_column("len")[0].as_py()
        5
    """
    return Expr(f.char_length(coerce_to_column(string)))


def chr(arg: Expr | str) -> Expr:
    """Converts the Unicode code point to a UTF8 character.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"a": [65]})
        >>> result = df.select(dfn.functions.chr("a").alias("chr"))
        >>> result.collect_column("chr")[0].as_py()
        'A'
    """
    return Expr(f.chr(coerce_to_column(arg)))


def coalesce(*args: Expr | str) -> Expr:
    """Returns the value of the first expr in ``args`` which is not NULL.

    Args:
        *args: Expressions to evaluate in order.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"a": [None, 1], "b": [2, 3]})
        >>> result = df.select(dfn.functions.coalesce("a", "b").alias("c"))
        >>> result.collect_column("c")[0].as_py()
        2
    """
    args = coerce_to_column_list(args)
    return Expr(f.coalesce(*args))


def cos(arg: Expr | str) -> Expr:
    """Returns the cosine of the argument.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"a": [0,-1,1]})
        >>> cos_df = df.select(dfn.functions.cos("a").alias("cos"))
        >>> cos_df.collect_column("cos")[0].as_py()
        1.0
    """
    return Expr(f.cos(coerce_to_column(arg)))


def cosh(arg: Expr | str) -> Expr:
    """Returns the hyperbolic cosine of the argument.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"a": [0,-1,1]})
        >>> cosh_df = df.select(dfn.functions.cosh("a").alias("cosh"))
        >>> cosh_df.collect_column("cosh")[0].as_py()
        1.0
    """
    return Expr(f.cosh(coerce_to_column(arg)))


def cot(arg: Expr | str) -> Expr:
    """Returns the cotangent of the argument.

    Examples:
        >>> from math import pi
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"a": [pi / 4]})
        >>> result = df.select(dfn.functions.cot("a").alias("cot"))
        >>> result.collect_column("cot")[0].as_py()
        1.0...
    """
    return Expr(f.cot(coerce_to_column(arg)))


def degrees(arg: Expr | str) -> Expr:
    """Converts the argument from radians to degrees.

    Examples:
        >>> from math import pi
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"a": [0,pi,2*pi]})
        >>> deg_df = df.select(dfn.functions.degrees("a").alias("deg"))
        >>> deg_df.collect_column("deg")[2].as_py()
        360.0
    """
    return Expr(f.degrees(coerce_to_column(arg)))


def ends_with(arg: Expr | str, suffix: Expr | str) -> Expr:
    """Returns true if the ``string`` ends with the ``suffix``, false otherwise.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"a": ["abc","b","c"]})
        >>> ends_with_df = df.select(
        ...     dfn.functions.ends_with("a", "c").alias("ends_with"))
        >>> ends_with_df.collect_column("ends_with")[0].as_py()
        True
    """
    suffix = coerce_to_literal(suffix)
    return Expr(f.ends_with(coerce_to_column(arg), suffix))


def exp(arg: Expr | str) -> Expr:
    """Returns the exponential of the argument.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"a": [0.0]})
        >>> result = df.select(dfn.functions.exp("a").alias("exp"))
        >>> result.collect_column("exp")[0].as_py()
        1.0
    """
    return Expr(f.exp(coerce_to_column(arg)))


def factorial(arg: Expr | str) -> Expr:
    """Returns the factorial of the argument.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"a": [3]})
        >>> result = df.select(dfn.functions.factorial("a").alias("factorial"))
        >>> result.collect_column("factorial")[0].as_py()
        6
    """
    return Expr(f.factorial(coerce_to_column(arg)))


def find_in_set(string: Expr | str, string_list: Expr | str) -> Expr:
    """Find a string in a list of strings.

    Returns a value in the range of 1 to N if the string is in the string list
    ``string_list`` consisting of N substrings.

    The string list is a string composed of substrings separated by ``,`` characters.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"a": ["b"]})
        >>> result = df.select(dfn.functions.find_in_set("a", "a,b,c").alias("pos"))
        >>> result.collect_column("pos")[0].as_py()
        2
    """
    string_list = coerce_to_literal(string_list)
    return Expr(f.find_in_set(coerce_to_column(string), string_list))


def floor(arg: Expr | str) -> Expr:
    """Returns the nearest integer less than or equal to the argument.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"a": [1.9]})
        >>> floor_df = df.select(dfn.functions.floor("a").alias("floor"))
        >>> floor_df.collect_column("floor")[0].as_py()
        1.0
    """
    return Expr(f.floor(coerce_to_column(arg)))


def gcd(x: Expr | str, y: Expr | str) -> Expr:
    """Returns the greatest common divisor.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"a": [12], "b": [8]})
        >>> result = df.select(dfn.functions.gcd("a", "b").alias("gcd"))
        >>> result.collect_column("gcd")[0].as_py()
        4
    """
    return Expr(f.gcd(coerce_to_column(x), coerce_to_column(y)))


def greatest(*args: Expr | str) -> Expr:
    """Returns the greatest value from a list of expressions.

    Returns NULL if all expressions are NULL.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"a": [1, 3], "b": [2, 1]})
        >>> result = df.select(dfn.functions.greatest("a", "b").alias("greatest"))
        >>> result.collect_column("greatest")[0].as_py()
        2
        >>> result.collect_column("greatest")[1].as_py()
        3
    """
    exprs = coerce_to_column_list(args)
    return Expr(f.greatest(*exprs))


def ifnull(x: Expr | str, y: Expr | str) -> Expr:
    """Returns ``x`` if ``x`` is not NULL. Otherwise returns ``y``.

    Args:
        x: Expression to return when it is not NULL.
        y: Fallback expression to return when ``x`` is NULL.

    See Also:
        This is an alias for :py:func:`nvl`.
    """
    return nvl(x, y)


def initcap(string: Expr | str) -> Expr:
    """Set the initial letter of each word to capital.

    Converts the first letter of each word in ``string`` to uppercase and the remaining
    characters to lowercase.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"a": ["the cat"]})
        >>> cap_df = df.select(dfn.functions.initcap("a").alias("cap"))
        >>> cap_df.collect_column("cap")[0].as_py()
        'The Cat'
    """
    return Expr(f.initcap(coerce_to_column(string)))


def instr(string: Expr | str, substring: Expr | str) -> Expr:
    """Finds the position from where the ``substring`` matches the ``string``.

    See Also:
        This is an alias for :py:func:`strpos`.
    """
    return strpos(string, substring)


def iszero(arg: Expr | str) -> Expr:
    """Returns true if a given number is +0.0 or -0.0 otherwise returns false.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"a": [0.0, 1.0]})
        >>> result = df.select(dfn.functions.iszero("a").alias("iz"))
        >>> result.collect_column("iz")[0].as_py()
        True
    """
    return Expr(f.iszero(coerce_to_column(arg)))


def lcm(x: Expr | str, y: Expr | str) -> Expr:
    """Returns the least common multiple.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"a": [4], "b": [6]})
        >>> result = df.select(dfn.functions.lcm("a", "b").alias("lcm"))
        >>> result.collect_column("lcm")[0].as_py()
        12
    """
    return Expr(f.lcm(coerce_to_column(x), coerce_to_column(y)))


def least(*args: Expr | str) -> Expr:
    """Returns the least value from a list of expressions.

    Returns NULL if all expressions are NULL.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"a": [1, 3], "b": [2, 1]})
        >>> result = df.select(dfn.functions.least("a", "b").alias("least"))
        >>> result.collect_column("least")[0].as_py()
        1
        >>> result.collect_column("least")[1].as_py()
        1
    """
    exprs = coerce_to_column_list(args)
    return Expr(f.least(*exprs))


def left(string: Expr | str, n: Expr | int) -> Expr:
    """Returns the first ``n`` characters in the ``string``.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"a": ["the cat"]})
        >>> left_df = df.select(dfn.functions.left("a", 3).alias("left"))
        >>> left_df.collect_column("left")[0].as_py()
        'the'
    """
    n = coerce_to_literal(n)
    return Expr(f.left(coerce_to_column(string), n))


def levenshtein(string1: Expr | str, string2: Expr | str) -> Expr:
    """Returns the Levenshtein distance between the two given strings.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"a": ["kitten"]})
        >>> result = df.select(dfn.functions.levenshtein("a", "sitting").alias("d"))
        >>> result.collect_column("d")[0].as_py()
        3
    """
    string2 = coerce_to_literal(string2)
    return Expr(f.levenshtein(coerce_to_column(string1), string2))


def ln(arg: Expr | str) -> Expr:
    """Returns the natural logarithm (base e) of the argument.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"a": [1.0]})
        >>> result = df.select(dfn.functions.ln("a").alias("ln"))
        >>> result.collect_column("ln")[0].as_py()
        0.0
    """
    return Expr(f.ln(coerce_to_column(arg)))


def log(base: Expr | int | float, num: Expr | str) -> Expr:  # noqa: PYI041
    """Returns the logarithm of a number for a particular ``base``.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"a": [100.0]})
        >>> result = df.select(dfn.functions.log(10.0, "a").alias("log"))
        >>> result.collect_column("log")[0].as_py()
        2.0
    """
    base = coerce_to_literal(base)
    return Expr(f.log(base, coerce_to_column(num)))


def log10(arg: Expr | str) -> Expr:
    """Base 10 logarithm of the argument.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"a": [100.0]})
        >>> result = df.select(dfn.functions.log10("a").alias("log10"))
        >>> result.collect_column("log10")[0].as_py()
        2.0
    """
    return Expr(f.log10(coerce_to_column(arg)))


def log2(arg: Expr | str) -> Expr:
    """Base 2 logarithm of the argument.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"a": [8.0]})
        >>> result = df.select(dfn.functions.log2("a").alias("log2"))
        >>> result.collect_column("log2")[0].as_py()
        3.0
    """
    return Expr(f.log2(coerce_to_column(arg)))


def lower(arg: Expr | str) -> Expr:
    """Converts a string to lowercase.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"a": ["THE CaT"]})
        >>> lower_df = df.select(dfn.functions.lower("a").alias("lower"))
        >>> lower_df.collect_column("lower")[0].as_py()
        'the cat'
    """
    return Expr(f.lower(coerce_to_column(arg)))


def lpad(
    string: Expr | str, count: Expr | int, characters: Expr | str | None = None
) -> Expr:
    """Add left padding to a string.

    Extends the string to length length by prepending the characters fill (a
    space by default). If the string is already longer than length then it is
    truncated (on the right).

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"a": ["the cat", "a hat"]})
        >>> lpad_df = df.select(dfn.functions.lpad("a", 6).alias("lpad"))
        >>> lpad_df.collect_column("lpad")[0].as_py()
        'the ca'
        >>> lpad_df.collect_column("lpad")[1].as_py()
        ' a hat'

        >>> result = df.select(
        ...     dfn.functions.lpad(
        ...         "a", 10, characters="."
        ...     ).alias("lpad"))
        >>> result.collect_column("lpad")[0].as_py()
        '...the cat'
    """
    count = coerce_to_literal(count)
    characters = coerce_to_literal(characters if characters is not None else " ")
    return Expr(f.lpad(coerce_to_column(string), count, characters))


def ltrim(arg: Expr | str) -> Expr:
    """Removes all characters, spaces by default, from the beginning of a string.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"a": [" a  "]})
        >>> trim_df = df.select(dfn.functions.ltrim("a").alias("trimmed"))
        >>> trim_df.collect_column("trimmed")[0].as_py()
        'a  '
    """
    return Expr(f.ltrim(coerce_to_column(arg)))


def md5(arg: Expr | str) -> Expr:
    """Computes an MD5 128-bit checksum for a string expression.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"a": ["hello"]})
        >>> result = df.select(dfn.functions.md5("a").alias("md5"))
        >>> result.collect_column("md5")[0].as_py()
        '5d41402abc4b2a76b9719d911017c592'
    """
    return Expr(f.md5(coerce_to_column(arg)))


def nanvl(x: Expr | str, y: Expr | str) -> Expr:
    """Returns ``x`` if ``x`` is not ``NaN``. Otherwise returns ``y``.

    Args:
        x: Expression to return when it is not NaN.
        y: Fallback expression to return when ``x`` is NaN.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"a": [np.nan, 1.0], "b": [0.0, 0.0]})
        >>> nanvl_df = df.select(dfn.functions.nanvl("a", "b").alias("nanvl"))
        >>> nanvl_df.collect_column("nanvl")[0].as_py()
        0.0
        >>> nanvl_df.collect_column("nanvl")[1].as_py()
        1.0
    """
    return Expr(f.nanvl(coerce_to_column(x), coerce_to_column(y)))


def nvl(x: Expr | str, y: Expr | str) -> Expr:
    """Returns ``x`` if ``x`` is not ``NULL``. Otherwise returns ``y``.

    Args:
        x: Expression to return when it is not NULL.
        y: Fallback expression to return when ``x`` is NULL.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"a": [None, 1], "b": [0, 0]})
        >>> nvl_df = df.select(dfn.functions.nvl("a", "b").alias("nvl"))
        >>> nvl_df.collect_column("nvl")[0].as_py()
        0
        >>> nvl_df.collect_column("nvl")[1].as_py()
        1
    """
    return Expr(f.nvl(coerce_to_column(x), coerce_to_column(y)))


def nvl2(x: Expr | str, y: Expr | str, z: Expr | str) -> Expr:
    """Returns ``y`` if ``x`` is not NULL. Otherwise returns ``z``.

    Args:
        x: Expression to check for NULL.
        y: Expression to return when ``x`` is not NULL.
        z: Expression to return when ``x`` is NULL.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"a": [None, 1], "b": [10, 20], "c": [30, 40]})
        >>> result = df.select(dfn.functions.nvl2("a", "b", "c").alias("nvl2"))
        >>> result.collect_column("nvl2")[0].as_py()
        30
        >>> result.collect_column("nvl2")[1].as_py()
        20
    """
    return Expr(f.nvl2(coerce_to_column(x), coerce_to_column(y), coerce_to_column(z)))


def octet_length(arg: Expr | str) -> Expr:
    """Returns the number of bytes of a string.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"a": ["hello"]})
        >>> result = df.select(dfn.functions.octet_length("a").alias("len"))
        >>> result.collect_column("len")[0].as_py()
        5
    """
    return Expr(f.octet_length(coerce_to_column(arg)))


def overlay(
    string: Expr | str,
    substring: Expr | str,
    start: Expr | int,
    length: Expr | int | None = None,
) -> Expr:
    """Replace a substring with a new substring.

    Replace the substring of string that starts at the ``start``'th character and
    extends for ``length`` characters with new substring.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"a": ["abcdef"]})
        >>> result = df.select(dfn.functions.overlay("a", "XY", 3, 2).alias("o"))
        >>> result.collect_column("o")[0].as_py()
        'abXYef'
    """
    substring = coerce_to_literal(substring)
    start = coerce_to_literal(start)
    if length is None:
        return Expr(f.overlay(coerce_to_column(string), substring, start))
    length = coerce_to_literal(length)
    return Expr(f.overlay(coerce_to_column(string), substring, start, length))


def pi() -> Expr:
    """Returns an approximate value of π.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"a": [1]})
        >>> from math import pi
        >>> result = df.select(
        ...     dfn.functions.pi().alias("pi")
        ... )
        >>> result.collect_column("pi")[0].as_py() == pi
        True
    """
    return Expr(f.pi())


def position(string: Expr | str, substring: Expr | str) -> Expr:
    """Finds the position from where the ``substring`` matches the ``string``.

    See Also:
        This is an alias for :py:func:`strpos`.
    """
    return strpos(string, substring)


def power(base: Expr | str, exponent: Expr | int | float) -> Expr:  # noqa: PYI041
    """Returns ``base`` raised to the power of ``exponent``.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"a": [2.0]})
        >>> result = df.select(dfn.functions.power("a", 3.0).alias("pow"))
        >>> result.collect_column("pow")[0].as_py()
        8.0
    """
    exponent = coerce_to_literal(exponent)
    return Expr(f.power(coerce_to_column(base), exponent))


def pow(base: Expr | str, exponent: Expr | int | float) -> Expr:  # noqa: PYI041
    """Returns ``base`` raised to the power of ``exponent``.

    See Also:
        This is an alias of :py:func:`power`.
    """
    return power(base, exponent)


def radians(arg: Expr | str) -> Expr:
    """Converts the argument from degrees to radians.

    Examples:
        >>> from math import pi
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"a": [180.0]})
        >>> result = df.select(dfn.functions.radians("a").alias("rad"))
        >>> result.collect_column("rad")[0].as_py() == pi
        True
    """
    return Expr(f.radians(coerce_to_column(arg)))


def regexp_like(
    string: Expr | str, regex: Expr | str, flags: Expr | str | None = None
) -> Expr:
    r"""Find if any regular expression (regex) matches exist.

    Tests a string using a regular expression returning true if at least one match,
    false otherwise.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"a": ["hello123"]})
        >>> result = df.select(
        ...     dfn.functions.regexp_like("a", "\\d+").alias("m")
        ... )
        >>> result.collect_column("m")[0].as_py()
        True

        Use ``flags`` for case-insensitive matching:

        >>> result = df.select(
        ...     dfn.functions.regexp_like(
        ...         "a", "HELLO", flags="i",
        ...     ).alias("m")
        ... )
        >>> result.collect_column("m")[0].as_py()
        True
    """
    regex = coerce_to_literal(regex)
    flags = coerce_to_literal_or_none(flags)
    return Expr(
        f.regexp_like(
            coerce_to_column(string),
            regex,
            flags,
        )
    )


def regexp_match(
    string: Expr | str, regex: Expr | str, flags: Expr | str | None = None
) -> Expr:
    r"""Perform regular expression (regex) matching.

    Returns an array with each element containing the leftmost-first match of the
    corresponding index in ``regex`` to string in ``string``.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"a": ["hello 42 world"]})
        >>> result = df.select(
        ...     dfn.functions.regexp_match("a", "(\\d+)").alias("m")
        ... )
        >>> result.collect_column("m")[0].as_py()
        ['42']

        Use ``flags`` for case-insensitive matching:

        >>> result = df.select(
        ...     dfn.functions.regexp_match(
        ...         "a", "(HELLO)", flags="i",
        ...     ).alias("m")
        ... )
        >>> result.collect_column("m")[0].as_py()
        ['hello']
    """
    regex = coerce_to_literal(regex)
    flags = coerce_to_literal_or_none(flags)
    return Expr(
        f.regexp_match(
            coerce_to_column(string),
            regex,
            flags,
        )
    )


def regexp_replace(
    string: Expr | str,
    pattern: Expr | str,
    replacement: Expr | str,
    flags: Expr | str | None = None,
) -> Expr:
    r"""Replaces substring(s) matching a PCRE-like regular expression.

    The full list of supported features and syntax can be found at
    <https://docs.rs/regex/latest/regex/#syntax>

    Supported flags with the addition of 'g' can be found at
    <https://docs.rs/regex/latest/regex/#grouping-and-flags>

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"a": ["hello 42"]})
        >>> result = df.select(
        ...     dfn.functions.regexp_replace(
        ...         "a", "\\d+", "XX"
        ...     ).alias("r")
        ... )
        >>> result.collect_column("r")[0].as_py()
        'hello XX'

        Use the ``g`` flag to replace all occurrences:

        >>> df = ctx.from_pydict({"a": ["a1 b2 c3"]})
        >>> result = df.select(
        ...     dfn.functions.regexp_replace(
        ...         "a", "\\d+", "X", flags="g",
        ...     ).alias("r")
        ... )
        >>> result.collect_column("r")[0].as_py()
        'aX bX cX'
    """
    pattern = coerce_to_literal(pattern)
    replacement = coerce_to_literal(replacement)
    flags = coerce_to_literal_or_none(flags)
    return Expr(
        f.regexp_replace(
            coerce_to_column(string),
            pattern,
            replacement,
            flags,
        )
    )


def regexp_count(
    string: Expr | str,
    pattern: Expr | str,
    start: Expr | int | None = None,
    flags: Expr | str | None = None,
) -> Expr:
    """Returns the number of matches in a string.

    Optional start position (the first position is 1) to search for the regular
    expression.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"a": ["abcabc"]})
        >>> result = df.select(dfn.functions.regexp_count("a", "abc").alias("c"))
        >>> result.collect_column("c")[0].as_py()
        2

        Use ``start`` to begin searching from a position, and
        ``flags`` for case-insensitive matching:

        >>> result = df.select(
        ...     dfn.functions.regexp_count(
        ...         "a", "ABC", start=4, flags="i",
        ...     ).alias("c"))
        >>> result.collect_column("c")[0].as_py()
        1
    """
    pattern = coerce_to_literal(pattern)
    start = coerce_to_literal_or_none(start)
    flags = coerce_to_literal_or_none(flags)
    return Expr(
        f.regexp_count(
            coerce_to_column(string),
            pattern,
            start,
            flags,
        )
    )


def regexp_instr(
    values: Expr | str,
    regex: Expr | str,
    start: Expr | int | None = None,
    n: Expr | int | None = None,
    flags: Expr | str | None = None,
    sub_expr: Expr | int | None = None,
) -> Expr:
    r"""Returns the position of a regular expression match in a string.

    Args:
        values: Data to search for the regular expression match.
        regex: Regular expression to search for.
        start: Optional position to start the search (the first position is 1).
        n: Optional occurrence of the match to find (the first occurrence is 1).
        flags: Optional regular expression flags to control regex behavior.
        sub_expr: Optionally capture group position instead of the entire match.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"a": ["hello 42 world"]})
        >>> result = df.select(
        ...     dfn.functions.regexp_instr("a", "\\d+").alias("pos")
        ... )
        >>> result.collect_column("pos")[0].as_py()
        7

        Use ``start`` to search from a position, ``n`` for the
        nth occurrence, and ``flags`` for case-insensitive mode:

        >>> df = ctx.from_pydict({"a": ["abc ABC abc"]})
        >>> result = df.select(
        ...     dfn.functions.regexp_instr(
        ...         "a", "abc",
        ...         start=2, n=1, flags="i",
        ...     ).alias("pos")
        ... )
        >>> result.collect_column("pos")[0].as_py()
        5

        Use ``sub_expr`` to get the position of a capture group:

        >>> result = df.select(
        ...     dfn.functions.regexp_instr(
        ...         "a", "(abc)", sub_expr=1,
        ...     ).alias("pos")
        ... )
        >>> result.collect_column("pos")[0].as_py()
        1
    """
    regex = coerce_to_literal(regex)
    start = coerce_to_literal_or_none(start)
    n = coerce_to_literal_or_none(n)
    flags = coerce_to_literal_or_none(flags)
    sub_expr = coerce_to_literal_or_none(sub_expr)

    return Expr(
        f.regexp_instr(
            coerce_to_column(values),
            regex,
            start,
            n,
            flags,
            sub_expr,
        )
    )


def repeat(string: Expr | str, n: Expr | int) -> Expr:
    """Repeats the ``string`` to ``n`` times.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"a": ["ha"]})
        >>> result = df.select(dfn.functions.repeat("a", 3).alias("r"))
        >>> result.collect_column("r")[0].as_py()
        'hahaha'
    """
    n = coerce_to_literal(n)
    return Expr(f.repeat(coerce_to_column(string), n))


def replace(string: Expr | str, from_val: Expr | str, to_val: Expr | str) -> Expr:
    """Replaces all occurrences of ``from_val`` with ``to_val`` in the ``string``.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"a": ["hello world"]})
        >>> result = df.select(dfn.functions.replace("a", "world", "there").alias("r"))
        >>> result.collect_column("r")[0].as_py()
        'hello there'
    """
    from_val = coerce_to_literal(from_val)
    to_val = coerce_to_literal(to_val)
    return Expr(f.replace(coerce_to_column(string), from_val, to_val))


def reverse(arg: Expr | str) -> Expr:
    """Reverse the string argument.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"a": ["hello"]})
        >>> result = df.select(dfn.functions.reverse("a").alias("r"))
        >>> result.collect_column("r")[0].as_py()
        'olleh'
    """
    return Expr(f.reverse(coerce_to_column(arg)))


def right(string: Expr | str, n: Expr | int) -> Expr:
    """Returns the last ``n`` characters in the ``string``.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"a": ["hello"]})
        >>> result = df.select(dfn.functions.right("a", 3).alias("r"))
        >>> result.collect_column("r")[0].as_py()
        'llo'
    """
    n = coerce_to_literal(n)
    return Expr(f.right(coerce_to_column(string), n))


def round(value: Expr | str, decimal_places: Expr | int | None = None) -> Expr:
    """Round the argument to the nearest integer.

    If the optional ``decimal_places`` is specified, round to the nearest number of
    decimal places. You can specify a negative number of decimal places. For example
    ``round(lit(125.2345), -2)`` would yield a value of ``100.0``.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"a": [1.567]})
        >>> result = df.select(dfn.functions.round("a", 2).alias("r"))
        >>> result.collect_column("r")[0].as_py()
        1.57
    """
    decimal_places = coerce_to_literal(
        decimal_places if decimal_places is not None else 0
    )
    return Expr(f.round(coerce_to_column(value), decimal_places))


def rpad(
    string: Expr | str, count: Expr | int, characters: Expr | str | None = None
) -> Expr:
    """Add right padding to a string.

    Extends the string to length length by appending the characters fill (a space
    by default). If the string is already longer than length then it is truncated.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"a": ["hi"]})
        >>> result = df.select(dfn.functions.rpad("a", 5, "!").alias("r"))
        >>> result.collect_column("r")[0].as_py()
        'hi!!!'
    """
    count = coerce_to_literal(count)
    characters = coerce_to_literal(characters if characters is not None else " ")
    return Expr(f.rpad(coerce_to_column(string), count, characters))


def rtrim(arg: Expr | str) -> Expr:
    """Removes all characters, spaces by default, from the end of a string.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"a": [" a  "]})
        >>> trim_df = df.select(dfn.functions.rtrim("a").alias("trimmed"))
        >>> trim_df.collect_column("trimmed")[0].as_py()
        ' a'
    """
    return Expr(f.rtrim(coerce_to_column(arg)))


def sha224(arg: Expr | str) -> Expr:
    """Computes the SHA-224 hash of a binary string.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"a": ["hello"]})
        >>> result = df.select(dfn.functions.sha224("a").alias("h"))
        >>> result.collect_column("h")[0].as_py().hex()
        'ea09ae9cc6768c50fcee903ed054556e5bfc8347907f12598aa24193'
    """
    return Expr(f.sha224(coerce_to_column(arg)))


def sha256(arg: Expr | str) -> Expr:
    """Computes the SHA-256 hash of a binary string.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"a": ["hello"]})
        >>> result = df.select(dfn.functions.sha256("a").alias("h"))
        >>> result.collect_column("h")[0].as_py().hex()
        '2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824'
    """
    return Expr(f.sha256(coerce_to_column(arg)))


def sha384(arg: Expr | str) -> Expr:
    """Computes the SHA-384 hash of a binary string.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"a": ["hello"]})
        >>> result = df.select(dfn.functions.sha384("a").alias("h"))
        >>> result.collect_column("h")[0].as_py().hex()
        '59e1748777448c69de6b800d7a33bbfb9ff1b...
    """
    return Expr(f.sha384(coerce_to_column(arg)))


def sha512(arg: Expr | str) -> Expr:
    """Computes the SHA-512 hash of a binary string.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"a": ["hello"]})
        >>> result = df.select(dfn.functions.sha512("a").alias("h"))
        >>> result.collect_column("h")[0].as_py().hex()
        '9b71d224bd62f3785d96d46ad3ea3d73319bfb...
    """
    return Expr(f.sha512(coerce_to_column(arg)))


def signum(arg: Expr | str) -> Expr:
    """Returns the sign of the argument (-1, 0, +1).

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"a": [-5.0, 0.0, 5.0]})
        >>> result = df.select(dfn.functions.signum("a").alias("s"))
        >>> result.collect_column("s").to_pylist()
        [-1.0, 0.0, 1.0]
    """
    return Expr(f.signum(coerce_to_column(arg)))


def sin(arg: Expr | str) -> Expr:
    """Returns the sine of the argument.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"a": [0.0]})
        >>> result = df.select(dfn.functions.sin("a").alias("sin"))
        >>> result.collect_column("sin")[0].as_py()
        0.0
    """
    return Expr(f.sin(coerce_to_column(arg)))


def sinh(arg: Expr | str) -> Expr:
    """Returns the hyperbolic sine of the argument.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"a": [0.0]})
        >>> result = df.select(dfn.functions.sinh("a").alias("sinh"))
        >>> result.collect_column("sinh")[0].as_py()
        0.0
    """
    return Expr(f.sinh(coerce_to_column(arg)))


def split_part(string: Expr | str, delimiter: Expr | str, index: Expr | int) -> Expr:
    """Split a string and return one part.

    Splits a string based on a delimiter and picks out the desired field based
    on the index.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"a": ["a,b,c"]})
        >>> result = df.select(dfn.functions.split_part("a", ",", 2).alias("s"))
        >>> result.collect_column("s")[0].as_py()
        'b'
    """
    delimiter = coerce_to_literal(delimiter)
    index = coerce_to_literal(index)
    return Expr(f.split_part(coerce_to_column(string), delimiter, index))


def sqrt(arg: Expr | str) -> Expr:
    """Returns the square root of the argument.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"a": [9.0]})
        >>> result = df.select(dfn.functions.sqrt("a").alias("sqrt"))
        >>> result.collect_column("sqrt")[0].as_py()
        3.0
    """
    return Expr(f.sqrt(coerce_to_column(arg)))


def starts_with(string: Expr | str, prefix: Expr | str) -> Expr:
    """Returns true if string starts with prefix.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"a": ["hello_from_datafusion"]})
        >>> result = df.select(dfn.functions.starts_with("a", "hello").alias("sw"))
        >>> result.collect_column("sw")[0].as_py()
        True
    """
    prefix = coerce_to_literal(prefix)
    return Expr(f.starts_with(coerce_to_column(string), prefix))


def strpos(string: Expr | str, substring: Expr | str) -> Expr:
    """Finds the position from where the ``substring`` matches the ``string``.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"a": ["hello"]})
        >>> result = df.select(dfn.functions.strpos("a", "llo").alias("pos"))
        >>> result.collect_column("pos")[0].as_py()
        3
    """
    substring = coerce_to_literal(substring)
    return Expr(f.strpos(coerce_to_column(string), substring))


def substr(string: Expr | str, position: Expr | int) -> Expr:
    """Substring from the ``position`` to the end.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"a": ["hello"]})
        >>> result = df.select(dfn.functions.substr("a", 3).alias("s"))
        >>> result.collect_column("s")[0].as_py()
        'llo'
    """
    position = coerce_to_literal(position)
    return Expr(f.substr(coerce_to_column(string), position))


def substr_index(string: Expr | str, delimiter: Expr | str, count: Expr | int) -> Expr:
    """Returns an indexed substring.

    The return will be the ``string`` from before ``count`` occurrences of
    ``delimiter``.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"a": ["a.b.c"]})
        >>> result = df.select(dfn.functions.substr_index("a", ".", 2).alias("s"))
        >>> result.collect_column("s")[0].as_py()
        'a.b'
    """
    delimiter = coerce_to_literal(delimiter)
    count = coerce_to_literal(count)
    return Expr(f.substr_index(coerce_to_column(string), delimiter, count))


def substring(string: Expr | str, position: Expr | int, length: Expr | int) -> Expr:
    """Substring from the ``position`` with ``length`` characters.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"a": ["hello world"]})
        >>> result = df.select(dfn.functions.substring("a", 1, 5).alias("s"))
        >>> result.collect_column("s")[0].as_py()
        'hello'
    """
    position = coerce_to_literal(position)
    length = coerce_to_literal(length)
    return Expr(f.substring(coerce_to_column(string), position, length))


def tan(arg: Expr | str) -> Expr:
    """Returns the tangent of the argument.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"a": [0.0]})
        >>> result = df.select(dfn.functions.tan("a").alias("tan"))
        >>> result.collect_column("tan")[0].as_py()
        0.0
    """
    return Expr(f.tan(coerce_to_column(arg)))


def tanh(arg: Expr | str) -> Expr:
    """Returns the hyperbolic tangent of the argument.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"a": [0.0]})
        >>> result = df.select(dfn.functions.tanh("a").alias("tanh"))
        >>> result.collect_column("tanh")[0].as_py()
        0.0
    """
    return Expr(f.tanh(coerce_to_column(arg)))


def to_hex(arg: Expr | str) -> Expr:
    """Converts an integer to a hexadecimal string.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"a": [255]})
        >>> result = df.select(dfn.functions.to_hex("a").alias("hex"))
        >>> result.collect_column("hex")[0].as_py()
        'ff'
    """
    return Expr(f.to_hex(coerce_to_column(arg)))


def now() -> Expr:
    """Returns the current timestamp in nanoseconds.

    This will use the same value for all instances of now() in same statement.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"a": [1]})
        >>> result = df.select(
        ...     dfn.functions.now().alias("now")
        ... )

        Use .value instead of .as_py() because nanosecond timestamps
        require pandas to convert to Python datetime objects.

        >>> result.collect_column("now")[0].value > 0
        True
    """
    return Expr(f.now())


def current_timestamp() -> Expr:
    """Returns the current timestamp in nanoseconds.

    See Also:
        This is an alias for :py:func:`now`.
    """
    return now()


def to_char(arg: Expr | str, formatter: Expr | str) -> Expr:
    """Returns a string representation of a date, time, timestamp or duration.

    For usage of ``formatter`` see the rust chrono package ``strftime`` package.

    [Documentation here.](https://docs.rs/chrono/latest/chrono/format/strftime/index.html)

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"a": ["2021-01-01T00:00:00"]})
        >>> result = df.select(
        ...     dfn.functions.to_char(
        ...         dfn.functions.to_timestamp(dfn.col("a")),
        ...         "%Y/%m/%d",
        ...     ).alias("formatted")
        ... )
        >>> result.collect_column("formatted")[0].as_py()
        '2021/01/01'
    """
    formatter = coerce_to_literal(formatter)
    return Expr(f.to_char(coerce_to_column(arg), formatter))


def date_format(arg: Expr | str, formatter: Expr | str) -> Expr:
    """Returns a string representation of a date, time, timestamp or duration.

    See Also:
        This is an alias for :py:func:`to_char`.
    """
    return to_char(arg, formatter)


def to_date(arg: Expr | str, *formatters: Expr | str) -> Expr:
    """Converts a value to a date (YYYY-MM-DD).

    Supports strings, numeric and timestamp types as input.
    Integers and doubles are interpreted as days since the unix epoch.
    Strings are parsed as YYYY-MM-DD (e.g. '2023-07-20')
    if ``formatters`` are not provided.

    For usage of ``formatters`` see the rust chrono package ``strftime`` package.

    [Documentation here.](https://docs.rs/chrono/latest/chrono/format/strftime/index.html)

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"a": ["2021-07-20"]})
        >>> result = df.select(dfn.functions.to_date("a").alias("dt"))
        >>> str(result.collect_column("dt")[0].as_py())
        '2021-07-20'

        Pass a format string as a bare ``str``:

        >>> df = ctx.from_pydict({"a": ["20-07-2021"]})
        >>> result = df.select(dfn.functions.to_date("a", "%d-%m-%Y").alias("dt"))
        >>> str(result.collect_column("dt")[0].as_py())
        '2021-07-20'
    """
    return Expr(f.to_date(coerce_to_column(arg), *coerce_to_literal_list(formatters)))


def to_local_time(*args: Expr | str) -> Expr:
    """Converts a timestamp with a timezone to a timestamp without a timezone.

    This function handles daylight saving time changes.
    """
    return Expr(f.to_local_time(*coerce_to_column_list(args)))


def to_time(arg: Expr | str, *formatters: Expr | str) -> Expr:
    """Converts a value to a time. Supports strings and timestamps as input.

    If ``formatters`` is not provided strings are parsed as HH:MM:SS, HH:MM or
    HH:MM:SS.nnnnnnnnn;

    For usage of ``formatters`` see the rust chrono package ``strftime`` package.

    [Documentation here.](https://docs.rs/chrono/latest/chrono/format/strftime/index.html)

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"a": ["14:30:00"]})
        >>> result = df.select(dfn.functions.to_time("a").alias("t"))
        >>> str(result.collect_column("t")[0].as_py())
        '14:30:00'

        Pass a format string as a bare ``str``:

        >>> df = ctx.from_pydict({"a": ["14h30m00s"]})
        >>> result = df.select(dfn.functions.to_time("a", "%Hh%Mm%Ss").alias("t"))
        >>> str(result.collect_column("t")[0].as_py())
        '14:30:00'
    """
    return Expr(f.to_time(coerce_to_column(arg), *coerce_to_literal_list(formatters)))


def to_timestamp(arg: Expr | str, *formatters: Expr | str) -> Expr:
    """Converts a string and optional formats to a ``Timestamp`` in nanoseconds.

    For usage of ``formatters`` see the rust chrono package ``strftime`` package.

    [Documentation here.](https://docs.rs/chrono/latest/chrono/format/strftime/index.html)

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"a": ["2021-01-01T00:00:00"]})
        >>> result = df.select(dfn.functions.to_timestamp("a").alias("ts"))
        >>> str(result.collect_column("ts")[0].as_py())
        '2021-01-01 00:00:00'

        Pass a format string as a bare ``str``:

        >>> df = ctx.from_pydict({"a": ["01/01/2021 00:00:00"]})
        >>> result = df.select(
        ...     dfn.functions.to_timestamp(
        ...         "a", "%d/%m/%Y %H:%M:%S"
        ...     ).alias("ts")
        ... )
        >>> str(result.collect_column("ts")[0].as_py())
        '2021-01-01 00:00:00'
    """
    return Expr(
        f.to_timestamp(coerce_to_column(arg), *coerce_to_literal_list(formatters))
    )


def to_timestamp_millis(arg: Expr | str, *formatters: Expr | str) -> Expr:
    """Converts a string and optional formats to a ``Timestamp`` in milliseconds.

    See :py:func:`to_timestamp` for a description on how to use formatters.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"a": ["2021-01-01T00:00:00"]})
        >>> result = df.select(dfn.functions.to_timestamp_millis("a").alias("ts"))
        >>> str(result.collect_column("ts")[0].as_py())
        '2021-01-01 00:00:00'

        Pass a format string as a bare ``str``:

        >>> df = ctx.from_pydict({"a": ["01/01/2021 00:00:00"]})
        >>> result = df.select(
        ...     dfn.functions.to_timestamp_millis(
        ...         "a", "%d/%m/%Y %H:%M:%S"
        ...     ).alias("ts")
        ... )
        >>> str(result.collect_column("ts")[0].as_py())
        '2021-01-01 00:00:00'
    """
    return Expr(
        f.to_timestamp_millis(
            coerce_to_column(arg), *coerce_to_literal_list(formatters)
        )
    )


def to_timestamp_micros(arg: Expr | str, *formatters: Expr | str) -> Expr:
    """Converts a string and optional formats to a ``Timestamp`` in microseconds.

    See :py:func:`to_timestamp` for a description on how to use formatters.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"a": ["2021-01-01T00:00:00"]})
        >>> result = df.select(dfn.functions.to_timestamp_micros("a").alias("ts"))
        >>> str(result.collect_column("ts")[0].as_py())
        '2021-01-01 00:00:00'

        Pass a format string as a bare ``str``:

        >>> df = ctx.from_pydict({"a": ["01/01/2021 00:00:00"]})
        >>> result = df.select(
        ...     dfn.functions.to_timestamp_micros(
        ...         "a", "%d/%m/%Y %H:%M:%S"
        ...     ).alias("ts")
        ... )
        >>> str(result.collect_column("ts")[0].as_py())
        '2021-01-01 00:00:00'
    """
    return Expr(
        f.to_timestamp_micros(
            coerce_to_column(arg), *coerce_to_literal_list(formatters)
        )
    )


def to_timestamp_nanos(arg: Expr | str, *formatters: Expr | str) -> Expr:
    """Converts a string and optional formats to a ``Timestamp`` in nanoseconds.

    See :py:func:`to_timestamp` for a description on how to use formatters.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"a": ["2021-01-01T00:00:00"]})
        >>> result = df.select(dfn.functions.to_timestamp_nanos("a").alias("ts"))
        >>> str(result.collect_column("ts")[0].as_py())
        '2021-01-01 00:00:00'

        Pass a format string as a bare ``str``:

        >>> df = ctx.from_pydict({"a": ["01/01/2021 00:00:00"]})
        >>> result = df.select(
        ...     dfn.functions.to_timestamp_nanos(
        ...         "a", "%d/%m/%Y %H:%M:%S"
        ...     ).alias("ts")
        ... )
        >>> str(result.collect_column("ts")[0].as_py())
        '2021-01-01 00:00:00'
    """
    return Expr(
        f.to_timestamp_nanos(coerce_to_column(arg), *coerce_to_literal_list(formatters))
    )


def to_timestamp_seconds(arg: Expr | str, *formatters: Expr | str) -> Expr:
    """Converts a string and optional formats to a ``Timestamp`` in seconds.

    See :py:func:`to_timestamp` for a description on how to use formatters.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"a": ["2021-01-01T00:00:00"]})
        >>> result = df.select(dfn.functions.to_timestamp_seconds("a").alias("ts"))
        >>> str(result.collect_column("ts")[0].as_py())
        '2021-01-01 00:00:00'

        Pass a format string as a bare ``str``:

        >>> df = ctx.from_pydict({"a": ["01/01/2021 00:00:00"]})
        >>> result = df.select(
        ...     dfn.functions.to_timestamp_seconds(
        ...         "a", "%d/%m/%Y %H:%M:%S"
        ...     ).alias("ts")
        ... )
        >>> str(result.collect_column("ts")[0].as_py())
        '2021-01-01 00:00:00'
    """
    return Expr(
        f.to_timestamp_seconds(
            coerce_to_column(arg), *coerce_to_literal_list(formatters)
        )
    )


def to_unixtime(string: Expr | str, *format_arguments: Expr | str) -> Expr:
    """Converts a string and optional formats to a Unixtime.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"a": ["1970-01-01T00:00:00"]})
        >>> result = df.select(dfn.functions.to_unixtime("a").alias("u"))
        >>> result.collect_column("u")[0].as_py()
        0

        Pass a format string as a bare ``str``:

        >>> df = ctx.from_pydict({"a": ["01/01/1970 00:00:00"]})
        >>> result = df.select(
        ...     dfn.functions.to_unixtime(
        ...         "a", "%d/%m/%Y %H:%M:%S"
        ...     ).alias("u")
        ... )
        >>> result.collect_column("u")[0].as_py()
        0
    """
    return Expr(
        f.to_unixtime(
            coerce_to_column(string),
            *coerce_to_literal_list(format_arguments),
        )
    )


def current_date() -> Expr:
    """Returns current UTC date as a Date32 value.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"a": [1]})
        >>> result = df.select(
        ...     dfn.functions.current_date().alias("d")
        ... )
        >>> result.collect_column("d")[0].as_py() is not None
        True
    """
    return Expr(f.current_date())


today = current_date


def current_time() -> Expr:
    """Returns current UTC time as a Time64 value.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"a": [1]})
        >>> result = df.select(
        ...     dfn.functions.current_time().alias("t")
        ... )

        Use .value instead of .as_py() because nanosecond timestamps
        require pandas to convert to Python datetime objects.

        >>> result.collect_column("t")[0].value > 0
        True
    """
    return Expr(f.current_time())


def datepart(part: Expr | str, date: Expr | str) -> Expr:
    """Return a specified part of a date.

    See Also:
        This is an alias for :py:func:`date_part`.
    """
    return _date_part(part, date, "datepart")


def date_part(part: Expr | str, date: Expr | str) -> Expr:
    """Extracts a subfield from the date.

    Args:
        part: The part of the date to extract. Must be one of ``"year"``,
            ``"month"``, ``"day"``, ``"hour"``, ``"minute"``, ``"second"``, etc.
        date: The date expression to extract from.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"a": ["2021-07-15T00:00:00"]})
        >>> df = df.select(dfn.functions.to_timestamp("a").alias("a"))
        >>> result = df.select(dfn.functions.date_part("year", "a").alias("y"))
        >>> result.collect_column("y")[0].as_py()
        2021
    """
    return _date_part(part, date, "date_part")


def _date_part(part: Expr | str, date: Expr | str, function_name: str) -> Expr:
    _warn_if_expr_for_literal_arg(part, function_name, "part")
    part = coerce_to_literal(part)
    return Expr(f.date_part(part, coerce_to_column(date)))


def extract(part: Expr | str, date: Expr | str) -> Expr:
    """Extracts a subfield from the date.

    See Also:
        This is an alias for :py:func:`date_part`.
    """
    return _date_part(part, date, "extract")


def date_trunc(part: Expr | str, date: Expr | str) -> Expr:
    """Truncates the date to a specified level of precision.

    Args:
        part: The precision to truncate to. Must be one of ``"year"``,
            ``"month"``, ``"day"``, ``"hour"``, ``"minute"``, ``"second"``, etc.
        date: The date expression to truncate.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"a": ["2021-07-15T12:34:56"]})
        >>> df = df.select(dfn.functions.to_timestamp("a").alias("a"))
        >>> result = df.select(dfn.functions.date_trunc("month", "a").alias("t"))
        >>> str(result.collect_column("t")[0].as_py())
        '2021-07-01 00:00:00'
    """
    return _date_trunc(part, date, "date_trunc")


def _date_trunc(part: Expr | str, date: Expr | str, function_name: str) -> Expr:
    _warn_if_expr_for_literal_arg(part, function_name, "part")
    part = coerce_to_literal(part)
    return Expr(f.date_trunc(part, coerce_to_column(date)))


def datetrunc(part: Expr | str, date: Expr | str) -> Expr:
    """Truncates the date to a specified level of precision.

    See Also:
        This is an alias for :py:func:`date_trunc`.
    """
    return _date_trunc(part, date, "datetrunc")


def date_bin(stride: Expr | str, source: Expr | str, origin: Expr | str) -> Expr:
    """Coerces an arbitrary timestamp to the start of the nearest specified interval.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"timestamp": ['2021-07-15 12:34:56', '2021-01-01']})
        >>> result = df.select(
        ...     dfn.functions.date_bin(
        ...         "15 minutes",
        ...         dfn.col("timestamp"),
        ...         "2001-01-01 00:00:00",
        ...     ).alias("b")
        ... )
        >>> str(result.collect_column("b")[0].as_py())
        '2021-07-15 12:30:00'
        >>> str(result.collect_column("b")[1].as_py())
        '2021-01-01 00:00:00'

        ``source`` may also be a bare literal:

        >>> result = df.select(
        ...     dfn.functions.date_bin(
        ...         "15 minutes", "2021-07-15 12:34:56", "2001-01-01 00:00:00"
        ...     ).alias("b")
        ... )
        >>> str(result.collect_column("b")[0].as_py())
        '2021-07-15 12:30:00'
    """
    # date_bin's planner coerces Utf8 (not Utf8View) literals to Interval/Timestamp,
    # so wrap bare strs via string_literal to force Utf8.
    stride = Expr.string_literal(stride) if isinstance(stride, str) else stride
    source = Expr.string_literal(source) if isinstance(source, str) else source
    origin = Expr.string_literal(origin) if isinstance(origin, str) else origin
    return Expr(f.date_bin(stride.expr, source.expr, origin.expr))


def make_date(year: Expr | int, month: Expr | int, day: Expr | int) -> Expr:
    """Make a date from year, month and day component parts.

    Examples:
        >>> from datetime import date
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"y": [2024], "m": [1], "d": [15]})
        >>> result = df.select(
        ...     dfn.functions.make_date(dfn.col("y"), dfn.col("m"),
        ...     dfn.col("d")).alias("dt"))
        >>> result.collect_column("dt")[0].as_py()
        datetime.date(2024, 1, 15)

        Pass bare ints for any component:

        >>> df = ctx.from_pydict({"y": [2024]})
        >>> result = df.select(
        ...     dfn.functions.make_date(dfn.col("y"), 1, 15).alias("dt"))
        >>> result.collect_column("dt")[0].as_py()
        datetime.date(2024, 1, 15)
    """
    year = coerce_to_literal(year)
    month = coerce_to_literal(month)
    day = coerce_to_literal(day)
    return Expr(f.make_date(year, month, day))


def make_time(hour: Expr | int, minute: Expr | int, second: Expr | int) -> Expr:
    """Make a time from hour, minute and second component parts.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"h": [12], "m": [30], "s": [0]})
        >>> result = df.select(
        ...     dfn.functions.make_time(dfn.col("h"), dfn.col("m"),
        ...     dfn.col("s")).alias("t"))
        >>> result.collect_column("t")[0].as_py()
        datetime.time(12, 30)

        Pass bare ints for any component:

        >>> df = ctx.from_pydict({"h": [12]})
        >>> result = df.select(
        ...     dfn.functions.make_time(dfn.col("h"), 30, 0).alias("t"))
        >>> result.collect_column("t")[0].as_py()
        datetime.time(12, 30)
    """
    hour = coerce_to_literal(hour)
    minute = coerce_to_literal(minute)
    second = coerce_to_literal(second)
    return Expr(f.make_time(hour, minute, second))


def translate(string: Expr | str, from_val: Expr | str, to_val: Expr | str) -> Expr:
    """Replaces the characters in ``from_val`` with the counterpart in ``to_val``.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"a": ["hello"]})
        >>> result = df.select(dfn.functions.translate("a", "helo", "HELO").alias("t"))
        >>> result.collect_column("t")[0].as_py()
        'HELLO'
    """
    from_val = coerce_to_literal(from_val)
    to_val = coerce_to_literal(to_val)
    return Expr(f.translate(coerce_to_column(string), from_val, to_val))


def trim(arg: Expr | str) -> Expr:
    """Removes all characters, spaces by default, from both sides of a string.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"a": ["  hello  "]})
        >>> result = df.select(dfn.functions.trim("a").alias("t"))
        >>> result.collect_column("t")[0].as_py()
        'hello'
    """
    return Expr(f.trim(coerce_to_column(arg)))


def trunc(num: Expr | str, precision: Expr | int | None = None) -> Expr:
    """Truncate the number toward zero with optional precision.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"a": [1.567]})
        >>> result = df.select(dfn.functions.trunc("a").alias("t"))
        >>> result.collect_column("t")[0].as_py()
        1.0

        >>> result = df.select(dfn.functions.trunc("a", precision=2).alias("t"))
        >>> result.collect_column("t")[0].as_py()
        1.56
    """
    if precision is not None:
        precision = coerce_to_literal(precision)
        return Expr(f.trunc(coerce_to_column(num), precision))
    return Expr(f.trunc(coerce_to_column(num)))


def upper(arg: Expr | str) -> Expr:
    """Converts a string to uppercase.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"a": ["hello"]})
        >>> result = df.select(dfn.functions.upper("a").alias("u"))
        >>> result.collect_column("u")[0].as_py()
        'HELLO'
    """
    return Expr(f.upper(coerce_to_column(arg)))


def make_array(*args: Expr | str) -> Expr:
    """Returns an array using the specified input expressions.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"a": [1]})
        >>> result = df.select(
        ...     dfn.functions.make_array(
        ...         dfn.lit(1), dfn.lit(2), dfn.lit(3)
        ...     ).alias("arr"))
        >>> result.collect_column("arr")[0].as_py()
        [1, 2, 3]
    """
    args = coerce_to_column_list(args)
    return Expr(f.make_array(args))


def make_list(*args: Expr | str) -> Expr:
    """Returns an array using the specified input expressions.

    See Also:
        This is an alias for :py:func:`make_array`.
    """
    return make_array(*args)


def array(*args: Expr | str) -> Expr:
    """Returns an array using the specified input expressions.

    See Also:
        This is an alias for :py:func:`make_array`.
    """
    return make_array(*args)


def range(start: Expr, stop: Expr, step: Expr) -> Expr:
    """Create a list of values in the range between start and stop.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"a": [1]})
        >>> result = df.select(
        ...     dfn.functions.range(dfn.lit(0), dfn.lit(5), dfn.lit(2)).alias("r"))
        >>> result.collect_column("r")[0].as_py()
        [0, 2, 4]
    """
    return Expr(f.range(ensure_expr(start), ensure_expr(stop), ensure_expr(step)))


def uuid() -> Expr:
    """Returns uuid v4 as a string value.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"a": [1]})
        >>> result = df.select(
        ...     dfn.functions.uuid().alias("u")
        ... )
        >>> len(result.collect_column("u")[0].as_py()) == 36
        True
    """
    return Expr(f.uuid())


def struct(*args: Expr | str) -> Expr:
    """Returns a struct with the given arguments.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"a": [1], "b": [2]})
        >>> result = df.select(dfn.functions.struct("a", "b").alias("s"))

        Children in the new struct will always be `c0`, ..., `cN-1`
        for `N` children.

        >>> result.collect_column("s")[0].as_py() == {"c0": 1, "c1": 2}
        True
    """
    args = coerce_to_column_list(args)
    return Expr(f.struct(*args))


def named_struct(name_pairs: list[tuple[str, Expr | str]]) -> Expr:
    """Returns a struct with the given names and arguments pairs.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"a": [1]})
        >>> result = df.select(
        ...     dfn.functions.named_struct(
        ...         [("x", dfn.lit(10)), ("y", dfn.lit(20))]
        ...     ).alias("s")
        ... )
        >>> result.collect_column("s")[0].as_py() == {"x": 10, "y": 20}
        True
    """
    raw_pairs = [
        raw
        for name, value in name_pairs
        for raw in (
            Expr.literal(pa.scalar(name, type=pa.string())).expr,
            coerce_to_column(value),
        )
    ]
    return Expr(f.named_struct(*raw_pairs))


def from_unixtime(arg: Expr | str) -> Expr:
    """Converts an integer to RFC3339 timestamp format string.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"a": [0]})
        >>> result = df.select(dfn.functions.from_unixtime("a").alias("ts"))
        >>> str(result.collect_column("ts")[0].as_py())
        '1970-01-01 00:00:00'
    """
    return Expr(f.from_unixtime(coerce_to_column(arg)))


def arrow_typeof(arg: Expr | str) -> Expr:
    """Returns the Arrow type of the expression.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"a": [1]})
        >>> result = df.select(dfn.functions.arrow_typeof("a").alias("t"))
        >>> result.collect_column("t")[0].as_py()
        'Int64'
    """
    return Expr(f.arrow_typeof(coerce_to_column(arg)))


def arrow_cast(expr: Expr | str, data_type: Expr | str | pa.DataType) -> Expr:
    """Casts an expression to a specified data type.

    The ``data_type`` can be a string, a ``pyarrow.DataType``, or an
    ``Expr``. For simple types, :py:meth:`Expr.cast()
    <datafusion.expr.Expr.cast>` is more concise
    (e.g., ``col("a").cast(pa.float64())``). Use ``arrow_cast`` when
    you want to specify the target type as a string using DataFusion's
    type syntax, which can be more readable for complex types like
    ``"Timestamp(Nanosecond, None)"``.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"a": [1]})
        >>> result = df.select(dfn.functions.arrow_cast("a", "Float64").alias("c"))
        >>> result.collect_column("c")[0].as_py()
        1.0

        >>> result = df.select(
        ...     dfn.functions.arrow_cast(
        ...         "a", data_type=pa.float64()
        ...     ).alias("c")
        ... )
        >>> result.collect_column("c")[0].as_py()
        1.0
    """
    _warn_if_expr_for_literal_arg(data_type, "arrow_cast", "data_type")
    if isinstance(data_type, pa.DataType):
        return Expr(coerce_to_column(expr)).cast(data_type)
    if isinstance(data_type, str):
        data_type = Expr.string_literal(data_type)
    return Expr(f.arrow_cast(coerce_to_column(expr), data_type.expr))


def arrow_try_cast(expr: Expr | str, data_type: Expr | str | pa.DataType) -> Expr:
    """Casts an expression to a specified data type, returning NULL on failure.

    Like :py:func:`arrow_cast` but produces NULL instead of erroring when the
    cast cannot be performed. The ``data_type`` may be a string in DataFusion
    type syntax (for example ``"Float64"``), a ``pyarrow.DataType``, or an
    ``Expr`` of string type.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"a": ["oops"]})
        >>> result = df.select(dfn.functions.arrow_try_cast("a", "Float64").alias("c"))
        >>> result.collect_column("c")[0].as_py() is None
        True

        >>> result = df.select(
        ...     dfn.functions.arrow_try_cast(
        ...         "a", data_type=pa.float64()
        ...     ).alias("c")
        ... )
        >>> result.collect_column("c")[0].as_py() is None
        True
    """
    _warn_if_expr_for_literal_arg(data_type, "arrow_try_cast", "data_type")
    if isinstance(data_type, pa.DataType):
        return Expr(coerce_to_column(expr)).try_cast(data_type)
    if isinstance(data_type, str):
        data_type = Expr.string_literal(data_type)
    return Expr(f.arrow_try_cast(coerce_to_column(expr), data_type.expr))


def arrow_field(expr: Expr | str) -> Expr:
    """Returns the Arrow field information of an expression as a struct.

    The returned struct contains the field's name, data type, nullability,
    and metadata.

    Examples:
        >>> field = pa.field("val", pa.int64(), metadata={"k": "v"})
        >>> schema = pa.schema([field])
        >>> batch = pa.RecordBatch.from_arrays([pa.array([1])], schema=schema)
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.create_dataframe([[batch]])
        >>> result = df.select(dfn.functions.arrow_field("val").alias("f"))
        >>> out = result.collect_column("f")[0].as_py()
        >>> out["name"], out["data_type"], out["nullable"], out["metadata"]
        ('val', 'Int64', True, [('k', 'v')])
    """
    return Expr(f.arrow_field(coerce_to_column(expr)))


def cast_to_type(value: Expr | str, type_ref: Expr | str) -> Expr:
    """Casts ``value`` to the data type of ``type_ref``.

    Only the *type* of ``type_ref`` is used; its value is ignored. This is
    useful when the target type comes from another column or expression
    rather than being known up-front. Casts that fail produce an error; use
    :py:func:`try_cast_to_type` for the NULL-on-failure variant.

    If the target type is known statically, prefer :py:func:`arrow_cast`
    (or :py:func:`arrow_try_cast` for the NULL-on-failure variant) and
    pass a type string or ``pyarrow.DataType`` directly.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"a": [1], "b": [1.0]})
        >>> result = df.select(dfn.functions.cast_to_type("a", "b").alias("c"))
        >>> result.collect_column("c")[0].as_py()
        1.0
    """
    return Expr(f.cast_to_type(coerce_to_column(value), coerce_to_column(type_ref)))


def try_cast_to_type(value: Expr | str, type_ref: Expr | str) -> Expr:
    """Casts ``value`` to the data type of ``type_ref``, NULL on failure.

    Like :py:func:`cast_to_type`, but casts that fail produce NULL instead
    of erroring. Only the *type* of ``type_ref`` is used; its value is
    ignored.

    If the target type is known statically, prefer :py:func:`arrow_try_cast`
    and pass a type string or ``pyarrow.DataType`` directly.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"a": ["oops"], "b": [1.0]})
        >>> result = df.select(dfn.functions.try_cast_to_type("a", "b").alias("c"))
        >>> result.collect_column("c")[0].as_py() is None
        True
    """
    return Expr(f.try_cast_to_type(coerce_to_column(value), coerce_to_column(type_ref)))


def arrow_metadata(expr: Expr | str, key: Expr | str | None = None) -> Expr:
    """Returns the metadata of the input expression.

    If called with one argument, returns a Map of all metadata key-value pairs.
    If called with two arguments, returns the value for the specified metadata key.

    Examples:
        >>> field = pa.field("val", pa.int64(), metadata={"k": "v"})
        >>> schema = pa.schema([field])
        >>> batch = pa.RecordBatch.from_arrays([pa.array([1])], schema=schema)
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.create_dataframe([[batch]])
        >>> result = df.select(dfn.functions.arrow_metadata("val").alias("meta"))
        >>> ("k", "v") in result.collect_column("meta")[0].as_py()
        True

        >>> result = df.select(
        ...     dfn.functions.arrow_metadata(
        ...         "val", key="k"
        ...     ).alias("meta_val")
        ... )
        >>> result.collect_column("meta_val")[0].as_py()
        'v'
    """
    if key is None:
        return Expr(f.arrow_metadata(coerce_to_column(expr)))
    _warn_if_expr_for_literal_arg(key, "arrow_metadata", "key")
    if isinstance(key, str):
        key = Expr.string_literal(key)
    return Expr(f.arrow_metadata(coerce_to_column(expr), key.expr))


def with_metadata(expr: Expr | str, metadata: dict[str, str]) -> Expr:
    """Attaches Arrow field metadata (key/value pairs) to the input expression.

    This is the inverse of :py:func:`arrow_metadata`. Existing metadata on the
    input field is preserved; new keys overwrite on collision. Keys must be
    non-empty strings; empty values are allowed.

    An empty ``metadata`` dict is a no-op and returns the input expression
    unchanged. Empty keys raise :py:class:`ValueError`.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"a": [1]})
        >>> result = df.select(
        ...     dfn.functions.with_metadata(
        ...         "a", {"unit": "ms"}
        ...     ).alias("a")
        ... )
        >>> result.select(
        ...     dfn.functions.arrow_metadata("a", "unit").alias("u")
        ... ).collect_column("u")[0].as_py()
        'ms'
    """
    if not metadata:
        return expr
    args = [coerce_to_column(expr)]
    for k, v in metadata.items():
        if not k:
            msg = "with_metadata keys must be non-empty strings"
            raise ValueError(msg)
        args.append(Expr.string_literal(k).expr)
        args.append(Expr.string_literal(v).expr)
    return Expr(f.with_metadata(*args))


def get_field(expr: Expr | str, *names: Expr | str) -> Expr:
    """Extracts a (possibly nested) field from a struct or map by name.

    Pass one name for a single-level lookup, or several names to walk a path
    of nested struct/map fields in a single ``get_field`` call. For a single
    static-string name, ``expr["field"]`` is a convenient shorthand; use
    ``get_field`` when the field name is a dynamic
    :py:class:`~datafusion.expr.Expr` or when traversing multiple levels at
    once.

    Args:
        expr: The struct or map expression to read from.
        *names: One or more field names (``str``) or expressions
            (:py:class:`~datafusion.expr.Expr`).

    Examples:
        Single-level lookup:

        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"a": [1], "b": [2]})
        >>> df = df.with_column(
        ...     "s",
        ...     F.named_struct([("x", dfn.col("a")), ("y", dfn.col("b"))]),
        ... )
        >>> result = df.select(F.get_field("s", "x").alias("x_val"))
        >>> result.collect_column("x_val")[0].as_py()
        1

        Equivalent using bracket syntax:

        >>> result = df.select(
        ...     dfn.col("s")["x"].alias("x_val")
        ... )
        >>> result.collect_column("x_val")[0].as_py()
        1

        Multi-level lookup:

        >>> df = df.with_column(
        ...     "outer",
        ...     F.named_struct([("inner", dfn.col("s"))]),
        ... )
        >>> result = df.select(F.get_field("outer", "inner", "x").alias("x_val"))
        >>> result.collect_column("x_val")[0].as_py()
        1
    """
    if not names:
        msg = "get_field requires at least one field name"
        raise ValueError(msg)
    resolved = [Expr.string_literal(n) if isinstance(n, str) else n for n in names]
    return Expr(f.get_field(coerce_to_column(expr), [n.expr for n in resolved]))


def union_extract(union_expr: Expr | str, field_name: Expr | str) -> Expr:
    """Extracts a value from a union type by field name.

    Returns the value of the named field if it is the currently selected
    variant, otherwise returns NULL.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> types = pa.array([0, 1, 0], type=pa.int8())
        >>> offsets = pa.array([0, 0, 1], type=pa.int32())
        >>> arr = pa.UnionArray.from_dense(
        ...     types, offsets, [pa.array([1, 2]), pa.array(["hi"])],
        ...     ["int", "str"], [0, 1],
        ... )
        >>> batch = pa.RecordBatch.from_arrays([arr], names=["u"])
        >>> df = ctx.create_dataframe([[batch]])
        >>> result = df.select(dfn.functions.union_extract("u", "int").alias("val"))
        >>> result.collect_column("val").to_pylist()
        [1, None, 2]
    """
    if isinstance(field_name, str):
        field_name = Expr.string_literal(field_name)
    return Expr(f.union_extract(coerce_to_column(union_expr), field_name.expr))


def union_tag(union_expr: Expr | str) -> Expr:
    """Returns the tag (active field name) of a union type.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> types = pa.array([0, 1, 0], type=pa.int8())
        >>> offsets = pa.array([0, 0, 1], type=pa.int32())
        >>> arr = pa.UnionArray.from_dense(
        ...     types, offsets, [pa.array([1, 2]), pa.array(["hi"])],
        ...     ["int", "str"], [0, 1],
        ... )
        >>> batch = pa.RecordBatch.from_arrays([arr], names=["u"])
        >>> df = ctx.create_dataframe([[batch]])
        >>> result = df.select(dfn.functions.union_tag("u").alias("tag"))
        >>> result.collect_column("tag").to_pylist()
        ['int', 'str', 'int']
    """
    return Expr(f.union_tag(coerce_to_column(union_expr)))


def version() -> Expr:
    """Returns the DataFusion version string.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.empty_table()
        >>> result = df.select(dfn.functions.version().alias("v"))
        >>> "Apache DataFusion" in result.collect_column("v")[0].as_py()
        True
    """
    return Expr(f.version())


def row(*args: Expr | str) -> Expr:
    """Returns a struct with the given arguments.

    See Also:
        This is an alias for :py:func:`struct`.
    """
    return struct(*args)


def random() -> Expr:
    """Returns a random value in the range ``0.0 <= x < 1.0``.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"a": [1]})
        >>> result = df.select(
        ...     dfn.functions.random().alias("r")
        ... )
        >>> val = result.collect_column("r")[0].as_py()
        >>> 0.0 <= val < 1.0
        True
    """
    return Expr(f.random())


def array_append(array: Expr | str, element: Expr) -> Expr:
    """Appends an element to the end of an array.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"a": [[1, 2, 3]]})
        >>> result = df.select(
        ...     dfn.functions.array_append("a", dfn.lit(4)).alias("result"))
        >>> result.collect_column("result")[0].as_py()
        [1, 2, 3, 4]
    """
    return Expr(f.array_append(coerce_to_column(array), ensure_expr(element)))


def array_push_back(array: Expr | str, element: Expr) -> Expr:
    """Appends an element to the end of an array.

    See Also:
        This is an alias for :py:func:`array_append`.
    """
    return array_append(array, element)


def list_append(array: Expr | str, element: Expr) -> Expr:
    """Appends an element to the end of an array.

    See Also:
        This is an alias for :py:func:`array_append`.
    """
    return array_append(array, element)


def list_push_back(array: Expr | str, element: Expr) -> Expr:
    """Appends an element to the end of an array.

    See Also:
        This is an alias for :py:func:`array_append`.
    """
    return array_append(array, element)


def array_concat(*args: Expr | str) -> Expr:
    """Concatenates the input arrays.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"a": [[1, 2]], "b": [[3, 4]]})
        >>> result = df.select(dfn.functions.array_concat("a", "b").alias("result"))
        >>> result.collect_column("result")[0].as_py()
        [1, 2, 3, 4]
    """
    args = coerce_to_column_list(args)
    return Expr(f.array_concat(args))


def array_cat(*args: Expr | str) -> Expr:
    """Concatenates the input arrays.

    See Also:
        This is an alias for :py:func:`array_concat`.
    """
    return array_concat(*args)


def array_dims(array: Expr | str) -> Expr:
    """Returns an array of the array's dimensions.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"a": [[1, 2, 3]]})
        >>> result = df.select(dfn.functions.array_dims("a").alias("result"))
        >>> result.collect_column("result")[0].as_py()
        [3]
    """
    return Expr(f.array_dims(coerce_to_column(array)))


def array_distinct(array: Expr | str) -> Expr:
    """Returns distinct values from the array after removing duplicates.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"a": [[1, 1, 2, 3]]})
        >>> result = df.select(dfn.functions.array_distinct("a").alias("result"))
        >>> sorted(
        ...     result.collect_column("result")[0].as_py()
        ... )
        [1, 2, 3]
    """
    return Expr(f.array_distinct(coerce_to_column(array)))


def array_compact(array: Expr | str) -> Expr:
    """Removes NULL values from the array.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"a": [[1, None, 2, None, 3]]})
        >>> result = df.select(dfn.functions.array_compact("a").alias("result"))
        >>> result.collect_column("result")[0].as_py()
        [1, 2, 3]
    """
    return Expr(f.array_compact(coerce_to_column(array)))


def array_normalize(array: Expr | str) -> Expr:
    """Scales a numeric array so it has Euclidean length 1.

    Treats the array as a vector and divides every element by the vector's
    Euclidean (L2) norm — the square root of the sum of the squared
    elements. The returned array points in the same direction as the input
    but has a magnitude of 1, which makes it suitable for cosine-similarity
    comparisons and other operations that expect unit vectors.

    For the input ``[3.0, 4.0]`` the L2 norm is ``sqrt(3**2 + 4**2) = 5``,
    so each element is divided by 5 to produce ``[0.6, 0.8]``.

    Normalizing the zero vector is undefined (it would divide by zero), so
    the function returns NULL for an all-zero input. NULL is also returned
    if any element of the input array is NULL.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"a": [[3.0, 4.0]]})
        >>> result = df.select(dfn.functions.array_normalize("a").alias("result"))
        >>> result.collect_column("result")[0].as_py()
        [0.6, 0.8]

        The zero vector has no direction to preserve, so the result is NULL:

        >>> df_zero = ctx.from_pydict({"a": [[0.0, 0.0]]})
        >>> result = df_zero.select(dfn.functions.array_normalize("a").alias("result"))
        >>> result.collect_column("result")[0].as_py() is None
        True
    """
    return Expr(f.array_normalize(coerce_to_column(array)))


def cosine_distance(array1: Expr | str, array2: Expr | str) -> Expr:
    """Measures how much two numeric arrays differ in direction.

    Treats each input as a vector and compares the angle between them,
    ignoring their magnitudes. The result is ``1 - cosine_similarity``,
    where cosine similarity is the dot product of the two vectors divided
    by the product of their Euclidean (L2) norms.

    The returned value ranges from 0 to 2:

    * ``0`` — vectors point in the same direction (any positive scaling
      of one yields the other).
    * ``1`` — vectors are orthogonal (no shared direction).
    * ``2`` — vectors point in exactly opposite directions.

    This is the standard distance metric for comparing embedding vectors
    (text, image, audio) where direction carries the meaning and overall
    magnitude does not.

    Both arrays must have the same length; otherwise execution fails. If
    either input is the zero vector the cosine is undefined and the
    function returns NULL.

    Examples:
        Identical vectors have distance ``0``:

        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict(
        ...     {"a": [[1.0, 2.0, 3.0]], "b": [[1.0, 2.0, 3.0]]}
        ... )
        >>> result = df.select(dfn.functions.cosine_distance("a", "b").alias("result"))
        >>> result.collect_column("result")[0].as_py()
        0.0

        Orthogonal vectors have distance ``1``:

        >>> df_orth = ctx.from_pydict(
        ...     {"a": [[1.0, 0.0]], "b": [[0.0, 1.0]]}
        ... )
        >>> result = df_orth.select(
        ...     dfn.functions.cosine_distance(
        ...         "a", "b"
        ...     ).alias("result")
        ... )
        >>> result.collect_column("result")[0].as_py()
        1.0
    """
    return Expr(f.cosine_distance(coerce_to_column(array1), coerce_to_column(array2)))


def inner_product(array1: Expr | str, array2: Expr | str) -> Expr:
    """Returns the inner (dot) product of two numeric arrays.

    Treats each input as a vector and returns the sum of the element-wise
    products: ``sum(array1[i] * array2[i])``. For ``[1, 2, 3]`` and
    ``[4, 5, 6]`` the result is ``1*4 + 2*5 + 3*6 = 32``.

    Also available as :py:func:`dot_product` (and as ``dot_product`` in
    raw SQL).

    Both arrays must have the same length; otherwise execution fails. NULL
    is returned when either input array is NULL or when any element of
    either array is NULL.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict(
        ...     {"a": [[1.0, 2.0, 3.0]], "b": [[4.0, 5.0, 6.0]]}
        ... )
        >>> result = df.select(dfn.functions.inner_product("a", "b").alias("result"))
        >>> result.collect_column("result")[0].as_py()
        32.0

        NULL elements propagate to NULL output:

        >>> df_null = ctx.from_pydict(
        ...     {"a": [[1.0, None, 3.0]], "b": [[4.0, 5.0, 6.0]]}
        ... )
        >>> result = df_null.select(
        ...     dfn.functions.inner_product(
        ...         "a", "b"
        ...     ).alias("result")
        ... )
        >>> result.collect_column("result")[0].as_py() is None
        True
    """
    return Expr(f.inner_product(coerce_to_column(array1), coerce_to_column(array2)))


def dot_product(array1: Expr | str, array2: Expr | str) -> Expr:
    """Returns the inner (dot) product of two numeric arrays.

    See Also:
        This is an alias for :py:func:`inner_product`.
    """
    return inner_product(array1, array2)


def list_cat(*args: Expr | str) -> Expr:
    """Concatenates the input arrays.

    See Also:
        This is an alias for :py:func:`array_concat`, :py:func:`array_cat`.
    """
    return array_concat(*args)


def list_concat(*args: Expr | str) -> Expr:
    """Concatenates the input arrays.

    See Also:
        This is an alias for :py:func:`array_concat`, :py:func:`array_cat`.
    """
    return array_concat(*args)


def list_distinct(array: Expr | str) -> Expr:
    """Returns distinct values from the array after removing duplicates.

    See Also:
        This is an alias for :py:func:`array_distinct`.
    """
    return array_distinct(array)


def list_compact(array: Expr | str) -> Expr:
    """Removes NULL values from the array.

    See Also:
        This is an alias for :py:func:`array_compact`.
    """
    return array_compact(array)


def list_normalize(array: Expr | str) -> Expr:
    """Scales a numeric array so it has Euclidean length 1.

    See Also:
        This is an alias for :py:func:`array_normalize`.
    """
    return array_normalize(array)


def list_dims(array: Expr | str) -> Expr:
    """Returns an array of the array's dimensions.

    See Also:
        This is an alias for :py:func:`array_dims`.
    """
    return array_dims(array)


def array_element(array: Expr | str, n: Expr | int) -> Expr:
    """Extracts the element with the index n from the array.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"a": [[10, 20, 30]]})
        >>> result = df.select(dfn.functions.array_element("a", 2).alias("result"))
        >>> result.collect_column("result")[0].as_py()
        20
    """
    n = coerce_to_literal(n)
    return Expr(f.array_element(coerce_to_column(array), n))


def array_empty(array: Expr | str) -> Expr:
    """Returns a boolean indicating whether the array is empty.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"a": [[1, 2]]})
        >>> result = df.select(dfn.functions.array_empty("a").alias("result"))
        >>> result.collect_column("result")[0].as_py()
        False
    """
    return Expr(f.array_empty(coerce_to_column(array)))


def list_empty(array: Expr | str) -> Expr:
    """Returns a boolean indicating whether the array is empty.

    See Also:
        This is an alias for :py:func:`array_empty`.
    """
    return array_empty(array)


def array_extract(array: Expr | str, n: Expr | int) -> Expr:
    """Extracts the element with the index n from the array.

    See Also:
        This is an alias for :py:func:`array_element`.
    """
    return array_element(array, n)


def list_element(array: Expr | str, n: Expr | int) -> Expr:
    """Extracts the element with the index n from the array.

    See Also:
        This is an alias for :py:func:`array_element`.
    """
    return array_element(array, n)


def list_extract(array: Expr | str, n: Expr | int) -> Expr:
    """Extracts the element with the index n from the array.

    See Also:
        This is an alias for :py:func:`array_element`.
    """
    return array_element(array, n)


def array_length(array: Expr | str) -> Expr:
    """Returns the length of the array.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"a": [[1, 2, 3]]})
        >>> result = df.select(dfn.functions.array_length("a").alias("result"))
        >>> result.collect_column("result")[0].as_py()
        3
    """
    return Expr(f.array_length(coerce_to_column(array)))


def list_length(array: Expr | str) -> Expr:
    """Returns the length of the array.

    See Also:
        This is an alias for :py:func:`array_length`.
    """
    return array_length(array)


def array_has(first_array: Expr | str, second_array: Expr | str) -> Expr:
    """Returns true if the element appears in the first array, otherwise false.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"a": [[1, 2, 3]]})
        >>> result = df.select(dfn.functions.array_has("a", dfn.lit(2)).alias("result"))
        >>> result.collect_column("result")[0].as_py()
        True
    """
    return Expr(
        f.array_has(coerce_to_column(first_array), coerce_to_column(second_array))
    )


def array_has_all(first_array: Expr | str, second_array: Expr | str) -> Expr:
    """Determines if there is complete overlap ``second_array`` in ``first_array``.

    Returns true if each element of the second array appears in the first array.
    Otherwise, it returns false.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"a": [[1, 2, 3]], "b": [[1, 2]]})
        >>> result = df.select(dfn.functions.array_has_all("a", "b").alias("result"))
        >>> result.collect_column("result")[0].as_py()
        True
    """
    return Expr(
        f.array_has_all(coerce_to_column(first_array), coerce_to_column(second_array))
    )


def array_has_any(first_array: Expr | str, second_array: Expr | str) -> Expr:
    """Determine if there is an overlap between ``first_array`` and ``second_array``.

    Returns true if at least one element of the second array appears in the first
    array. Otherwise, it returns false.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"a": [[1, 2, 3]], "b": [[2, 5]]})
        >>> result = df.select(dfn.functions.array_has_any("a", "b").alias("result"))
        >>> result.collect_column("result")[0].as_py()
        True
    """
    return Expr(
        f.array_has_any(coerce_to_column(first_array), coerce_to_column(second_array))
    )


def array_contains(array: Expr | str, element: Expr) -> Expr:
    """Returns true if the element appears in the array, otherwise false.

    See Also:
        This is an alias for :py:func:`array_has`.
    """
    return array_has(array, element)


def list_has(array: Expr | str, element: Expr) -> Expr:
    """Returns true if the element appears in the array, otherwise false.

    See Also:
        This is an alias for :py:func:`array_has`.
    """
    return array_has(array, element)


def list_has_all(first_array: Expr | str, second_array: Expr | str) -> Expr:
    """Determines if there is complete overlap ``second_array`` in ``first_array``.

    See Also:
        This is an alias for :py:func:`array_has_all`.
    """
    return array_has_all(first_array, second_array)


def list_has_any(first_array: Expr | str, second_array: Expr | str) -> Expr:
    """Determine if there is an overlap between ``first_array`` and ``second_array``.

    See Also:
        This is an alias for :py:func:`array_has_any`.
    """
    return array_has_any(first_array, second_array)


def arrays_overlap(first_array: Expr | str, second_array: Expr | str) -> Expr:
    """Returns true if any element appears in both arrays.

    See Also:
        This is an alias for :py:func:`array_has_any`.
    """
    return array_has_any(first_array, second_array)


def list_overlap(first_array: Expr | str, second_array: Expr | str) -> Expr:
    """Returns true if any element appears in both arrays.

    See Also:
        This is an alias for :py:func:`array_has_any`.
    """
    return array_has_any(first_array, second_array)


def list_contains(array: Expr | str, element: Expr) -> Expr:
    """Returns true if the element appears in the array, otherwise false.

    See Also:
        This is an alias for :py:func:`array_has`.
    """
    return array_has(array, element)


def array_position(array: Expr | str, element: Expr, index: int | None = 1) -> Expr:
    """Return the position of the first occurrence of ``element`` in ``array``.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"a": [[10, 20, 30]]})
        >>> result = df.select(
        ...     dfn.functions.array_position(
        ...         "a", dfn.lit(20)
        ...     ).alias("result"))
        >>> result.collect_column("result")[0].as_py()
        2

        Use ``index`` to start searching from a given position:

        >>> df = ctx.from_pydict({"a": [[10, 20, 10, 20]]})
        >>> result = df.select(
        ...     dfn.functions.array_position(
        ...         "a", dfn.lit(20), index=3,
        ...     ).alias("result"))
        >>> result.collect_column("result")[0].as_py()
        4
    """
    return Expr(f.array_position(coerce_to_column(array), ensure_expr(element), index))


def array_indexof(array: Expr | str, element: Expr, index: int | None = 1) -> Expr:
    """Return the position of the first occurrence of ``element`` in ``array``.

    See Also:
        This is an alias for :py:func:`array_position`.
    """
    return array_position(array, element, index)


def list_position(array: Expr | str, element: Expr, index: int | None = 1) -> Expr:
    """Return the position of the first occurrence of ``element`` in ``array``.

    See Also:
        This is an alias for :py:func:`array_position`.
    """
    return array_position(array, element, index)


def list_indexof(array: Expr | str, element: Expr, index: int | None = 1) -> Expr:
    """Return the position of the first occurrence of ``element`` in ``array``.

    See Also:
        This is an alias for :py:func:`array_position`.
    """
    return array_position(array, element, index)


def array_positions(array: Expr | str, element: Expr) -> Expr:
    """Searches for an element in the array and returns all occurrences.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"a": [[1, 2, 1]]})
        >>> result = df.select(
        ...     dfn.functions.array_positions("a", dfn.lit(1)).alias("result"))
        >>> result.collect_column("result")[0].as_py()
        [1, 3]
    """
    return Expr(f.array_positions(coerce_to_column(array), ensure_expr(element)))


def list_positions(array: Expr | str, element: Expr) -> Expr:
    """Searches for an element in the array and returns all occurrences.

    See Also:
        This is an alias for :py:func:`array_positions`.
    """
    return array_positions(array, element)


def array_ndims(array: Expr | str) -> Expr:
    """Returns the number of dimensions of the array.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"a": [[1, 2, 3]]})
        >>> result = df.select(dfn.functions.array_ndims("a").alias("result"))
        >>> result.collect_column("result")[0].as_py()
        1
    """
    return Expr(f.array_ndims(coerce_to_column(array)))


def list_ndims(array: Expr | str) -> Expr:
    """Returns the number of dimensions of the array.

    See Also:
        This is an alias for :py:func:`array_ndims`.
    """
    return array_ndims(array)


def array_prepend(element: Expr, array: Expr | str) -> Expr:
    """Prepends an element to the beginning of an array.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"a": [[1, 2]]})
        >>> result = df.select(
        ...     dfn.functions.array_prepend(dfn.lit(0), "a").alias("result"))
        >>> result.collect_column("result")[0].as_py()
        [0, 1, 2]
    """
    return Expr(f.array_prepend(ensure_expr(element), coerce_to_column(array)))


def array_push_front(element: Expr, array: Expr | str) -> Expr:
    """Prepends an element to the beginning of an array.

    See Also:
        This is an alias for :py:func:`array_prepend`.
    """
    return array_prepend(element, array)


def list_prepend(element: Expr, array: Expr | str) -> Expr:
    """Prepends an element to the beginning of an array.

    See Also:
        This is an alias for :py:func:`array_prepend`.
    """
    return array_prepend(element, array)


def list_push_front(element: Expr, array: Expr | str) -> Expr:
    """Prepends an element to the beginning of an array.

    See Also:
        This is an alias for :py:func:`array_prepend`.
    """
    return array_prepend(element, array)


def array_pop_back(array: Expr | str) -> Expr:
    """Returns the array without the last element.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"a": [[1, 2, 3]]})
        >>> result = df.select(dfn.functions.array_pop_back("a").alias("result"))
        >>> result.collect_column("result")[0].as_py()
        [1, 2]
    """
    return Expr(f.array_pop_back(coerce_to_column(array)))


def array_pop_front(array: Expr | str) -> Expr:
    """Returns the array without the first element.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"a": [[1, 2, 3]]})
        >>> result = df.select(dfn.functions.array_pop_front("a").alias("result"))
        >>> result.collect_column("result")[0].as_py()
        [2, 3]
    """
    return Expr(f.array_pop_front(coerce_to_column(array)))


def list_pop_back(array: Expr | str) -> Expr:
    """Returns the array without the last element.

    See Also:
        This is an alias for :py:func:`array_pop_back`.
    """
    return array_pop_back(array)


def list_pop_front(array: Expr | str) -> Expr:
    """Returns the array without the first element.

    See Also:
        This is an alias for :py:func:`array_pop_front`.
    """
    return array_pop_front(array)


def array_remove(array: Expr | str, element: Expr) -> Expr:
    """Removes the first element from the array equal to the given value.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"a": [[1, 2, 1]]})
        >>> result = df.select(
        ...     dfn.functions.array_remove("a", dfn.lit(1)).alias("result"))
        >>> result.collect_column("result")[0].as_py()
        [2, 1]
    """
    return Expr(f.array_remove(coerce_to_column(array), ensure_expr(element)))


def list_remove(array: Expr | str, element: Expr) -> Expr:
    """Removes the first element from the array equal to the given value.

    See Also:
        This is an alias for :py:func:`array_remove`.
    """
    return array_remove(array, element)


def array_remove_n(array: Expr | str, element: Expr, max: Expr | int) -> Expr:
    """Removes the first ``max`` elements from the array equal to the given value.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"a": [[1, 2, 1, 1]]})
        >>> result = df.select(
        ...     dfn.functions.array_remove_n(
        ...         "a", dfn.lit(1), 2
        ...     ).alias("result"))
        >>> result.collect_column("result")[0].as_py()
        [2, 1]
    """
    max = coerce_to_literal(max)
    return Expr(f.array_remove_n(coerce_to_column(array), ensure_expr(element), max))


def list_remove_n(array: Expr | str, element: Expr, max: Expr | int) -> Expr:
    """Removes the first ``max`` elements from the array equal to the given value.

    See Also:
        This is an alias for :py:func:`array_remove_n`.
    """
    return array_remove_n(array, element, max)


def array_remove_all(array: Expr | str, element: Expr) -> Expr:
    """Removes all elements from the array equal to the given value.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"a": [[1, 2, 1]]})
        >>> result = df.select(
        ...     dfn.functions.array_remove_all(
        ...         "a", dfn.lit(1)
        ...     ).alias("result"))
        >>> result.collect_column("result")[0].as_py()
        [2]
    """
    return Expr(f.array_remove_all(coerce_to_column(array), ensure_expr(element)))


def list_remove_all(array: Expr | str, element: Expr) -> Expr:
    """Removes all elements from the array equal to the given value.

    See Also:
        This is an alias for :py:func:`array_remove_all`.
    """
    return array_remove_all(array, element)


def array_repeat(element: Expr, count: Expr | int) -> Expr:
    """Returns an array containing ``element`` ``count`` times.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"a": [1]})
        >>> result = df.select(
        ...     dfn.functions.array_repeat(dfn.lit(3), 3).alias("result"))
        >>> result.collect_column("result")[0].as_py()
        [3, 3, 3]
    """
    count = coerce_to_literal(count)
    return Expr(f.array_repeat(ensure_expr(element), count))


def list_repeat(element: Expr, count: Expr | int) -> Expr:
    """Returns an array containing ``element`` ``count`` times.

    See Also:
        This is an alias for :py:func:`array_repeat`.
    """
    return array_repeat(element, count)


def array_replace(array: Expr | str, from_val: Expr, to_val: Expr) -> Expr:
    """Replaces the first occurrence of ``from_val`` with ``to_val``.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"a": [[1, 2, 1]]})
        >>> result = df.select(
        ...     dfn.functions.array_replace("a", dfn.lit(1),
        ...     dfn.lit(9)).alias("result"))
        >>> result.collect_column("result")[0].as_py()
        [9, 2, 1]
    """
    return Expr(
        f.array_replace(
            coerce_to_column(array), ensure_expr(from_val), ensure_expr(to_val)
        )
    )


def list_replace(array: Expr | str, from_val: Expr, to_val: Expr) -> Expr:
    """Replaces the first occurrence of ``from_val`` with ``to_val``.

    See Also:
        This is an alias for :py:func:`array_replace`.
    """
    return array_replace(array, from_val, to_val)


def array_replace_n(
    array: Expr | str, from_val: Expr, to_val: Expr, max: Expr | int
) -> Expr:
    """Replace ``n`` occurrences of ``from_val`` with ``to_val``.

    Replaces the first ``max`` occurrences of the specified element with another
    specified element.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"a": [[1, 2, 1, 1]]})
        >>> result = df.select(
        ...     dfn.functions.array_replace_n(
        ...         "a", dfn.lit(1), dfn.lit(9), 2
        ...     ).alias("result"))
        >>> result.collect_column("result")[0].as_py()
        [9, 2, 9, 1]
    """
    max = coerce_to_literal(max)
    return Expr(
        f.array_replace_n(
            coerce_to_column(array),
            ensure_expr(from_val),
            ensure_expr(to_val),
            max,
        )
    )


def list_replace_n(
    array: Expr | str, from_val: Expr, to_val: Expr, max: Expr | int
) -> Expr:
    """Replace ``n`` occurrences of ``from_val`` with ``to_val``.

    Replaces the first ``max`` occurrences of the specified element with another
    specified element.

    See Also:
        This is an alias for :py:func:`array_replace_n`.
    """
    return array_replace_n(array, from_val, to_val, max)


def array_replace_all(array: Expr | str, from_val: Expr, to_val: Expr) -> Expr:
    """Replaces all occurrences of ``from_val`` with ``to_val``.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"a": [[1, 2, 1]]})
        >>> result = df.select(
        ...     dfn.functions.array_replace_all("a", dfn.lit(1),
        ...     dfn.lit(9)).alias("result"))
        >>> result.collect_column("result")[0].as_py()
        [9, 2, 9]
    """
    return Expr(
        f.array_replace_all(
            coerce_to_column(array), ensure_expr(from_val), ensure_expr(to_val)
        )
    )


def list_replace_all(array: Expr | str, from_val: Expr, to_val: Expr) -> Expr:
    """Replaces all occurrences of ``from_val`` with ``to_val``.

    See Also:
        This is an alias for :py:func:`array_replace_all`.
    """
    return array_replace_all(array, from_val, to_val)


def array_sort(
    array: Expr | str, descending: bool = False, null_first: bool = False
) -> Expr:
    """Sort an array.

    Args:
        array: The input array to sort.
        descending: If True, sorts in descending order.
        null_first: If True, nulls will be returned at the beginning of the array.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"a": [[3, 1, 2]]})
        >>> result = df.select(dfn.functions.array_sort("a").alias("result"))
        >>> result.collect_column("result")[0].as_py()
        [1, 2, 3]

        >>> df = ctx.from_pydict({"a": [[3, None, 1]]})
        >>> result = df.select(
        ...     dfn.functions.array_sort(
        ...         "a", descending=True, null_first=True,
        ...     ).alias("result"))
        >>> result.collect_column("result")[0].as_py()
        [None, 3, 1]
    """
    desc = "DESC" if descending else "ASC"
    nulls_first = "NULLS FIRST" if null_first else "NULLS LAST"
    return Expr(
        f.array_sort(
            coerce_to_column(array),
            Expr.literal(pa.scalar(desc, type=pa.string())).expr,
            Expr.literal(pa.scalar(nulls_first, type=pa.string())).expr,
        )
    )


def list_sort(
    array: Expr | str, descending: bool = False, null_first: bool = False
) -> Expr:
    """Sorts the array.

    See Also:
        This is an alias for :py:func:`array_sort`.
    """
    return array_sort(array, descending=descending, null_first=null_first)


def array_slice(
    array: Expr | str,
    begin: Expr | int,
    end: Expr | int,
    stride: Expr | int | None = None,
) -> Expr:
    """Returns a slice of the array.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"a": [[1, 2, 3, 4]]})
        >>> result = df.select(dfn.functions.array_slice("a", 2, 3).alias("result"))
        >>> result.collect_column("result")[0].as_py()
        [2, 3]

        Use ``stride`` to skip elements:

        >>> result = df.select(
        ...     dfn.functions.array_slice(
        ...         "a", 1, 4, stride=2,
        ...     ).alias("result"))
        >>> result.collect_column("result")[0].as_py()
        [1, 3]
    """
    begin = coerce_to_literal(begin)
    end = coerce_to_literal(end)
    stride = coerce_to_literal_or_none(stride)
    return Expr(
        f.array_slice(
            coerce_to_column(array),
            begin,
            end,
            stride,
        )
    )


def list_slice(
    array: Expr | str,
    begin: Expr | int,
    end: Expr | int,
    stride: Expr | int | None = None,
) -> Expr:
    """Returns a slice of the array.

    See Also:
        This is an alias for :py:func:`array_slice`.
    """
    return array_slice(array, begin, end, stride)


def array_intersect(array1: Expr | str, array2: Expr | str) -> Expr:
    """Returns the intersection of ``array1`` and ``array2``.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"a": [[1, 2, 3]], "b": [[2, 3, 4]]})
        >>> result = df.select(dfn.functions.array_intersect("a", "b").alias("result"))
        >>> sorted(
        ...     result.collect_column("result")[0].as_py()
        ... )
        [2, 3]
    """
    return Expr(f.array_intersect(coerce_to_column(array1), coerce_to_column(array2)))


def list_intersect(array1: Expr | str, array2: Expr | str) -> Expr:
    """Returns an the intersection of ``array1`` and ``array2``.

    See Also:
        This is an alias for :py:func:`array_intersect`.
    """
    return array_intersect(array1, array2)


def array_union(array1: Expr | str, array2: Expr | str) -> Expr:
    """Returns an array of the elements in the union of array1 and array2.

    Duplicate rows will not be returned.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"a": [[1, 2, 3]], "b": [[2, 3, 4]]})
        >>> result = df.select(dfn.functions.array_union("a", "b").alias("result"))
        >>> sorted(
        ...     result.collect_column("result")[0].as_py()
        ... )
        [1, 2, 3, 4]
    """
    return Expr(f.array_union(coerce_to_column(array1), coerce_to_column(array2)))


def list_union(array1: Expr | str, array2: Expr | str) -> Expr:
    """Returns an array of the elements in the union of array1 and array2.

    Duplicate rows will not be returned.

    See Also:
        This is an alias for :py:func:`array_union`.
    """
    return array_union(array1, array2)


def array_except(array1: Expr | str, array2: Expr | str) -> Expr:
    """Returns the elements that appear in ``array1`` but not in ``array2``.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"a": [[1, 2, 3]], "b": [[2, 3, 4]]})
        >>> result = df.select(dfn.functions.array_except("a", "b").alias("result"))
        >>> result.collect_column("result")[0].as_py()
        [1]
    """
    return Expr(f.array_except(coerce_to_column(array1), coerce_to_column(array2)))


def list_except(array1: Expr | str, array2: Expr | str) -> Expr:
    """Returns the elements that appear in ``array1`` but not in the ``array2``.

    See Also:
        This is an alias for :py:func:`array_except`.
    """
    return array_except(array1, array2)


def array_resize(array: Expr | str, size: Expr | int, value: Expr) -> Expr:
    """Returns an array with the specified size filled.

    If ``size`` is greater than the ``array`` length, the additional entries will
    be filled with the given ``value``.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"a": [[1, 2]]})
        >>> result = df.select(
        ...     dfn.functions.array_resize("a", 4, dfn.lit(0)).alias("result"))
        >>> result.collect_column("result")[0].as_py()
        [1, 2, 0, 0]
    """
    size = coerce_to_literal(size)
    return Expr(f.array_resize(coerce_to_column(array), size, ensure_expr(value)))


def list_resize(array: Expr | str, size: Expr | int, value: Expr) -> Expr:
    """Returns an array with the specified size filled.

    If ``size`` is greater than the ``array`` length, the additional entries will be
    filled with the given ``value``.

    See Also:
        This is an alias for :py:func:`array_resize`.
    """
    return array_resize(array, size, value)


def array_any_value(array: Expr | str) -> Expr:
    """Returns the first non-null element in the array.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"a": [[None, 2, 3]]})
        >>> result = df.select(dfn.functions.array_any_value("a").alias("result"))
        >>> result.collect_column("result")[0].as_py()
        2
    """
    return Expr(f.array_any_value(coerce_to_column(array)))


def list_any_value(array: Expr | str) -> Expr:
    """Returns the first non-null element in the array.

    See Also:
        This is an alias for :py:func:`array_any_value`.
    """
    return array_any_value(array)


def array_distance(array1: Expr | str, array2: Expr | str) -> Expr:
    """Returns the Euclidean distance between two numeric arrays.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"a": [[1.0, 2.0]], "b": [[1.0, 4.0]]})
        >>> result = df.select(dfn.functions.array_distance("a", "b").alias("result"))
        >>> result.collect_column("result")[0].as_py()
        2.0
    """
    return Expr(f.array_distance(coerce_to_column(array1), coerce_to_column(array2)))


def list_distance(array1: Expr | str, array2: Expr | str) -> Expr:
    """Returns the Euclidean distance between two numeric arrays.

    See Also:
        This is an alias for :py:func:`array_distance`.
    """
    return array_distance(array1, array2)


def array_max(array: Expr | str) -> Expr:
    """Returns the maximum value in the array.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"a": [[1, 2, 3]]})
        >>> result = df.select(dfn.functions.array_max("a").alias("result"))
        >>> result.collect_column("result")[0].as_py()
        3
    """
    return Expr(f.array_max(coerce_to_column(array)))


def list_max(array: Expr | str) -> Expr:
    """Returns the maximum value in the array.

    See Also:
        This is an alias for :py:func:`array_max`.
    """
    return array_max(array)


def array_min(array: Expr | str) -> Expr:
    """Returns the minimum value in the array.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"a": [[1, 2, 3]]})
        >>> result = df.select(dfn.functions.array_min("a").alias("result"))
        >>> result.collect_column("result")[0].as_py()
        1
    """
    return Expr(f.array_min(coerce_to_column(array)))


def list_min(array: Expr | str) -> Expr:
    """Returns the minimum value in the array.

    See Also:
        This is an alias for :py:func:`array_min`.
    """
    return array_min(array)


def array_reverse(array: Expr | str) -> Expr:
    """Reverses the order of elements in the array.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"a": [[1, 2, 3]]})
        >>> result = df.select(dfn.functions.array_reverse("a").alias("result"))
        >>> result.collect_column("result")[0].as_py()
        [3, 2, 1]
    """
    return Expr(f.array_reverse(coerce_to_column(array)))


def list_reverse(array: Expr | str) -> Expr:
    """Reverses the order of elements in the array.

    See Also:
        This is an alias for :py:func:`array_reverse`.
    """
    return array_reverse(array)


def arrays_zip(*arrays: Expr | str) -> Expr:
    """Combines multiple arrays into a single array of structs.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"a": [[1, 2]], "b": [[3, 4]]})
        >>> result = df.select(dfn.functions.arrays_zip("a", "b").alias("result"))
        >>> result.collect_column("result")[0].as_py()
        [{'1': 1, '2': 3}, {'1': 2, '2': 4}]
    """
    args = coerce_to_column_list(arrays)
    return Expr(f.arrays_zip(args))


def list_zip(*arrays: Expr | str) -> Expr:
    """Combines multiple arrays into a single array of structs.

    See Also:
        This is an alias for :py:func:`arrays_zip`.
    """
    return arrays_zip(*arrays)


def string_to_array(
    string: Expr | str, delimiter: Expr | str, null_string: Expr | str | None = None
) -> Expr:
    """Splits a string based on a delimiter and returns an array of parts.

    Any parts matching the optional ``null_string`` will be replaced with ``NULL``.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"a": ["hello,world"]})
        >>> result = df.select(dfn.functions.string_to_array("a", ",").alias("result"))
        >>> result.collect_column("result")[0].as_py()
        ['hello', 'world']

        Replace parts matching a ``null_string`` with ``NULL``:

        >>> result = df.select(
        ...     dfn.functions.string_to_array(
        ...         "a", ",", null_string="world",
        ...     ).alias("result"))
        >>> result.collect_column("result")[0].as_py()
        ['hello', None]
    """
    delimiter = coerce_to_literal(delimiter)
    null_string = coerce_to_literal_or_none(null_string)
    return Expr(
        f.string_to_array(
            coerce_to_column(string),
            delimiter,
            null_string,
        )
    )


def string_to_list(
    string: Expr | str, delimiter: Expr | str, null_string: Expr | str | None = None
) -> Expr:
    """Splits a string based on a delimiter and returns an array of parts.

    See Also:
        This is an alias for :py:func:`string_to_array`.
    """
    return string_to_array(string, delimiter, null_string)


def gen_series(start: Expr, stop: Expr, step: Expr | None = None) -> Expr:
    """Creates a list of values in the range between start and stop.

    Unlike :py:func:`range`, this includes the upper bound.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"a": [0]})
        >>> result = df.select(
        ...     dfn.functions.gen_series(
        ...         dfn.lit(1), dfn.lit(5),
        ...     ).alias("result"))
        >>> result.collect_column("result")[0].as_py()
        [1, 2, 3, 4, 5]

        Specify a custom ``step``:

        >>> result = df.select(
        ...     dfn.functions.gen_series(
        ...         dfn.lit(1), dfn.lit(10), step=dfn.lit(3),
        ...     ).alias("result"))
        >>> result.collect_column("result")[0].as_py()
        [1, 4, 7, 10]
    """
    step_expr = ensure_expr_or_none(step)
    return Expr(f.gen_series(ensure_expr(start), ensure_expr(stop), step_expr))


def generate_series(start: Expr, stop: Expr, step: Expr | None = None) -> Expr:
    """Creates a list of values in the range between start and stop.

    Unlike :py:func:`range`, this includes the upper bound.

    See Also:
        This is an alias for :py:func:`gen_series`.
    """
    return gen_series(start, stop, step)


def flatten(array: Expr | str) -> Expr:
    """Flattens an array of arrays into a single array.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"a": [[[1, 2], [3, 4]]]})
        >>> result = df.select(dfn.functions.flatten("a").alias("result"))
        >>> result.collect_column("result")[0].as_py()
        [1, 2, 3, 4]
    """
    return Expr(f.flatten(coerce_to_column(array)))


def cardinality(array: Expr | str) -> Expr:
    """Returns the total number of elements in the array.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"a": [[1, 2, 3]]})
        >>> result = df.select(dfn.functions.cardinality("a").alias("result"))
        >>> result.collect_column("result")[0].as_py()
        3
    """
    return Expr(f.cardinality(coerce_to_column(array)))


def empty(array: Expr | str) -> Expr:
    """Returns true if the array is empty.

    See Also:
        This is an alias for :py:func:`array_empty`.
    """
    return array_empty(array)


# map functions


def make_map(*args: Any) -> Expr:
    """Returns a map expression.

    Supports three calling conventions:

    - ``make_map({"a": 1, "b": 2})`` — from a Python dictionary.
    - ``make_map([keys], [values])`` — from a list of keys and a list of
      their associated values.  Both lists must be the same length.
    - ``make_map(k1, v1, k2, v2, ...)`` — from alternating keys and their
      associated values.

    Keys and values that are not already :py:class:`~datafusion.expr.Expr`
    are automatically converted to literal expressions.

    Examples:
        From a dictionary:

        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"a": [1]})
        >>> result = df.select(
        ...     dfn.functions.make_map({"a": 1, "b": 2}).alias("m"))
        >>> result.collect_column("m")[0].as_py()
        [('a', 1), ('b', 2)]

        From two lists:

        >>> df = ctx.from_pydict({"key": ["x", "y"], "val": [10, 20]})
        >>> df = df.select(
        ...     dfn.functions.make_map(
        ...         [dfn.col("key")], [dfn.col("val")]
        ...     ).alias("m"))
        >>> df.collect_column("m")[0].as_py()
        [('x', 10)]

        From alternating keys and values:

        >>> df = ctx.from_pydict({"a": [1]})
        >>> result = df.select(
        ...     dfn.functions.make_map("x", 1, "y", 2).alias("m"))
        >>> result.collect_column("m")[0].as_py()
        [('x', 1), ('y', 2)]
    """
    if len(args) == 1 and isinstance(args[0], dict):
        key_list = list(args[0].keys())
        value_list = list(args[0].values())
    elif (
        len(args) == 2  # noqa: PLR2004
        and isinstance(args[0], list)
        and isinstance(args[1], list)
    ):
        if len(args[0]) != len(args[1]):
            msg = "make_map requires key and value lists to be the same length"
            raise ValueError(msg)
        key_list = args[0]
        value_list = args[1]
    elif len(args) >= 2 and len(args) % 2 == 0:  # noqa: PLR2004
        key_list = list(args[0::2])
        value_list = list(args[1::2])
    else:
        msg = (
            "make_map expects a dict, two lists, or an even number of "
            "key-value arguments"
        )
        raise ValueError(msg)

    key_exprs = [k if isinstance(k, Expr) else Expr.literal(k) for k in key_list]
    val_exprs = [v if isinstance(v, Expr) else Expr.literal(v) for v in value_list]
    return Expr(f.make_map([k.expr for k in key_exprs], [v.expr for v in val_exprs]))


def map_keys(map: Expr | str) -> Expr:
    """Returns a list of all keys in the map.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"a": [1]})
        >>> df = df.select(
        ...     dfn.functions.make_map({"x": 1, "y": 2}).alias("m"))
        >>> result = df.select(dfn.functions.map_keys("m").alias("keys"))
        >>> result.collect_column("keys")[0].as_py()
        ['x', 'y']
    """
    return Expr(f.map_keys(coerce_to_column(map)))


def map_values(map: Expr | str) -> Expr:
    """Returns a list of all values in the map.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"a": [1]})
        >>> df = df.select(
        ...     dfn.functions.make_map({"x": 1, "y": 2}).alias("m"))
        >>> result = df.select(dfn.functions.map_values("m").alias("vals"))
        >>> result.collect_column("vals")[0].as_py()
        [1, 2]
    """
    return Expr(f.map_values(coerce_to_column(map)))


def map_extract(map: Expr | str, key: Expr) -> Expr:
    """Returns the value for a given key in the map.

    Returns ``[None]`` if the key is absent.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"a": [1]})
        >>> df = df.select(
        ...     dfn.functions.make_map({"x": 1, "y": 2}).alias("m"))
        >>> result = df.select(
        ...     dfn.functions.map_extract(
        ...         "m", dfn.lit("x")
        ...     ).alias("val"))
        >>> result.collect_column("val")[0].as_py()
        [1]
    """
    return Expr(f.map_extract(coerce_to_column(map), ensure_expr(key)))


def map_entries(map: Expr | str) -> Expr:
    """Returns a list of all entries (key-value struct pairs) in the map.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"a": [1]})
        >>> df = df.select(
        ...     dfn.functions.make_map({"x": 1, "y": 2}).alias("m"))
        >>> result = df.select(dfn.functions.map_entries("m").alias("entries"))
        >>> result.collect_column("entries")[0].as_py()
        [{'key': 'x', 'value': 1}, {'key': 'y', 'value': 2}]
    """
    return Expr(f.map_entries(coerce_to_column(map)))


def element_at(map: Expr | str, key: Expr) -> Expr:
    """Returns the value for a given key in the map.

    Returns ``[None]`` if the key is absent.

    See Also:
        This is an alias for :py:func:`map_extract`.
    """
    return map_extract(map, key)


# aggregate functions
def approx_distinct(
    expression: Expr | str,
    filter: Expr | str | None = None,
) -> Expr:
    """Returns the approximate number of distinct values.

    This aggregate function is similar to :py:func:`count` with distinct set, but it
    will approximate the number of distinct entries. It may return significantly faster
    than :py:func:`count` for some DataFrames.

    If using the builder functions described in ref:`_aggregation` this function ignores
    the options ``order_by``, ``null_treatment``, and ``distinct``.

    Args:
        expression: Values to check for distinct entries
        filter: If provided, only compute against rows for which the filter is True

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"a": [1, 1, 2, 3]})
        >>> result = df.aggregate([], [dfn.functions.approx_distinct("a").alias("v")])
        >>> result.collect_column("v")[0].as_py() == 3
        True

        >>> result = df.aggregate(
        ...     [], [dfn.functions.approx_distinct(
        ...         "a",
        ...         filter=dfn.col("a") > dfn.lit(1)
        ...     ).alias("v")])
        >>> result.collect_column("v")[0].as_py() == 2
        True
    """
    filter_raw = coerce_to_column_or_none(filter)

    return Expr(f.approx_distinct(coerce_to_column(expression), filter=filter_raw))


def approx_median(expression: Expr | str, filter: Expr | str | None = None) -> Expr:
    """Returns the approximate median value.

    This aggregate function is similar to :py:func:`median`, but it will only
    approximate the median. It may return significantly faster for some DataFrames.

    If using the builder functions described in ref:`_aggregation` this function ignores
    the options ``order_by`` and ``null_treatment``, and ``distinct``.

    Args:
        expression: Values to find the median for
        filter: If provided, only compute against rows for which the filter is True

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"a": [1.0, 2.0, 3.0]})
        >>> result = df.aggregate([], [dfn.functions.approx_median("a").alias("v")])
        >>> result.collect_column("v")[0].as_py()
        2.0

        >>> result = df.aggregate(
        ...     [], [dfn.functions.approx_median(
        ...         "a",
        ...         filter=dfn.col("a") > dfn.lit(1.0)
        ...     ).alias("v")])
        >>> result.collect_column("v")[0].as_py()
        2.5
    """
    filter_raw = coerce_to_column_or_none(filter)
    return Expr(f.approx_median(coerce_to_column(expression), filter=filter_raw))


def approx_percentile_cont(
    sort_expression: Expr | str | SortExpr,
    percentile: float,
    num_centroids: int | None = None,
    filter: Expr | str | None = None,
) -> Expr:
    """Returns the value that is approximately at a given percentile of ``expr``.

    This aggregate function assumes the input values form a continuous distribution.
    Suppose you have a DataFrame which consists of 100 different test scores. If you
    called this function with a percentile of 0.9, it would return the value of the
    test score that is above 90% of the other test scores. The returned value may be
    between two of the values.

    This function uses the [t-digest](https://arxiv.org/abs/1902.04023) algorithm to
    compute the percentile. You can limit the number of bins used in this algorithm by
    setting the ``num_centroids`` parameter.

    If using the builder functions described in ref:`_aggregation` this function ignores
    the options ``order_by``, ``null_treatment``, and ``distinct``.

    Args:
        sort_expression: Values for which to find the approximate percentile
        percentile: This must be between 0.0 and 1.0, inclusive
        num_centroids: Max bin size for the t-digest algorithm
        filter: If provided, only compute against rows for which the filter is True

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"a": [1.0, 2.0, 3.0, 4.0, 5.0]})
        >>> result = df.aggregate(
        ...     [], [dfn.functions.approx_percentile_cont(
        ...         "a", 0.5
        ...     ).alias("v")])
        >>> result.collect_column("v")[0].as_py()
        3.0

        >>> result = df.aggregate(
        ...     [], [dfn.functions.approx_percentile_cont(
        ...         "a", 0.5,
        ...         num_centroids=10,
        ...         filter=dfn.col("a") > dfn.lit(1.0),
        ...     ).alias("v")])
        >>> result.collect_column("v")[0].as_py()
        3.5
    """
    sort_expr_raw = sort_or_default(sort_expression)
    filter_raw = coerce_to_column_or_none(filter)
    return Expr(
        f.approx_percentile_cont(
            sort_expr_raw, percentile, num_centroids=num_centroids, filter=filter_raw
        )
    )


def approx_percentile_cont_with_weight(
    sort_expression: Expr | str | SortExpr,
    weight: Expr | str,
    percentile: float,
    num_centroids: int | None = None,
    filter: Expr | str | None = None,
) -> Expr:
    """Returns the value of the weighted approximate percentile.

    This aggregate function is similar to :py:func:`approx_percentile_cont` except that
    it uses the associated associated weights.

    If using the builder functions described in ref:`_aggregation` this function ignores
    the options ``order_by``, ``null_treatment``, and ``distinct``.

    Args:
        sort_expression: Values for which to find the approximate percentile
        weight: Relative weight for each of the values in ``expression``
        percentile: This must be between 0.0 and 1.0, inclusive
        num_centroids: Max bin size for the t-digest algorithm
        filter: If provided, only compute against rows for which the filter is True

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"a": [1.0, 2.0, 3.0], "w": [1.0, 1.0, 1.0]})
        >>> result = df.aggregate(
        ...     [], [dfn.functions.approx_percentile_cont_with_weight(
        ...         "a", "w", 0.5
        ...     ).alias("v")])
        >>> result.collect_column("v")[0].as_py()
        2.0

        >>> result = df.aggregate(
        ...     [], [dfn.functions.approx_percentile_cont_with_weight(
        ...         "a", "w", 0.5,
        ...         num_centroids=10,
        ...         filter=dfn.col("a") > dfn.lit(1.0),
        ...     ).alias("v")])
        >>> result.collect_column("v")[0].as_py()
        2.5
    """
    sort_expr_raw = sort_or_default(sort_expression)
    filter_raw = coerce_to_column_or_none(filter)
    return Expr(
        f.approx_percentile_cont_with_weight(
            sort_expr_raw,
            coerce_to_column(weight),
            percentile,
            num_centroids=num_centroids,
            filter=filter_raw,
        )
    )


def percentile_cont(
    sort_expression: Expr | str | SortExpr,
    percentile: float,
    filter: Expr | str | None = None,
) -> Expr:
    """Computes the exact percentile of input values using continuous interpolation.

    Unlike :py:func:`approx_percentile_cont`, this function computes the exact
    percentile value rather than an approximation.

    If using the builder functions described in ref:`_aggregation` this function ignores
    the options ``order_by``, ``null_treatment``, and ``distinct``.

    Args:
        sort_expression: Values for which to find the percentile
        percentile: This must be between 0.0 and 1.0, inclusive
        filter: If provided, only compute against rows for which the filter is True

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"a": [1.0, 2.0, 3.0, 4.0, 5.0]})
        >>> result = df.aggregate(
        ...     [], [dfn.functions.percentile_cont(
        ...         "a", 0.5
        ...     ).alias("v")])
        >>> result.collect_column("v")[0].as_py()
        3.0

        >>> result = df.aggregate(
        ...     [], [dfn.functions.percentile_cont(
        ...         "a", 0.5,
        ...         filter=dfn.col("a") > dfn.lit(1.0),
        ...     ).alias("v")])
        >>> result.collect_column("v")[0].as_py()
        3.5
    """
    sort_expr_raw = sort_or_default(sort_expression)
    filter_raw = coerce_to_column_or_none(filter)
    return Expr(f.percentile_cont(sort_expr_raw, percentile, filter=filter_raw))


def quantile_cont(
    sort_expression: Expr | str | SortExpr,
    percentile: float,
    filter: Expr | str | None = None,
) -> Expr:
    """Computes the exact percentile of input values using continuous interpolation.

    See Also:
        This is an alias for :py:func:`percentile_cont`.
    """
    return percentile_cont(sort_expression, percentile, filter)


def array_agg(
    expression: Expr | str,
    distinct: bool = False,
    filter: Expr | str | None = None,
    order_by: list[SortKey] | SortKey | None = None,
) -> Expr:
    """Aggregate values into an array.

    Currently ``distinct`` and ``order_by`` cannot be used together. As a work around,
    consider :py:func:`array_sort` after aggregation.
    [Issue Tracker](https://github.com/apache/datafusion/issues/12371)

    If using the builder functions described in ref:`_aggregation` this function ignores
    the option ``null_treatment``.

    Args:
        expression: Values to combine into an array
        distinct: If True, a single entry for each distinct value will be in the result
        filter: If provided, only compute against rows for which the filter is True
        order_by: Order the resultant array values. Accepts column names or expressions.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"a": [1, 2, 3]})
        >>> result = df.aggregate([], [dfn.functions.array_agg("a").alias("v")])
        >>> result.collect_column("v")[0].as_py()
        [1, 2, 3]

        >>> df = ctx.from_pydict({"a": [3, 1, 2, 1]})
        >>> result = df.aggregate(
        ...     [], [dfn.functions.array_agg(
        ...         "a", distinct=True,
        ...     ).alias("v")])
        >>> sorted(result.collect_column("v")[0].as_py())
        [1, 2, 3]

        >>> result = df.aggregate(
        ...     [], [dfn.functions.array_agg(
        ...         "a",
        ...         filter=dfn.col("a") > dfn.lit(1),
        ...         order_by="a",
        ...     ).alias("v")])
        >>> result.collect_column("v")[0].as_py()
        [2, 3]
    """
    order_by_raw = sort_list_to_raw_sort_list(order_by)
    filter_raw = coerce_to_column_or_none(filter)

    return Expr(
        f.array_agg(
            coerce_to_column(expression),
            distinct=distinct,
            filter=filter_raw,
            order_by=order_by_raw,
        )
    )


def grouping(
    expression: Expr | str,
    distinct: bool = False,
    filter: Expr | str | None = None,
) -> Expr:
    """Indicates whether a column is aggregated across in the current row.

    Returns 0 when the column is part of the grouping key for that row
    (i.e., the row contains per-group results for that column). Returns 1
    when the column is *not* part of the grouping key (i.e., the row's
    aggregate spans all values of that column).

    This function is meaningful with
    :py:meth:`GroupingSet.rollup <datafusion.expr.GroupingSet.rollup>`,
    :py:meth:`GroupingSet.cube <datafusion.expr.GroupingSet.cube>`, or
    :py:meth:`GroupingSet.grouping_sets <datafusion.expr.GroupingSet.grouping_sets>`,
    where different rows are grouped by different subsets of columns. In a
    default aggregation without grouping sets every column is always part
    of the key, so ``grouping()`` always returns 0.

    .. warning::

        Due to an upstream DataFusion limitation
        (`#21411 <https://github.com/apache/datafusion/issues/21411>`_),
        ``.alias()`` cannot be applied directly to a ``grouping()``
        expression. Doing so will raise an error at execution time. To
        rename the column, use
        :py:meth:`~datafusion.dataframe.DataFrame.with_column_renamed`
        on the result DataFrame instead.

    Args:
        expression: The column to check grouping status for
        distinct: If True, compute on distinct values only
        filter: If provided, only compute against rows for which the filter is True

    Examples:
        With :py:meth:`~datafusion.expr.GroupingSet.rollup`, the result
        includes both per-group rows (``grouping(a) = 0``) and a
        grand-total row where ``a`` is aggregated across
        (``grouping(a) = 1``):

        >>> from datafusion.expr import GroupingSet
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"a": [1, 1, 2], "b": [10, 20, 30]})
        >>> result = df.aggregate(
        ...     [GroupingSet.rollup(dfn.col("a"))],
        ...     [dfn.functions.sum("b").alias("s"),
        ...      dfn.functions.grouping("a")],
        ... ).sort(dfn.col("a").sort(nulls_first=False))
        >>> result.collect_column("s").to_pylist()
        [30, 30, 60]

    See Also:
        :py:class:`~datafusion.expr.GroupingSet`
    """
    filter_raw = coerce_to_column_or_none(filter)
    return Expr(
        f.grouping(coerce_to_column(expression), distinct=distinct, filter=filter_raw)
    )


def avg(
    expression: Expr | str,
    distinct: bool = False,
    filter: Expr | str | None = None,
) -> Expr:
    """Returns the average value.

    This aggregate function expects a numeric expression and will return a float.

    If using the builder functions described in ref:`_aggregation` this function ignores
    the options ``order_by`` and ``null_treatment``.

    Args:
        expression: Values to combine into an array
        distinct: If True, duplicate values are removed before averaging.
        filter: If provided, only compute against rows for which the filter is True

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"a": [1.0, 2.0, 3.0]})
        >>> result = df.aggregate([], [dfn.functions.avg("a").alias("v")])
        >>> result.collect_column("v")[0].as_py()
        2.0

        >>> result = df.aggregate(
        ...     [], [dfn.functions.avg(
        ...         "a",
        ...         filter=dfn.col("a") > dfn.lit(1.0)
        ...     ).alias("v")])
        >>> result.collect_column("v")[0].as_py()
        2.5

        >>> df = ctx.from_pydict({"a": [1.0, 1.0, 2.0, 3.0]})
        >>> result = df.aggregate(
        ...     [], [dfn.functions.avg(
        ...         "a", distinct=True,
        ...     ).alias("v")])
        >>> result.collect_column("v")[0].as_py()
        2.0
    """
    filter_raw = coerce_to_column_or_none(filter)
    return Expr(
        f.avg(coerce_to_column(expression), distinct=distinct, filter=filter_raw)
    )


def corr(
    value_y: Expr | str, value_x: Expr | str, filter: Expr | str | None = None
) -> Expr:
    """Returns the correlation coefficient between ``value1`` and ``value2``.

    This aggregate function expects both values to be numeric and will return a float.

    If using the builder functions described in ref:`_aggregation` this function ignores
    the options ``order_by``, ``null_treatment``, and ``distinct``.

    Args:
        value_y: The dependent variable for correlation
        value_x: The independent variable for correlation
        filter: If provided, only compute against rows for which the filter is True

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"a": [1.0, 2.0, 3.0], "b": [1.0, 2.0, 3.0]})
        >>> result = df.aggregate([], [dfn.functions.corr("a", "b").alias("v")])
        >>> result.collect_column("v")[0].as_py()
        1.0

        >>> result = df.aggregate(
        ...     [], [dfn.functions.corr(
        ...         "a", "b",
        ...         filter=dfn.col("a") > dfn.lit(1.0)
        ...     ).alias("v")])
        >>> result.collect_column("v")[0].as_py()
        1.0
    """
    filter_raw = coerce_to_column_or_none(filter)
    return Expr(
        f.corr(coerce_to_column(value_y), coerce_to_column(value_x), filter=filter_raw)
    )


def count(
    expressions: Expr | str | list[Expr | str] | None = None,
    distinct: bool = False,
    filter: Expr | str | None = None,
) -> Expr:
    """Returns the number of rows that match the given arguments.

    This aggregate function will count the non-null rows provided in the expression.

    If using the builder functions described in ref:`_aggregation` this function ignores
    the options ``order_by`` and ``null_treatment``.

    Args:
        expressions: Argument to perform bitwise calculation on
        distinct: If True, a single entry for each distinct value will be in the result
        filter: If provided, only compute against rows for which the filter is True

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"a": [1, 2, 3]})
        >>> result = df.aggregate([], [dfn.functions.count("a").alias("v")])
        >>> result.collect_column("v")[0].as_py()
        3

        >>> df = ctx.from_pydict({"a": [1, 1, 2, 3]})
        >>> result = df.aggregate(
        ...     [], [dfn.functions.count(
        ...         "a", distinct=True,
        ...         filter=dfn.col("a") > dfn.lit(1),
        ...     ).alias("v")])
        >>> result.collect_column("v")[0].as_py()
        2
    """
    filter_raw = coerce_to_column_or_none(filter)

    if expressions is None:
        args = [Expr.literal(1).expr]
    elif isinstance(expressions, list):
        args = coerce_to_column_list(expressions)
    else:
        args = [coerce_to_column(expressions)]

    return Expr(f.count(*args, distinct=distinct, filter=filter_raw))


def covar_pop(
    value_y: Expr | str, value_x: Expr | str, filter: Expr | str | None = None
) -> Expr:
    """Computes the population covariance.

    This aggregate function expects both values to be numeric and will return a float.

    If using the builder functions described in ref:`_aggregation` this function ignores
    the options ``order_by``, ``null_treatment``, and ``distinct``.

    Args:
        value_y: The dependent variable for covariance
        value_x: The independent variable for covariance
        filter: If provided, only compute against rows for which the filter is True

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"a": [1.0, 5.0, 10.0], "b": [1.0, 2.0, 3.0]})
        >>> result = df.aggregate([], [dfn.functions.covar_pop("a", "b").alias("v")])
        >>> result.collect_column("v")[0].as_py()
        3.0

        >>> df = ctx.from_pydict(
        ...     {"a": [0.0, 1.0, 3.0], "b": [0.0, 1.0, 3.0]})
        >>> result = df.aggregate(
        ...     [],
        ...     [dfn.functions.covar_pop(
        ...         "a", "b",
        ...         filter=dfn.col("a") > dfn.lit(0.0)
        ...     ).alias("v")]
        ... )
        >>> result.collect_column("v")[0].as_py()
        1.0
    """
    filter_raw = coerce_to_column_or_none(filter)
    return Expr(
        f.covar_pop(
            coerce_to_column(value_y), coerce_to_column(value_x), filter=filter_raw
        )
    )


def covar_samp(
    value_y: Expr | str, value_x: Expr | str, filter: Expr | str | None = None
) -> Expr:
    """Computes the sample covariance.

    This aggregate function expects both values to be numeric and will return a float.

    If using the builder functions described in ref:`_aggregation` this function ignores
    the options ``order_by``, ``null_treatment``, and ``distinct``.

    Args:
        value_y: The dependent variable for covariance
        value_x: The independent variable for covariance
        filter: If provided, only compute against rows for which the filter is True

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
        >>> result = df.aggregate([], [dfn.functions.covar_samp("a", "b").alias("v")])
        >>> result.collect_column("v")[0].as_py()
        1.0

        >>> result = df.aggregate(
        ...     [], [dfn.functions.covar_samp(
        ...         "a", "b",
        ...         filter=dfn.col("a") > dfn.lit(1.0)
        ...     ).alias("v")])
        >>> result.collect_column("v")[0].as_py()
        0.5
    """
    filter_raw = coerce_to_column_or_none(filter)
    return Expr(
        f.covar_samp(
            coerce_to_column(value_y), coerce_to_column(value_x), filter=filter_raw
        )
    )


def covar(
    value_y: Expr | str, value_x: Expr | str, filter: Expr | str | None = None
) -> Expr:
    """Computes the sample covariance.

    See Also:
        This is an alias for :py:func:`covar_samp`.
    """
    return covar_samp(value_y, value_x, filter)


def max(expression: Expr | str, filter: Expr | str | None = None) -> Expr:
    """Aggregate function that returns the maximum value of the argument.

    If using the builder functions described in ref:`_aggregation` this function ignores
    the options ``order_by``, ``null_treatment``, and ``distinct``.

    Args:
        expression: The value to find the maximum of
        filter: If provided, only compute against rows for which the filter is True

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"a": [1, 2, 3]})
        >>> result = df.aggregate([], [dfn.functions.max("a").alias("v")])
        >>> result.collect_column("v")[0].as_py()
        3

        >>> result = df.aggregate(
        ...     [], [dfn.functions.max(
        ...         "a",
        ...         filter=dfn.col("a") < dfn.lit(3)
        ...     ).alias("v")])
        >>> result.collect_column("v")[0].as_py()
        2
    """
    filter_raw = coerce_to_column_or_none(filter)
    return Expr(f.max(coerce_to_column(expression), filter=filter_raw))


def mean(expression: Expr | str, filter: Expr | str | None = None) -> Expr:
    """Returns the average (mean) value of the argument.

    See Also:
        This is an alias for :py:func:`avg`.
    """
    return avg(expression, filter=filter)


def median(
    expression: Expr | str, distinct: bool = False, filter: Expr | str | None = None
) -> Expr:
    """Computes the median of a set of numbers.

    This aggregate function returns the median value of the expression for the given
    aggregate function.

    If using the builder functions described in ref:`_aggregation` this function ignores
    the options ``order_by`` and ``null_treatment``.

    Args:
        expression: The value to compute the median of
        distinct: If True, a single entry for each distinct value will be in the result
        filter: If provided, only compute against rows for which the filter is True

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"a": [1.0, 2.0, 3.0]})
        >>> result = df.aggregate([], [dfn.functions.median("a").alias("v")])
        >>> result.collect_column("v")[0].as_py()
        2.0

        >>> df = ctx.from_pydict({"a": [1.0, 1.0, 2.0, 3.0]})
        >>> result = df.aggregate(
        ...     [], [dfn.functions.median(
        ...         "a", distinct=True,
        ...         filter=dfn.col("a") < dfn.lit(3.0),
        ...     ).alias("v")])
        >>> result.collect_column("v")[0].as_py()
        1.5
    """
    filter_raw = coerce_to_column_or_none(filter)
    return Expr(
        f.median(coerce_to_column(expression), distinct=distinct, filter=filter_raw)
    )


def min(expression: Expr | str, filter: Expr | str | None = None) -> Expr:
    """Aggregate function that returns the minimum value of the argument.

    If using the builder functions described in ref:`_aggregation` this function ignores
    the options ``order_by``, ``null_treatment``, and ``distinct``.

    Args:
        expression: The value to find the minimum of
        filter: If provided, only compute against rows for which the filter is True

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"a": [1, 2, 3]})
        >>> result = df.aggregate([], [dfn.functions.min("a").alias("v")])
        >>> result.collect_column("v")[0].as_py()
        1

        >>> result = df.aggregate(
        ...     [], [dfn.functions.min(
        ...         "a",
        ...         filter=dfn.col("a") > dfn.lit(1)
        ...     ).alias("v")])
        >>> result.collect_column("v")[0].as_py()
        2
    """
    filter_raw = coerce_to_column_or_none(filter)
    return Expr(f.min(coerce_to_column(expression), filter=filter_raw))


def sum(
    expression: Expr | str,
    distinct: bool = False,
    filter: Expr | str | None = None,
) -> Expr:
    """Computes the sum of a set of numbers.

    This aggregate function expects a numeric expression.

    If using the builder functions described in ref:`_aggregation` this function ignores
    the options ``order_by`` and ``null_treatment``.

    Args:
        expression: Values to combine into an array
        distinct: If True, duplicate values are removed before summing.
        filter: If provided, only compute against rows for which the filter is True

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"a": [1, 2, 3]})
        >>> result = df.aggregate([], [dfn.functions.sum("a").alias("v")])
        >>> result.collect_column("v")[0].as_py()
        6

        >>> result = df.aggregate(
        ...     [], [dfn.functions.sum(
        ...         "a",
        ...         filter=dfn.col("a") > dfn.lit(1)
        ...     ).alias("v")])
        >>> result.collect_column("v")[0].as_py()
        5

        ``filter`` also takes the name of a boolean column:

        >>> flagged = ctx.from_pydict({"a": [1, 2, 3], "keep": [True, False, True]})
        >>> result = flagged.aggregate(
        ...     [], [dfn.functions.sum("a", filter="keep").alias("v")])
        >>> result.collect_column("v")[0].as_py()
        4

        >>> df = ctx.from_pydict({"a": [1, 1, 2, 3]})
        >>> result = df.aggregate(
        ...     [], [dfn.functions.sum(
        ...         "a", distinct=True,
        ...     ).alias("v")])
        >>> result.collect_column("v")[0].as_py()
        6
    """
    filter_raw = coerce_to_column_or_none(filter)
    return Expr(
        f.sum(coerce_to_column(expression), distinct=distinct, filter=filter_raw)
    )


def stddev(expression: Expr | str, filter: Expr | str | None = None) -> Expr:
    """Computes the standard deviation of the argument.

    If using the builder functions described in ref:`_aggregation` this function ignores
    the options ``order_by``, ``null_treatment``, and ``distinct``.

    Args:
        expression: The value to find the minimum of
        filter: If provided, only compute against rows for which the filter is True

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"a": [2.0, 4.0, 6.0]})
        >>> result = df.aggregate([], [dfn.functions.stddev("a").alias("v")])
        >>> result.collect_column("v")[0].as_py()
        2.0

        >>> result = df.aggregate(
        ...     [], [dfn.functions.stddev(
        ...         "a",
        ...         filter=dfn.col("a") > dfn.lit(2.0)
        ...     ).alias("v")])
        >>> result.collect_column("v")[0].as_py()
        1.41...
    """
    filter_raw = coerce_to_column_or_none(filter)
    return Expr(f.stddev(coerce_to_column(expression), filter=filter_raw))


def stddev_pop(expression: Expr | str, filter: Expr | str | None = None) -> Expr:
    """Computes the population standard deviation of the argument.

    If using the builder functions described in ref:`_aggregation` this function ignores
    the options ``order_by``, ``null_treatment``, and ``distinct``.

    Args:
        expression: The value to find the minimum of
        filter: If provided, only compute against rows for which the filter is True

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"a": [0.0, 1.0, 3.0]})
        >>> result = df.aggregate([], [dfn.functions.stddev_pop("a").alias("v")])
        >>> result.collect_column("v")[0].as_py()
        1.247...

        >>> df = ctx.from_pydict({"a": [0.0, 1.0, 3.0]})
        >>> result = df.aggregate(
        ...     [], [dfn.functions.stddev_pop(
        ...         "a",
        ...         filter=dfn.col("a") > dfn.lit(0.0)
        ...     ).alias("v")]
        ... )
        >>> result.collect_column("v")[0].as_py()
        1.0
    """
    filter_raw = coerce_to_column_or_none(filter)
    return Expr(f.stddev_pop(coerce_to_column(expression), filter=filter_raw))


def stddev_samp(arg: Expr | str, filter: Expr | str | None = None) -> Expr:
    """Computes the sample standard deviation of the argument.

    See Also:
        This is an alias for :py:func:`stddev`.
    """
    return stddev(arg, filter=filter)


def var(expression: Expr | str, filter: Expr | str | None = None) -> Expr:
    """Computes the sample variance of the argument.

    See Also:
        This is an alias for :py:func:`var_samp`.
    """
    return var_samp(expression, filter)


def var_pop(expression: Expr | str, filter: Expr | str | None = None) -> Expr:
    """Computes the population variance of the argument.

    If using the builder functions described in ref:`_aggregation` this function ignores
    the options ``order_by``, ``null_treatment``, and ``distinct``.

    Args:
        expression: The variable to compute the variance for
        filter: If provided, only compute against rows for which the filter is True

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"a": [-1.0, 0.0, 2.0]})
        >>> result = df.aggregate([], [dfn.functions.var_pop("a").alias("v")])
        >>> result.collect_column("v")[0].as_py()
        1.555...

        >>> result = df.aggregate(
        ...     [], [dfn.functions.var_pop(
        ...         "a",
        ...         filter=dfn.col("a") > dfn.lit(-1.0)
        ...     ).alias("v")])
        >>> result.collect_column("v")[0].as_py()
        1.0
    """
    filter_raw = coerce_to_column_or_none(filter)
    return Expr(f.var_pop(coerce_to_column(expression), filter=filter_raw))


def var_population(expression: Expr | str, filter: Expr | str | None = None) -> Expr:
    """Computes the population variance of the argument.

    See Also:
        This is an alias for :py:func:`var_pop`.
    """
    return var_pop(expression, filter)


def var_samp(expression: Expr | str, filter: Expr | str | None = None) -> Expr:
    """Computes the sample variance of the argument.

    If using the builder functions described in ref:`_aggregation` this function ignores
    the options ``order_by``, ``null_treatment``, and ``distinct``.

    Args:
        expression: The variable to compute the variance for
        filter: If provided, only compute against rows for which the filter is True

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"a": [1.0, 2.0, 3.0]})
        >>> result = df.aggregate([], [dfn.functions.var_samp("a").alias("v")])
        >>> result.collect_column("v")[0].as_py()
        1.0

        >>> result = df.aggregate(
        ...     [], [dfn.functions.var_samp(
        ...         "a",
        ...         filter=dfn.col("a") > dfn.lit(1.0)
        ...     ).alias("v")])
        >>> result.collect_column("v")[0].as_py()
        0.5
    """
    filter_raw = coerce_to_column_or_none(filter)
    return Expr(f.var_sample(coerce_to_column(expression), filter=filter_raw))


def var_sample(expression: Expr | str, filter: Expr | str | None = None) -> Expr:
    """Computes the sample variance of the argument.

    See Also:
        This is an alias for :py:func:`var_samp`.
    """
    return var_samp(expression, filter)


def regr_avgx(
    y: Expr | str,
    x: Expr | str,
    filter: Expr | str | None = None,
) -> Expr:
    """Computes the average of the independent variable ``x``.

    This is a linear regression aggregate function. Only non-null pairs of the inputs
    are evaluated.

    If using the builder functions described in ref:`_aggregation` this function ignores
    the options ``order_by``, ``null_treatment``, and ``distinct``.

    Args:
        y: The linear regression dependent variable
        x: The linear regression independent variable
        filter: If provided, only compute against rows for which the filter is True

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"y": [1.0, 2.0, 3.0], "x": [4.0, 5.0, 6.0]})
        >>> result = df.aggregate([], [dfn.functions.regr_avgx("y", "x").alias("v")])
        >>> result.collect_column("v")[0].as_py()
        5.0

        >>> result = df.aggregate(
        ...     [], [dfn.functions.regr_avgx(
        ...         "y", "x",
        ...         filter=dfn.col("y") > dfn.lit(1.0)
        ...     ).alias("v")])
        >>> result.collect_column("v")[0].as_py()
        5.5
    """
    filter_raw = coerce_to_column_or_none(filter)

    return Expr(
        f.regr_avgx(coerce_to_column(y), coerce_to_column(x), filter=filter_raw)
    )


def regr_avgy(
    y: Expr | str,
    x: Expr | str,
    filter: Expr | str | None = None,
) -> Expr:
    """Computes the average of the dependent variable ``y``.

    This is a linear regression aggregate function. Only non-null pairs of the inputs
    are evaluated.

    If using the builder functions described in ref:`_aggregation` this function ignores
    the options ``order_by``, ``null_treatment``, and ``distinct``.

    Args:
        y: The linear regression dependent variable
        x: The linear regression independent variable
        filter: If provided, only compute against rows for which the filter is True

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"y": [1.0, 2.0, 3.0], "x": [4.0, 5.0, 6.0]})
        >>> result = df.aggregate([], [dfn.functions.regr_avgy("y", "x").alias("v")])
        >>> result.collect_column("v")[0].as_py()
        2.0

        >>> result = df.aggregate(
        ...     [], [dfn.functions.regr_avgy(
        ...         "y", "x",
        ...         filter=dfn.col("y") > dfn.lit(1.0)
        ...     ).alias("v")])
        >>> result.collect_column("v")[0].as_py()
        2.5
    """
    filter_raw = coerce_to_column_or_none(filter)

    return Expr(
        f.regr_avgy(coerce_to_column(y), coerce_to_column(x), filter=filter_raw)
    )


def regr_count(
    y: Expr | str,
    x: Expr | str,
    filter: Expr | str | None = None,
) -> Expr:
    """Counts the number of rows in which both expressions are not null.

    This is a linear regression aggregate function. Only non-null pairs of the inputs
    are evaluated.

    If using the builder functions described in ref:`_aggregation` this function ignores
    the options ``order_by``, ``null_treatment``, and ``distinct``.

    Args:
        y: The linear regression dependent variable
        x: The linear regression independent variable
        filter: If provided, only compute against rows for which the filter is True

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"y": [1.0, 2.0, 3.0], "x": [4.0, 5.0, 6.0]})
        >>> result = df.aggregate([], [dfn.functions.regr_count("y", "x").alias("v")])
        >>> result.collect_column("v")[0].as_py()
        3

        >>> result = df.aggregate(
        ...     [], [dfn.functions.regr_count(
        ...         "y", "x",
        ...         filter=dfn.col("y") > dfn.lit(1.0)
        ...     ).alias("v")])
        >>> result.collect_column("v")[0].as_py()
        2
    """
    filter_raw = coerce_to_column_or_none(filter)

    return Expr(
        f.regr_count(coerce_to_column(y), coerce_to_column(x), filter=filter_raw)
    )


def regr_intercept(
    y: Expr | str,
    x: Expr | str,
    filter: Expr | str | None = None,
) -> Expr:
    """Computes the intercept from the linear regression.

    This is a linear regression aggregate function. Only non-null pairs of the inputs
    are evaluated.

    If using the builder functions described in ref:`_aggregation` this function ignores
    the options ``order_by``, ``null_treatment``, and ``distinct``.

    Args:
        y: The linear regression dependent variable
        x: The linear regression independent variable
        filter: If provided, only compute against rows for which the filter is True

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"y": [2.0, 4.0, 6.0], "x": [4.0, 16.0, 36.0]})
        >>> result = df.aggregate(
        ...     [],
        ...     [dfn.functions.regr_intercept(
        ...         "y", "x"
        ...     ).alias("v")])
        >>> result.collect_column("v")[0].as_py()
        1.714...

        >>> result = df.aggregate(
        ...     [],
        ...     [dfn.functions.regr_intercept(
        ...         "y", "x",
        ...         filter=dfn.col("y") > dfn.lit(2.0)
        ...     ).alias("v")])
        >>> result.collect_column("v")[0].as_py()
        2.4
    """
    filter_raw = coerce_to_column_or_none(filter)

    return Expr(
        f.regr_intercept(coerce_to_column(y), coerce_to_column(x), filter=filter_raw)
    )


def regr_r2(
    y: Expr | str,
    x: Expr | str,
    filter: Expr | str | None = None,
) -> Expr:
    """Computes the R-squared value from linear regression.

    This is a linear regression aggregate function. Only non-null pairs of the inputs
    are evaluated.

    If using the builder functions described in ref:`_aggregation` this function ignores
    the options ``order_by``, ``null_treatment``, and ``distinct``.

    Args:
        y: The linear regression dependent variable
        x: The linear regression independent variable
        filter: If provided, only compute against rows for which the filter is True

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"y": [2.0, 4.0, 6.0], "x": [4.0, 16.0, 36.0]})
        >>> result = df.aggregate([], [dfn.functions.regr_r2("y", "x").alias("v")])
        >>> result.collect_column("v")[0].as_py()
        0.9795...

        >>> result = df.aggregate(
        ...     [], [dfn.functions.regr_r2(
        ...         "y", "x",
        ...         filter=dfn.col("y") > dfn.lit(2.0)
        ...     ).alias("v")])
        >>> result.collect_column("v")[0].as_py()
        1.0
    """
    filter_raw = coerce_to_column_or_none(filter)

    return Expr(f.regr_r2(coerce_to_column(y), coerce_to_column(x), filter=filter_raw))


def regr_slope(
    y: Expr | str,
    x: Expr | str,
    filter: Expr | str | None = None,
) -> Expr:
    """Computes the slope from linear regression.

    This is a linear regression aggregate function. Only non-null pairs of the inputs
    are evaluated.

    If using the builder functions described in ref:`_aggregation` this function ignores
    the options ``order_by``, ``null_treatment``, and ``distinct``.

    Args:
        y: The linear regression dependent variable
        x: The linear regression independent variable
        filter: If provided, only compute against rows for which the filter is True

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"y": [2.0, 4.0, 6.0], "x": [4.0, 16.0, 36.0]})
        >>> result = df.aggregate([], [dfn.functions.regr_slope("y", "x").alias("v")])
        >>> result.collect_column("v")[0].as_py()
        0.122...

        >>> result = df.aggregate(
        ...     [], [dfn.functions.regr_slope(
        ...         "y", "x",
        ...         filter=dfn.col("y") > dfn.lit(2.0)
        ...     ).alias("v")])
        >>> result.collect_column("v")[0].as_py()
        0.1
    """
    filter_raw = coerce_to_column_or_none(filter)

    return Expr(
        f.regr_slope(coerce_to_column(y), coerce_to_column(x), filter=filter_raw)
    )


def regr_sxx(
    y: Expr | str,
    x: Expr | str,
    filter: Expr | str | None = None,
) -> Expr:
    """Computes the sum of squares of the independent variable ``x``.

    This is a linear regression aggregate function. Only non-null pairs of the inputs
    are evaluated.

    If using the builder functions described in ref:`_aggregation` this function ignores
    the options ``order_by``, ``null_treatment``, and ``distinct``.

    Args:
        y: The linear regression dependent variable
        x: The linear regression independent variable
        filter: If provided, only compute against rows for which the filter is True

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"y": [1.0, 2.0, 3.0], "x": [1.0, 2.0, 3.0]})
        >>> result = df.aggregate([], [dfn.functions.regr_sxx("y", "x").alias("v")])
        >>> result.collect_column("v")[0].as_py()
        2.0

        >>> result = df.aggregate(
        ...     [], [dfn.functions.regr_sxx(
        ...         "y", "x",
        ...         filter=dfn.col("y") > dfn.lit(1.0)
        ...     ).alias("v")])
        >>> result.collect_column("v")[0].as_py()
        0.5
    """
    filter_raw = coerce_to_column_or_none(filter)

    return Expr(f.regr_sxx(coerce_to_column(y), coerce_to_column(x), filter=filter_raw))


def regr_sxy(
    y: Expr | str,
    x: Expr | str,
    filter: Expr | str | None = None,
) -> Expr:
    """Computes the sum of products of pairs of numbers.

    This is a linear regression aggregate function. Only non-null pairs of the inputs
    are evaluated.

    If using the builder functions described in ref:`_aggregation` this function ignores
    the options ``order_by``, ``null_treatment``, and ``distinct``.

    Args:
        y: The linear regression dependent variable
        x: The linear regression independent variable
        filter: If provided, only compute against rows for which the filter is True

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"y": [1.0, 2.0, 3.0], "x": [1.0, 2.0, 3.0]})
        >>> result = df.aggregate([], [dfn.functions.regr_sxy("y", "x").alias("v")])
        >>> result.collect_column("v")[0].as_py()
        2.0

        >>> result = df.aggregate(
        ...     [], [dfn.functions.regr_sxy(
        ...         "y", "x",
        ...         filter=dfn.col("y") > dfn.lit(1.0)
        ...     ).alias("v")])
        >>> result.collect_column("v")[0].as_py()
        0.5
    """
    filter_raw = coerce_to_column_or_none(filter)

    return Expr(f.regr_sxy(coerce_to_column(y), coerce_to_column(x), filter=filter_raw))


def regr_syy(
    y: Expr | str,
    x: Expr | str,
    filter: Expr | str | None = None,
) -> Expr:
    """Computes the sum of squares of the dependent variable ``y``.

    This is a linear regression aggregate function. Only non-null pairs of the inputs
    are evaluated.

    If using the builder functions described in ref:`_aggregation` this function ignores
    the options ``order_by``, ``null_treatment``, and ``distinct``.

    Args:
        y: The linear regression dependent variable
        x: The linear regression independent variable
        filter: If provided, only compute against rows for which the filter is True

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"y": [1.0, 2.0, 3.0], "x": [1.0, 2.0, 3.0]})
        >>> result = df.aggregate([], [dfn.functions.regr_syy("y", "x").alias("v")])
        >>> result.collect_column("v")[0].as_py()
        2.0

        >>> result = df.aggregate(
        ...     [], [dfn.functions.regr_syy(
        ...         "y", "x",
        ...         filter=dfn.col("y") > dfn.lit(1.0)
        ...     ).alias("v")])
        >>> result.collect_column("v")[0].as_py()
        0.5
    """
    filter_raw = coerce_to_column_or_none(filter)

    return Expr(f.regr_syy(coerce_to_column(y), coerce_to_column(x), filter=filter_raw))


def first_value(
    expression: Expr | str,
    filter: Expr | str | None = None,
    order_by: list[SortKey] | SortKey | None = None,
    null_treatment: NullTreatment = NullTreatment.RESPECT_NULLS,
) -> Expr:
    """Returns the first value in a group of values.

    This aggregate function will return the first value in the partition.

    If using the builder functions described in ref:`_aggregation` this function ignores
    the option ``distinct``.

    Args:
        expression: Argument to perform bitwise calculation on
        filter: If provided, only compute against rows for which the filter is True
        order_by: Set the ordering of the expression to evaluate. Accepts
            column names or expressions.
        null_treatment: Assign whether to respect or ignore null values.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"a": [10, 20, 30]})
        >>> result = df.aggregate([], [dfn.functions.first_value("a").alias("v")])
        >>> result.collect_column("v")[0].as_py()
        10

        >>> df = ctx.from_pydict({"a": [None, 20, 10]})
        >>> result = df.aggregate(
        ...     [], [dfn.functions.first_value(
        ...         "a",
        ...         filter=dfn.col("a") > dfn.lit(10),
        ...         order_by="a",
        ...         null_treatment=dfn.common.NullTreatment.IGNORE_NULLS,
        ...     ).alias("v")]
        ... )
        >>> result.collect_column("v")[0].as_py()
        20
    """
    order_by_raw = sort_list_to_raw_sort_list(order_by)
    filter_raw = coerce_to_column_or_none(filter)

    return Expr(
        f.first_value(
            coerce_to_column(expression),
            filter=filter_raw,
            order_by=order_by_raw,
            null_treatment=null_treatment.value,
        )
    )


def last_value(
    expression: Expr | str,
    filter: Expr | str | None = None,
    order_by: list[SortKey] | SortKey | None = None,
    null_treatment: NullTreatment = NullTreatment.RESPECT_NULLS,
) -> Expr:
    """Returns the last value in a group of values.

    This aggregate function will return the last value in the partition.

    If using the builder functions described in ref:`_aggregation` this function ignores
    the option ``distinct``.

    Args:
        expression: Argument to perform bitwise calculation on
        filter: If provided, only compute against rows for which the filter is True
        order_by: Set the ordering of the expression to evaluate. Accepts
            column names or expressions.
        null_treatment: Assign whether to respect or ignore null values.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"a": [10, 20, 30]})
        >>> result = df.aggregate([], [dfn.functions.last_value("a").alias("v")])
        >>> result.collect_column("v")[0].as_py()
        30

        >>> df = ctx.from_pydict({"a": [None, 20, 10]})
        >>> result = df.aggregate(
        ...     [], [dfn.functions.last_value(
        ...         "a",
        ...         filter=dfn.col("a") > dfn.lit(10),
        ...         order_by="a",
        ...         null_treatment=dfn.common.NullTreatment.IGNORE_NULLS,
        ...     ).alias("v")]
        ... )
        >>> result.collect_column("v")[0].as_py()
        20
    """
    order_by_raw = sort_list_to_raw_sort_list(order_by)
    filter_raw = coerce_to_column_or_none(filter)

    return Expr(
        f.last_value(
            coerce_to_column(expression),
            filter=filter_raw,
            order_by=order_by_raw,
            null_treatment=null_treatment.value,
        )
    )


def nth_value(
    expression: Expr | str,
    n: int,
    filter: Expr | str | None = None,
    order_by: list[SortKey] | SortKey | None = None,
    null_treatment: NullTreatment = NullTreatment.RESPECT_NULLS,
) -> Expr:
    """Returns the n-th value in a group of values.

    This aggregate function will return the n-th value in the partition.

    If using the builder functions described in ref:`_aggregation` this function ignores
    the option ``distinct``.

    Args:
        expression: Argument to perform bitwise calculation on
        n: Index of value to return. Starts at 1.
        filter: If provided, only compute against rows for which the filter is True
        order_by: Set the ordering of the expression to evaluate. Accepts
            column names or expressions.
        null_treatment: Assign whether to respect or ignore null values.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"a": [10, 20, 30]})
        >>> result = df.aggregate([], [dfn.functions.nth_value("a", 1).alias("v")])
        >>> result.collect_column("v")[0].as_py()
        10

        >>> result = df.aggregate(
        ...     [], [dfn.functions.nth_value(
        ...         "a", 1,
        ...         filter=dfn.col("a") > dfn.lit(10),
        ...         order_by="a",
        ...         null_treatment=dfn.common.NullTreatment.IGNORE_NULLS,
        ...     ).alias("v")]
        ... )
        >>> result.collect_column("v")[0].as_py()
        20
    """
    order_by_raw = sort_list_to_raw_sort_list(order_by)
    filter_raw = coerce_to_column_or_none(filter)

    return Expr(
        f.nth_value(
            coerce_to_column(expression),
            n,
            filter=filter_raw,
            order_by=order_by_raw,
            null_treatment=null_treatment.value,
        )
    )


def bit_and(expression: Expr | str, filter: Expr | str | None = None) -> Expr:
    """Computes the bitwise AND of the argument.

    This aggregate function will bitwise compare every value in the input partition.

    If using the builder functions described in ref:`_aggregation` this function ignores
    the options ``order_by``, ``null_treatment``, and ``distinct``.

    Args:
        expression: Argument to perform bitwise calculation on
        filter: If provided, only compute against rows for which the filter is True

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"a": [7, 3]})
        >>> result = df.aggregate([], [dfn.functions.bit_and("a").alias("v")])
        >>> result.collect_column("v")[0].as_py()
        3

        >>> df = ctx.from_pydict({"a": [7, 5, 3]})
        >>> result = df.aggregate(
        ...     [], [dfn.functions.bit_and(
        ...         "a",
        ...         filter=dfn.col("a") > dfn.lit(3)
        ...     ).alias("v")])
        >>> result.collect_column("v")[0].as_py()
        5
    """
    filter_raw = coerce_to_column_or_none(filter)
    return Expr(f.bit_and(coerce_to_column(expression), filter=filter_raw))


def bit_or(expression: Expr | str, filter: Expr | str | None = None) -> Expr:
    """Computes the bitwise OR of the argument.

    This aggregate function will bitwise compare every value in the input partition.

    If using the builder functions described in ref:`_aggregation` this function ignores
    the options ``order_by``, ``null_treatment``, and ``distinct``.

    Args:
        expression: Argument to perform bitwise calculation on
        filter: If provided, only compute against rows for which the filter is True

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"a": [1, 2]})
        >>> result = df.aggregate([], [dfn.functions.bit_or("a").alias("v")])
        >>> result.collect_column("v")[0].as_py()
        3

        >>> df = ctx.from_pydict({"a": [1, 2, 4]})
        >>> result = df.aggregate(
        ...     [], [dfn.functions.bit_or(
        ...         "a",
        ...         filter=dfn.col("a") > dfn.lit(1)
        ...     ).alias("v")]
        ... )
        >>> result.collect_column("v")[0].as_py()
        6
    """
    filter_raw = coerce_to_column_or_none(filter)
    return Expr(f.bit_or(coerce_to_column(expression), filter=filter_raw))


def bit_xor(
    expression: Expr | str, distinct: bool = False, filter: Expr | str | None = None
) -> Expr:
    """Computes the bitwise XOR of the argument.

    This aggregate function will bitwise compare every value in the input partition.

    If using the builder functions described in ref:`_aggregation` this function ignores
    the options ``order_by`` and ``null_treatment``.

    Args:
        expression: Argument to perform bitwise calculation on
        distinct: If True, evaluate each unique value of expression only once
        filter: If provided, only compute against rows for which the filter is True

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"a": [5, 3]})
        >>> result = df.aggregate([], [dfn.functions.bit_xor("a").alias("v")])
        >>> result.collect_column("v")[0].as_py()
        6

        >>> df = ctx.from_pydict({"a": [5, 5, 3]})
        >>> result = df.aggregate(
        ...     [], [dfn.functions.bit_xor(
        ...         "a", distinct=True,
        ...         filter=dfn.col("a") > dfn.lit(3),
        ...     ).alias("v")]
        ... )
        >>> result.collect_column("v")[0].as_py()
        5
    """
    filter_raw = coerce_to_column_or_none(filter)
    return Expr(
        f.bit_xor(coerce_to_column(expression), distinct=distinct, filter=filter_raw)
    )


def bool_and(expression: Expr | str, filter: Expr | str | None = None) -> Expr:
    """Computes the boolean AND of the argument.

    This aggregate function will compare every value in the input partition. These are
    expected to be boolean values.

    If using the builder functions described in ref:`_aggregation` this function ignores
    the options ``order_by``, ``null_treatment``, and ``distinct``.

    Args:
        expression: Argument to perform calculation on
        filter: If provided, only compute against rows for which the filter is True

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"a": [True, True, False]})
        >>> result = df.aggregate([], [dfn.functions.bool_and("a").alias("v")])
        >>> result.collect_column("v")[0].as_py()
        False

        >>> df = ctx.from_pydict(
        ...     {"a": [True, True, False], "b": [1, 2, 3]})
        >>> result = df.aggregate(
        ...     [], [dfn.functions.bool_and(
        ...         "a",
        ...         filter=dfn.col("b") < dfn.lit(3)
        ...     ).alias("v")]
        ... )
        >>> result.collect_column("v")[0].as_py()
        True
    """
    filter_raw = coerce_to_column_or_none(filter)
    return Expr(f.bool_and(coerce_to_column(expression), filter=filter_raw))


def bool_or(expression: Expr | str, filter: Expr | str | None = None) -> Expr:
    """Computes the boolean OR of the argument.

    This aggregate function will compare every value in the input partition. These are
    expected to be boolean values.

    If using the builder functions described in ref:`_aggregation` this function ignores
    the options ``order_by``, ``null_treatment``, and ``distinct``.

    Args:
        expression: Argument to perform calculation on
        filter: If provided, only compute against rows for which the filter is True

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"a": [False, False, True]})
        >>> result = df.aggregate([], [dfn.functions.bool_or("a").alias("v")])
        >>> result.collect_column("v")[0].as_py()
        True

        >>> df = ctx.from_pydict(
        ...     {"a": [False, False, True], "b": [1, 2, 3]})
        >>> result = df.aggregate(
        ...     [], [dfn.functions.bool_or(
        ...         "a",
        ...         filter=dfn.col("b") < dfn.lit(3)
        ...     ).alias("v")]
        ... )
        >>> result.collect_column("v")[0].as_py()
        False
    """
    filter_raw = coerce_to_column_or_none(filter)
    return Expr(f.bool_or(coerce_to_column(expression), filter=filter_raw))


def lead(
    arg: Expr | str,
    shift_offset: int = 1,
    default_value: Any | None = None,
    partition_by: list[Expr | str] | Expr | str | None = None,
    order_by: list[SortKey] | SortKey | None = None,
) -> Expr:
    """Create a lead window function.

    Lead operation will return the argument that is in the next shift_offset-th row in
    the partition. For example ``lead(col("b"), shift_offset=3, default_value=5)`` will
    return the 3rd following value in column ``b``. At the end of the partition, where
    no further values can be returned it will return the default value of 5.

    Here is an example of both the ``lead`` and :py:func:`datafusion.functions.lag`
    functions on a simple DataFrame::

        +--------+------+-----+
        | points | lead | lag |
        +--------+------+-----+
        | 100    | 100  |     |
        | 100    | 50   | 100 |
        | 50     | 25   | 100 |
        | 25     |      | 50  |
        +--------+------+-----+

    To set window function parameters use the window builder approach described in the
    ref:`_window_functions` online documentation.

    Args:
        arg: Value to return
        shift_offset: Number of rows following the current row.
        default_value: Value to return if shift_offet row does not exist.
        partition_by: Expressions to partition the window frame on.
        order_by: Set ordering within the window frame. Accepts
            column names or expressions.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"a": [1, 2, 3]})
        >>> result = df.select(
        ...     dfn.col("a"),
        ...     dfn.functions.lead(
        ...         "a", shift_offset=1,
        ...         default_value=0, order_by="a"
        ...     ).alias("lead"))
        >>> result.sort(dfn.col("a")).collect_column("lead").to_pylist()
        [2, 3, 0]

        >>> df = ctx.from_pydict({"g": ["a", "a", "b"], "v": [1, 2, 3]})
        >>> result = df.select(
        ...     dfn.col("g"), dfn.col("v"),
        ...     dfn.functions.lead(
        ...         "v", shift_offset=1, default_value=0,
        ...         partition_by="g", order_by="v",
        ...     ).alias("lead"))
        >>> result.sort(dfn.col("g"), dfn.col("v")).collect_column("lead").to_pylist()
        [2, 0, 0]
    """
    if not isinstance(default_value, pa.Scalar) and default_value is not None:
        default_value = pa.scalar(default_value)

    partition_by_raw = expr_list_to_raw_expr_list(partition_by)
    order_by_raw = sort_list_to_raw_sort_list(order_by)

    return Expr(
        f.lead(
            coerce_to_column(arg),
            shift_offset,
            default_value,
            partition_by=partition_by_raw,
            order_by=order_by_raw,
        )
    )


def lag(
    arg: Expr | str,
    shift_offset: int = 1,
    default_value: Any | None = None,
    partition_by: list[Expr | str] | Expr | str | None = None,
    order_by: list[SortKey] | SortKey | None = None,
) -> Expr:
    """Create a lag window function.

    Lag operation will return the argument that is in the previous shift_offset-th row
    in the partition. For example ``lag(col("b"), shift_offset=3, default_value=5)``
    will return the 3rd previous value in column ``b``. At the beginning of the
    partition, where no values can be returned it will return the default value of 5.

    Here is an example of both the ``lag`` and :py:func:`datafusion.functions.lead`
    functions on a simple DataFrame::

        +--------+------+-----+
        | points | lead | lag |
        +--------+------+-----+
        | 100    | 100  |     |
        | 100    | 50   | 100 |
        | 50     | 25   | 100 |
        | 25     |      | 50  |
        +--------+------+-----+

    Args:
        arg: Value to return
        shift_offset: Number of rows before the current row.
        default_value: Value to return if shift_offet row does not exist.
        partition_by: Expressions to partition the window frame on.
        order_by: Set ordering within the window frame. Accepts
            column names or expressions.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"a": [1, 2, 3]})
        >>> result = df.select(
        ...     dfn.col("a"),
        ...     dfn.functions.lag(
        ...         "a", shift_offset=1,
        ...         default_value=0, order_by="a"
        ...     ).alias("lag"))
        >>> result.sort(dfn.col("a")).collect_column("lag").to_pylist()
        [0, 1, 2]

        >>> df = ctx.from_pydict({"g": ["a", "a", "b"], "v": [1, 2, 3]})
        >>> result = df.select(
        ...     dfn.col("g"), dfn.col("v"),
        ...     dfn.functions.lag(
        ...         "v", shift_offset=1, default_value=0,
        ...         partition_by="g", order_by="v",
        ...     ).alias("lag"))
        >>> result.sort(dfn.col("g"), dfn.col("v")).collect_column("lag").to_pylist()
        [0, 1, 0]
    """
    if not isinstance(default_value, pa.Scalar):
        default_value = pa.scalar(default_value)

    partition_by_raw = expr_list_to_raw_expr_list(partition_by)
    order_by_raw = sort_list_to_raw_sort_list(order_by)

    return Expr(
        f.lag(
            coerce_to_column(arg),
            shift_offset,
            default_value,
            partition_by=partition_by_raw,
            order_by=order_by_raw,
        )
    )


def row_number(
    partition_by: list[Expr | str] | Expr | str | None = None,
    order_by: list[SortKey] | SortKey | None = None,
) -> Expr:
    """Create a row number window function.

    Returns the row number of the window function.

    Here is an example of the ``row_number`` on a simple DataFrame::

        +--------+------------+
        | points | row number |
        +--------+------------+
        | 100    | 1          |
        | 100    | 2          |
        | 50     | 3          |
        | 25     | 4          |
        +--------+------------+

    Args:
        partition_by: Expressions to partition the window frame on.
        order_by: Set ordering within the window frame. Accepts
            column names or expressions.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"a": [10, 20, 30]})
        >>> result = df.select(
        ...     dfn.col("a"),
        ...     dfn.functions.row_number(
        ...         order_by="a"
        ...     ).alias("rn"))
        >>> result.sort(dfn.col("a")).collect_column("rn").to_pylist()
        [1, 2, 3]

        >>> df = ctx.from_pydict(
        ...     {"g": ["a", "a", "b", "b"], "v": [1, 2, 3, 4]})
        >>> result = df.select(
        ...     dfn.col("g"), dfn.col("v"),
        ...     dfn.functions.row_number(
        ...         partition_by="g", order_by="v",
        ...     ).alias("rn"))
        >>> result.sort(dfn.col("g"), dfn.col("v")).collect_column("rn").to_pylist()
        [1, 2, 1, 2]
    """
    partition_by_raw = expr_list_to_raw_expr_list(partition_by)
    order_by_raw = sort_list_to_raw_sort_list(order_by)

    return Expr(
        f.row_number(
            partition_by=partition_by_raw,
            order_by=order_by_raw,
        )
    )


def rank(
    partition_by: list[Expr | str] | Expr | str | None = None,
    order_by: list[SortKey] | SortKey | None = None,
) -> Expr:
    """Create a rank window function.

    Returns the rank based upon the window order. Consecutive equal values will receive
    the same rank, but the next different value will not be consecutive but rather the
    number of rows that precede it plus one. This is similar to Olympic medals. If two
    people tie for gold, the next place is bronze. There would be no silver medal. Here
    is an example of a dataframe with a window ordered by descending ``points`` and the
    associated rank.

    You should set ``order_by`` to produce meaningful results::

        +--------+------+
        | points | rank |
        +--------+------+
        | 100    | 1    |
        | 100    | 1    |
        | 50     | 3    |
        | 25     | 4    |
        +--------+------+

    Args:
        partition_by: Expressions to partition the window frame on.
        order_by: Set ordering within the window frame. Accepts
            column names or expressions.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"a": [10, 10, 20]})
        >>> result = df.select(
        ...     dfn.col("a"),
        ...     dfn.functions.rank(
        ...         order_by="a"
        ...     ).alias("rnk")
        ... )
        >>> result.sort(dfn.col("a")).collect_column("rnk").to_pylist()
        [1, 1, 3]

        >>> df = ctx.from_pydict(
        ...     {"g": ["a", "a", "b", "b"], "v": [1, 1, 2, 3]})
        >>> result = df.select(
        ...     dfn.col("g"), dfn.col("v"),
        ...     dfn.functions.rank(
        ...         partition_by="g", order_by="v",
        ...     ).alias("rnk"))
        >>> result.sort(dfn.col("g"), dfn.col("v")).collect_column("rnk").to_pylist()
        [1, 1, 1, 2]
    """
    partition_by_raw = expr_list_to_raw_expr_list(partition_by)
    order_by_raw = sort_list_to_raw_sort_list(order_by)

    return Expr(
        f.rank(
            partition_by=partition_by_raw,
            order_by=order_by_raw,
        )
    )


def dense_rank(
    partition_by: list[Expr | str] | Expr | str | None = None,
    order_by: list[SortKey] | SortKey | None = None,
) -> Expr:
    """Create a dense_rank window function.

    This window function is similar to :py:func:`rank` except that the returned values
    will be consecutive. Here is an example of a dataframe with a window ordered by
    descending ``points`` and the associated dense rank::

        +--------+------------+
        | points | dense_rank |
        +--------+------------+
        | 100    | 1          |
        | 100    | 1          |
        | 50     | 2          |
        | 25     | 3          |
        +--------+------------+

    Args:
        partition_by: Expressions to partition the window frame on.
        order_by: Set ordering within the window frame. Accepts
            column names or expressions.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"a": [10, 10, 20]})
        >>> result = df.select(
        ...     dfn.col("a"),
        ...     dfn.functions.dense_rank(
        ...         order_by="a"
        ...     ).alias("dr"))
        >>> result.sort(dfn.col("a")).collect_column("dr").to_pylist()
        [1, 1, 2]

        >>> df = ctx.from_pydict(
        ...     {"g": ["a", "a", "b", "b"], "v": [1, 1, 2, 3]})
        >>> result = df.select(
        ...     dfn.col("g"), dfn.col("v"),
        ...     dfn.functions.dense_rank(
        ...         partition_by="g", order_by="v",
        ...     ).alias("dr"))
        >>> result.sort(dfn.col("g"), dfn.col("v")).collect_column("dr").to_pylist()
        [1, 1, 1, 2]
    """
    partition_by_raw = expr_list_to_raw_expr_list(partition_by)
    order_by_raw = sort_list_to_raw_sort_list(order_by)

    return Expr(
        f.dense_rank(
            partition_by=partition_by_raw,
            order_by=order_by_raw,
        )
    )


def percent_rank(
    partition_by: list[Expr | str] | Expr | str | None = None,
    order_by: list[SortKey] | SortKey | None = None,
) -> Expr:
    """Create a percent_rank window function.

    This window function is similar to :py:func:`rank` except that the returned values
    are the percentage from 0.0 to 1.0 from first to last. Here is an example of a
    dataframe with a window ordered by descending ``points`` and the associated percent
    rank::

        +--------+--------------+
        | points | percent_rank |
        +--------+--------------+
        | 100    | 0.0          |
        | 100    | 0.0          |
        | 50     | 0.666667     |
        | 25     | 1.0          |
        +--------+--------------+

    Args:
        partition_by: Expressions to partition the window frame on.
        order_by: Set ordering within the window frame. Accepts
            column names or expressions.


    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"a": [10, 20, 30]})
        >>> result = df.select(
        ...     dfn.col("a"),
        ...     dfn.functions.percent_rank(
        ...         order_by="a"
        ...     ).alias("pr"))
        >>> result.sort(dfn.col("a")).collect_column("pr").to_pylist()
        [0.0, 0.5, 1.0]

        >>> df = ctx.from_pydict(
        ...     {"g": ["a", "a", "a", "b", "b"], "v": [1, 2, 3, 4, 5]})
        >>> result = df.select(
        ...     dfn.col("g"), dfn.col("v"),
        ...     dfn.functions.percent_rank(
        ...         partition_by="g", order_by="v",
        ...     ).alias("pr"))
        >>> result.sort(dfn.col("g"), dfn.col("v")).collect_column("pr").to_pylist()
        [0.0, 0.5, 1.0, 0.0, 1.0]
    """
    partition_by_raw = expr_list_to_raw_expr_list(partition_by)
    order_by_raw = sort_list_to_raw_sort_list(order_by)

    return Expr(
        f.percent_rank(
            partition_by=partition_by_raw,
            order_by=order_by_raw,
        )
    )


def cume_dist(
    partition_by: list[Expr | str] | Expr | str | None = None,
    order_by: list[SortKey] | SortKey | None = None,
) -> Expr:
    """Create a cumulative distribution window function.

    This window function is similar to :py:func:`rank` except that the returned values
    are the ratio of the row number to the total number of rows. Here is an example of a
    dataframe with a window ordered by descending ``points`` and the associated
    cumulative distribution::

        +--------+-----------+
        | points | cume_dist |
        +--------+-----------+
        | 100    | 0.5       |
        | 100    | 0.5       |
        | 50     | 0.75      |
        | 25     | 1.0       |
        +--------+-----------+

    Args:
        partition_by: Expressions to partition the window frame on.
        order_by: Set ordering within the window frame. Accepts
            column names or expressions.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"a": [1., 2., 2., 3.]})
        >>> result = df.select(
        ...     dfn.col("a"),
        ...     dfn.functions.cume_dist(
        ...         order_by="a"
        ...     ).alias("cd")
        ... )
        >>> result.collect_column("cd").to_pylist()
        [0.25..., 0.75..., 0.75..., 1.0...]

        >>> df = ctx.from_pydict(
        ...     {"g": ["a", "a", "b", "b"], "v": [1, 2, 3, 4]})
        >>> result = df.select(
        ...     dfn.col("g"), dfn.col("v"),
        ...     dfn.functions.cume_dist(
        ...         partition_by="g", order_by="v",
        ...     ).alias("cd"))
        >>> result.sort(dfn.col("g"), dfn.col("v")).collect_column("cd").to_pylist()
        [0.5, 1.0, 0.5, 1.0]
    """
    partition_by_raw = expr_list_to_raw_expr_list(partition_by)
    order_by_raw = sort_list_to_raw_sort_list(order_by)

    return Expr(
        f.cume_dist(
            partition_by=partition_by_raw,
            order_by=order_by_raw,
        )
    )


def ntile(
    groups: int,
    partition_by: list[Expr | str] | Expr | str | None = None,
    order_by: list[SortKey] | SortKey | None = None,
) -> Expr:
    """Create a n-tile window function.

    This window function orders the window frame into a give number of groups based on
    the ordering criteria. It then returns which group the current row is assigned to.
    Here is an example of a dataframe with a window ordered by descending ``points``
    and the associated n-tile function::

        +--------+-------+
        | points | ntile |
        +--------+-------+
        | 120    | 1     |
        | 100    | 1     |
        | 80     | 2     |
        | 60     | 2     |
        | 40     | 3     |
        | 20     | 3     |
        +--------+-------+

    Args:
        groups: Number of groups for the n-tile to be divided into.
        partition_by: Expressions to partition the window frame on.
        order_by: Set ordering within the window frame. Accepts
            column names or expressions.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"a": [10, 20, 30, 40]})
        >>> result = df.select(
        ...     dfn.col("a"),
        ...     dfn.functions.ntile(
        ...         2, order_by="a"
        ...     ).alias("nt"))
        >>> result.sort(dfn.col("a")).collect_column("nt").to_pylist()
        [1, 1, 2, 2]

        >>> df = ctx.from_pydict(
        ...     {"g": ["a", "a", "b", "b"], "v": [1, 2, 3, 4]})
        >>> result = df.select(
        ...     dfn.col("g"), dfn.col("v"),
        ...     dfn.functions.ntile(
        ...         2, partition_by="g", order_by="v",
        ...     ).alias("nt"))
        >>> result.sort(dfn.col("g"), dfn.col("v")).collect_column("nt").to_pylist()
        [1, 2, 1, 2]
    """
    partition_by_raw = expr_list_to_raw_expr_list(partition_by)
    order_by_raw = sort_list_to_raw_sort_list(order_by)

    return Expr(
        f.ntile(
            Expr.literal(groups).expr,
            partition_by=partition_by_raw,
            order_by=order_by_raw,
        )
    )


def string_agg(
    expression: Expr | str,
    delimiter: str,
    filter: Expr | str | None = None,
    order_by: list[SortKey] | SortKey | None = None,
) -> Expr:
    """Concatenates the input strings.

    This aggregate function will concatenate input strings, ignoring null values, and
    separating them with the specified delimiter. Non-string values will be converted to
    their string equivalents.

    If using the builder functions described in ref:`_aggregation` this function ignores
    the options ``distinct`` and ``null_treatment``.

    Args:
        expression: Argument to perform bitwise calculation on
        delimiter: Text to place between each value of expression
        filter: If provided, only compute against rows for which the filter is True
        order_by: Set the ordering of the expression to evaluate. Accepts
            column names or expressions.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"a": ["x", "y", "z"]})
        >>> result = df.aggregate(
        ...     [], [dfn.functions.string_agg(
        ...         "a", ",", order_by="a"
        ...     ).alias("s")])
        >>> result.collect_column("s")[0].as_py()
        'x,y,z'

        >>> result = df.aggregate(
        ...     [], [dfn.functions.string_agg(
        ...         "a", ",",
        ...         filter=dfn.col("a") > dfn.lit("x"),
        ...         order_by="a",
        ...     ).alias("s")])
        >>> result.collect_column("s")[0].as_py()
        'y,z'
    """
    order_by_raw = sort_list_to_raw_sort_list(order_by)
    filter_raw = coerce_to_column_or_none(filter)

    return Expr(
        f.string_agg(
            coerce_to_column(expression),
            delimiter,
            filter=filter_raw,
            order_by=order_by_raw,
        )
    )
