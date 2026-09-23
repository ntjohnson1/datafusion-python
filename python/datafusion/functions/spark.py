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

"""Spark-compatible function bindings.

These functions mirror the semantics of their Apache Spark counterparts
exactly. Some override DataFusion built-ins (``substring`` is 1-indexed,
``concat`` propagates NULL, ``round`` uses HALF_UP rounding, etc.), which is
why they live in a separate namespace rather than replacing the defaults.

For DataFrame use, import this module and call functions directly. For SQL
use, call :py:meth:`datafusion.SessionContext.enable_spark_functions` to
register the Spark UDFs by name (overriding any built-ins with matching
names) before issuing SQL queries.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pyarrow as pa

from datafusion._internal import functions as _functions
from datafusion.expr import (
    Expr,
    _to_raw_expr,
    _to_raw_expr_list,
    _to_raw_expr_or_none,
    coerce_to_expr,
    sort_list_to_raw_sort_list,
)

if TYPE_CHECKING:
    from datafusion.common import NullTreatment
    from datafusion.expr import SortKey

_f = _functions.spark

# Reused int32 literal so optional-arg defaults don't rebuild it per call.
_ZERO_I32 = Expr.literal(pa.scalar(0, type=pa.int32()))


def _filter_raw(filter: Expr | str | None) -> Any:
    return _to_raw_expr_or_none(filter)


def _coerce_i32(value: Expr | int | None) -> Expr | None:
    """Coerce a native ``int`` to an int32 literal, passing ``Expr``/``None`` through.

    Several Spark datetime and interval builders require 32-bit integer
    inputs, so a bare ``int`` must become an int32 literal rather than the
    int64 default that :meth:`Expr.literal` would produce.
    """
    if value is None or isinstance(value, Expr):
        return value
    return Expr.literal(pa.scalar(value, type=pa.int32()))


# ---------------------------------------------------------------------------
# Aggregate functions
# ---------------------------------------------------------------------------


def avg(
    col: Expr | str,
    distinct: bool | None = None,
    filter: Expr | str | None = None,
    order_by: list[SortKey] | SortKey | None = None,
    null_treatment: NullTreatment | None = None,
) -> Expr:
    """Spark ``avg``: returns the mean of a numeric column.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"a": [1.0, 2.0, 3.0]})
        >>> r = df.aggregate([], [dfn.functions.spark.avg("a").alias("v")])
        >>> r.collect_column("v")[0].as_py()
        2.0
    """
    return Expr(
        _f.avg(
            _to_raw_expr(col),
            distinct=distinct,
            filter=_filter_raw(filter),
            order_by=sort_list_to_raw_sort_list(order_by),
            null_treatment=null_treatment,
        )
    )


def try_sum(
    col: Expr | str,
    distinct: bool | None = None,
    filter: Expr | str | None = None,
    order_by: list[SortKey] | SortKey | None = None,
    null_treatment: NullTreatment | None = None,
) -> Expr:
    """Spark ``try_sum``: sum that returns NULL on overflow.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"a": [1, 2, 3]})
        >>> r = df.aggregate([], [dfn.functions.spark.try_sum("a").alias("v")])
        >>> r.collect_column("v")[0].as_py()
        6
    """
    return Expr(
        _f.try_sum(
            _to_raw_expr(col),
            distinct=distinct,
            filter=_filter_raw(filter),
            order_by=sort_list_to_raw_sort_list(order_by),
            null_treatment=null_treatment,
        )
    )


def collect_list(
    col: Expr | str,
    distinct: bool | None = None,
    filter: Expr | str | None = None,
    order_by: list[SortKey] | SortKey | None = None,
    null_treatment: NullTreatment | None = None,
) -> Expr:
    """Spark ``collect_list``: collect values into an array (preserves dups).

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"a": [1, 2, 2]})
        >>> r = df.aggregate([], [dfn.functions.spark.collect_list("a").alias("v")])
        >>> sorted(r.collect_column("v")[0].as_py())
        [1, 2, 2]
    """
    return Expr(
        _f.collect_list(
            _to_raw_expr(col),
            distinct=distinct,
            filter=_filter_raw(filter),
            order_by=sort_list_to_raw_sort_list(order_by),
            null_treatment=null_treatment,
        )
    )


def collect_set(
    col: Expr | str,
    distinct: bool | None = None,
    filter: Expr | str | None = None,
    order_by: list[SortKey] | SortKey | None = None,
    null_treatment: NullTreatment | None = None,
) -> Expr:
    """Spark ``collect_set``: collect distinct values into an array.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"a": [1, 2, 2, 3]})
        >>> r = df.aggregate([], [dfn.functions.spark.collect_set("a").alias("v")])
        >>> sorted(r.collect_column("v")[0].as_py())
        [1, 2, 3]
    """
    return Expr(
        _f.collect_set(
            _to_raw_expr(col),
            distinct=distinct,
            filter=_filter_raw(filter),
            order_by=sort_list_to_raw_sort_list(order_by),
            null_treatment=null_treatment,
        )
    )


# ---------------------------------------------------------------------------
# Array functions
# ---------------------------------------------------------------------------


def array_contains(col: Expr | str, value: Expr | Any) -> Expr:
    """Spark ``array_contains``: true if the array contains the element.

    ``value`` accepts a native Python literal or an :class:`Expr`.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"x": [1]})
        >>> r = df.select(
        ...     dfn.functions.spark.array_contains(
        ...         dfn.functions.spark.array(dfn.lit(1), dfn.lit(2)), 1
        ...     ).alias("v")
        ... )
        >>> r.collect_column("v")[0].as_py()
        True
    """
    return Expr(_f.array_contains(_to_raw_expr(col), coerce_to_expr(value).expr))


def array(*cols: Expr | str) -> Expr:
    """Spark ``array``: builds an array from the given elements.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"x": [1]})
        >>> r = df.select(
        ...     dfn.functions.spark.array(
        ...         dfn.lit(1), dfn.lit(2), dfn.lit(3)
        ...     ).alias("v")
        ... )
        >>> r.collect_column("v")[0].as_py()
        [1, 2, 3]
    """
    return Expr(_f.array(*_to_raw_expr_list(cols)))


def shuffle(col: Expr | str, seed: int | None = None) -> Expr:
    """Spark ``shuffle``: returns a random permutation of the input array.

    ``seed`` is accepted for pyspark parity but is not yet wired through the
    Rust binding; passing a non-``None`` value raises ``NotImplementedError``.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"x": [1]})
        >>> r = df.select(
        ...     dfn.functions.spark.shuffle(
        ...         dfn.functions.spark.array(dfn.lit(1), dfn.lit(2), dfn.lit(3))
        ...     ).alias("v")
        ... )
        >>> sorted(r.collect_column("v")[0].as_py())
        [1, 2, 3]
    """
    if seed is not None:
        msg = "shuffle(seed=...) is not yet supported by the Spark UDF binding"
        raise NotImplementedError(msg)
    return Expr(_f.shuffle(_to_raw_expr(col)))


def array_repeat(col: Expr | str, count: Expr | int) -> Expr:
    """Spark ``array_repeat``: array of ``element`` repeated ``count`` times.

    ``count`` accepts a native ``int`` or an :class:`Expr`.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"x": [1]})
        >>> r = df.select(
        ...     dfn.functions.spark.array_repeat(dfn.lit("a"), 3).alias("v"))
        >>> r.collect_column("v")[0].as_py()
        ['a', 'a', 'a']
    """
    return Expr(_f.array_repeat(_to_raw_expr(col), coerce_to_expr(count).expr))


def slice(x: Expr | str, start: Expr | int, length: Expr | int) -> Expr:
    """Spark ``slice``: subset of the array from 1-indexed ``start`` with ``length``.

    Negative ``start`` counts from the end. ``start`` and ``length`` accept
    native ``int`` values or :class:`Expr`.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"x": [[1, 2, 3, 4]]})
        >>> r = df.select(dfn.functions.spark.slice("x", 2, 2).alias("v"))
        >>> r.collect_column("v")[0].as_py()
        [2, 3]
    """
    return Expr(
        _f.slice(
            _to_raw_expr(x), coerce_to_expr(start).expr, coerce_to_expr(length).expr
        )
    )


# ---------------------------------------------------------------------------
# Bitmap functions
# ---------------------------------------------------------------------------


def bitmap_count(col: Expr | str) -> Expr:
    r"""Spark ``bitmap_count``: number of set bits in a bitmap.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"x": [1]})
        >>> r = df.select(
        ...     dfn.functions.spark.bitmap_count(dfn.lit(b"\xff")).alias("v"))
        >>> r.collect_column("v")[0].as_py()
        8
    """
    return Expr(_f.bitmap_count(_to_raw_expr(col)))


def bitmap_bit_position(col: Expr | str) -> Expr:
    """Spark ``bitmap_bit_position``: bit position for a child expression.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"x": [1]})
        >>> r = df.select(
        ...     dfn.functions.spark.bitmap_bit_position(dfn.lit(15)).alias("v"))
        >>> r.collect_column("v")[0].as_py()
        14
    """
    return Expr(_f.bitmap_bit_position(_to_raw_expr(col)))


def bitmap_bucket_number(col: Expr | str) -> Expr:
    """Spark ``bitmap_bucket_number``: bucket number for a child expression.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"x": [1]})
        >>> r = df.select(
        ...     dfn.functions.spark.bitmap_bucket_number(dfn.lit(15)).alias("v"))
        >>> r.collect_column("v")[0].as_py()
        1
    """
    return Expr(_f.bitmap_bucket_number(_to_raw_expr(col)))


# ---------------------------------------------------------------------------
# Bitwise functions
# ---------------------------------------------------------------------------


def bit_get(col: Expr | str, pos: Expr | str) -> Expr:
    """Spark ``bit_get``: returns the bit (0 or 1) at ``pos``.

    A bare ``str`` ``pos`` is treated as a column name (matching pyspark),
    not a literal; pass :func:`~datafusion.lit` for a literal position.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"x": [1]})
        >>> r = df.select(
        ...     dfn.functions.spark.bit_get(dfn.lit(5), dfn.lit(0)).alias("v"))
        >>> r.collect_column("v")[0].as_py()
        1
    """
    return Expr(_f.bit_get(_to_raw_expr(col), _to_raw_expr(pos)))


def bit_count(col: Expr | str) -> Expr:
    """Spark ``bit_count``: number of bits set in the integer's binary form.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"x": [1]})
        >>> r = df.select(dfn.functions.spark.bit_count(dfn.lit(7)).alias("v"))
        >>> r.collect_column("v")[0].as_py()
        3
    """
    return Expr(_f.bit_count(_to_raw_expr(col)))


def bitwise_not(col: Expr | str) -> Expr:
    """Spark ``~``: bitwise NOT.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"x": [1]})
        >>> r = df.select(dfn.functions.spark.bitwise_not(dfn.lit(0)).alias("v"))
        >>> r.collect_column("v")[0].as_py()
        -1
    """
    return Expr(_f.bitwise_not(_to_raw_expr(col)))


def shiftleft(col: Expr | str, numBits: Expr | int) -> Expr:  # noqa: N803
    """Spark ``shiftleft``: ``value`` shifted left by ``shift`` bits.

    ``numBits`` accepts a native ``int`` or an :class:`Expr`.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"x": [1]})
        >>> r = df.select(
        ...     dfn.functions.spark.shiftleft(dfn.lit(1), 3).alias("v"))
        >>> r.collect_column("v")[0].as_py()
        8
    """
    return Expr(_f.shiftleft(_to_raw_expr(col), coerce_to_expr(numBits).expr))


def shiftright(col: Expr | str, numBits: Expr | int) -> Expr:  # noqa: N803
    """Spark ``shiftright``: arithmetic right shift.

    ``numBits`` accepts a native ``int`` or an :class:`Expr`.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"x": [1]})
        >>> r = df.select(
        ...     dfn.functions.spark.shiftright(dfn.lit(8), 2).alias("v"))
        >>> r.collect_column("v")[0].as_py()
        2
    """
    return Expr(_f.shiftright(_to_raw_expr(col), coerce_to_expr(numBits).expr))


def shiftrightunsigned(col: Expr | str, numBits: Expr | int) -> Expr:  # noqa: N803
    """Spark ``shiftrightunsigned``: logical (unsigned) right shift.

    ``numBits`` accepts a native ``int`` or an :class:`Expr`.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"x": [1]})
        >>> r = df.select(
        ...     dfn.functions.spark.shiftrightunsigned(dfn.lit(8), 2).alias("v"))
        >>> r.collect_column("v")[0].as_py()
        2
    """
    return Expr(_f.shiftrightunsigned(_to_raw_expr(col), coerce_to_expr(numBits).expr))


# ---------------------------------------------------------------------------
# Collection / Conditional / Conversion
# ---------------------------------------------------------------------------


def size(col: Expr | str) -> Expr:
    """Spark ``size``: length of an array or map.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"x": [1]})
        >>> r = df.select(
        ...     dfn.functions.spark.size(
        ...         dfn.functions.spark.array(dfn.lit(1), dfn.lit(2), dfn.lit(3))
        ...     ).alias("v")
        ... )
        >>> r.collect_column("v")[0].as_py()
        3
    """
    return Expr(_f.size(_to_raw_expr(col)))


def if_(condition: Expr | str, if_true: Expr | Any, if_false: Expr | Any) -> Expr:
    """Spark ``if``: returns ``if_true`` when ``condition`` is true, else ``if_false``.

    Exposed as ``if_`` because ``if`` is a Python keyword. ``if_true`` and
    ``if_false`` accept native Python literals or :class:`Expr`.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"x": [1]})
        >>> r = df.select(
        ...     dfn.functions.spark.if_(
        ...         dfn.lit(2) > dfn.lit(1), "big", "small"
        ...     ).alias("v")
        ... )
        >>> r.collect_column("v")[0].as_py()
        'big'
    """
    return Expr(
        _f.if_(
            _to_raw_expr(condition),
            coerce_to_expr(if_true).expr,
            coerce_to_expr(if_false).expr,
        )
    )


def spark_cast(arg: Expr | str, type_str: Expr | str) -> Expr:
    """Spark ``cast``: cast ``arg`` to the type named by ``type_str``.

    Uses Spark cast semantics (e.g. overflow returns NULL, not error).
    ``type_str`` accepts a native ``str`` or an :class:`Expr`.

    Currently only supports casting numeric values to ``"timestamp"``.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"x": [1]})
        >>> r = df.select(
        ...     dfn.functions.spark.spark_cast(
        ...         dfn.lit(1579098645), "timestamp"
        ...     ).alias("v")
        ... )
        >>> r.collect_column("v")[0].as_py().isoformat()
        '2020-01-15T14:30:45+00:00'
    """
    return Expr(_f.spark_cast(_to_raw_expr(arg), coerce_to_expr(type_str).expr))


# ---------------------------------------------------------------------------
# Datetime functions
# ---------------------------------------------------------------------------


def add_months(start: Expr | str, months: Expr | int) -> Expr:
    """Spark ``add_months``: date + N months.

    ``months`` accepts a native ``int`` or an :class:`Expr`.

    Examples:
        >>> import pyarrow as pa
        >>> from datetime import date
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"x": [1]})
        >>> d = dfn.lit(pa.scalar(date(2020, 1, 15), type=pa.date32()))
        >>> r = df.select(dfn.functions.spark.add_months(d, 2).alias("v"))
        >>> r.collect_column("v")[0].as_py()
        datetime.date(2020, 3, 15)
    """
    return Expr(_f.add_months(_to_raw_expr(start), _coerce_i32(months).expr))


def date_add(start: Expr | str, days: Expr | int) -> Expr:
    """Spark ``date_add``: date + N days.

    ``days`` accepts a native ``int`` or an :class:`Expr`.

    Examples:
        >>> import pyarrow as pa
        >>> from datetime import date
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"x": [1]})
        >>> d = dfn.lit(pa.scalar(date(2020, 1, 15), type=pa.date32()))
        >>> r = df.select(dfn.functions.spark.date_add(d, 5).alias("v"))
        >>> r.collect_column("v")[0].as_py()
        datetime.date(2020, 1, 20)
    """
    return Expr(_f.date_add(_to_raw_expr(start), _coerce_i32(days).expr))


def date_sub(start: Expr | str, days: Expr | int) -> Expr:
    """Spark ``date_sub``: date - N days.

    ``days`` accepts a native ``int`` or an :class:`Expr`.

    Examples:
        >>> import pyarrow as pa
        >>> from datetime import date
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"x": [1]})
        >>> d = dfn.lit(pa.scalar(date(2020, 1, 15), type=pa.date32()))
        >>> r = df.select(dfn.functions.spark.date_sub(d, 5).alias("v"))
        >>> r.collect_column("v")[0].as_py()
        datetime.date(2020, 1, 10)
    """
    return Expr(_f.date_sub(_to_raw_expr(start), _coerce_i32(days).expr))


def hour(col: Expr | str) -> Expr:
    """Spark ``hour``: extract hour component of a timestamp.

    Examples:
        >>> import pyarrow as pa
        >>> from datetime import datetime
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"x": [1]})
        >>> ts = dfn.lit(
        ...     pa.scalar(datetime(2020, 1, 15, 14, 30, 45),
        ...               type=pa.timestamp('us')))
        >>> r = df.select(dfn.functions.spark.hour(ts).alias("v"))
        >>> r.collect_column("v")[0].as_py()
        14
    """
    return Expr(_f.hour(_to_raw_expr(col)))


def minute(col: Expr | str) -> Expr:
    """Spark ``minute``: extract minute component of a timestamp.

    Examples:
        >>> import pyarrow as pa
        >>> from datetime import datetime
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"x": [1]})
        >>> ts = dfn.lit(
        ...     pa.scalar(datetime(2020, 1, 15, 14, 30, 45),
        ...               type=pa.timestamp('us')))
        >>> r = df.select(dfn.functions.spark.minute(ts).alias("v"))
        >>> r.collect_column("v")[0].as_py()
        30
    """
    return Expr(_f.minute(_to_raw_expr(col)))


def second(col: Expr | str) -> Expr:
    """Spark ``second``: extract second component of a timestamp.

    Examples:
        >>> import pyarrow as pa
        >>> from datetime import datetime
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"x": [1]})
        >>> ts = dfn.lit(
        ...     pa.scalar(datetime(2020, 1, 15, 14, 30, 45),
        ...               type=pa.timestamp('us')))
        >>> r = df.select(dfn.functions.spark.second(ts).alias("v"))
        >>> r.collect_column("v")[0].as_py()
        45
    """
    return Expr(_f.second(_to_raw_expr(col)))


def last_day(col: Expr | str) -> Expr:
    """Spark ``last_day``: last day of the month containing the date.

    Examples:
        >>> import pyarrow as pa
        >>> from datetime import date
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"x": [1]})
        >>> d = dfn.lit(pa.scalar(date(2020, 1, 15), type=pa.date32()))
        >>> r = df.select(dfn.functions.spark.last_day(d).alias("v"))
        >>> r.collect_column("v")[0].as_py()
        datetime.date(2020, 1, 31)
    """
    return Expr(_f.last_day(_to_raw_expr(col)))


def make_dt_interval(
    days: Expr | int | None = None,
    hours: Expr | int | None = None,
    mins: Expr | int | None = None,
    secs: Expr | float | None = None,
) -> Expr:
    """Spark ``make_dt_interval``: day-time interval from components.

    All parts are optional; omitted parts default to zero, matching pyspark.
    Integer parts accept a native ``int`` and ``secs`` accepts a ``float``,
    or any part may be an :class:`Expr`.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"x": [1]})
        >>> r = df.select(dfn.functions.spark.make_dt_interval().alias("v"))
        >>> r.collect_column("v")[0].as_py()
        datetime.timedelta(0)

        >>> r = df.select(
        ...     dfn.functions.spark.make_dt_interval(
        ...         days=1, hours=2, mins=3, secs=4.5
        ...     ).alias("v")
        ... )
        >>> r.collect_column("v")[0].as_py()
        datetime.timedelta(days=1, seconds=7384, microseconds=500000)
    """
    return Expr(
        _f.make_dt_interval(
            (_coerce_i32(days) if days is not None else _ZERO_I32).expr,
            (_coerce_i32(hours) if hours is not None else _ZERO_I32).expr,
            (_coerce_i32(mins) if mins is not None else _ZERO_I32).expr,
            (coerce_to_expr(secs) if secs is not None else Expr.literal(0.0)).expr,
        )
    )


def make_interval(
    years: Expr | int | None = None,
    months: Expr | int | None = None,
    weeks: Expr | int | None = None,
    days: Expr | int | None = None,
    hours: Expr | int | None = None,
    mins: Expr | int | None = None,
    secs: Expr | float | None = None,
) -> Expr:
    """Spark ``make_interval``: interval from year/month/week/day/hour/min/sec parts.

    All parts are optional; omitted parts default to zero, matching pyspark.
    Integer parts accept a native ``int`` and ``secs`` accepts a ``float``,
    or any part may be an :class:`Expr`.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"x": [1]})
        >>> r = df.select(dfn.functions.spark.make_interval().alias("v"))
        >>> r.collect_column("v")[0].as_py().months
        0

        >>> r = df.select(dfn.functions.spark.make_interval(years=1).alias("v"))
        >>> r.collect_column("v")[0].as_py().months
        12
    """
    return Expr(
        _f.make_interval(
            (_coerce_i32(years) if years is not None else _ZERO_I32).expr,
            (_coerce_i32(months) if months is not None else _ZERO_I32).expr,
            (_coerce_i32(weeks) if weeks is not None else _ZERO_I32).expr,
            (_coerce_i32(days) if days is not None else _ZERO_I32).expr,
            (_coerce_i32(hours) if hours is not None else _ZERO_I32).expr,
            (_coerce_i32(mins) if mins is not None else _ZERO_I32).expr,
            (coerce_to_expr(secs) if secs is not None else Expr.literal(0.0)).expr,
        )
    )


def next_day(date: Expr | str, dayOfWeek: Expr | str) -> Expr:  # noqa: N803
    """Spark ``next_day``: first date after ``start_date`` named ``day_of_week``.

    ``dayOfWeek`` accepts a native ``str`` or an :class:`Expr`.

    Examples:
        >>> import pyarrow as pa
        >>> from datetime import date
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"x": [1]})
        >>> d = dfn.lit(pa.scalar(date(2020, 1, 15), type=pa.date32()))
        >>> r = df.select(dfn.functions.spark.next_day(d, "Mon").alias("v"))
        >>> r.collect_column("v")[0].as_py()
        datetime.date(2020, 1, 20)
    """
    return Expr(_f.next_day(_to_raw_expr(date), coerce_to_expr(dayOfWeek).expr))


def date_diff(end: Expr | str, start: Expr | str) -> Expr:
    """Spark ``date_diff``: number of days from ``start_date`` to ``end_date``.

    Examples:
        >>> import pyarrow as pa
        >>> from datetime import date
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"x": [1]})
        >>> d = dfn.lit(pa.scalar(date(2020, 1, 15), type=pa.date32()))
        >>> end = dfn.lit(pa.scalar(date(2020, 1, 20), type=pa.date32()))
        >>> r = df.select(dfn.functions.spark.date_diff(end, d).alias("v"))
        >>> r.collect_column("v")[0].as_py()
        5
    """
    return Expr(_f.date_diff(_to_raw_expr(end), _to_raw_expr(start)))


def date_trunc(format: Expr | str, timestamp: Expr | str) -> Expr:
    """Spark ``date_trunc``: truncate timestamp to unit ``fmt``.

    ``format`` accepts a native ``str`` or an :class:`Expr`.

    Examples:
        >>> import pyarrow as pa
        >>> from datetime import datetime
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"x": [1]})
        >>> ts = dfn.lit(
        ...     pa.scalar(datetime(2020, 1, 15, 14, 30, 45),
        ...               type=pa.timestamp('us')))
        >>> r = df.select(
        ...     dfn.functions.spark.date_trunc("month", ts).alias("v"))
        >>> r.collect_column("v")[0].as_py()
        datetime.datetime(2020, 1, 1, 0, 0)
    """
    return Expr(_f.date_trunc(coerce_to_expr(format).expr, _to_raw_expr(timestamp)))


def time_trunc(unit: Expr | str, time: Expr | str) -> Expr:
    """Spark ``time_trunc``: truncate time value to unit ``fmt``.

    A bare ``str`` ``unit`` is treated as a column name (matching pyspark),
    not a literal; pass :func:`~datafusion.lit` for a literal unit.

    Examples:
        >>> import pyarrow as pa
        >>> from datetime import time
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"x": [1]})
        >>> t = dfn.lit(pa.scalar(time(14, 30, 45), type=pa.time64('us')))
        >>> r = df.select(
        ...     dfn.functions.spark.time_trunc(dfn.lit("hour"), t).alias("v"))
        >>> r.collect_column("v")[0].as_py()
        datetime.time(14, 0)
    """
    return Expr(_f.time_trunc(_to_raw_expr(unit), _to_raw_expr(time)))


def trunc(date: Expr | str, format: Expr | str) -> Expr:
    """Spark ``trunc``: truncate date to unit ``fmt``.

    ``format`` accepts a native ``str`` or an :class:`Expr`.

    Examples:
        >>> import pyarrow as pa
        >>> from datetime import date
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"x": [1]})
        >>> d = dfn.lit(pa.scalar(date(2020, 1, 15), type=pa.date32()))
        >>> r = df.select(dfn.functions.spark.trunc(d, "YEAR").alias("v"))
        >>> r.collect_column("v")[0].as_py()
        datetime.date(2020, 1, 1)
    """
    return Expr(_f.trunc(_to_raw_expr(date), coerce_to_expr(format).expr))


def date_part(field: Expr | str, source: Expr | str) -> Expr:
    """Spark ``date_part``: extract ``field`` from a date/time/timestamp.

    ``field`` accepts a native ``str`` or an :class:`Expr`.

    Examples:
        >>> import pyarrow as pa
        >>> from datetime import date
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"x": [1]})
        >>> d = dfn.lit(pa.scalar(date(2020, 1, 15), type=pa.date32()))
        >>> r = df.select(
        ...     dfn.functions.spark.date_part("year", d).alias("v"))
        >>> r.collect_column("v")[0].as_py()
        2020
    """
    return Expr(_f.date_part(coerce_to_expr(field).expr, _to_raw_expr(source)))


def from_utc_timestamp(timestamp: Expr | str, tz: Expr | str) -> Expr:
    """Spark ``from_utc_timestamp``: interpret ``ts`` as UTC, convert to ``tz``.

    ``tz`` accepts a native ``str`` or an :class:`Expr`.

    Examples:
        >>> import pyarrow as pa
        >>> from datetime import datetime
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"x": [1]})
        >>> ts = dfn.lit(
        ...     pa.scalar(datetime(2020, 1, 15, 14, 30, 45),
        ...               type=pa.timestamp('us')))
        >>> r = df.select(
        ...     dfn.functions.spark.from_utc_timestamp(ts, "UTC").alias("v"))
        >>> r.collect_column("v")[0].as_py()
        datetime.datetime(2020, 1, 15, 14, 30, 45)
    """
    return Expr(_f.from_utc_timestamp(_to_raw_expr(timestamp), coerce_to_expr(tz).expr))


def to_utc_timestamp(timestamp: Expr | str, tz: Expr | str) -> Expr:
    """Spark ``to_utc_timestamp``: interpret ``ts`` as ``tz``, convert to UTC.

    ``tz`` accepts a native ``str`` or an :class:`Expr`.

    Examples:
        >>> import pyarrow as pa
        >>> from datetime import datetime
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"x": [1]})
        >>> ts = dfn.lit(
        ...     pa.scalar(datetime(2020, 1, 15, 14, 30, 45),
        ...               type=pa.timestamp('us')))
        >>> r = df.select(
        ...     dfn.functions.spark.to_utc_timestamp(ts, "UTC").alias("v"))
        >>> r.collect_column("v")[0].as_py()
        datetime.datetime(2020, 1, 15, 14, 30, 45)
    """
    return Expr(_f.to_utc_timestamp(_to_raw_expr(timestamp), coerce_to_expr(tz).expr))


def unix_date(col: Expr | str) -> Expr:
    """Spark ``unix_date``: days since 1970-01-01 for ``dt``.

    Examples:
        >>> import pyarrow as pa
        >>> from datetime import date
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"x": [1]})
        >>> d = dfn.lit(pa.scalar(date(2020, 1, 15), type=pa.date32()))
        >>> r = df.select(dfn.functions.spark.unix_date(d).alias("v"))
        >>> r.collect_column("v")[0].as_py()
        18276
    """
    return Expr(_f.unix_date(_to_raw_expr(col)))


def unix_micros(col: Expr | str) -> Expr:
    """Spark ``unix_micros``: microseconds since epoch for ``ts``.

    Examples:
        >>> import pyarrow as pa
        >>> from datetime import datetime
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"x": [1]})
        >>> ts = dfn.lit(
        ...     pa.scalar(datetime(2020, 1, 15, 14, 30, 45),
        ...               type=pa.timestamp('us')))
        >>> r = df.select(dfn.functions.spark.unix_micros(ts).alias("v"))
        >>> r.collect_column("v")[0].as_py()
        1579098645000000
    """
    return Expr(_f.unix_micros(_to_raw_expr(col)))


def unix_millis(col: Expr | str) -> Expr:
    """Spark ``unix_millis``: milliseconds since epoch for ``ts``.

    Examples:
        >>> import pyarrow as pa
        >>> from datetime import datetime
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"x": [1]})
        >>> ts = dfn.lit(
        ...     pa.scalar(datetime(2020, 1, 15, 14, 30, 45),
        ...               type=pa.timestamp('us')))
        >>> r = df.select(dfn.functions.spark.unix_millis(ts).alias("v"))
        >>> r.collect_column("v")[0].as_py()
        1579098645000
    """
    return Expr(_f.unix_millis(_to_raw_expr(col)))


def unix_seconds(col: Expr | str) -> Expr:
    """Spark ``unix_seconds``: seconds since epoch for ``ts``.

    Examples:
        >>> import pyarrow as pa
        >>> from datetime import datetime
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"x": [1]})
        >>> ts = dfn.lit(
        ...     pa.scalar(datetime(2020, 1, 15, 14, 30, 45),
        ...               type=pa.timestamp('us')))
        >>> r = df.select(dfn.functions.spark.unix_seconds(ts).alias("v"))
        >>> r.collect_column("v")[0].as_py()
        1579098645
    """
    return Expr(_f.unix_seconds(_to_raw_expr(col)))


# ---------------------------------------------------------------------------
# Hash functions
# ---------------------------------------------------------------------------


def crc32(col: Expr | str) -> Expr:
    """Spark ``crc32``: cyclic redundancy check value as a bigint.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"s": ["ABC"]})
        >>> r = df.select(dfn.functions.spark.crc32("s").alias("v"))
        >>> r.collect_column("v")[0].as_py()
        2743272264
    """
    return Expr(_f.crc32(_to_raw_expr(col)))


def sha1(col: Expr | str) -> Expr:
    """Spark ``sha1``: SHA-1 hash as a hex string.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"s": ["hello"]})
        >>> r = df.select(dfn.functions.spark.sha1("s").alias("v"))
        >>> r.collect_column("v")[0].as_py()
        'aaf4c61ddcc5e8a2dabede0f3b482cd9aea9434d'
    """
    return Expr(_f.sha1(_to_raw_expr(col)))


def sha2(col: Expr | str, numBits: Expr | int) -> Expr:  # noqa: N803
    """Spark ``sha2``: SHA-2 family hash (224, 256, 384, 512). Bit length 0 = 256.

    ``numBits`` accepts a native ``int`` or an :class:`Expr`.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"s": ["hello"]})
        >>> r = df.select(dfn.functions.spark.sha2("s", 256).alias("v"))
        >>> r.collect_column("v")[0].as_py()
        '2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824'
    """
    return Expr(_f.sha2(_to_raw_expr(col), coerce_to_expr(numBits).expr))


def xxhash64(*cols: Expr | str) -> Expr:
    """Spark ``xxhash64``: 64-bit xxHash of the arguments.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"x": [1]})
        >>> r = df.select(
        ...     dfn.functions.spark.xxhash64(dfn.lit("hello")).alias("v"))
        >>> r.collect_column("v")[0].as_py()
        -4367754540140381902
    """
    return Expr(_f.xxhash64(*_to_raw_expr_list(cols)))


# ---------------------------------------------------------------------------
# JSON functions
# ---------------------------------------------------------------------------


def json_tuple(col: Expr | str, *fields: Expr | str) -> Expr:
    """Spark ``json_tuple``: extract top-level fields from a JSON string.

    Each field name accepts a native ``str`` or an :class:`Expr`.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"x": [1]})
        >>> r = df.select(
        ...     dfn.functions.spark.json_tuple(
        ...         dfn.lit('{"a":1,"b":"x"}'), "a", "b"
        ...     ).alias("v")
        ... )
        >>> r.collect_column("v")[0].as_py()
        {'c0': '1', 'c1': 'x'}
    """
    return Expr(
        _f.json_tuple(_to_raw_expr(col), *[coerce_to_expr(f).expr for f in fields])
    )


# ---------------------------------------------------------------------------
# Map functions
# ---------------------------------------------------------------------------


def map_from_arrays(col1: Expr | str, col2: Expr | str) -> Expr:
    """Spark ``map_from_arrays``: build a map from parallel key/value arrays.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"x": [1]})
        >>> keys = dfn.functions.spark.array(dfn.lit("a"), dfn.lit("b"))
        >>> vals = dfn.functions.spark.array(dfn.lit(1), dfn.lit(2))
        >>> r = df.select(
        ...     dfn.functions.spark.map_from_arrays(keys, vals).alias("v"))
        >>> r.collect_column("v")[0].as_py()
        [('a', 1), ('b', 2)]
    """
    return Expr(_f.map_from_arrays(_to_raw_expr(col1), _to_raw_expr(col2)))


def map_from_entries(col: Expr | str) -> Expr:
    """Spark ``map_from_entries``: build a map from an array of key/value structs.

    ``col`` must be an array whose elements are two-field structs; the first
    field becomes the map key and the second the value.

    Examples:
        >>> import pyarrow as pa
        >>> ctx = dfn.SessionContext()
        >>> entry_type = pa.list_(
        ...     pa.struct([("key", pa.string()), ("value", pa.int64())]))
        >>> entries = pa.array(
        ...     [[{"key": "a", "value": 1}, {"key": "b", "value": 2}]],
        ...     type=entry_type)
        >>> df = ctx.from_arrow(pa.record_batch([entries], names=["e"]))
        >>> r = df.select(dfn.functions.spark.map_from_entries("e").alias("v"))
        >>> r.collect_column("v")[0].as_py()
        [('a', 1), ('b', 2)]
    """
    return Expr(_f.map_from_entries(_to_raw_expr(col)))


def str_to_map(
    text: Expr | str,
    pairDelim: Expr | str | None = None,  # noqa: N803
    keyValueDelim: Expr | str | None = None,  # noqa: N803
) -> Expr:
    """Spark ``str_to_map``: split text into key/value pairs using delimiters.

    Delimiters default to ``","`` and ``":"`` when omitted, matching pyspark.
    Parameter names match ``pyspark.sql.functions.str_to_map``; pyspark types
    the delimiters as column-or-name, so a bare ``str`` is treated as a column
    name. Pass :func:`~datafusion.lit` for a literal delimiter.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"x": [1]})
        >>> r = df.select(
        ...     dfn.functions.spark.str_to_map(dfn.lit("a:1,b:2")).alias("v"))
        >>> r.collect_column("v")[0].as_py()
        [('a', '1'), ('b', '2')]

        >>> r = df.select(
        ...     dfn.functions.spark.str_to_map(
        ...         dfn.lit("a=1;b=2"),
        ...         pairDelim=dfn.lit(";"),
        ...         keyValueDelim=dfn.lit("="),
        ...     ).alias("v")
        ... )
        >>> r.collect_column("v")[0].as_py()
        [('a', '1'), ('b', '2')]
    """
    pd = _to_raw_expr(pairDelim) if pairDelim is not None else Expr.literal(",").expr
    kvd = (
        _to_raw_expr(keyValueDelim)
        if keyValueDelim is not None
        else Expr.literal(":").expr
    )
    return Expr(_f.str_to_map(_to_raw_expr(text), pd, kvd))


# ---------------------------------------------------------------------------
# Math functions
# ---------------------------------------------------------------------------


def abs(col: Expr | str) -> Expr:
    """Spark ``abs``: absolute value.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"x": [1]})
        >>> r = df.select(dfn.functions.spark.abs(dfn.lit(-5)).alias("v"))
        >>> r.collect_column("v")[0].as_py()
        5
    """
    return Expr(_f.abs(_to_raw_expr(col)))


def ceil(col: Expr | str) -> Expr:
    """Spark ``ceil``: smallest integer ≥ arg.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"x": [1]})
        >>> r = df.select(dfn.functions.spark.ceil(dfn.lit(1.2)).alias("v"))
        >>> r.collect_column("v")[0].as_py()
        2
    """
    return Expr(_f.ceil(_to_raw_expr(col)))


def expm1(col: Expr | str) -> Expr:
    """Spark ``expm1``: exp(arg) - 1.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"x": [1]})
        >>> r = df.select(dfn.functions.spark.expm1(dfn.lit(0.0)).alias("v"))
        >>> r.collect_column("v")[0].as_py()
        0.0
    """
    return Expr(_f.expm1(_to_raw_expr(col)))


def factorial(col: Expr | str) -> Expr:
    """Spark ``factorial``: n! for n in [0..20], else NULL.

    Examples:
        >>> import pyarrow as pa
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"x": [1]})
        >>> r = df.select(
        ...     dfn.functions.spark.factorial(
        ...         dfn.lit(pa.scalar(5, type=pa.int32()))
        ...     ).alias("v")
        ... )
        >>> r.collect_column("v")[0].as_py()
        120
    """
    return Expr(_f.factorial(_to_raw_expr(col)))


def floor(col: Expr | str) -> Expr:
    """Spark ``floor``: largest integer ≤ arg.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"x": [1]})
        >>> r = df.select(dfn.functions.spark.floor(dfn.lit(1.8)).alias("v"))
        >>> r.collect_column("v")[0].as_py()
        1
    """
    return Expr(_f.floor(_to_raw_expr(col)))


def hex(col: Expr | str) -> Expr:
    """Spark ``hex``: hexadecimal representation.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"x": [1]})
        >>> r = df.select(dfn.functions.spark.hex(dfn.lit(255)).alias("v"))
        >>> r.collect_column("v")[0].as_py()
        'FF'
    """
    return Expr(_f.hex(_to_raw_expr(col)))


def modulus(dividend: Expr | float, divisor: Expr | float) -> Expr:
    """Spark ``mod``: remainder of ``dividend / divisor`` (sign follows dividend).

    ``dividend`` and ``divisor`` accept native numbers or :class:`Expr`.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"x": [1]})
        >>> r = df.select(
        ...     dfn.functions.spark.modulus(dfn.lit(10), dfn.lit(3)).alias("v"))
        >>> r.collect_column("v")[0].as_py()
        1
    """
    return Expr(_f.modulus(coerce_to_expr(dividend).expr, coerce_to_expr(divisor).expr))


def pmod(dividend: Expr | float, divisor: Expr | float) -> Expr:
    """Spark ``pmod``: positive remainder of division.

    ``dividend`` and ``divisor`` accept native numbers or :class:`Expr`.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"x": [1]})
        >>> r = df.select(
        ...     dfn.functions.spark.pmod(dfn.lit(-1), dfn.lit(3)).alias("v"))
        >>> r.collect_column("v")[0].as_py()
        2
    """
    return Expr(_f.pmod(coerce_to_expr(dividend).expr, coerce_to_expr(divisor).expr))


def rint(col: Expr | str) -> Expr:
    """Spark ``rint``: round to nearest mathematical integer (as double).

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"x": [1]})
        >>> r = df.select(dfn.functions.spark.rint(dfn.lit(2.5)).alias("v"))
        >>> r.collect_column("v")[0].as_py()
        2.0
    """
    return Expr(_f.rint(_to_raw_expr(col)))


def round(col: Expr | str, scale: Expr | int | None = None) -> Expr:
    """Spark ``round``: round to ``scale`` decimal places, HALF_UP rounding.

    ``scale`` defaults to zero when omitted, matching pyspark, and accepts a
    native ``int`` or an :class:`Expr`.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"x": [1]})
        >>> r = df.select(dfn.functions.spark.round(dfn.lit(2.5)).alias("v"))
        >>> r.collect_column("v")[0].as_py()
        3.0

        >>> r = df.select(
        ...     dfn.functions.spark.round(dfn.lit(2.345), scale=2).alias("v"))
        >>> r.collect_column("v")[0].as_py()
        2.35
    """
    scale_expr = coerce_to_expr(scale) if scale is not None else _ZERO_I32
    return Expr(_f.round(_to_raw_expr(col), scale_expr.expr))


def unhex(col: Expr | str) -> Expr:
    r"""Spark ``unhex``: convert hexadecimal string to binary.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"x": [1]})
        >>> r = df.select(dfn.functions.spark.unhex(dfn.lit("FF")).alias("v"))
        >>> r.collect_column("v")[0].as_py()
        b'\xff'
    """
    return Expr(_f.unhex(_to_raw_expr(col)))


def width_bucket(
    v: Expr | str,
    min: Expr | str,
    max: Expr | str,
    numBucket: Expr | int,  # noqa: N803
) -> Expr:
    """Spark ``width_bucket``: bucket number for ``value`` in equi-width histogram.

    ``numBucket`` accepts a native ``int`` or an :class:`Expr`.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"x": [1]})
        >>> r = df.select(
        ...     dfn.functions.spark.width_bucket(
        ...         dfn.lit(5.0), dfn.lit(0.0), dfn.lit(10.0), 5
        ...     ).alias("v")
        ... )
        >>> r.collect_column("v")[0].as_py()
        3
    """
    return Expr(
        _f.width_bucket(
            _to_raw_expr(v),
            _to_raw_expr(min),
            _to_raw_expr(max),
            coerce_to_expr(numBucket).expr,
        )
    )


def csc(col: Expr | str) -> Expr:
    """Spark ``csc``: cosecant.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"x": [1]})
        >>> r = df.select(dfn.functions.spark.csc(dfn.lit(1.5708)).alias("v"))
        >>> f"{r.collect_column('v')[0].as_py():.4f}"
        '1.0000'
    """
    return Expr(_f.csc(_to_raw_expr(col)))


def sec(col: Expr | str) -> Expr:
    """Spark ``sec``: secant.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"x": [1]})
        >>> r = df.select(dfn.functions.spark.sec(dfn.lit(0.0)).alias("v"))
        >>> r.collect_column("v")[0].as_py()
        1.0
    """
    return Expr(_f.sec(_to_raw_expr(col)))


def negative(col: Expr | str) -> Expr:
    """Spark ``negative``: unary minus.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"x": [1]})
        >>> r = df.select(dfn.functions.spark.negative(dfn.lit(3)).alias("v"))
        >>> r.collect_column("v")[0].as_py()
        -3
    """
    return Expr(_f.negative(_to_raw_expr(col)))


def bin(col: Expr | str) -> Expr:
    """Spark ``bin``: binary string representation of a long.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"x": [1]})
        >>> r = df.select(dfn.functions.spark.bin(dfn.lit(7)).alias("v"))
        >>> r.collect_column("v")[0].as_py()
        '111'
    """
    return Expr(_f.bin(_to_raw_expr(col)))


# ---------------------------------------------------------------------------
# String functions
# ---------------------------------------------------------------------------


def ascii(col: Expr | str) -> Expr:
    """Spark ``ascii``: code point of the first character.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"x": [1]})
        >>> r = df.select(dfn.functions.spark.ascii(dfn.lit("A")).alias("v"))
        >>> r.collect_column("v")[0].as_py()
        65
    """
    return Expr(_f.ascii(_to_raw_expr(col)))


def base64(col: Expr | str) -> Expr:
    """Spark ``base64``: encode binary as a base64 string.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"x": [1]})
        >>> r = df.select(dfn.functions.spark.base64(dfn.lit(b"hi")).alias("v"))
        >>> r.collect_column("v")[0].as_py()
        'aGk='
    """
    return Expr(_f.base64(_to_raw_expr(col)))


def char(col: Expr | str) -> Expr:
    """Spark ``char``: ASCII character for a code point (mod 256).

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"x": [1]})
        >>> r = df.select(dfn.functions.spark.char(dfn.lit(65)).alias("v"))
        >>> r.collect_column("v")[0].as_py()
        'A'
    """
    return Expr(_f.char(_to_raw_expr(col)))


def concat(*cols: Expr | str) -> Expr:
    """Spark ``concat``: concatenates strings; NULL if any input is NULL.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"x": [1]})
        >>> r = df.select(
        ...     dfn.functions.spark.concat(dfn.lit("a"), dfn.lit("b")).alias("v"))
        >>> r.collect_column("v")[0].as_py()
        'ab'
    """
    return Expr(_f.concat(*_to_raw_expr_list(cols)))


def elt(*inputs: Expr | str) -> Expr:
    """Spark ``elt``: returns the n-th input (1-indexed).

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"x": [1]})
        >>> r = df.select(
        ...     dfn.functions.spark.elt(
        ...         dfn.lit(2), dfn.lit("a"), dfn.lit("b")
        ...     ).alias("v")
        ... )
        >>> r.collect_column("v")[0].as_py()
        'b'
    """
    return Expr(_f.elt(*_to_raw_expr_list(inputs)))


def ilike(
    str: Expr | str,
    pattern: Expr | str,
    escapeChar: str | None = None,  # noqa: N803
) -> Expr:
    """Spark ``ilike``: case-insensitive pattern match.

    A bare ``str`` ``pattern`` is treated as a column name (matching pyspark),
    not a literal; pass :func:`~datafusion.lit` for a literal pattern.
    ``escapeChar`` is accepted for pyspark parity but is not yet wired through
    the Rust binding; passing a non-``None`` value raises ``NotImplementedError``.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"x": [1]})
        >>> r = df.select(
        ...     dfn.functions.spark.ilike(dfn.lit("HELLO"), dfn.lit("h%")).alias("v"))
        >>> r.collect_column("v")[0].as_py()
        True
    """
    if escapeChar is not None:
        msg = "ilike(escapeChar=...) is not yet supported by the Spark UDF binding"
        raise NotImplementedError(msg)
    return Expr(_f.ilike(_to_raw_expr(str), _to_raw_expr(pattern)))


def length(col: Expr | str) -> Expr:
    """Spark ``length``: character length of a string, or byte length of binary.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"x": [1]})
        >>> r = df.select(dfn.functions.spark.length(dfn.lit("hello")).alias("v"))
        >>> r.collect_column("v")[0].as_py()
        5
    """
    return Expr(_f.length(_to_raw_expr(col)))


def like(
    str: Expr | str,
    pattern: Expr | str,
    escapeChar: str | None = None,  # noqa: N803
) -> Expr:
    """Spark ``like``: case-sensitive pattern match.

    A bare ``str`` ``pattern`` is treated as a column name (matching pyspark),
    not a literal; pass :func:`~datafusion.lit` for a literal pattern.
    ``escapeChar`` is accepted for pyspark parity but is not yet wired through
    the Rust binding; passing a non-``None`` value raises ``NotImplementedError``.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"x": [1]})
        >>> r = df.select(
        ...     dfn.functions.spark.like(dfn.lit("hello"), dfn.lit("h%")).alias("v"))
        >>> r.collect_column("v")[0].as_py()
        True
    """
    if escapeChar is not None:
        msg = "like(escapeChar=...) is not yet supported by the Spark UDF binding"
        raise NotImplementedError(msg)
    return Expr(_f.like(_to_raw_expr(str), _to_raw_expr(pattern)))


def luhn_check(col: Expr | str) -> Expr:
    """Spark ``luhn_check``: true if the digit string passes the Luhn check.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"x": [1]})
        >>> r = df.select(
        ...     dfn.functions.spark.luhn_check(
        ...         dfn.lit("4111111111111111")
        ...     ).alias("v")
        ... )
        >>> r.collect_column("v")[0].as_py()
        True
    """
    return Expr(_f.luhn_check(_to_raw_expr(col)))


def format_string(format: str | Expr, *cols: Expr | str) -> Expr:
    """Spark ``format_string``: printf-style format string.

    ``format`` is the printf-style template (a plain ``str`` is auto-promoted
    to a literal expression); remaining args are values to substitute.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"x": [1]})
        >>> r = df.select(
        ...     dfn.functions.spark.format_string(
        ...         "%d-%s", dfn.lit(42), dfn.lit("hi")
        ...     ).alias("v")
        ... )
        >>> r.collect_column("v")[0].as_py()
        '42-hi'
    """
    fmt_expr = format if isinstance(format, Expr) else Expr.literal(format)
    return Expr(_f.format_string(fmt_expr.expr, *_to_raw_expr_list(cols)))


def space(col: Expr | int) -> Expr:
    """Spark ``space``: string of n spaces.

    ``col`` accepts a native ``int`` or an :class:`Expr`.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"x": [1]})
        >>> r = df.select(dfn.functions.spark.space(3).alias("v"))
        >>> r.collect_column("v")[0].as_py()
        '   '
    """
    return Expr(_f.space(_coerce_i32(col).expr))


def substring(str: Expr | str, pos: Expr | int, len: Expr | int) -> Expr:
    """Spark ``substring``: 1-indexed substring starting at ``pos`` of given ``length``.

    Negative ``pos`` counts from the end. ``pos`` and ``len`` accept native
    ``int`` values or :class:`Expr`.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"x": [1]})
        >>> r = df.select(
        ...     dfn.functions.spark.substring(dfn.lit("hello"), 1, 3).alias("v"))
        >>> r.collect_column("v")[0].as_py()
        'hel'
    """
    return Expr(
        _f.substring(
            _to_raw_expr(str), coerce_to_expr(pos).expr, coerce_to_expr(len).expr
        )
    )


def unbase64(col: Expr | str) -> Expr:
    """Spark ``unbase64``: decode a base64 string to binary.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"x": [1]})
        >>> r = df.select(
        ...     dfn.functions.spark.unbase64(dfn.lit("aGk=")).alias("v"))
        >>> r.collect_column("v")[0].as_py()
        b'hi'
    """
    return Expr(_f.unbase64(_to_raw_expr(col)))


def soundex(col: Expr | str) -> Expr:
    """Spark ``soundex``: Soundex phonetic code.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"x": [1]})
        >>> r = df.select(dfn.functions.spark.soundex(dfn.lit("Robert")).alias("v"))
        >>> r.collect_column("v")[0].as_py()
        'R163'
    """
    return Expr(_f.soundex(_to_raw_expr(col)))


def is_valid_utf8(str: Expr | str) -> Expr:
    """Spark ``is_valid_utf8``: true if the string is valid UTF-8.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"x": [1]})
        >>> r = df.select(
        ...     dfn.functions.spark.is_valid_utf8(dfn.lit("hello")).alias("v"))
        >>> r.collect_column("v")[0].as_py()
        True
    """
    return Expr(_f.is_valid_utf8(_to_raw_expr(str)))


def make_valid_utf8(str: Expr | str) -> Expr:
    """Spark ``make_valid_utf8``: replace invalid UTF-8 bytes with U+FFFD.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"x": [1]})
        >>> r = df.select(
        ...     dfn.functions.spark.make_valid_utf8(dfn.lit("hello")).alias("v"))
        >>> r.collect_column("v")[0].as_py()
        'hello'
    """
    return Expr(_f.make_valid_utf8(_to_raw_expr(str)))


# ---------------------------------------------------------------------------
# URL functions
# ---------------------------------------------------------------------------


def parse_url(
    url: Expr | str,
    partToExtract: Expr | str,  # noqa: N803
    key: Expr | str | None = None,
) -> Expr:
    """Spark ``parse_url``: extract a part from a URL; errors on invalid URLs.

    ``partToExtract`` is one of ``"HOST"``, ``"PATH"``, ``"QUERY"``,
    ``"REF"``, ``"PROTOCOL"``, ``"FILE"``, ``"AUTHORITY"``, ``"USERINFO"``.
    Pass ``key`` only with ``"QUERY"`` to extract a single parameter. Bare
    ``str`` values for ``partToExtract``/``key`` are treated as column names
    (matching pyspark); pass :func:`~datafusion.lit` for a literal.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"x": [1]})
        >>> r = df.select(
        ...     dfn.functions.spark.parse_url(
        ...         dfn.lit("http://example.com/path?q=1"), dfn.lit("HOST")
        ...     ).alias("v")
        ... )
        >>> r.collect_column("v")[0].as_py()
        'example.com'

        >>> r = df.select(
        ...     dfn.functions.spark.parse_url(
        ...         dfn.lit("http://example.com/path?q=1"),
        ...         dfn.lit("QUERY"),
        ...         key=dfn.lit("q"),
        ...     ).alias("v")
        ... )
        >>> r.collect_column("v")[0].as_py()
        '1'
    """
    if key is None:
        return Expr(_f.parse_url(_to_raw_expr(url), _to_raw_expr(partToExtract)))
    return Expr(
        _f.parse_url(_to_raw_expr(url), _to_raw_expr(partToExtract), _to_raw_expr(key))
    )


def try_parse_url(
    url: Expr | str,
    partToExtract: Expr | str,  # noqa: N803
    key: Expr | str | None = None,
) -> Expr:
    """Spark ``try_parse_url``: like ``parse_url`` but returns NULL on invalid URLs.

    Bare ``str`` values for ``partToExtract``/``key`` are treated as column
    names (matching pyspark); pass :func:`~datafusion.lit` for a literal.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"x": [1]})
        >>> r = df.select(
        ...     dfn.functions.spark.try_parse_url(
        ...         dfn.lit("http://example.com/"), dfn.lit("HOST")
        ...     ).alias("v")
        ... )
        >>> r.collect_column("v")[0].as_py()
        'example.com'
    """
    if key is None:
        return Expr(_f.try_parse_url(_to_raw_expr(url), _to_raw_expr(partToExtract)))
    return Expr(
        _f.try_parse_url(
            _to_raw_expr(url), _to_raw_expr(partToExtract), _to_raw_expr(key)
        )
    )


def url_decode(str: Expr | str) -> Expr:
    """Spark ``url_decode``: decode an application/x-www-form-urlencoded string.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"x": [1]})
        >>> r = df.select(
        ...     dfn.functions.spark.url_decode(dfn.lit("a%20b")).alias("v"))
        >>> r.collect_column("v")[0].as_py()
        'a b'
    """
    return Expr(_f.url_decode(_to_raw_expr(str)))


def try_url_decode(str: Expr | str) -> Expr:
    """Spark ``try_url_decode``: like ``url_decode``; returns NULL on invalid input.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"x": [1]})
        >>> r = df.select(
        ...     dfn.functions.spark.try_url_decode(dfn.lit("a%20b")).alias("v"))
        >>> r.collect_column("v")[0].as_py()
        'a b'
    """
    return Expr(_f.try_url_decode(_to_raw_expr(str)))


def url_encode(str: Expr | str) -> Expr:
    """Spark ``url_encode``: encode a string in application/x-www-form-urlencoded.

    Examples:
        >>> ctx = dfn.SessionContext()
        >>> df = ctx.from_pydict({"x": [1]})
        >>> r = df.select(
        ...     dfn.functions.spark.url_encode(dfn.lit("a b")).alias("v"))
        >>> r.collect_column("v")[0].as_py()
        'a+b'
    """
    return Expr(_f.url_encode(_to_raw_expr(str)))


__all__ = [
    # Math
    "abs",
    # Datetime
    "add_months",
    "array",
    # Array
    "array_contains",
    "array_repeat",
    # String
    "ascii",
    # Aggregate
    "avg",
    "base64",
    "bin",
    "bit_count",
    # Bitwise
    "bit_get",
    "bitmap_bit_position",
    "bitmap_bucket_number",
    # Bitmap
    "bitmap_count",
    "bitwise_not",
    "ceil",
    "char",
    "collect_list",
    "collect_set",
    "concat",
    # Hash
    "crc32",
    "csc",
    "date_add",
    "date_diff",
    "date_part",
    "date_sub",
    "date_trunc",
    "elt",
    "expm1",
    "factorial",
    "floor",
    "format_string",
    "from_utc_timestamp",
    "hex",
    "hour",
    "if_",
    "ilike",
    "is_valid_utf8",
    # JSON
    "json_tuple",
    "last_day",
    "length",
    "like",
    "luhn_check",
    "make_dt_interval",
    "make_interval",
    "make_valid_utf8",
    # Map
    "map_from_arrays",
    "map_from_entries",
    "minute",
    "modulus",
    "negative",
    "next_day",
    # URL
    "parse_url",
    "pmod",
    "rint",
    "round",
    "sec",
    "second",
    "sha1",
    "sha2",
    "shiftleft",
    "shiftright",
    "shiftrightunsigned",
    "shuffle",
    # Collection / Conditional / Conversion
    "size",
    "slice",
    "soundex",
    "space",
    "spark_cast",
    "str_to_map",
    "substring",
    "time_trunc",
    "to_utc_timestamp",
    "trunc",
    "try_parse_url",
    "try_sum",
    "try_url_decode",
    "unbase64",
    "unhex",
    "unix_date",
    "unix_micros",
    "unix_millis",
    "unix_seconds",
    "url_decode",
    "url_encode",
    "width_bucket",
    "xxhash64",
]
