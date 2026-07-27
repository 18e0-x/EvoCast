from __future__ import annotations

import re
from typing import Optional

from pandas.tseries.frequencies import to_offset
from ts_benchmark.data.utils import FREQ_MAP as TFB_FREQ_MAP


_HUMAN_FREQUENCY_ALIASES = {
    "secondly": "1s",
    "sec": "1s",
    "second": "1s",
    "seconds": "1s",
    "s": "1s",
    "minutely": "1min",
    "minute": "1min",
    "minutes": "1min",
    "min": "1min",
    "t": "1min",
    "hourly": "1h",
    "hour": "1h",
    "hours": "1h",
    "h": "1h",
    "daily": "1d",
    "day": "1d",
    "days": "1d",
    "d": "1d",
    "weekly": "1w",
    "week": "1w",
    "weeks": "1w",
    "w": "1w",
    "monthly": "1mo",
    "month": "1mo",
    "months": "1mo",
    "m": "1mo",
    "quarterly": "1q",
    "quarter": "1q",
    "quarters": "1q",
    "q": "1q",
    "yearly": "1y",
    "year": "1y",
    "years": "1y",
    "y": "1y",
}

_FIXED_SUFFIX_SECONDS = {
    "s": 1,
    "sec": 1,
    "secs": 1,
    "second": 1,
    "seconds": 1,
    "t": 60,
    "min": 60,
    "mins": 60,
    "minute": 60,
    "minutes": 60,
    "h": 3600,
    "hr": 3600,
    "hrs": 3600,
    "hour": 3600,
    "hours": 3600,
    "d": 86400,
    "day": 86400,
    "days": 86400,
    "w": 604800,
    "week": 604800,
    "weeks": 604800,
}

_MONTHLIKE_SECONDS = 30 * 86400
_QUARTERLIKE_SECONDS = 91 * 86400
_YEARLIKE_SECONDS = 365 * 86400
_CANONICAL_LABELS_BY_SECONDS = {
    60: "minutely",
    3600: "hourly",
    86400: "daily",
    604800: "weekly",
}
_PANDAS_ALIAS_TO_CANONICAL = {
    str(alias).upper(): normalize
    for alias, normalize in TFB_FREQ_MAP.items()
}
_PANDAS_ALIAS_TO_CANONICAL.update({
    "YE": "yearly",
    "YE-DEC": "yearly",
    "YS": "yearly",
    "YS-JAN": "yearly",
    "QE": "quarterly",
    "QE-DEC": "quarterly",
    "BQE": "quarterly",
    "BQE-DEC": "quarterly",
    "ME": "monthly",
    "BME": "monthly",
    "SME": "monthly",
    "CBME": "monthly",
})


def normalize_frequency_label(value: Optional[str]) -> Optional[str]:
    text = str(value or "").strip()
    if not text:
        return None

    lowered = text.lower()
    if lowered == "other":
        return "other"
    if lowered in _HUMAN_FREQUENCY_ALIASES:
        return _HUMAN_FREQUENCY_ALIASES[lowered]

    alias_label = canonical_frequency_label_from_pandas_alias(text)
    if alias_label:
        return alias_label

    parsed_seconds = parse_frequency_seconds(text)
    if parsed_seconds is not None:
        return frequency_label_from_seconds(parsed_seconds)

    return lowered


def parse_frequency_seconds(value: Optional[str]) -> Optional[int]:
    text = str(value or "").strip()
    if not text:
        return None

    lowered = text.lower()
    if lowered in {"1mo", "monthly"}:
        return None
    if lowered in {"1q", "quarterly"}:
        return None
    if lowered in {"1y", "yearly"}:
        return None
    if lowered in _HUMAN_FREQUENCY_ALIASES:
        lowered = _HUMAN_FREQUENCY_ALIASES[lowered]
        if lowered in {"1mo", "1q", "1y"}:
            return None

    match = re.fullmatch(r"(\d+)\s*([a-zA-Z]+)", lowered)
    if match:
        count = int(match.group(1))
        unit = match.group(2)
        if unit in _FIXED_SUFFIX_SECONDS:
            return count * _FIXED_SUFFIX_SECONDS[unit]

    offset = _safe_to_offset(text)
    if offset is None:
        return None
    try:
        nanos = offset.nanos
    except ValueError:
        return None
    if nanos <= 0:
        return None
    return int(nanos // 1_000_000_000)


def frequency_label_from_seconds(seconds: int) -> str:
    if seconds <= 0:
        return "other"
    if seconds in _CANONICAL_LABELS_BY_SECONDS:
        return _CANONICAL_LABELS_BY_SECONDS[seconds]
    if seconds < 60:
        return f"{seconds}s"
    if seconds % 60 == 0 and seconds < 3600:
        minutes = seconds // 60
        return f"{minutes}min"
    if seconds % 3600 == 0 and seconds < 86400:
        hours = seconds // 3600
        return f"{hours}h"
    if seconds % 86400 == 0 and seconds < 604800:
        days = seconds // 86400
        return f"{days}d"
    if seconds % 604800 == 0:
        weeks = seconds // 604800
        return f"{weeks}w"
    return f"{seconds}s"


def resolve_mase_seasonality_generic(value: Optional[str]) -> int:
    label = normalize_frequency_label(value)
    if not label or label == "other":
        return 1

    seconds = parse_frequency_seconds(label)
    if seconds is not None:
        if seconds < 86400:
            return max(1, int(round(86400 / seconds)))
        if seconds < 604800:
            return max(1, int(round(604800 / seconds)))
        if seconds < _MONTHLIKE_SECONDS:
            return 1
        if seconds < _QUARTERLIKE_SECONDS:
            return 12
        if seconds < _YEARLIKE_SECONDS:
            return 4
        return 1

    if label.endswith("mo"):
        return 12
    if label.endswith("q"):
        return 4
    if label.endswith("y"):
        return 1
    return 1


def canonical_frequency_label_from_pandas_alias(alias: Optional[str]) -> Optional[str]:
    text = str(alias or "").strip()
    if not text:
        return None
    normalized = text.upper()
    if normalized in _PANDAS_ALIAS_TO_CANONICAL:
        return _PANDAS_ALIAS_TO_CANONICAL[normalized]
    base = normalized.split("-", 1)[0]
    if base in _PANDAS_ALIAS_TO_CANONICAL:
        return _PANDAS_ALIAS_TO_CANONICAL[base]

    seconds = parse_frequency_seconds(text)
    if seconds is not None:
        return frequency_label_from_seconds(seconds)
    return None


def resolve_tfb_runtime_freq_token(value: Optional[str]) -> str:
    label = normalize_frequency_label(value)
    seconds = parse_frequency_seconds(label)
    if seconds is not None:
        if seconds < 60:
            return "s"
        if seconds < 3600:
            return "t"
        if seconds < 86400:
            return "h"
        if seconds < 604800:
            return "d"
        return "w"

    if label in {"monthly", "1mo", "quarterly", "1q"}:
        return "m"
    if label in {"yearly", "1y"}:
        return "a"
    return "h"


def is_subdaily_frequency(value: Optional[str]) -> bool:
    seconds = parse_frequency_seconds(value)
    return seconds is not None and seconds < 86400


def describe_horizon_span(horizon: int, frequency: Optional[str]) -> Optional[str]:
    seconds = parse_frequency_seconds(frequency)
    if seconds is None or horizon <= 0:
        return None
    total_seconds = horizon * seconds
    return _humanize_duration(total_seconds)


def _humanize_duration(total_seconds: int) -> str:
    if total_seconds % _YEARLIKE_SECONDS == 0:
        years = total_seconds / _YEARLIKE_SECONDS
        return f"{years:.2f} years"
    if total_seconds % _MONTHLIKE_SECONDS == 0:
        months = total_seconds / _MONTHLIKE_SECONDS
        return f"{months:.2f} months"
    if total_seconds % 604800 == 0:
        weeks = total_seconds / 604800
        return f"{weeks:.2f} weeks"
    if total_seconds % 86400 == 0:
        days = total_seconds / 86400
        return f"{days:.2f} days"
    if total_seconds % 3600 == 0:
        hours = total_seconds / 3600
        return f"{hours:.2f} hours"
    if total_seconds % 60 == 0:
        minutes = total_seconds / 60
        return f"{minutes:.2f} minutes"
    return f"{total_seconds:.2f} seconds"


def _safe_to_offset(value: str):
    try:
        return to_offset(value)
    except (TypeError, ValueError):
        return None
