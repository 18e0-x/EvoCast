import os
import re

import pandas as pd

FREQ_MAP = {
    "Y": "yearly",
    "A": "yearly",
    "A-DEC": "yearly",
    "A-JAN": "yearly",
    "A-FEB": "yearly",
    "A-MAR": "yearly",
    "A-APR": "yearly",
    "A-MAY": "yearly",
    "A-JUN": "yearly",
    "A-JUL": "yearly",
    "A-AUG": "yearly",
    "A-SEP": "yearly",
    "A-OCT": "yearly",
    "A-NOV": "yearly",
    "AS-DEC": "yearly",
    "AS-JAN": "yearly",
    "AS-FEB": "yearly",
    "AS-MAR": "yearly",
    "AS-APR": "yearly",
    "AS-MAY": "yearly",
    "AS-JUN": "yearly",
    "AS-JUL": "yearly",
    "AS-AUG": "yearly",
    "AS-SEP": "yearly",
    "AS-OCT": "yearly",
    "AS-NOV": "yearly",
    "BA-DEC": "yearly",
    "BA-JAN": "yearly",
    "BA-FEB": "yearly",
    "BA-MAR": "yearly",
    "BA-APR": "yearly",
    "BA-MAY": "yearly",
    "BA-JUN": "yearly",
    "BA-JUL": "yearly",
    "BA-AUG": "yearly",
    "BA-SEP": "yearly",
    "BA-OCT": "yearly",
    "BA-NOV": "yearly",
    "BAS-DEC": "yearly",
    "BAS-JAN": "yearly",
    "BAS-FEB": "yearly",
    "BAS-MAR": "yearly",
    "BAS-APR": "yearly",
    "BAS-MAY": "yearly",
    "BAS-JUN": "yearly",
    "BAS-JUL": "yearly",
    "BAS-AUG": "yearly",
    "BAS-SEP": "yearly",
    "BAS-OCT": "yearly",
    "BAS-NOV": "yearly",
    "Q": "quarterly",
    "Q-DEC": "quarterly",
    "Q-JAN": "quarterly",
    "Q-FEB": "quarterly",
    "Q-MAR": "quarterly",
    "Q-APR": "quarterly",
    "Q-MAY": "quarterly",
    "Q-JUN": "quarterly",
    "Q-JUL": "quarterly",
    "Q-AUG": "quarterly",
    "Q-SEP": "quarterly",
    "Q-OCT": "quarterly",
    "Q-NOV": "quarterly",
    "QS-DEC": "quarterly",
    "QS-JAN": "quarterly",
    "QS-FEB": "quarterly",
    "QS-MAR": "quarterly",
    "QS-APR": "quarterly",
    "QS-MAY": "quarterly",
    "QS-JUN": "quarterly",
    "QS-JUL": "quarterly",
    "QS-AUG": "quarterly",
    "QS-SEP": "quarterly",
    "QS-OCT": "quarterly",
    "QS-NOV": "quarterly",
    "BQ-DEC": "quarterly",
    "BQ-JAN": "quarterly",
    "BQ-FEB": "quarterly",
    "BQ-MAR": "quarterly",
    "BQ-APR": "quarterly",
    "BQ-MAY": "quarterly",
    "BQ-JUN": "quarterly",
    "BQ-JUL": "quarterly",
    "BQ-AUG": "quarterly",
    "BQ-SEP": "quarterly",
    "BQ-OCT": "quarterly",
    "BQ-NOV": "quarterly",
    "BQS-DEC": "quarterly",
    "BQS-JAN": "quarterly",
    "BQS-FEB": "quarterly",
    "BQS-MAR": "quarterly",
    "BQS-APR": "quarterly",
    "BQS-MAY": "quarterly",
    "BQS-JUN": "quarterly",
    "BQS-JUL": "quarterly",
    "BQS-AUG": "quarterly",
    "BQS-SEP": "quarterly",
    "BQS-OCT": "quarterly",
    "BQS-NOV": "quarterly",
    "M": "monthly",
    "BM": "monthly",
    "CBM": "monthly",
    "MS": "monthly",
    "BMS": "monthly",
    "CBMS": "monthly",
    "W": "weekly",
    "W-SUN": "weekly",
    "W-MON": "weekly",
    "W-TUE": "weekly",
    "W-WED": "weekly",
    "W-THU": "weekly",
    "W-FRI": "weekly",
    "W-SAT": "weekly",
    "D": "daily",
    "B": "daily",
    "C": "daily",
    "H": "hourly",
    "UNKNOWN": "other",
}


def _parse_datetime_series(values: pd.Series) -> pd.Series:
    """Parse common timestamp formats with a day-first fallback.

    Some user datasets use `dd/mm/YYYY HH:MM` while benchmark datasets often use
    `mm/dd/YYYY HH:MM` or ISO-like formats. We first try pandas' default parser,
    then retry with `dayfirst=True` if needed.
    """
    slash_preference = _infer_slash_dayfirst_preference(values)
    if slash_preference is True:
        return pd.to_datetime(values, errors="coerce", dayfirst=True)
    if slash_preference is False:
        return pd.to_datetime(values, errors="coerce", dayfirst=False)

    default_parsed = pd.to_datetime(values, errors="coerce", dayfirst=False)
    dayfirst_parsed = pd.to_datetime(values, errors="coerce", dayfirst=True)

    default_valid = int(default_parsed.notna().sum())
    dayfirst_valid = int(dayfirst_parsed.notna().sum())

    if dayfirst_valid > default_valid:
        return dayfirst_parsed
    if default_valid > dayfirst_valid:
        return default_parsed

    def _delta_irregularity(parsed: pd.Series) -> float:
        deltas = parsed.dropna().diff().dropna().dt.total_seconds()
        deltas = deltas[deltas > 0]
        if deltas.empty:
            return float("inf")
        median = float(deltas.median())
        if median <= 0:
            return float("inf")
        return float(deltas.max()) / median

    if _looks_like_ambiguous_slash_datetime(values):
        default_irregularity = _delta_irregularity(default_parsed)
        dayfirst_irregularity = _delta_irregularity(dayfirst_parsed)
        if dayfirst_irregularity < default_irregularity:
            return dayfirst_parsed

    return default_parsed


def _infer_slash_dayfirst_preference(values: pd.Series, sample_size: int = 4096):
    sample = values.dropna().astype(str).head(sample_size)
    saw_ambiguous = False

    for value in sample:
        match = re.match(
            r"^\s*(\d{1,2})/(\d{1,2})/(\d{4})(?:\s+\d{1,2}:\d{2}(?::\d{2})?)?\s*$",
            value,
        )
        if not match:
            continue
        first = int(match.group(1))
        second = int(match.group(2))
        if first > 12 and second <= 12:
            return True
        if second > 12 and first <= 12:
            return False
        if first <= 12 and second <= 12:
            saw_ambiguous = True

    return None if saw_ambiguous else None


def _looks_like_ambiguous_slash_datetime(values: pd.Series, sample_size: int = 64) -> bool:
    sample = values.dropna().astype(str).head(sample_size)
    if sample.empty:
        return False
    matched = sample.str.match(r"^\d{1,2}/\d{1,2}/\d{4}(?:\s+\d{1,2}:\d{2}(?::\d{2})?)?$")
    return bool(matched.any())


def _coerce_wide_numeric_columns(data: pd.DataFrame) -> pd.DataFrame:
    """Keep only forecasting-safe numeric columns from a wide-format frame.

    User CSVs may contain descriptive text columns alongside numeric signals.
    The forecasting stack expects a purely numeric matrix, so object/category
    columns must not flow downstream as-is.
    """
    numeric_data = pd.DataFrame(index=data.index)

    for col in data.columns:
        series = data[col]
        if pd.api.types.is_numeric_dtype(series):
            numeric_data[col] = series
            continue

        normalized = series
        if pd.api.types.is_string_dtype(series) or series.dtype == object:
            normalized = series.astype(str).str.replace(",", "", regex=False).str.strip()
            normalized = normalized.mask(series.isna(), other=pd.NA)
            normalized = normalized.replace({"": pd.NA})

        converted = pd.to_numeric(normalized, errors="coerce")
        non_empty_mask = normalized.notna() if hasattr(normalized, "notna") else series.notna()
        if non_empty_mask.any() and converted[non_empty_mask].notna().all():
            numeric_data[col] = converted

    if numeric_data.shape[1] == 0:
        raise ValueError(
            "No numeric forecasting columns remain after dropping non-numeric columns "
            f"from wide-format CSV: {list(data.columns)}"
        )

    return numeric_data


def read_data(path: str, nrows=None) -> pd.DataFrame:
    """
    Read the data file and return DataFrame.
    According to the provided file path, read the data file and return the corresponding DataFrame.
    :param path: The path to the data file.
    :return:  The DataFrame of the content of the data file.
    """
    data = pd.read_csv(path)

    # Wide-format CSV (no "cols" column): already in target shape.
    if "cols" not in data.columns:
        if data.columns[0] == "date":
            data["date"] = _parse_datetime_series(data["date"])
            data.set_index("date", inplace=True)
            data.sort_index(inplace=True)
        data = _coerce_wide_numeric_columns(data)
        if nrows is not None and isinstance(nrows, int) and data.shape[0] >= nrows:
            data = data.iloc[:nrows, :]
        return data

    label_exists = "label" in data["cols"].values

    all_points = data.shape[0]

    columns = data.columns

    if columns[0] == "date":
        n_points = data.iloc[:, 2].value_counts().max()
    else:
        n_points = data.iloc[:, 1].value_counts().max()

    is_univariate = n_points == all_points

    n_cols = all_points // n_points
    df = pd.DataFrame()

    cols_name = data["cols"].unique()

    if columns[0] == "date" and not is_univariate:
        df["date"] = data.iloc[:n_points, 0]
        col_data = {
            cols_name[j]: data.iloc[j * n_points : (j + 1) * n_points, 1].tolist()
            for j in range(n_cols)
        }
        df = pd.concat([df, pd.DataFrame(col_data)], axis=1)
        df["date"] = _parse_datetime_series(df["date"])
        df.set_index("date", inplace=True)
        df.sort_index(inplace=True)

    elif columns[0] != "date" and not is_univariate:
        col_data = {
            cols_name[j]: data.iloc[j * n_points : (j + 1) * n_points, 0].tolist()
            for j in range(n_cols)
        }
        df = pd.concat([df, pd.DataFrame(col_data)], axis=1)

    elif columns[0] == "date" and is_univariate:
        df["date"] = data.iloc[:, 0]
        df[cols_name[0]] = data.iloc[:, 1]

        df["date"] = _parse_datetime_series(df["date"])
        df.set_index("date", inplace=True)
        df.sort_index(inplace=True)

    else:
        df[cols_name[0]] = data.iloc[:, 0]

    if label_exists:
        # Get the column name of the last column
        last_col_name = df.columns[-1]
        # Renaming the last column as "label"
        df.rename(columns={last_col_name: "label"}, inplace=True)

    if nrows is not None and isinstance(nrows, int) and df.shape[0] >= nrows:
        df = df.iloc[:nrows, :]

    return df


def load_series_info(file_path: str) -> dict:
    """
    get series info
    :param file_path: series file path
    :return: series info
    :rtype: dict
    """
    data = read_data(file_path)
    file_name = os.path.basename(file_path)
    freq = "other"
    if isinstance(data.index, pd.DatetimeIndex):
        try:
            inferred = pd.infer_freq(data.index)
        except ValueError:
            inferred = None
        if inferred is not None:
            freq = FREQ_MAP.get(inferred, "other")
        elif len(data.index) >= 2:
            deltas = data.index.to_series().diff().dropna()
            positive_deltas = deltas[deltas > pd.Timedelta(0)]
            if not positive_deltas.empty:
                median_delta = positive_deltas.median()
                seconds = float(median_delta.total_seconds())
                day_seconds = 24 * 60 * 60
                if seconds < day_seconds:
                    freq = "hourly"
                elif seconds < 7 * day_seconds:
                    freq = "daily"
                elif seconds < 28 * day_seconds:
                    freq = "weekly"
                elif seconds < 92 * day_seconds:
                    freq = "monthly"
                elif seconds < 366 * day_seconds:
                    freq = "quarterly"
                else:
                    freq = "yearly"
    if_univariate = data.shape[1] == 1
    return {
        "file_name": file_name,
        "freq": freq,
        "if_univariate": if_univariate,
        "size": "user",
        "length": data.shape[0],
        "trend": "",
        "seasonal": "",
        "stationary": "",
        "transition": "",
        "shifting": "",
        "correlation": "",
    }
