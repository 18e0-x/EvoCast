# Standard library imports
import os
import warnings
from typing import Dict, Iterable, List, Optional, Tuple

# Third-party imports
import numpy as np
import pandas as pd
from scipy.signal import argrelextrema
from scipy.stats import entropy, kurtosis, norm, skew
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.stl._stl import STL

warnings.filterwarnings("ignore")

DEFAULT_PERIODS = [4, 7, 12, 24, 48, 52, 96, 144, 168, 336, 672, 1008, 1440]
TFB_CHARACTERISTIC_COLUMNS = [
    "Correlation",
    "Transition",
    "Shifting",
    "Seasonality",
    "Trend",
    "Stationarity",
    "Short_term_jsd",
    "Long_term_jsd",
]


class TimeSeriesFeatureExtractor:
    def read_data(self, path: str, nrows: Optional[int] = None) -> pd.DataFrame:
        data = pd.read_csv(path)
        lowered = [str(col).strip().lower() for col in data.columns]

        if "cols" in lowered:
            cols_col = data.columns[lowered.index("cols")]
            value_col = self._value_column(data, lowered)
            time_col = self._time_column(data, lowered)
            frame = data.copy()
            if time_col:
                frame[time_col] = pd.to_datetime(frame[time_col], errors="coerce")
                frame = frame.dropna(subset=[time_col])
                pivot = frame.pivot_table(index=time_col, columns=cols_col, values=value_col, aggfunc="first")
                pivot = pivot.sort_index()
            else:
                frame["_order"] = frame.groupby(cols_col).cumcount()
                pivot = frame.pivot(index="_order", columns=cols_col, values=value_col).sort_index()
            result = self._coerce_numeric_frame(pivot)
        else:
            result = self._coerce_numeric_frame(data)

        if nrows is not None and isinstance(nrows, int) and result.shape[0] >= nrows:
            result = result.iloc[:nrows, :]
        if result.empty:
            raise ValueError(f"No numeric time-series columns found in {path}")
        return result

    @staticmethod
    def _value_column(data: pd.DataFrame, lowered: List[str]) -> str:
        for candidate in ("data", "value", "target"):
            if candidate in lowered:
                return data.columns[lowered.index(candidate)]
        for col in data.columns:
            if str(col).strip().lower() not in {"date", "time", "timestamp", "datetime", "cols", "label"}:
                return col
        raise ValueError("No value column found in TFB long-table dataset.")

    @staticmethod
    def _time_column(data: pd.DataFrame, lowered: List[str]) -> Optional[str]:
        for candidate in ("date", "time", "timestamp", "datetime"):
            if candidate in lowered:
                return data.columns[lowered.index(candidate)]
        return None

    @staticmethod
    def _coerce_numeric_frame(frame: pd.DataFrame) -> pd.DataFrame:
        numeric_cols: List[str] = []
        for col in frame.columns:
            lowered = str(col).strip().lower()
            if lowered in {"date", "time", "timestamp", "datetime", "label"}:
                continue
            series = pd.to_numeric(frame[col], errors="coerce")
            if series.notna().sum() >= max(8, int(len(series) * 0.5)):
                numeric_cols.append(col)
        if not numeric_cols:
            return pd.DataFrame()
        return frame[numeric_cols].apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all")

    def adjust_period(self, period_value: int) -> int:
        standard_periods = (
            (4, 1),
            (7, 1),
            (12, 2),
            (24, 3),
            (48, 4),
            (52, 2),
            (96, 10),
            (144, 10),
            (168, 10),
            (336, 50),
            (672, 20),
            (720, 20),
            (1008, 100),
            (1440, 200),
            (8766, 500),
            (10080, 500),
            (21600, 2000),
            (43200, 2000),
        )
        for target, tolerance in standard_periods:
            if abs(period_value - target) <= tolerance:
                return target
        return period_value

    def fft_transfer(self, timeseries: np.ndarray, fmin: float = 0.2) -> Tuple[np.ndarray, np.ndarray]:
        series = np.asarray(timeseries, dtype=float)
        yf = np.abs(np.fft.fft(series))
        normalized = yf / max(len(series), 1)
        half = normalized[: len(series) // 2] * 2
        extrema = argrelextrema(half, np.greater)[0]
        amplitudes = half[extrema]
        keep = amplitudes >= fmin
        extrema = extrema[keep]
        amplitudes = amplitudes[keep]
        extrema = extrema[extrema > 0]
        amplitudes = amplitudes[: len(extrema)]
        if extrema.size == 0:
            return np.array([], dtype=float), np.array([], dtype=float)
        return len(series) / extrema, amplitudes

    def count_inversions(self, series: np.ndarray) -> int:
        def merge_sort(values: List[float]) -> Tuple[List[float], int]:
            if len(values) <= 1:
                return values, 0
            mid = len(values) // 2
            left, left_count = merge_sort(values[:mid])
            right, right_count = merge_sort(values[mid:])
            merged: List[float] = []
            count = left_count + right_count
            i = j = 0
            while i < len(left) and j < len(right):
                if left[i] <= right[j]:
                    merged.append(left[i])
                    i += 1
                else:
                    merged.append(right[j])
                    j += 1
                    count += len(left) - i
            merged.extend(left[i:])
            merged.extend(right[j:])
            return merged, count

        _, inversions = merge_sort(np.asarray(series, dtype=float).tolist())
        return inversions

    @staticmethod
    def count_peaks_and_valleys(sequence: np.ndarray) -> int:
        values = np.asarray(sequence, dtype=float)
        total = 0
        for i in range(1, len(values) - 1):
            if values[i] > values[i - 1] and values[i] > values[i + 1]:
                total += 1
            elif values[i] < values[i - 1] and values[i] < values[i + 1]:
                total += 1
        return total

    @staticmethod
    def count_series(sequence: np.ndarray, threshold: float) -> int:
        values = np.asarray(sequence, dtype=float)
        if values.size == 0:
            return 0
        count = 1
        previous = values[0] > threshold
        for value in values[1:]:
            current = value > threshold
            if current != previous:
                count += 1
                previous = current
        return count

    def extract_other_features(self, series_value: np.ndarray) -> List[float]:
        values = np.asarray(series_value, dtype=float)
        mean = float(np.nanmean(values))
        return [
            float(skew(values, nan_policy="omit")),
            float(kurtosis(values, nan_policy="omit")),
            float(abs((np.nanstd(values) / mean) * 100)) if mean != 0 else 0.0,
            float(np.nanstd(np.diff(values))) if values.size > 1 else 0.0,
            float(self.count_inversions(values) / max(len(values), 1)),
            float(self.count_peaks_and_valleys(values) / max(len(values), 1)),
            float(self.count_series(values, np.nanmedian(values)) / max(len(values), 1)),
        ]

    def feature_extract(self, path: str) -> pd.DataFrame:
        data = self.read_data(path)
        rows: List[Dict[str, float]] = []
        for col in data.columns:
            series = data[col].dropna().astype(float).to_numpy()
            if series.size < 8:
                continue
            characteristics = self.series_characteristics(series)
            other = self.extract_other_features(series)
            rows.append(
                {
                    "length": float(series.size),
                    "period_value1": characteristics["period_value1"],
                    "seasonal_strength1": characteristics["seasonal_strength1"],
                    "trend_strength1": characteristics["trend_strength1"],
                    "period_value2": characteristics["period_value2"],
                    "seasonal_strength2": characteristics["seasonal_strength2"],
                    "trend_strength2": characteristics["trend_strength2"],
                    "period_value3": characteristics["period_value3"],
                    "seasonal_strength3": characteristics["seasonal_strength3"],
                    "trend_strength3": characteristics["trend_strength3"],
                    "if_season": characteristics["seasonal_strength1"] >= 0.9,
                    "if_trend": characteristics["trend_strength1"] >= 0.85,
                    "ADF:p-value": characteristics["adf_p_value"],
                    "KPSS:p-value": characteristics["kpss_p_value"],
                    "stability": characteristics["adf_p_value"] <= 0.05 or characteristics["kpss_p_value"] >= 0.05,
                    "skewness": other[0],
                    "kurt": other[1],
                    "rsd": other[2],
                    "std_of_first_derivative": other[3],
                    "inversions": other[4],
                    "turning_points": other[5],
                    "series_in_series": other[6],
                }
            )
        return pd.DataFrame(rows)

    def feature_extract_for_series(self, series: np.ndarray) -> Dict[str, float]:
        characteristics = self.series_characteristics(series)
        return {
            "seasonal_strength1": characteristics["seasonal_strength1"],
            "trend_strength1": characteristics["trend_strength1"],
            "ADF:p-value": characteristics["adf_p_value"],
        }

    def series_characteristics(self, series: np.ndarray) -> Dict[str, float]:
        periods = self._candidate_periods(series)
        stl_rows = self._stl_strengths(series, periods)
        while len(stl_rows) < 3:
            stl_rows.append((0.0, 0.0, 0.0))
        adf_p = self._adf_pvalue(series)
        kpss_p = self._kpss_pvalue(series)
        return {
            "period_value1": stl_rows[0][0],
            "seasonal_strength1": stl_rows[0][1],
            "trend_strength1": stl_rows[0][2],
            "period_value2": stl_rows[1][0],
            "seasonal_strength2": stl_rows[1][1],
            "trend_strength2": stl_rows[1][2],
            "period_value3": stl_rows[2][0],
            "seasonal_strength3": stl_rows[2][1],
            "trend_strength3": stl_rows[2][2],
            "adf_p_value": adf_p,
            "kpss_p_value": kpss_p,
        }

    def _candidate_periods(self, series: np.ndarray) -> List[int]:
        values = np.asarray(series, dtype=float)
        centered = values - float(np.nanmean(values))
        periods, amplitudes = self.fft_transfer(centered, fmin=0.0)
        order = np.argsort(amplitudes)[::-1] if amplitudes.size else []
        result: List[int] = []
        for idx in order[:16]:
            period = self.adjust_period(int(round(periods[idx])))
            if period >= 4 and period not in result:
                result.append(period)
        for period in DEFAULT_PERIODS:
            if period not in result:
                result.append(period)
        return result

    @staticmethod
    def _stl_strengths(series: np.ndarray, periods: Iterable[int]) -> List[Tuple[float, float, float]]:
        values = pd.Series(np.asarray(series, dtype=float))
        upper = max(int(values.size / 3), 12)
        rows: List[Tuple[float, float, float]] = []
        for period in periods:
            if period < 4 or period >= upper:
                continue
            try:
                result = STL(values, period=int(period), robust=True).fit()
            except Exception:
                continue
            resid = result.resid.to_numpy()
            detrend = (values - result.trend).to_numpy()
            deseasonal = (values - result.seasonal).to_numpy()
            resid_var = float(np.nanvar(resid))
            detrend_var = float(np.nanvar(detrend))
            deseasonal_var = float(np.nanvar(deseasonal))
            seasonal = 0.0 if detrend_var == 0 else max(0.0, 1.0 - resid_var / detrend_var)
            trend = 0.0 if deseasonal_var == 0 else max(0.0, 1.0 - resid_var / deseasonal_var)
            rows.append((float(period), float(np.clip(seasonal, 0.0, 1.0)), float(np.clip(trend, 0.0, 1.0))))
        rows.sort(key=lambda item: item[1], reverse=True)
        return rows[:3]

    @staticmethod
    def _adf_pvalue(series: np.ndarray) -> float:
        try:
            return float(adfuller(np.asarray(series, dtype=float), autolag="AIC")[1])
        except Exception:
            return 1.0

    @staticmethod
    def _kpss_pvalue(series: np.ndarray) -> float:
        try:
            return float(kpss(np.asarray(series, dtype=float), regression="c", nlags="auto")[1])
        except Exception:
            return 0.0


class StatisticalCalculator:
    @staticmethod
    def compute_correlation(data: pd.DataFrame) -> Optional[float]:
        numeric = data.select_dtypes(include=[np.number]).dropna(axis=1, how="all")
        if numeric.shape[0] <= 1 or numeric.shape[1] <= 1:
            return None
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(numeric.fillna(numeric.mean()))
        corr = np.corrcoef(scaled)
        if corr.ndim != 2:
            return None
        values = corr[np.triu_indices_from(corr, k=1)]
        values = np.abs(values[np.isfinite(values)])
        if values.size == 0:
            return None
        mean = float(np.mean(values))
        var = float(np.var(values))
        return float(np.clip(2 * (mean + 1 / (var + 2)) / 3, 0.0, 1.0))

    @staticmethod
    def calculate_jsd_for_window(data: np.ndarray, window_size: int) -> float:
        values = np.asarray(data, dtype=float)
        values = values[np.isfinite(values)]
        if values.size < max(window_size, 8):
            return 0.0
        jsd_list: List[float] = []
        num_windows = values.size // window_size
        for i in range(num_windows):
            window = values[i * window_size : (i + 1) * window_size]
            sigma = float(np.nanstd(window))
            if sigma == 0:
                jsd_list.append(0.0)
                continue
            hist, edges = np.histogram(window, bins="stone", density=True)
            centers = (edges[:-1] + edges[1:]) / 2
            pdf = norm.pdf(centers, float(np.nanmean(window)), sigma)
            jsd_list.append(StatisticalCalculator.js_divergence(hist, pdf))
        return float(np.mean(jsd_list)) if jsd_list else 0.0

    @staticmethod
    def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
        left = np.clip(np.asarray(p, dtype=float), 0.0, None)
        right = np.clip(np.asarray(q, dtype=float), 0.0, None)
        if left.sum() <= 0 or right.sum() <= 0:
            return 0.0
        left = left / left.sum()
        right = right / right.sum()
        middle = 0.5 * (left + right)
        return float(0.5 * (entropy(left, middle) + entropy(right, middle)))

    @staticmethod
    def calculate_jsd_multivariate(df: pd.DataFrame, window_size: int) -> List[float]:
        return [
            StatisticalCalculator.calculate_jsd_for_window(pd.to_numeric(df[col], errors="coerce").dropna().to_numpy(), window_size)
            for col in df.columns
        ]

    @staticmethod
    def calculate_jsd(filename: str) -> pd.DataFrame:
        feature_extractor = TimeSeriesFeatureExtractor()
        df = feature_extractor.read_data(path=filename)
        return pd.DataFrame(
            data={
                "short_term_jsd": StatisticalCalculator.calculate_jsd_multivariate(df, 30),
                "long_term_jsd": StatisticalCalculator.calculate_jsd_multivariate(df, 336),
            }
        )

    @staticmethod
    def transition_score(series: np.ndarray) -> float:
        values = np.asarray(series, dtype=float)
        values = values[np.isfinite(values)]
        if values.size < 8:
            return 0.0
        q1, q2 = np.nanquantile(values, [1 / 3, 2 / 3])
        symbols = np.where(values <= q1, 0, np.where(values <= q2, 1, 2))
        matrix = np.zeros((3, 3), dtype=float)
        for left, right in zip(symbols[:-1], symbols[1:]):
            matrix[int(left), int(right)] += 1.0
        if matrix.sum() <= 0:
            return 0.0
        matrix = matrix / matrix.sum()
        return float(np.clip(np.trace(matrix), 0.0, 1.0))

    @staticmethod
    def shifting_score(series: np.ndarray) -> float:
        values = np.asarray(series, dtype=float)
        values = values[np.isfinite(values)]
        if values.size < 64:
            return 0.0
        window = max(16, min(128, values.size // 8))
        scores: List[float] = []
        for start in range(0, values.size - 2 * window + 1, window):
            left = values[start : start + window]
            right = values[start + window : start + 2 * window]
            bins = min(24, max(8, int(np.sqrt(left.size + right.size))))
            hist_l, edges = np.histogram(left, bins=bins, density=True)
            hist_r, _ = np.histogram(right, bins=edges, density=True)
            scores.append(StatisticalCalculator.js_divergence(hist_l, hist_r))
        return float(np.clip(np.mean(scores) if scores else 0.0, 0.0, 1.0))


class TimeSeriesProcessor:
    def __init__(self, output_dir: str = "characteristics"):
        self.feature_extractor = TimeSeriesFeatureExtractor()
        self.stat_calculator = StatisticalCalculator()
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def process_file(self, file_path: str) -> pd.DataFrame:
        data = self.feature_extractor.read_data(file_path)
        rows: List[Dict[str, float]] = []
        for col in data.columns:
            series = data[col].dropna().astype(float).to_numpy()
            if series.size < 8:
                continue
            features = self.feature_extractor.feature_extract_for_series(series)
            rows.append(features)
        return pd.DataFrame(rows)

    def process_path(self, file_path: str) -> None:
        if os.path.isfile(file_path) and file_path.lower().endswith(".csv"):
            self._process_single_file(file_path)
        elif os.path.isdir(file_path):
            self._process_directory(file_path)
        else:
            raise ValueError(f"Invalid path: {file_path}")

    def _process_single_file(self, file_path: str) -> None:
        file_basename = os.path.splitext(os.path.basename(file_path))[0]
        data = self.feature_extractor.read_data(file_path)
        is_univariate = data.shape[1] == 1
        feature_extract_result = self.feature_extractor.feature_extract(file_path)
        process_result = self._core_feature_frame(data)
        jsd_result = self.stat_calculator.calculate_jsd(file_path)
        result = pd.concat(
            [
                jsd_result,
                feature_extract_result.loc[:, ["seasonal_strength1", "trend_strength1", "ADF:p-value"]],
                process_result,
            ],
            axis=1,
        )
        if not is_univariate:
            mean_results = self._calculate_mean_results(result)
            self._save_mean_results(mean_results, file_basename)
        self._save_basic_results(result, file_basename)

    def _core_feature_frame(self, data: pd.DataFrame) -> pd.DataFrame:
        rows: List[Dict[str, float]] = []
        for col in data.columns:
            series = pd.to_numeric(data[col], errors="coerce").dropna().to_numpy(dtype=float)
            rows.append(
                {
                    "SB_TransitionMatrix_3ac_sumdiagcov": self.stat_calculator.transition_score(series),
                    "DN_OutlierInclude_p_001_mdrmd": self.stat_calculator.shifting_score(series),
                    "mean": float(np.nanmean(series)) if series.size else 0.0,
                    "var": float(np.nanvar(series)) if series.size else 0.0,
                }
            )
        return pd.DataFrame(rows)

    def _save_basic_results(self, result: pd.DataFrame, file_prefix: str) -> None:
        all_features_filename = os.path.join(self.output_dir, f"All_characteristics_{file_prefix}.csv")
        features_filename = os.path.join(self.output_dir, f"TFB_characteristics_{file_prefix}.csv")
        result.to_csv(all_features_filename, index=False)
        result = result.copy()
        result["correlation"] = None
        result = result[["correlation"] + [col for col in result.columns if col != "correlation"]]
        dropandrename_dataframe(result).to_csv(features_filename, index=False)

    def _calculate_mean_results(self, result: pd.DataFrame) -> pd.DataFrame:
        numeric_cols = result.select_dtypes(include=[np.number]).columns
        mean_result = result[numeric_cols].mean().to_frame().transpose()
        mean_result["correlation"] = self.stat_calculator.compute_correlation(result)
        return mean_result

    def _save_mean_results(self, mean_result: pd.DataFrame, file_prefix: str) -> None:
        mean_all_features_filename = os.path.join(self.output_dir, f"mean_All_characteristics_{file_prefix}.csv")
        mean_features_filename = os.path.join(self.output_dir, f"mean_TFB_characteristics_{file_prefix}.csv")
        mean_result.to_csv(mean_all_features_filename, index=False)
        dropandrename_dataframe(mean_result).to_csv(mean_features_filename, index=False)

    def _process_directory(self, dir_path: str) -> None:
        file_names = [
            name
            for name in os.listdir(dir_path)
            if os.path.isfile(os.path.join(dir_path, name)) and name.lower().endswith(".csv")
        ]
        if not file_names:
            raise ValueError(f"No CSV files found in: {dir_path}")
        for file_name in file_names:
            self._process_single_file(os.path.join(dir_path, file_name))


def dropandrename_dataframe(result_df: pd.DataFrame) -> pd.DataFrame:
    result = result_df.loc[
        :,
        [
            "correlation",
            "SB_TransitionMatrix_3ac_sumdiagcov",
            "DN_OutlierInclude_p_001_mdrmd",
            "seasonal_strength1",
            "trend_strength1",
            "ADF:p-value",
            "short_term_jsd",
            "long_term_jsd",
        ],
    ].copy()
    result.columns = TFB_CHARACTERISTIC_COLUMNS
    return result


if __name__ == "__main__":
    processor = TimeSeriesProcessor(output_dir="characteristics")
    file_path = r"./DemoDatasets/Exchange.csv"
    processor.process_path(file_path)
