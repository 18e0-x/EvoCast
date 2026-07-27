# -*- coding: utf-8 -*-

import numpy as np

__all__ = [
    "mae",
    "mse",
    "rmse",
    "mape",
    "smape",
    "mase",
    "wape",
    "msmape",
    "mae_norm",
    "mse_norm",
    "rmse_norm",
    "mape_norm",
    "smape_norm",
    "mase_norm",
    "wape_norm",
    "msmape_norm",
]


def _error(actual: np.ndarray, predicted: np.ndarray, **kwargs):
    """Simple error"""
    return actual - predicted


def _safe_mase_scale(hist_data: np.ndarray, seasonality: int) -> np.ndarray:
    """Compute the MASE denominator once with vectorized numpy ops."""
    hist = np.asarray(hist_data)
    if hist.ndim == 1:
        hist = hist[:, None]
    if seasonality <= 0 or len(hist) <= seasonality:
        raise ValueError("seasonality must be positive and smaller than history length")
    diffs = np.abs(hist[seasonality:] - hist[:-seasonality])
    scale = diffs.sum(axis=0)
    scale = np.where(scale == 0, np.nan, scale)
    return scale


def _percentage_error(actual: np.ndarray, predicted: np.ndarray, **kwargs):
    """Percentage error"""
    return (actual - predicted) / actual


def mse(actual: np.ndarray, predicted: np.ndarray, **kwargs):
    """Mean Squared Error"""
    return np.mean(np.square(_error(actual, predicted)))


def rmse(actual: np.ndarray, predicted: np.ndarray, **kwargs):
    """Root Mean Squared Error"""
    return np.sqrt(mse(actual, predicted))


def mae(actual: np.ndarray, predicted: np.ndarray, **kwargs):
    """Mean Absolute Error"""

    return np.mean(np.abs(_error(actual, predicted)))


def mase(
    actual: np.ndarray,
    predicted: np.ndarray,
    hist_data: np.ndarray,
    seasonality: int = 1,
    **kwargs
):
    """
    Mean Absolute Scaled Error
    Baseline (benchmark) is computed with naive forecasting (shifted by @seasonality)
    """
    if hist_data is None or seasonality <= 0 or len(hist_data) <= seasonality:
        return np.nan
    actual = np.asarray(actual)
    predicted = np.asarray(predicted)
    if actual.ndim == 1:
        actual = actual[:, None]
    if predicted.ndim == 1:
        predicted = predicted[:, None]
    scale = (len(predicted) / (len(hist_data) - seasonality)) * _safe_mase_scale(
        hist_data, seasonality
    )
    numer = np.abs(actual - predicted).sum(axis=0)
    value = numer / scale
    return float(np.nanmean(value))


def mape(actual: np.ndarray, predicted: np.ndarray, **kwargs):
    """
    Mean Absolute Percentage Error
    Properties:
        + Easy to interpret
        + Scale independent
        - Biased, not symmetric
        - Undefined when actual[t] == 0
    """
    actual = np.asarray(actual)
    predicted = np.asarray(predicted)
    valid = np.isfinite(actual) & (np.abs(actual) > np.finfo(float).eps)
    if not np.any(valid):
        return np.nan
    return np.mean(np.abs((actual[valid] - predicted[valid]) / actual[valid])) * 100


def smape(actual: np.ndarray, predicted: np.ndarray, **kwargs):
    """
    Symmetric Mean Absolute Percentage Error
    """
    return (
        np.mean(
            2.0 * np.abs(actual - predicted) / ((np.abs(actual) + np.abs(predicted)))
        )
        * 100
    )


def wape(actual: np.ndarray, predicted: np.ndarray, **kwargs):
    """Masked weighted absolute percentage error (WAPE)

    Args:
        preds (torch.Tensor): predicted values
        labels (torch.Tensor): labels
    Returns:
        torch.Tensor: masked mean absolute error
    """
    loss = np.sum(np.abs(actual - predicted)) / np.sum(np.abs(actual)) * 100
    return loss


def msmape(actual: np.ndarray, predicted: np.ndarray, epsilon: float = 0.1, **kwargs):
    """
    Function to calculate series wise smape values

    Parameters
    forecasts - a matrix containing forecasts for a set of series
                no: of rows should be equal to number of series and no: of columns should be equal to the forecast horizon
    test_set - a matrix with the same dimensions as 'forecasts' containing the actual values corresponding with them
    """

    comparator = np.full_like(actual, 0.5 + epsilon)
    denom = np.maximum(comparator, np.abs(predicted) + np.abs(actual) + epsilon)
    msmape_per_series = np.mean(2 * np.abs(predicted - actual) / denom) * 100
    return msmape_per_series


def _error_norm(actual: np.ndarray, predicted: np.ndarray, scaler: object, **kwargs):
    """Simple error"""
    return scaler.transform(actual) - scaler.transform(predicted)


def _percentage_error_norm(
    actual: np.ndarray, predicted: np.ndarray, scaler: object, **kwargs
):
    """Percentage error"""
    return (scaler.transform(actual) - scaler.transform(predicted)) / scaler.transform(
        actual
    )


def mse_norm(actual: np.ndarray, predicted: np.ndarray, scaler: object, **kwargs):
    """Mean Squared Error"""
    return np.mean(np.square(_error_norm(actual, predicted, scaler)))


def rmse_norm(actual: np.ndarray, predicted: np.ndarray, scaler: object, **kwargs):
    """Root Mean Squared Error"""
    return np.sqrt(mse_norm(actual, predicted, scaler))


def mae_norm(actual: np.ndarray, predicted: np.ndarray, scaler: object, **kwargs):
    """Mean Absolute Error"""

    return np.mean(np.abs(_error_norm(actual, predicted, scaler)))


def mase_norm(
    actual: np.ndarray,
    predicted: np.ndarray,
    scaler: object,
    hist_data: np.ndarray,
    seasonality: int = 1,
    **kwargs
):
    """
    Mean Absolute Scaled Error
    Baseline (benchmark) is computed with naive forecasting (shifted by @seasonality)
    """
    if hist_data is None or seasonality <= 0 or len(hist_data) <= seasonality:
        return np.nan
    actual = scaler.transform(actual)
    predicted = scaler.transform(predicted)
    hist_data = scaler.transform(hist_data)
    scale = (len(predicted) / (len(hist_data) - seasonality)) * _safe_mase_scale(
        hist_data, seasonality
    )
    numer = np.abs(actual - predicted).sum(axis=0)
    value = numer / scale
    return float(np.nanmean(value))


def mape_norm(actual: np.ndarray, predicted: np.ndarray, scaler: object, **kwargs):
    """
    Mean Absolute Percentage Error
    Properties:
        + Easy to interpret
        + Scale independent
        - Biased, not symmetric
        - Undefined when actual[t] == 0
    """
    # MAPE is already scale-free; centering with StandardScaler changes its denominator.
    return mape(actual, predicted)


def smape_norm(actual: np.ndarray, predicted: np.ndarray, scaler: object, **kwargs):
    """
    Symmetric Mean Absolute Percentage Error
    """
    actual = scaler.transform(actual)
    predicted = scaler.transform(predicted)
    return (
        np.mean(
            2.0 * np.abs(actual - predicted) / ((np.abs(actual) + np.abs(predicted)))
        )
        * 100
    )


def wape_norm(actual: np.ndarray, predicted: np.ndarray, scaler: object, **kwargs):
    """Masked weighted absolute percentage error (WAPE)

    Args:
        preds (torch.Tensor): predicted values
        labels (torch.Tensor): labels
    Returns:
        torch.Tensor: masked mean absolute error
    """
    actual = scaler.transform(actual)
    predicted = scaler.transform(predicted)
    loss = np.sum(np.abs(actual - predicted)) / np.sum(np.abs(actual)) * 100
    return loss


def msmape_norm(
    actual: np.ndarray,
    predicted: np.ndarray,
    scaler: object,
    epsilon: float = 0.1,
    **kwargs
):
    """
    Function to calculate series wise smape values

    Parameters
    forecasts - a matrix containing forecasts for a set of series
                no: of rows should be equal to number of series and no: of columns should be equal to the forecast horizon
    test_set - a matrix with the same dimensions as 'forecasts' containing the actual values corresponding with them
    """
    actual = scaler.transform(actual)
    predicted = scaler.transform(predicted)
    comparator = np.full_like(actual, 0.5 + epsilon)
    denom = np.maximum(comparator, np.abs(predicted) + np.abs(actual) + epsilon)
    msmape_per_series = np.mean(2 * np.abs(predicted - actual) / denom) * 100
    return msmape_per_series
