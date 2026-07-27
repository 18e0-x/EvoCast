"""Error taxonomy for EvoCast runs.

Classifies every run outcome into a structured label.
Controls repair policy: engineering bugs can enter debug,
scientific failures should NOT trigger automatic code repair.
"""

from enum import Enum
from typing import Optional, Dict


class ErrorLabel(str, Enum):
    SUCCESS = "success"
    IMPORT_ERROR = "import_error"
    FRAMEWORK_PROTOCOL_ERROR = "framework_protocol_error"
    CONFIG_ERROR = "config_error"
    SHAPE_CONSTRAINT_VIOLATION = "shape_constraint_violation"
    MISSING_REQUIRED_HPARAM = "missing_required_hparam"
    ENVIRONMENT_INCOMPATIBLE = "environment_incompatible"
    RUNTIME_ERROR = "runtime_error"
    OOM = "oom"
    TIMEOUT = "timeout"
    NAN_LOSS = "nan_loss"
    METRIC_MISSING = "metric_missing"
    METRIC_REGRESSION = "metric_regression"
    NO_IMPROVEMENT = "no_improvement"
    PROTECTED_FILE_VIOLATION = "protected_file_violation"
    INVALID_VARIANT_CONTRACT = "invalid_variant_contract"
    UNKNOWN_ERROR = "unknown_error"


# Errors that are reasonable to attempt automatic code repair on.
REPAIRABLE_LABELS = {
    ErrorLabel.IMPORT_ERROR,
    ErrorLabel.CONFIG_ERROR,
    ErrorLabel.SHAPE_CONSTRAINT_VIOLATION,
    ErrorLabel.MISSING_REQUIRED_HPARAM,
    ErrorLabel.ENVIRONMENT_INCOMPATIBLE,
    ErrorLabel.RUNTIME_ERROR,
    ErrorLabel.METRIC_MISSING,
    ErrorLabel.INVALID_VARIANT_CONTRACT,
}

# Errors that should NOT trigger automatic code repair.
NON_REPAIRABLE_LABELS = {
    ErrorLabel.SUCCESS,
    ErrorLabel.OOM,
    ErrorLabel.TIMEOUT,
    ErrorLabel.NAN_LOSS,
    ErrorLabel.METRIC_REGRESSION,
    ErrorLabel.NO_IMPROVEMENT,
    ErrorLabel.PROTECTED_FILE_VIOLATION,
    ErrorLabel.FRAMEWORK_PROTOCOL_ERROR,
    ErrorLabel.UNKNOWN_ERROR,
}

ENGINEERING_LABELS = {
    ErrorLabel.IMPORT_ERROR,
    ErrorLabel.CONFIG_ERROR,
    ErrorLabel.SHAPE_CONSTRAINT_VIOLATION,
    ErrorLabel.MISSING_REQUIRED_HPARAM,
    ErrorLabel.RUNTIME_ERROR,
    ErrorLabel.METRIC_MISSING,
    ErrorLabel.INVALID_VARIANT_CONTRACT,
}

SCIENTIFIC_FAILURE_LABELS = {
    ErrorLabel.METRIC_REGRESSION,
    ErrorLabel.NO_IMPROVEMENT,
    ErrorLabel.NAN_LOSS,
}

ENGINEERING_ERROR_TYPES = {label.value for label in ENGINEERING_LABELS}
SCIENTIFIC_FAILURE_TYPES = {label.value for label in SCIENTIFIC_FAILURE_LABELS}


def is_repairable(label: ErrorLabel) -> bool:
    """Whether an error label justifies automatic code repair attempts."""
    return label in REPAIRABLE_LABELS


def is_engineering_failure(label: ErrorLabel) -> bool:
    """Whether the failure is engineering-side and potentially debuggable."""
    return label in ENGINEERING_LABELS


def is_scientific_failure(label: ErrorLabel) -> bool:
    """Whether the failure is scientific (model quality) not engineering."""
    return label in SCIENTIFIC_FAILURE_LABELS


def is_engineering_error_type(error_type: str) -> bool:
    """String form helper for journal-driven policy code."""
    return error_type in ENGINEERING_ERROR_TYPES


def is_scientific_error_type(error_type: str) -> bool:
    """String form helper for journal-driven policy code."""
    return error_type in SCIENTIFIC_FAILURE_TYPES


def classify_exception(exc: Exception) -> ErrorLabel:
    """Classify a Python exception into an error label."""
    exc_type = type(exc).__name__
    exc_msg = str(exc).lower()

    if isinstance(exc, ImportError) or isinstance(exc, ModuleNotFoundError):
        return ErrorLabel.IMPORT_ERROR
    if isinstance(exc, MemoryError):
        return ErrorLabel.OOM
    if isinstance(exc, TimeoutError):
        return ErrorLabel.TIMEOUT
    if "oom" in exc_msg or "out of memory" in exc_msg or "cuda out of memory" in exc_msg:
        return ErrorLabel.OOM
    if "timeout" in exc_msg:
        return ErrorLabel.TIMEOUT
    if "invalid decimal literal" in exc_msg:
        return ErrorLabel.FRAMEWORK_PROTOCOL_ERROR
    if "nan" in exc_msg or "NaN" in str(exc):
        return ErrorLabel.NAN_LOSS
    if isinstance(exc, ValueError) or isinstance(exc, TypeError):
        if "config" in exc_msg.lower():
            return ErrorLabel.CONFIG_ERROR
        return ErrorLabel.RUNTIME_ERROR
    if isinstance(exc, RuntimeError):
        return ErrorLabel.RUNTIME_ERROR

    return ErrorLabel.UNKNOWN_ERROR


def classify_from_result(
    success: bool,
    error: Optional[Exception] = None,
    metrics: Optional[Dict] = None,
    objective_metric: Optional[str] = None,
    baseline_value: Optional[float] = None,
    is_better_direction: str = "lower",
) -> ErrorLabel:
    """Classify a run outcome holistically.

    Args:
        success: Whether the pipeline ran without exceptions.
        error: The exception if one occurred.
        metrics: Parsed metrics dict.
        objective_metric: The key of the objective metric.
        baseline_value: The baseline value to compare against.
        is_better_direction: "lower" or "higher".
    """
    if not success:
        if error is not None:
            return classify_exception(error)
        return ErrorLabel.RUNTIME_ERROR

    if metrics is None or not metrics:
        return ErrorLabel.METRIC_MISSING

    if objective_metric and objective_metric not in metrics:
        return ErrorLabel.METRIC_MISSING

    if baseline_value is not None and objective_metric:
        current = metrics.get(objective_metric)
        if current is not None:
            if is_better_direction == "lower" and current > baseline_value:
                return ErrorLabel.METRIC_REGRESSION
            elif is_better_direction == "higher" and current < baseline_value:
                return ErrorLabel.METRIC_REGRESSION
            elif abs(current - baseline_value) < 1e-8:
                return ErrorLabel.NO_IMPROVEMENT

    return ErrorLabel.SUCCESS


def make_error_entry(
    label: ErrorLabel,
    message: str = "",
    traceback_str: str = "",
    traceback_path: str = "",
) -> Dict:
    """Create a structured error entry for the trial journal."""
    return {
        "error_type": label.value,
        "is_repairable": is_repairable(label),
        "is_scientific_failure": is_scientific_failure(label),
        "message": (message or "")[:2000],
        "traceback": (traceback_str or "")[:5000],
        "traceback_path": traceback_path,
    }
