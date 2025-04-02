import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

def calculate_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculates Mean Absolute Percentage Error, handling zeros and NaNs."""
    y_true, y_pred = np.array(y_true, dtype=float), np.array(y_pred, dtype=float)

    # Filter pairs where true is zero/NaN, or pred is NaN
    mask = (y_true != 0) & (~np.isnan(y_true)) & (~np.isnan(y_pred))
    removed_count = len(y_true) - np.sum(mask)
    if removed_count > 0:
         logger.debug(f"MAPE Calc: Ignoring {removed_count} pair(s) due to zero/NaN true or NaN pred.")

    y_true_filtered = y_true[mask]
    y_pred_filtered = y_pred[mask]

    if len(y_true_filtered) == 0:
        logger.warning("MAPE Warning: No valid points for MAPE after filtering zeros/NaNs.")
        return np.nan

    try:
        mape = np.mean(np.abs((y_true_filtered - y_pred_filtered) / y_true_filtered)) * 100
        return mape
    except ZeroDivisionError: # Should be caught by mask, but safety check
        logger.error("MAPE Error: Division by zero encountered unexpectedly.")
        return np.nan

def evaluate_forecast(
    y_true_series: pd.Series,
    y_pred_series: pd.Series,
    model_name: str,
    metrics_to_calculate: List[str] # e.g., ['MAE', 'RMSE', 'MAPE']
) -> Dict[str, Any]:
    """Calculates specified evaluation metrics for a forecast series."""
    results: Dict[str, Any] = {'Model': model_name}

    # Ensure indices match
    y_pred_aligned: pd.Series = y_pred_series.reindex(y_true_series.index)

    # Get values only where prediction is not NaN
    valid_pred_mask: pd.Series = ~y_pred_aligned.isnull()
    y_true_eval: np.ndarray = y_true_series[valid_pred_mask].values
    y_pred_eval: np.ndarray = y_pred_aligned[valid_pred_mask].values

    num_valid_points = len(y_true_eval)
    num_total_points = len(y_true_series)
    results['Num_Forecast_Points'] = num_total_points # Total points in test set
    results['Num_Valid_Points_Eval'] = num_valid_points # Points used for metrics

    if num_valid_points == 0:
        logger.warning(f"{model_name} Eval Warning: No valid (non-NaN) predictions matching test index. Cannot calc metrics.")
        for metric in metrics_to_calculate: results[metric] = np.nan # Init metrics to NaN
        return results

    if num_valid_points < num_total_points:
         logger.warning(f"{model_name} Eval Warning: Only {num_valid_points}/{num_total_points} points used for eval due to NaN preds.")


    metric_log_parts = []
    calculation_errors = []

    # Calculate requested metrics
    if 'MAE' in metrics_to_calculate:
        try:
            mae = mean_absolute_error(y_true_eval, y_pred_eval)
            results['MAE'] = mae
            metric_log_parts.append(f"MAE: {mae:.4f}")
        except Exception as e: results['MAE'] = np.nan; calculation_errors.append(f"MAE Error: {e}")

    if 'RMSE' in metrics_to_calculate:
        try:
            rmse = np.sqrt(mean_squared_error(y_true_eval, y_pred_eval))
            results['RMSE'] = rmse
            metric_log_parts.append(f"RMSE: {rmse:.4f}")
        except Exception as e: results['RMSE'] = np.nan; calculation_errors.append(f"RMSE Error: {e}")

    if 'MAPE' in metrics_to_calculate:
        try:
            mape = calculate_mape(y_true_eval, y_pred_eval) # Uses filtered values internally
            results['MAPE'] = mape
            metric_log_parts.append(f"MAPE: {mape:.4f}%")
        except Exception as e: results['MAPE'] = np.nan; calculation_errors.append(f"MAPE Error: {e}")

    # Log results and errors
    if metric_log_parts: logger.info(f"{model_name} Evaluation - {', '.join(metric_log_parts)}")
    if calculation_errors: logger.error(f"{model_name} Evaluation Errors: {'; '.join(calculation_errors)}")

    return results