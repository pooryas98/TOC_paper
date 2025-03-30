# --- START OF FILE src/evaluation.py ---

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from typing import Dict, Any # Added for type hinting

def calculate_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculates Mean Absolute Percentage Error, handling zeros and NaNs."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    y_true = y_true.astype(float)
    y_pred = y_pred.astype(float)

    # Filter out pairs where true value is zero or NaN, or prediction is NaN
    mask: np.ndarray = (y_true != 0) & (~np.isnan(y_true)) & (~np.isnan(y_pred))
    removed_count: int = len(y_true) - np.sum(mask)
    if removed_count > 0:
         # Using warnings.warn might be better if this needs more visibility
         print(f"MAPE Warning: Ignoring {removed_count} pair(s) due to zero/NaN true values or NaN predictions.")

    y_true_filtered: np.ndarray = y_true[mask]
    y_pred_filtered: np.ndarray = y_pred[mask]

    if len(y_true_filtered) == 0:
        print("MAPE Error: No valid points for MAPE calculation after filtering.")
        return np.nan # Return NaN if calculation is impossible

    mape: float = np.mean(np.abs((y_true_filtered - y_pred_filtered) / y_true_filtered)) * 100
    return mape

def evaluate_forecast(y_true_series: pd.Series, y_pred_series: pd.Series, model_name: str) -> Dict[str, Any]:
    """Calculates MAE, RMSE, MAPE for a forecast series against the true series."""

    # Align prediction series to the true series index, filling missing predictions with NaN
    y_pred_aligned: pd.Series = y_pred_series.reindex(y_true_series.index)

    # Get actual values and predictions where prediction is not NaN
    valid_pred_mask: pd.Series = ~y_pred_aligned.isnull()
    y_true_eval: np.ndarray = y_true_series[valid_pred_mask].values
    y_pred_eval: np.ndarray = y_pred_aligned[valid_pred_mask].values

    results: Dict[str, Any] = {'Model': model_name, 'MAE': np.nan, 'RMSE': np.nan, 'MAPE': np.nan}

    if len(y_true_eval) == 0:
        print(f"{model_name} Evaluation Error: No valid (non-NaN) predictions found matching the test set index.")
        return results # Return dict with NaNs

    try:
        mae: float = mean_absolute_error(y_true_eval, y_pred_eval)
        rmse: float = np.sqrt(mean_squared_error(y_true_eval, y_pred_eval))
        mape: float = calculate_mape(y_true_eval, y_pred_eval) # Use filtered values for MAPE

        print(f"{model_name} Evaluation - MAE: {mae:.4f}, RMSE: {rmse:.4f}, MAPE: {mape:.4f}%")
        results.update({'MAE': mae, 'RMSE': rmse, 'MAPE': mape})
        return results
    except Exception as e:
         print(f"Error calculating metrics for {model_name}: {e}")
         return results # Return dict with NaNs if metric calculation fails

# --- END OF FILE src/evaluation.py ---