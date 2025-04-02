import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX, SARIMAXResultsWrapper
import warnings
import logging
import pickle # For saving model
import os
from typing import Tuple, Optional, Any, Dict

# Optional: Import pmdarima
try: import pmdarima as pm; _pmdarima_available = True
except ImportError: _pmdarima_available = False

logger = logging.getLogger(__name__)

def get_seasonal_period(freq_str: Optional[str], manual_s: Optional[int] = None) -> int:
    """Determines seasonal period 's' from frequency string or manual override."""
    if manual_s is not None and manual_s > 1:
        logger.info(f"Using manual SARIMA seasonal period S = {manual_s}")
        return manual_s
    if not freq_str:
        logger.warning("SARIMA: Data freq not determined. Cannot infer 's'. Assuming s=1 (non-seasonal).")
        return 1

    # Map common frequency strings to seasonal periods
    freq_base = freq_str.upper().split('-')[0]
    s_map = {'A': 1, 'Y': 1, 'Q': 4, 'M': 12, 'W': 52, 'D': 7, 'B': 5, 'H': 24, 'T': 60*24, 'S': 3600*24}
    s = s_map.get(freq_base[0], 1) # Default to 1 if not found

    logger.info(f"Inferred SARIMA seasonal period S = {s} from freq '{freq_str}'")
    return s

def run_sarima(
    train_data: pd.DataFrame,
    test_periods: int,
    test_index: pd.DatetimeIndex,
    config_params: Dict[str, Any]
) -> Tuple[pd.DataFrame, Optional[Dict[str, Any]], Optional[SARIMAXResultsWrapper]]:
    """
    Trains SARIMA (manual or auto) and returns forecasts, parameters, and fitted model.
    Returns: forecast DataFrame, parameter dict, fitted model object (or Nones on failure).
    """
    logger.info("--- Running SARIMA ---")
    model_params_used: Dict[str, Any] = {'model_type': 'SARIMA'}
    fitted_model: Optional[SARIMAXResultsWrapper] = None
    nan_df = pd.DataFrame(np.nan, index=test_index, columns=['yhat', 'yhat_lower', 'yhat_upper'])

    if train_data['y'].isnull().any():
        logger.error("SARIMA Error: Training data contains NaNs. SARIMAX cannot fit. Check imputation.")
        return nan_df, model_params_used, None # Cannot proceed

    seasonal_period: int = get_seasonal_period(train_data.index.freqstr, config_params.get('SARIMA_MANUAL_S'))
    model_params_used['seasonal_period_used'] = seasonal_period

    # Set default orders (manual)
    final_order: Tuple[int, int, int] = config_params['SARIMA_ORDER']
    final_seasonal_order = config_params['SARIMA_SEASONAL_ORDER_NOS'] + (seasonal_period,) if seasonal_period > 1 else (0, 0, 0, 0)

    use_auto = config_params['USE_AUTO_ARIMA']
    model_params_used['auto_arima_attempted'] = use_auto

    if use_auto:
        if not _pmdarima_available:
            logger.error("SARIMA Error: USE_AUTO_ARIMA=True, but pmdarima not installed. Install or set USE_AUTO_ARIMA=False.")
            return nan_df, model_params_used, None

        logger.info("Attempting auto_arima...")
        is_seasonal_search: bool = config_params['SARIMA_AUTO_SEASONAL'] and seasonal_period > 1
        try:
            # Check if enough samples for seasonal search
            min_samples_needed = seasonal_period * 2 if is_seasonal_search else 2
            if len(train_data) < min_samples_needed:
                if is_seasonal_search: logger.warning(f"Not enough samples ({len(train_data)}) for auto_arima seasonal search (m={seasonal_period}). Disabling.")
                is_seasonal_search = False

            m_param = seasonal_period if is_seasonal_search else 1
            D_param = None if is_seasonal_search else 0 # Let auto find D if seasonal

            auto_model = pm.auto_arima(
                train_data['y'],
                start_p=config_params['AUTO_ARIMA_START_P'], max_p=config_params['AUTO_ARIMA_MAX_P'],
                start_q=config_params['AUTO_ARIMA_START_Q'], max_q=config_params['AUTO_ARIMA_MAX_Q'],
                max_d=config_params['AUTO_ARIMA_MAX_D'],
                m=m_param, seasonal=is_seasonal_search,
                start_P=config_params['AUTO_ARIMA_START_SP'], max_P=config_params['AUTO_ARIMA_MAX_SP'],
                start_Q=config_params['AUTO_ARIMA_START_SQ'], max_Q=config_params['AUTO_ARIMA_MAX_SQ'],
                max_D=config_params['AUTO_ARIMA_MAX_SD'],
                d=None, D=D_param, # Let auto find differencing orders
                test=config_params['AUTO_ARIMA_TEST'], information_criterion=config_params['AUTO_ARIMA_IC'],
                stepwise=config_params['AUTO_ARIMA_STEPWISE'], suppress_warnings=True,
                error_action='ignore', trace=(logger.getEffectiveLevel() <= logging.DEBUG) # Show trace only if DEBUG level
            )
            final_order = auto_model.order
            final_seasonal_order = auto_model.seasonal_order
            model_params_used.update({'auto_arima_successful': True, 'auto_arima_order': final_order, 'auto_arima_seasonal_order': final_seasonal_order})
            logger.info(f"auto_arima identified Order: {final_order} Seasonal Order: {final_seasonal_order}")

        except Exception as e: # Fallback to manual if auto_arima fails
            logger.error(f"auto_arima failed: {e}. Falling back to manual orders.")
            model_params_used.update({'auto_arima_successful': False, 'auto_arima_error': str(e)})
            final_order = config_params['SARIMA_ORDER'] # Reset to manual defaults
            final_seasonal_order = config_params['SARIMA_SEASONAL_ORDER_NOS'] + (seasonal_period,) if seasonal_period > 1 else (0, 0, 0, 0)
            logger.info(f"Using manual Order: {final_order} Seasonal Order: {final_seasonal_order}")

    else: # Use Manual Orders explicitly
        logger.info(f"Using manual Order: {final_order}")
        if seasonal_period > 1: logger.info(f"Using manual Seasonal Order: {final_seasonal_order}")
        else: logger.info("Using Non-Seasonal SARIMA (s=1)")

    # Store final orders used & enforce settings
    model_params_used.update({ 'final_order': final_order, 'final_seasonal_order': final_seasonal_order,
                               'enforce_stationarity': config_params['SARIMA_ENFORCE_STATIONARITY'],
                               'enforce_invertibility': config_params['SARIMA_ENFORCE_INVERTIBILITY'] })

    # --- Fit Final SARIMAX Model ---
    y_train = train_data['y'] # Use directly
    if y_train.index.freq is None: # Check frequency before fitting
         logger.warning("SARIMA: Train index freq not set. Attempting to infer.")
         inferred_freq = pd.infer_freq(y_train.index)
         if inferred_freq:
             y_train = y_train.copy() # Work on copy to set freq
             try: y_train.index.freq = inferred_freq; logger.info(f"Set index freq to inferred: {inferred_freq}")
             except Exception as freq_err: logger.error(f"Failed to set inferred freq '{inferred_freq}': {freq_err}."); return nan_df, model_params_used, None
         else: logger.error("SARIMA Error: Cannot determine frequency. Forecast index may be incorrect."); return nan_df, model_params_used, None

    try:
        logger.info(f"Fitting SARIMAX{final_order}{final_seasonal_order}")
        model = SARIMAX(y_train, order=final_order, seasonal_order=final_seasonal_order,
                        enforce_stationarity=config_params['SARIMA_ENFORCE_STATIONARITY'],
                        enforce_invertibility=config_params['SARIMA_ENFORCE_INVERTIBILITY'])
        fitted_model = model.fit(disp=False)
        logger.info("SARIMAX fitting complete.")
        # logger.debug(f"SARIMAX Model Summary:\n{fitted_model.summary()}") # Log summary if debugging

        # Predict for the test period range
        pred = fitted_model.get_prediction(start=test_index[0], end=test_index[-1])
        forecast_summary: pd.DataFrame = pred.summary_frame(alpha=1.0 - config_params.get('PROPHET_INTERVAL_WIDTH', 0.95))

        # Check alignment and reindex if necessary
        if not forecast_summary.index.equals(test_index):
            logger.warning("SARIMA forecast index mismatch test_index. Reindexing.")
            forecast_summary = forecast_summary.reindex(test_index)

        # Extract relevant columns and rename for consistency
        forecast_values: pd.DataFrame = forecast_summary[['mean', 'mean_ci_lower', 'mean_ci_upper']].rename(
            columns={'mean': 'yhat', 'mean_ci_lower': 'yhat_lower', 'mean_ci_upper': 'yhat_upper'}
        )
        forecast_values.index = test_index # Ensure correct final index

        if forecast_values['yhat'].isnull().any(): logger.warning("SARIMA forecast contains NaN values.")

        logger.info("SARIMA forecast generated successfully.")
        return forecast_values, model_params_used, fitted_model

    except (ValueError, np.linalg.LinAlgError) as fit_err: # Catch common statsmodels errors
         logger.error(f"SARIMA Final Fit/Forecast Error: {type(fit_err).__name__}: {fit_err}")
         logger.error("Check model orders, data properties (constants, collinearity).")
         model_params_used['fit_error'] = f"{type(fit_err).__name__}: {fit_err}"
         return nan_df, model_params_used, None
    except Exception as e: # Catch any other errors
        logger.error(f"SARIMA Final Fit/Forecast Unexpected Error: {type(e).__name__}: {e}", exc_info=True)
        model_params_used['fit_error'] = f"Unexpected {type(e).__name__}: {e}"
        return nan_df, model_params_used, None

def save_sarima_model(model: SARIMAXResultsWrapper, file_path: str):
    """Saves a fitted SARIMAX model using pickle."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as f:
            pickle.dump(model, f)
        logger.info(f"SARIMA model saved to: {file_path}")
    except Exception as e:
        logger.error(f"Error saving SARIMA model to {file_path}: {e}", exc_info=True)