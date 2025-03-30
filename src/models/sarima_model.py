# --- START OF FILE src/models/sarima_model.py ---

import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX, SARIMAXResultsWrapper
# Removed STLForecast import as it wasn't used
import warnings
from .. import config # Import config
from typing import Tuple, Optional, Any # Added for type hinting

# Attempt to import pmdarima, but allow running without it if auto_arima is disabled
try:
    import pmdarima as pm
    _pmdarima_available = True
except ImportError:
    _pmdarima_available = False
    # Warning is now printed only if USE_AUTO_ARIMA is True and pmdarima is missing


def get_seasonal_period(freq_str: Optional[str], manual_s: Optional[int] = None) -> int:
    """Determines the seasonal period 's' based on frequency string or manual override."""
    if manual_s is not None and manual_s > 1:
        print(f"Using manually configured SARIMA seasonal period S = {manual_s}")
        return manual_s
    if not freq_str:
        warnings.warn("SARIMA: Data frequency not determined. Cannot infer seasonal period 's'. Assuming s=1 (non-seasonal).")
        return 1

    # Normalize frequency string (uppercase, take first char after potential offset like '-')
    # Handles 'MS', 'M', 'QS', 'Q', 'AS', 'A', 'D', 'H', 'T', 'MIN', 'S' etc.
    # Keep 'W' for weekly
    freq_base = freq_str.upper().split('-')[0]
    if freq_base == 'W':
        s = 52 # Weekly
    else:
        # Take the first character for M, Q, A, D, H, T, S, B
        freq_char = freq_base[0]
        s_map = {
            'A': 1,  # Annual (or year-end) -> Non-seasonal for m
            'Y': 1,  # Yearly -> Non-seasonal for m
            'Q': 4,  # Quarterly
            'M': 12, # Monthly
            'W': 52, # Weekly (handled above, but include for completeness)
            'D': 7,  # Daily
            'B': 5,  # Business Day
            'H': 24, # Hourly
            'T': 60 * 24, # Minutely -> Daily seasonality
            'S': 60 * 60 * 24 # Secondly -> Daily seasonality
        }
        s = s_map.get(freq_char, 1) # Default to 1 (non-seasonal) if char not found

    print(f"Inferred SARIMA seasonal period S = {s} based on frequency '{freq_str}' (processed as '{freq_base}')")
    return s


def run_sarima(train_data: pd.DataFrame, test_periods: int, test_index: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Trains SARIMA model (manual or auto) and returns forecasts with confidence intervals.
    Returns a DataFrame with columns ['yhat', 'yhat_lower', 'yhat_upper'].
    """
    print("Running SARIMA...")
    # Prepare output DataFrame in case of errors
    nan_df = pd.DataFrame(np.nan, index=test_index, columns=['yhat', 'yhat_lower', 'yhat_upper'])

    # Check for NaNs in training data - SARIMAX cannot handle them
    if train_data['y'].isnull().any():
        warnings.warn("SARIMA Warning: Training data contains NaN values. SARIMAX cannot fit. Returning NaNs. Consider imputing data.")
        return nan_df

    seasonal_period: int = get_seasonal_period(train_data.index.freqstr, config.SARIMA_MANUAL_S)
    final_order: Tuple[int, int, int] = config.SARIMA_ORDER
    # Initialize seasonal order correctly based on inferred/manual s
    final_seasonal_order: Tuple[int, int, int, int]
    if seasonal_period > 1:
        final_seasonal_order = config.SARIMA_SEASONAL_ORDER_NOS + (seasonal_period,)
    else:
        final_seasonal_order = (0, 0, 0, 0) # Non-seasonal

    # --- Determine SARIMA Orders ---
    if config.USE_AUTO_ARIMA:
        if not _pmdarima_available:
            print("SARIMA Error: USE_AUTO_ARIMA is True, but pmdarima is not installed. Install it (`pip install pmdarima`) or set USE_AUTO_ARIMA=False in .env.")
            return nan_df # Return NaN DataFrame

        print("Attempting auto_arima...")
        # Determine if seasonal search should be enabled based on config AND inferred period > 1
        is_seasonal_search: bool = config.SARIMA_AUTO_SEASONAL and seasonal_period > 1
        try:
            # Check if enough data for seasonal differencing if seasonal search is enabled
            min_samples_needed = seasonal_period * 2 if is_seasonal_search else 2
            if len(train_data) < min_samples_needed:
                if is_seasonal_search: # Only warn if we intended to search seasonally
                    warnings.warn(f"Not enough training samples ({len(train_data)}) for auto_arima seasonal search (m={seasonal_period}). Disabling seasonal component search.")
                is_seasonal_search = False # Force non-seasonal search if not enough data

            # Set m=1 if not doing seasonal search to avoid errors/warnings in pmdarima
            m_param = seasonal_period if is_seasonal_search else 1

            auto_model = pm.auto_arima(
                train_data['y'],
                start_p=1, start_q=1,
                max_p=3, max_q=3, # Adjust max orders as needed
                m=m_param,        # Use calculated m
                seasonal=is_seasonal_search, # Enable/disable seasonal search correctly
                start_P=0, start_Q=0,
                max_P=2, max_Q=2,
                d=None, D=None if is_seasonal_search else 0, # Let auto find d, force D=0 if non-seasonal search
                test='adf',           # Use ADF test to find optimal 'd'
                stepwise=True,        # Use efficient stepwise search
                suppress_warnings=True,# Suppress convergence warnings etc. from auto_arima
                error_action='ignore', # Continue trying other models on error
                trace=False,          # Set to True to see models tried
                information_criterion='aic' # Or 'bic'
            )
            final_order = auto_model.order
            # auto_arima returns (0,0,0,0) if seasonal=False, otherwise the seasonal order including m
            final_seasonal_order = auto_model.seasonal_order
            print(f"auto_arima identified Order: {final_order} and Seasonal Order: {final_seasonal_order}")

        except Exception as e:
            print(f"auto_arima failed: {e}. Falling back to manual orders from config.")
            # Fallback logic (use manual orders if auto fails)
            final_order = config.SARIMA_ORDER
            # Re-calculate seasonal order based on inferred period and manual P,D,Q
            if seasonal_period > 1:
                final_seasonal_order = config.SARIMA_SEASONAL_ORDER_NOS + (seasonal_period,)
            else:
                 final_seasonal_order = (0, 0, 0, 0)
            print(f"Using manual Order: {final_order}")
            print(f"Using manual Seasonal Order: {final_seasonal_order}")

    else: # Use Manual Orders
        print(f"Using manual Order: {config.SARIMA_ORDER}")
        final_order = config.SARIMA_ORDER
        # Seasonal order already calculated based on inferred/manual s and config P,D,Q
        if seasonal_period > 1:
            print(f"Using manual Seasonal Order: {final_seasonal_order}")
        else:
            print("Using Non-Seasonal SARIMA (s=1)")


    # --- Fit Final SARIMAX Model ---
    # Check if data frequency is set, required for forecasting index generation by statsmodels
    current_freq = train_data.index.freq
    if current_freq is None:
         warnings.warn("SARIMA Warning: Data frequency is not set on the training index. Forecast index might be incorrect. Attempting to set inferred frequency.")
         inferred_freq = pd.infer_freq(train_data.index)
         if inferred_freq:
             # Create a copy to avoid SettingWithCopyWarning if train_data is used elsewhere
             y_train = train_data['y'].copy()
             y_train.index.freq = inferred_freq
             print(f"Set index frequency to inferred value: {inferred_freq} on copied series")
         else:
             print("SARIMA Error: Cannot determine frequency. SARIMAX forecast may fail or produce incorrect index.")
             return nan_df
    else:
        # If frequency exists, use the original series
        y_train = train_data['y']


    try:
        model = SARIMAX(y_train,
                        order=final_order,
                        seasonal_order=final_seasonal_order,
                        enforce_stationarity=False,
                        enforce_invertibility=False)
        # Use fit_kwargs to suppress convergence warnings if desired during final fit
        results: SARIMAXResultsWrapper = model.fit(disp=False) # disp=False hides convergence messages

        # Get forecast object and confidence intervals
        # Ensure start and end dates match the desired test_index coverage
        # Use the actual test_index provided
        start_index = test_index[0]
        end_index = test_index[-1]

        # Use get_prediction for reliable index handling with time series index
        pred = results.get_prediction(start=start_index, end=end_index)
        forecast_summary: pd.DataFrame = pred.summary_frame(alpha=0.05) # 95% CI

        # Ensure the forecast summary index aligns perfectly with the test index
        # Important if get_prediction returns slightly different index points
        forecast_summary = forecast_summary.reindex(test_index)

        # Extract mean forecast and CIs
        # Rename columns for consistency with Prophet and plotting function
        forecast_values: pd.DataFrame = forecast_summary[['mean', 'mean_ci_lower', 'mean_ci_upper']].rename(
            columns={'mean': 'yhat', 'mean_ci_lower': 'yhat_lower', 'mean_ci_upper': 'yhat_upper'}
        )

        # Double check index assignment
        forecast_values.index = test_index
        print("SARIMA Finished.")
        return forecast_values

    except ValueError as ve:
         # Handle specific errors like non-stationarity if enforce=True or matrix issues
         print(f"SARIMA Final Fit/Forecast ValueError: {ve}")
         print("This might be due to model orders, lack of data, or data properties (e.g., constants).")
         return nan_df
    except np.linalg.LinAlgError as lae:
         print(f"SARIMA Final Fit/Forecast LinAlgError: {lae}")
         print("This often indicates issues with the data or model specification (e.g., perfect multicollinearity in exogenous variables, non-invertible matrices).")
         return nan_df
    except Exception as e:
        print(f"SARIMA Final Fit/Forecast Unexpected Error: {type(e).__name__}: {e}")
        return nan_df # Return NaN DataFrame
# --- END OF FILE src/models/sarima_model.py ---