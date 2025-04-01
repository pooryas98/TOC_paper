import pandas as pd
import numpy as np
from prophet import Prophet
import warnings
import logging
import pickle # For saving model
import os # For saving model path
from typing import Optional, Dict, Any, Tuple

logger = logging.getLogger(__name__)

# Optional: Import holidays if needed
_holidays_available = False
try:
    import holidays
    _holidays_available = True
except ImportError:
    pass

def run_prophet(
    train_data: pd.DataFrame,
    test_periods: int,
    test_index: pd.DatetimeIndex,
    config_params: Dict[str, Any]
) -> Tuple[pd.DataFrame, Optional[Dict[str, Any]], Optional[Prophet]]:
    """
    Trains Prophet model and returns forecasts, parameters used, and the fitted model.
    Returns forecast DataFrame, parameter dict, and fitted model object (or Nones on failure).
    """
    logger.info("--- Running Prophet ---")
    model_params_used: Dict[str, Any] = {'model_type': 'Prophet'}
    fitted_model: Optional[Prophet] = None
    nan_df = pd.DataFrame(np.nan, index=test_index, columns=['yhat', 'yhat_lower', 'yhat_upper'])

    if train_data['y'].isnull().any():
        logger.warning("Prophet: Training data contains NaN values after potential imputation. Prophet can handle them, but results might be affected.")
        # No need to return early, Prophet handles internal NaNs

    prophet_train_df: pd.DataFrame = train_data.reset_index().rename(
        columns={'index': 'ds', train_data.index.name: 'ds', 'y': 'y'}
    )

    if not pd.api.types.is_datetime64_any_dtype(prophet_train_df['ds']):
         try:
             prophet_train_df['ds'] = pd.to_datetime(prophet_train_df['ds'])
         except Exception as e:
             logger.error(f"Prophet Error: Could not convert 'ds' column to datetime: {e}")
             return nan_df, model_params_used, None

    # Handle logistic growth requirements
    growth = config_params['PROPHET_GROWTH']
    cap = config_params.get('PROPHET_CAP')
    floor = config_params.get('PROPHET_FLOOR')
    if growth == 'logistic':
        if cap is None:
            logger.error("Prophet Error: Growth is 'logistic' but PROPHET_CAP is not set.")
            return nan_df, model_params_used, None
        prophet_train_df['cap'] = cap
        if floor is not None:
            prophet_train_df['floor'] = floor
        model_params_used['cap'] = cap
        model_params_used['floor'] = floor

    model_params_used['growth'] = growth
    model_params_used['seasonality_mode'] = config_params['PROPHET_SEASONALITY_MODE']
    model_params_used['interval_width'] = config_params['PROPHET_INTERVAL_WIDTH']
    model_params_used['yearly_seasonality'] = config_params['PROPHET_ADD_YEARLY_SEASONALITY']
    model_params_used['weekly_seasonality'] = config_params['PROPHET_ADD_WEEKLY_SEASONALITY']
    model_params_used['daily_seasonality'] = config_params['PROPHET_ADD_DAILY_SEASONALITY']

    # Handle Holidays
    country_holidays = config_params.get('PROPHET_COUNTRY_HOLIDAYS')
    prophet_holidays = None
    if country_holidays:
        if _holidays_available:
             # Ensure we cover the full range (train + test) for holiday generation
             min_date = prophet_train_df['ds'].min()
             max_date_forecast = test_index.max() # Use the actual end date of the forecast period
             year_list = list(range(min_date.year, max_date_forecast.year + 1))

             try:
                 # Use holidays library to generate holidays DataFrame
                 country_holidays_obj = holidays.country_holidays(country_holidays, years=year_list)
                 if country_holidays_obj:
                     prophet_holidays = pd.DataFrame(list(country_holidays_obj.items()), columns=['ds', 'holiday'])
                     prophet_holidays['ds'] = pd.to_datetime(prophet_holidays['ds'])
                     # Filter holidays to be within the actual data + forecast range
                     prophet_holidays = prophet_holidays[(prophet_holidays['ds'] >= min_date) & (prophet_holidays['ds'] <= max_date_forecast)]
                     logger.info(f"Using built-in country holidays for: {country_holidays} ({len(prophet_holidays)} instances found)")
                     model_params_used['country_holidays_used'] = country_holidays
                 else:
                      logger.warning(f"Could not retrieve holidays for country code: {country_holidays}")
                      model_params_used['country_holidays_used'] = f"{country_holidays} (Not Found)"

             except Exception as holiday_err:
                  logger.error(f"Error processing holidays for {country_holidays}: {holiday_err}")
                  model_params_used['country_holidays_used'] = f"{country_holidays} (Error: {holiday_err})"
        else:
            logger.warning("PROPHET_COUNTRY_HOLIDAYS is set, but 'holidays' library not installed. Skipping holidays.")
            model_params_used['country_holidays_used'] = f"{country_holidays} (Library Missing)"
    else:
         model_params_used['country_holidays_used'] = None


    try:
        model = Prophet(
            growth=growth,
            yearly_seasonality=config_params['PROPHET_ADD_YEARLY_SEASONALITY'],
            weekly_seasonality=config_params['PROPHET_ADD_WEEKLY_SEASONALITY'],
            daily_seasonality=config_params['PROPHET_ADD_DAILY_SEASONALITY'],
            seasonality_mode=config_params['PROPHET_SEASONALITY_MODE'],
            interval_width=config_params['PROPHET_INTERVAL_WIDTH'],
            holidays=prophet_holidays
        )


        logger.info("Fitting Prophet model...")
        fitted_model = model.fit(prophet_train_df)
        logger.info("Prophet fitting complete.")

        # Determine frequency for future dataframe
        freq_str: Optional[str] = train_data.index.freqstr
        if not freq_str:
            freq_str = pd.infer_freq(train_data.index)
            if freq_str:
                logger.warning(f"Prophet using inferred frequency for future dataframe: {freq_str}")
            else:
                logger.error("Prophet Error: Cannot determine frequency for make_future_dataframe.")
                model_params_used['fit_error'] = "Frequency Undetermined"
                return nan_df, model_params_used, fitted_model # Return model even if predict fails

        if not freq_str: # Double check after inference attempt
             model_params_used['fit_error'] = "Frequency Undetermined"
             return nan_df, model_params_used, fitted_model

        future: pd.DataFrame = model.make_future_dataframe(periods=test_periods, freq=freq_str)

        # Add cap/floor to future dataframe if logistic growth
        if growth == 'logistic':
            future['cap'] = cap
            if floor is not None:
                future['floor'] = floor


        if future.empty or len(future) != len(prophet_train_df) + test_periods:
             raise ValueError(f"Failed to create future dataframe correctly. Expected {len(prophet_train_df) + test_periods}, got {len(future)}.")

        logger.info("Generating Prophet forecast...")
        forecast: pd.DataFrame = model.predict(future)
        logger.info("Prophet forecast generated.")

        forecast_subset: pd.DataFrame = forecast.set_index('ds')[['yhat', 'yhat_lower', 'yhat_upper']].iloc[-test_periods:]

        if len(forecast_subset) != test_periods:
             logger.warning(f"Prophet Warning: Forecast subset length ({len(forecast_subset)}) does not match test_periods ({test_periods}). Attempting reindex.")
             forecast_subset = forecast_subset.reindex(test_index) # Align explicitly
        else:
            # Assign the correct index from the test set
            forecast_subset.index = test_index

        if forecast_subset['yhat'].isnull().any():
            logger.warning("Prophet forecast contains NaN values. Check model stability or data.")

        logger.info("Prophet Finished.")
        return forecast_subset, model_params_used, fitted_model

    except Exception as e:
        logger.error(f"Prophet Error during fit/predict: {type(e).__name__}: {e}", exc_info=True)
        model_params_used['fit_error'] = f"{type(e).__name__}: {e}"
        # Return model if it was fitted before predict failed
        return nan_df, model_params_used, fitted_model

def save_prophet_model(model: Prophet, file_path: str):
    """Saves a fitted Prophet model using pickle."""
    # Prophet models can sometimes have issues with standard pickle, especially across versions.
    # Using model.__dict__ might be more robust but requires care on loading. Sticking to pickle for now.
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as f:
            pickle.dump(model, f)
        logger.info(f"Prophet model saved successfully to: {file_path}")
    except Exception as e:
        logger.error(f"Error saving Prophet model to {file_path}: {e}", exc_info=True)

