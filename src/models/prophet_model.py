# --- START OF FILE src/models/prophet_model.py ---

import pandas as pd
import numpy as np
from prophet import Prophet # Use the newer package name 'prophet'
import warnings
from typing import Optional # Added for type hinting

def run_prophet(train_data: pd.DataFrame, test_periods: int, test_index: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Trains Prophet model and returns forecasts along with confidence intervals.
    Returns a DataFrame with columns ['yhat', 'yhat_lower', 'yhat_upper'].
    """
    print("Running Prophet...")
    # Prepare output DataFrame in case of errors
    nan_df = pd.DataFrame(np.nan, index=test_index, columns=['yhat', 'yhat_lower', 'yhat_upper'])

    # Check for NaNs in training data - Prophet can handle them, but warns
    if train_data['y'].isnull().any():
        warnings.warn("Prophet Warning: Training data contains NaN values. Prophet may proceed but results might be affected.")
        # Optional: Impute NaNs before passing to Prophet if desired
        # train_data = train_data.fillna(method='ffill') # Example imputation

    # Prophet requires columns 'ds' and 'y'
    prophet_train_df: pd.DataFrame = train_data.reset_index().rename(
        columns={'index': 'ds', train_data.index.name: 'ds', 'y': 'y'}
    )

    # Ensure 'ds' column is datetime
    if not pd.api.types.is_datetime64_any_dtype(prophet_train_df['ds']):
         try:
             prophet_train_df['ds'] = pd.to_datetime(prophet_train_df['ds'])
         except Exception as e:
             print(f"Prophet Error: Could not convert 'ds' column to datetime: {e}")
             return nan_df

    # Get frequency for Prophet's future dataframe
    freq_str: Optional[str] = train_data.index.freqstr
    if not freq_str:
        freq_str = pd.infer_freq(train_data.index)
        if freq_str:
             warnings.warn(f"Prophet using inferred frequency: {freq_str}")
        else:
            # Cannot make future dataframe without frequency
             print("Prophet Error: Cannot determine frequency for make_future_dataframe. Ensure data has consistent intervals or set frequency.")
             return nan_df # Return DataFrame with NaNs

    model = Prophet()
    try:
        # Suppress informational messages from Prophet during fit/predict if desired
        # import logging
        # logging.getLogger('prophet').setLevel(logging.WARNING) # Suppress INFO messages
        # logging.getLogger('cmdstanpy').setLevel(logging.WARNING) # Suppress cmdstanpy messages

        model.fit(prophet_train_df)

        # Create future dataframe using the established frequency
        future: pd.DataFrame = model.make_future_dataframe(periods=test_periods, freq=freq_str)

        # Check if future dataframe generation was successful
        if future.empty or len(future) != len(prophet_train_df) + test_periods:
             raise ValueError("Failed to create future dataframe correctly.")

        forecast: pd.DataFrame = model.predict(future)

        # Extract forecast and CIs for the test period
        # Ensure 'ds' is the index before slicing
        forecast_subset: pd.DataFrame = forecast.set_index('ds')[['yhat', 'yhat_lower', 'yhat_upper']].iloc[-test_periods:]

        # Check if the subset extraction worked and aligns with expected length
        if len(forecast_subset) != test_periods:
             warnings.warn(f"Prophet Warning: Forecast subset length ({len(forecast_subset)}) does not match test_periods ({test_periods}). Alignment issue?")
             # Attempt to reindex anyway, might result in NaNs if ds values don't match
             forecast_subset = forecast_subset.reindex(test_index)
        else:
            # Ensure forecast index aligns with test data index
            forecast_subset.index = test_index

        print("Prophet Finished.")
        return forecast_subset # Return DataFrame

    except Exception as e:
        print(f"Prophet Error during fit/predict: {e}")
        # Return DataFrame with NaNs on error
        return nan_df
# --- END OF FILE src/models/prophet_model.py ---