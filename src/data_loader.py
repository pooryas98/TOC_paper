# src/data_loader.py

import pandas as pd
import os
import warnings
import logging # Use logging
from typing import Tuple, Optional, Dict, Any # Added Dict, Any

logger = logging.getLogger(__name__)

def apply_imputation(df: pd.DataFrame, method: str, value_col: str = 'y') -> pd.DataFrame:
    """Applies imputation to the specified column."""
    if method == 'none' or method is None:
        return df

    original_nan_count = df[value_col].isnull().sum()
    if original_nan_count == 0:
        logger.info("No NaNs found, imputation not needed.")
        return df

    logger.info(f"Applying imputation method '{method}' to column '{value_col}' ({original_nan_count} NaNs present).")
    df_imputed = df.copy() # Work on a copy

    if method == 'ffill':
        df_imputed[value_col] = df_imputed[value_col].ffill()
    elif method == 'bfill':
        df_imputed[value_col] = df_imputed[value_col].bfill()
    elif method == 'mean':
        fill_value = df_imputed[value_col].mean()
        df_imputed[value_col] = df_imputed[value_col].fillna(fill_value)
        logger.info(f"Imputed with mean: {fill_value:.4f}")
    elif method == 'median':
        fill_value = df_imputed[value_col].median()
        df_imputed[value_col] = df_imputed[value_col].fillna(fill_value)
        logger.info(f"Imputed with median: {fill_value:.4f}")
    elif method == 'interpolate':
        df_imputed[value_col] = df_imputed[value_col].interpolate(method='linear', limit_direction='both') # Limit direction handles edges
    else:
        logger.warning(f"Unknown imputation method '{method}'. No imputation applied.")
        return df # Return original if method unknown

    # Verify imputation effectiveness
    final_nan_count = df_imputed[value_col].isnull().sum()
    if final_nan_count > 0:
        logger.warning(f"Imputation method '{method}' finished, but {final_nan_count} NaNs still remain (check method/data).")
    else:
        logger.info("Imputation complete. No remaining NaNs.")

    return df_imputed


def load_and_prepare_data(
    file_path: str,
    date_column: str,
    value_column: str,
    config_params: Dict[str, Any] # Pass config
) -> Tuple[Optional[pd.DataFrame], Optional[Dict[str, Any]]]:
    """Loads, validates, imputes, and prepares time series data from a standard CSV."""
    logger.info(f"Attempting to load data from: {file_path}")
    data_params: Dict[str, Any] = {
        'file_path': file_path,
        'date_column_config': date_column,
        'value_column_config': value_column,
        'requested_freq': config_params['TIME_SERIES_FREQUENCY'],
        'imputation_method': config_params['DATA_IMPUTATION_METHOD']
        # Removed csv_header_* params as they are no longer relevant
    }

    if not os.path.exists(file_path):
        logger.error(f"Fatal Error: The file '{file_path}' was not found.")
        data_params['status'] = 'Error - File not found'
        return None, data_params

    try:
        # --- MODIFICATION START ---
        # Simplified loading for standard CSV with header
        df: pd.DataFrame = pd.read_csv(
            file_path,
            parse_dates=[date_column], # Parse the date column specified in config
            index_col=date_column      # Set the date column specified in config as index
        )
        logger.info(f"CSV file loaded successfully. Columns found: {df.columns.tolist()}")

        # Check if the configured value column exists
        if value_column not in df.columns:
             logger.error(f"Fatal Error: Value column '{value_column}' configured in '.env' not found in CSV file columns: {df.columns.tolist()}")
             data_params['status'] = f"Error - Value column '{value_column}' not found"
             return None, data_params

        # Select only the value column and rename it to 'y' for internal consistency
        df = df[[value_column]].rename(columns={value_column: 'y'})
        # --- MODIFICATION END ---

    except KeyError as e: # Might occur if date_column is wrong in config/CSV
        logger.error(f"KeyError during CSV loading: {e}. Check DATE_COLUMN ('{date_column}') in config matches the CSV header.")
        data_params['status'] = f'Error - KeyError loading index/date column: {e}'
        return None, data_params
    except ValueError as e: # Might occur during date parsing
        logger.error(f"ValueError during CSV loading/parsing: {e}. Check date format in column '{date_column}'.")
        data_params['status'] = f'Error - ValueError loading/parsing: {e}'
        return None, data_params
    except Exception as e:
        logger.error(f"Error loading or processing CSV: {e}", exc_info=True)
        data_params['status'] = f'Error - CSV Load/Process: {e}'
        return None, data_params

    # --- Validation and Cleaning (Keep this section) ---
    df = df.sort_index()
    if not isinstance(df.index, pd.DatetimeIndex):
         # This check might be redundant now with parse_dates/index_col, but keep as safeguard
         logger.error("Fatal Error: Index was not parsed as DatetimeIndex. Check date format in CSV or loading parameters.")
         data_params['status'] = 'Error - Index not DatetimeIndex after load'
         return None, data_params

    if not pd.api.types.is_numeric_dtype(df['y']):
        logger.warning(f"Value column '{value_column}' (loaded as 'y') is not numeric. Attempting conversion.")
        original_dtype = df['y'].dtype
        df['y'] = pd.to_numeric(df['y'], errors='coerce')
        converted_nan_count = df['y'].isnull().sum()
        if converted_nan_count > 0:
            logger.warning(f"Coercion to numeric introduced {converted_nan_count} NaNs.")
        data_params['value_col_original_dtype'] = str(original_dtype)

    # --- Imputation (Before frequency setting) (Keep this section) ---
    imputation_method = config_params['DATA_IMPUTATION_METHOD']
    initial_nan_count = df['y'].isnull().sum()
    data_params['nan_count_before_imputation'] = initial_nan_count
    if initial_nan_count > 0 and imputation_method != 'none':
         df = apply_imputation(df, imputation_method, 'y')
         data_params['nan_count_after_imputation'] = df['y'].isnull().sum()
    elif imputation_method == 'none' and initial_nan_count > 0:
        logger.warning(f"Initial data has {initial_nan_count} NaNs and imputation is 'none'. Models like SARIMA may fail.")

    # --- Frequency Handling (Keep this section) ---
    specified_freq = config_params['TIME_SERIES_FREQUENCY']
    original_freq = None
    try:
        # Try inferring frequency even if one is specified, for comparison/logging
        original_freq = pd.infer_freq(df.index)
        logger.info(f"Inferred frequency from data: {original_freq}")
    except Exception as infer_err:
        # Catch potential errors during inference on irregular data
        logger.warning(f"Could not infer frequency: {infer_err}")

    final_freq_to_set: Optional[str] = None
    if specified_freq:
        logger.info(f"Frequency specified in config: {specified_freq}")
        if original_freq == specified_freq:
            logger.info("Data frequency matches specified frequency.")
            final_freq_to_set = specified_freq
        elif original_freq:
             # Decide strategy: Warn and use inferred? Warn and use specified? Force specified?
             # Current logic prioritizes inferred if it exists and differs. Let's keep it.
             logger.warning(f"Data's inferred freq ('{original_freq}') differs from specified freq ('{specified_freq}'). "
                           f"Using inferred frequency ('{original_freq}') to avoid potential issues.")
             final_freq_to_set = original_freq
             # Alternative: Force specified freq
             # logger.warning(f"Data's inferred freq ('{original_freq}') differs from specified freq ('{specified_freq}'). "
             #               f"Attempting to set specified frequency '{specified_freq}'.")
             # final_freq_to_set = specified_freq
        else: # No inferred freq, but one was specified
            logger.warning(f"Could not infer frequency, but '{specified_freq}' was specified. Attempting to set specified frequency.")
            final_freq_to_set = specified_freq
    elif original_freq: # No specified freq, use inferred if available
         logger.info(f"Using inferred frequency: {original_freq}")
         final_freq_to_set = original_freq
    else: # No specified freq, no inferred freq
        logger.warning("Could not infer time series frequency, and none was specified. Models relying on frequency might error or behave unexpectedly.")

    # Set frequency attribute if determined
    if final_freq_to_set:
        try:
            # Create a copy to avoid SettingWithCopyWarning if df is a slice
            # Though current logic should return the full df
            df_copy = df.copy()
            df_copy.index = pd.DatetimeIndex(df_copy.index, freq=final_freq_to_set)
            df = df_copy # Assign back
            # Verify it was set
            if df.index.freqstr != final_freq_to_set:
                 logger.warning(f"Attempted to set frequency to '{final_freq_to_set}', but index.freqstr is now '{df.index.freqstr}'.")

        except ValueError as set_freq_err: # Catch errors like non-monotonic index
             logger.error(f"Failed to set frequency '{final_freq_to_set}' on index: {set_freq_err}. Frequency will remain None. Check if data is sorted and has regular intervals.")
             final_freq_to_set = None # Reset if setting failed
             df.index.freq = None # Ensure it's None
    else:
         df.index.freq = None # Ensure it's None if not set

    final_freq_str: Optional[str] = df.index.freqstr # Read back the set frequency
    logger.info(f"Final frequency attribute set on DataFrame index: {final_freq_str}")
    data_params['final_frequency_set'] = final_freq_str


    # --- Final Checks (Keep this section) ---
    nan_count_final = df['y'].isnull().sum()
    if nan_count_final > 0:
         logger.warning(f"Data still contains {nan_count_final} NaNs after all processing. Ensure this is intended or handled by models.")
    data_params['nan_count_final'] = nan_count_final

    data_params['data_start_date'] = df.index.min().strftime('%Y-%m-%d %H:%M:%S')
    data_params['data_end_date'] = df.index.max().strftime('%Y-%m-%d %H:%M:%S')
    data_params['num_observations'] = len(df)
    data_params['status'] = 'Loaded Successfully'

    logger.info(f"Data loaded: {len(df)} observations from {data_params['data_start_date']} to {data_params['data_end_date']}.")
    logger.debug(f"Final data head:\n{df.head()}")
    return df, data_params


# Updated split function (Keep this section as is)
def split_data_train_val_test(df: pd.DataFrame, validation_size: int, test_size: int) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], pd.DataFrame]:
    """Splits data into training, validation, and testing sets."""
    n: int = len(df)
    if validation_size < 0 or test_size <= 0:
        raise ValueError("validation_size must be >= 0 and test_size must be > 0")
    min_train_size = 1 # Need at least one point for training baseline
    # Add specific check for NN_STEPS here? No, handle within NN module if train is too small.
    if n < test_size + validation_size + min_train_size:
        raise ValueError(f"Not enough data points ({n}) for the requested validation ({validation_size}), test ({test_size}), and min train ({min_train_size}) sizes.")

    test_split_idx: int = n - test_size
    val_split_idx: int = test_split_idx - validation_size

    test_df: pd.DataFrame = df.iloc[test_split_idx:]
    train_df: pd.DataFrame
    val_df: Optional[pd.DataFrame] = None # Initialize val_df

    if validation_size > 0:
        if val_split_idx < 0: # Check if validation pushes into negative index
             raise ValueError(f"Validation size ({validation_size}) too large for available data before test set.")
        val_df = df.iloc[val_split_idx:test_split_idx]
        train_df = df.iloc[:val_split_idx]
        if train_df.empty:
             raise ValueError(f"Train/Val/Test split resulted in an empty training set (n={n}, val={validation_size}, test={test_size}).")
        logger.info(f"Splitting data: Train {len(train_df)} ({train_df.index.min()} - {train_df.index.max()})")
        logger.info(f"             : Validation {len(val_df)} ({val_df.index.min()} - {val_df.index.max()})")
    else: # No validation set
        if test_split_idx <= 0: # Train split needs at least one point
             raise ValueError(f"Test size ({test_size}) too large, leaving no data for training set (n={n}).")
        train_df = df.iloc[:test_split_idx]
        # No need to check train_df.empty here because test_split_idx > 0 ensures it
        logger.info(f"Splitting data: Train {len(train_df)} ({train_df.index.min()} - {train_df.index.max()})")
        logger.info("             : Validation set disabled (size=0).")

    logger.info(f"             : Test {len(test_df)} ({test_df.index.min()} - {test_df.index.max()})")

    return train_df, val_df, test_df