# src/data_loader.py

import pandas as pd
import os
import warnings
import logging
from typing import Tuple, Optional, Dict, Any

logger = logging.getLogger(__name__)

def apply_imputation(df: pd.DataFrame, method: str, value_col: str = 'y') -> pd.DataFrame:
    """Applies imputation to the specified column."""
    if method == 'none' or method is None: return df

    original_nan_count = df[value_col].isnull().sum()
    if original_nan_count == 0:
        logger.info("No NaNs found, imputation not needed.")
        return df

    logger.info(f"Applying imputation '{method}' to '{value_col}' ({original_nan_count} NaNs present).")
    df_imputed = df.copy()

    if method == 'ffill': df_imputed[value_col] = df_imputed[value_col].ffill()
    elif method == 'bfill': df_imputed[value_col] = df_imputed[value_col].bfill()
    elif method == 'mean':
        fill_value = df_imputed[value_col].mean()
        df_imputed[value_col] = df_imputed[value_col].fillna(fill_value)
        logger.info(f"Imputed with mean: {fill_value:.4f}")
    elif method == 'median':
        fill_value = df_imputed[value_col].median()
        df_imputed[value_col] = df_imputed[value_col].fillna(fill_value)
        logger.info(f"Imputed with median: {fill_value:.4f}")
    elif method == 'interpolate':
        df_imputed[value_col] = df_imputed[value_col].interpolate(method='linear', limit_direction='both')
    else:
        logger.warning(f"Unknown imputation method '{method}'. No imputation applied.")
        return df # Return original

    # Verify imputation
    final_nan_count = df_imputed[value_col].isnull().sum()
    if final_nan_count > 0:
        logger.warning(f"Imputation '{method}' finished, but {final_nan_count} NaNs still remain.")
    else:
        logger.info("Imputation complete. No remaining NaNs.")
    return df_imputed


def load_and_prepare_data(
    file_path: str,
    date_column: str,
    value_column: str,
    config_params: Dict[str, Any]
) -> Tuple[Optional[pd.DataFrame], Optional[Dict[str, Any]]]:
    """Loads, validates, imputes, and prepares time series data from a CSV."""
    logger.info(f"Loading data from: {file_path}")
    data_params: Dict[str, Any] = {
        'file_path': file_path,
        'date_column_config': date_column,
        'value_column_config': value_column,
        'requested_freq': config_params['TIME_SERIES_FREQUENCY'],
        'imputation_method': config_params['DATA_IMPUTATION_METHOD']
    }

    if not os.path.exists(file_path):
        logger.error(f"Fatal Error: File '{file_path}' not found.")
        data_params['status'] = 'Error - File not found'
        return None, data_params

    try:
        # Load standard CSV with header, parse date column, set as index
        df: pd.DataFrame = pd.read_csv(
            file_path,
            parse_dates=[date_column],
            index_col=date_column
        )
        logger.info(f"CSV loaded. Columns: {df.columns.tolist()}")

        # Check if value column exists
        if value_column not in df.columns:
             logger.error(f"Fatal Error: Value column '{value_column}' (from config) not found in CSV columns: {df.columns.tolist()}")
             data_params['status'] = f"Error - Value column '{value_column}' not found"
             return None, data_params

        # Select only value column, rename to 'y' internally
        df = df[[value_column]].rename(columns={value_column: 'y'})

    except KeyError as e: # Catch wrong date_column
        logger.error(f"KeyError during CSV load: {e}. Check DATE_COLUMN ('{date_column}') matches CSV header.")
        data_params['status'] = f'Error - KeyError loading index/date column: {e}'
        return None, data_params
    except ValueError as e: # Catch date parsing errors
        logger.error(f"ValueError during CSV load/parse: {e}. Check date format in '{date_column}'.")
        data_params['status'] = f'Error - ValueError loading/parsing: {e}'
        return None, data_params
    except Exception as e:
        logger.error(f"Error loading/processing CSV: {e}", exc_info=True)
        data_params['status'] = f'Error - CSV Load/Process: {e}'
        return None, data_params

    # --- Validation and Cleaning ---
    df = df.sort_index()
    if not isinstance(df.index, pd.DatetimeIndex):
         logger.error("Fatal Error: Index not parsed as DatetimeIndex. Check date format or loading.")
         data_params['status'] = 'Error - Index not DatetimeIndex'
         return None, data_params

    if not pd.api.types.is_numeric_dtype(df['y']):
        logger.warning(f"Value column '{value_column}' ('y') not numeric. Attempting conversion.")
        original_dtype = df['y'].dtype
        df['y'] = pd.to_numeric(df['y'], errors='coerce')
        converted_nan_count = df['y'].isnull().sum()
        if converted_nan_count > 0: logger.warning(f"Coercion to numeric introduced {converted_nan_count} NaNs.")
        data_params['value_col_original_dtype'] = str(original_dtype)

    # --- Imputation (Before frequency setting) ---
    imputation_method = config_params['DATA_IMPUTATION_METHOD']
    initial_nan_count = df['y'].isnull().sum()
    data_params['nan_count_before_imputation'] = initial_nan_count
    if initial_nan_count > 0 and imputation_method != 'none':
         df = apply_imputation(df, imputation_method, 'y')
         data_params['nan_count_after_imputation'] = df['y'].isnull().sum()
    elif imputation_method == 'none' and initial_nan_count > 0:
        logger.warning(f"Initial data has {initial_nan_count} NaNs; imputation is 'none'. Some models may fail.")

    # --- Frequency Handling ---
    specified_freq = config_params['TIME_SERIES_FREQUENCY']
    original_freq = None
    try: # Try inferring frequency
        original_freq = pd.infer_freq(df.index)
        logger.info(f"Inferred frequency: {original_freq}")
    except Exception as infer_err: logger.warning(f"Could not infer frequency: {infer_err}")

    final_freq_to_set: Optional[str] = None
    if specified_freq:
        logger.info(f"Frequency specified in config: {specified_freq}")
        if original_freq == specified_freq:
            logger.info("Data frequency matches specified.")
            final_freq_to_set = specified_freq
        elif original_freq: # Prioritize inferred if differs
             logger.warning(f"Inferred freq ('{original_freq}') differs from specified ('{specified_freq}'). Using inferred.")
             final_freq_to_set = original_freq
        else: # No inferred, but specified
            logger.warning(f"Could not infer freq; attempting to set specified '{specified_freq}'.")
            final_freq_to_set = specified_freq
    elif original_freq: # No specified, use inferred
         logger.info(f"Using inferred frequency: {original_freq}")
         final_freq_to_set = original_freq
    else: # No specified, no inferred
        logger.warning("Could not infer or find specified frequency. Models relying on freq might error.")

    # Set frequency attribute if determined
    if final_freq_to_set:
        try:
            df_copy = df.copy() # Avoid SettingWithCopyWarning
            df_copy.index = pd.DatetimeIndex(df_copy.index, freq=final_freq_to_set)
            df = df_copy
            if df.index.freqstr != final_freq_to_set: # Verify
                 logger.warning(f"Attempted set freq '{final_freq_to_set}', but index.freqstr is '{df.index.freqstr}'.")
        except ValueError as set_freq_err: # Catch non-monotonic etc.
             logger.error(f"Failed to set frequency '{final_freq_to_set}': {set_freq_err}. Frequency remains None.")
             final_freq_to_set = None
             df.index.freq = None
    else: df.index.freq = None # Ensure None if not set

    final_freq_str: Optional[str] = df.index.freqstr
    logger.info(f"Final frequency set on DataFrame index: {final_freq_str}")
    data_params['final_frequency_set'] = final_freq_str

    # --- Final Checks ---
    nan_count_final = df['y'].isnull().sum()
    if nan_count_final > 0:
         logger.warning(f"Data still contains {nan_count_final} NaNs after processing.")
    data_params['nan_count_final'] = nan_count_final
    data_params['data_start_date'] = df.index.min().strftime('%Y-%m-%d %H:%M:%S')
    data_params['data_end_date'] = df.index.max().strftime('%Y-%m-%d %H:%M:%S')
    data_params['num_observations'] = len(df)
    data_params['status'] = 'Loaded Successfully'

    logger.info(f"Data loaded: {len(df)} obs from {data_params['data_start_date']} to {data_params['data_end_date']}.")
    logger.debug(f"Final data head:\n{df.head()}")
    return df, data_params


# Split function
def split_data_train_val_test(df: pd.DataFrame, validation_size: int, test_size: int) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], pd.DataFrame]:
    """Splits data into training, validation, and testing sets."""
    n: int = len(df)
    if validation_size < 0 or test_size <= 0:
        raise ValueError("validation_size must be >= 0 and test_size must be > 0")
    min_train_size = 1
    if n < test_size + validation_size + min_train_size:
        raise ValueError(f"Not enough data ({n}) for requested validation ({validation_size}), test ({test_size}), and min train ({min_train_size}) sizes.")

    test_split_idx: int = n - test_size
    val_split_idx: int = test_split_idx - validation_size

    test_df: pd.DataFrame = df.iloc[test_split_idx:]
    train_df: pd.DataFrame
    val_df: Optional[pd.DataFrame] = None

    if validation_size > 0:
        if val_split_idx < 0: raise ValueError(f"Validation size ({validation_size}) too large.")
        val_df = df.iloc[val_split_idx:test_split_idx]
        train_df = df.iloc[:val_split_idx]
        if train_df.empty: raise ValueError(f"Split resulted in empty training set (n={n}, val={validation_size}, test={test_size}).")
        logger.info(f"Splitting data: Train {len(train_df)} ({train_df.index.min()} - {train_df.index.max()})")
        logger.info(f"             : Validation {len(val_df)} ({val_df.index.min()} - {val_df.index.max()})")
    else: # No validation set
        if test_split_idx <= 0: raise ValueError(f"Test size ({test_size}) too large, no data for training (n={n}).")
        train_df = df.iloc[:test_split_idx]
        logger.info(f"Splitting data: Train {len(train_df)} ({train_df.index.min()} - {train_df.index.max()})")
        logger.info("             : Validation set disabled (size=0).")

    logger.info(f"             : Test {len(test_df)} ({test_df.index.min()} - {test_df.index.max()})")
    return train_df, val_df, test_df