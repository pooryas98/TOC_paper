# In data_loader.py

import pandas as pd
import os
import warnings
from typing import Tuple, Optional # Added for type hinting

def load_and_prepare_data(file_path: str, date_column: str, value_column: str, freq: Optional[str] = None) -> pd.DataFrame:
    """Loads, validates, and prepares time series data from a CSV file."""
    print(f"Attempting to load data from: {file_path}")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Error: The file '{file_path}' was not found.")

    try:
        # Load data, assuming the first column is the date if date_column matches the first header
        # Need to handle the header row being split across columns in the CSV
        df_header_check = pd.read_csv(file_path, nrows=0) # Read only header
        # Assume date is first col, value is second if names match CSV header
        usecols = [df_header_check.columns[0], df_header_check.columns[1]]
        if df_header_check.columns[0] != date_column or df_header_check.columns[1] != value_column:
             warnings.warn(f"CSV header ({df_header_check.columns[:2]}) doesn't match .env config ({date_column}, {value_column}). Using first two columns.")

        date_col_name: str = df_header_check.columns[0]
        value_col_name_original: str = df_header_check.columns[1]

        # Skip the header row explicitly now
        df: pd.DataFrame = pd.read_csv(
            file_path,
            skiprows=1,
            header=None,
            names=[date_col_name, value_col_name_original],
            usecols=[0, 1],
            parse_dates=[0],
            index_col=0
        )
        print("CSV file loaded successfully (using first two columns).")

    except KeyError as e:
         # This error is less likely now with direct column indexing, but keep for safety
        raise KeyError(f"Error: Problem identifying columns. Check date_column/value_column config ({date_column}, {value_column}) vs CSV header.")
    except Exception as e:
        raise Exception(f"Error loading CSV: {e}")

    # Rename value column to 'y' internally
    df = df[[value_col_name_original]].rename(columns={value_col_name_original: 'y'})
    df = df.sort_index()

    if not isinstance(df.index, pd.DatetimeIndex):
         raise TypeError("Index was not parsed as DatetimeIndex. Check date format in CSV and config.")
    if not pd.api.types.is_numeric_dtype(df['y']):
        try:
            df['y'] = pd.to_numeric(df['y'])
            warnings.warn(f"Value column '{value_col_name_original}' converted to numeric.")
        except ValueError as e:
            raise TypeError(f"Error: Value column '{value_col_name_original}' could not be converted to numeric: {e}")

    nan_count_initial: int = df['y'].isnull().sum()
    if nan_count_initial > 0:
        warnings.warn(f"Found {nan_count_initial} missing value(s) in '{value_col_name_original}' *before* frequency handling. Consider imputation. Models might fail.")
        # OPTIONAL: Add imputation here if needed, e.g., df['y'] = df['y'].fillna(method='ffill')

    if not isinstance(df.index, pd.DatetimeIndex):
        # This check is somewhat redundant after the explicit parsing/sorting but ensures type safety
        raise TypeError("Error: Index is not a DatetimeIndex after initial processing.")

    # --- Frequency Handling ---
    original_freq: Optional[str] = pd.infer_freq(df.index)
    print(f"Inferred frequency from data: {original_freq}")

    final_freq_to_set: Optional[str] = None # Initialize
    if freq: # If frequency is specified in .env
        print(f"Frequency specified in config: {freq}")
        if original_freq == freq:
            print("Data frequency matches specified frequency. Setting index frequency attribute.")
            final_freq_to_set = freq
        else:
            warnings.warn(f"Data's inferred frequency ('{original_freq}') differs from specified frequency ('{freq}'). "
                          f"Proceeding with inferred frequency ('{original_freq}') to avoid introducing NaNs with asfreq(). "
                          "Ensure the specified frequency in .env matches the data's actual time interval ('MS' for this dataset).")
            # Use the inferred frequency if it exists, otherwise leave as None
            final_freq_to_set = original_freq if original_freq else None
    elif original_freq: # If not specified, but inference worked
         print(f"Using inferred frequency: {original_freq}")
         final_freq_to_set = original_freq
    else: # Not specified and inference failed
        warnings.warn("Could not infer time series frequency, and none was specified. Models relying on frequency might error or behave unexpectedly.")
        # final_freq_to_set remains None

    # Set frequency attribute if determined
    if final_freq_to_set:
        df.index.freq = final_freq_to_set
    else:
        # Ensure freq is None if it couldn't be determined/set
        df.index.freq = None

    # --- Final Checks ---
    final_freq_str: Optional[str] = df.index.freqstr
    print(f"Final frequency attribute set to: {final_freq_str}")

    # Check for NaNs *after* all processing
    nan_count_final: int = df['y'].isnull().sum()
    if nan_count_final > nan_count_initial:
         warnings.warn(f"WARNING: {nan_count_final - nan_count_initial} NaNs were introduced during processing (check imputation/logic). Total NaNs: {nan_count_final}")
    elif nan_count_final > 0:
         warnings.warn(f"Data still contains {nan_count_final} NaNs after processing. Consider imputation if not done already.")


    print(f"Data date range: {df.index.min()} to {df.index.max()}")
    print(f"Number of observations: {len(df)}")
    return df

# Updated split function
def split_data_train_val_test(df: pd.DataFrame, validation_size: int, test_size: int) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], pd.DataFrame]:
    """Splits data into training, validation, and testing sets."""
    n: int = len(df)
    if validation_size < 0 or test_size <= 0:
        raise ValueError("validation_size must be >= 0 and test_size must be > 0")
    if n < test_size + validation_size + 1: # Need at least one point for training
        raise ValueError(f"Not enough data points ({n}) for the requested validation ({validation_size}) and test ({test_size}) sizes.")

    test_split_idx: int = n - test_size
    val_split_idx: int = test_split_idx - validation_size

    test_df: pd.DataFrame = df.iloc[test_split_idx:] # Use iloc for integer-based slicing
    train_df: pd.DataFrame
    val_df: Optional[pd.DataFrame]

    if validation_size > 0:
        val_df = df.iloc[val_split_idx:test_split_idx]
        train_df = df.iloc[:val_split_idx]
        print(f"Training data shape:   {train_df.shape}")
        print(f"Validation data shape: {val_df.shape}")
    else: # No validation set
        val_df = None # Explicitly return None for clarity
        train_df = df.iloc[:test_split_idx]
        print(f"Training data shape:   {train_df.shape}")
        print("Validation set disabled (size=0).")

    print(f"Testing data shape:    {test_df.shape}")

    return train_df, val_df, test_df
# --- END OF FILE data_loader.py ---