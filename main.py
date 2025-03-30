# --- START OF FILE main.py ---

import pandas as pd
import numpy as np
import tensorflow as tf
import warnings
import sys
import time # For timing models
import os # For creating directories potentially
from typing import Dict, List, Tuple, Optional, Any # Added for type hinting

# Import project modules
from src import config
from src import data_loader
from src.models import sarima_model, prophet_model, nn_models
from src import evaluation
from src import plotting

def set_seeds(seed: int) -> None:
    """Sets random seeds for reproducibility."""
    np.random.seed(seed)
    tf.random.set_seed(seed)
    # Add other library seeds if necessary (e.g., random.seed(seed))
    print(f"Set random seeds to: {seed}")

def run_comparison() -> None:
    """Runs the full forecasting model comparison workflow."""
    print("--- Starting Forecasting Comparison ---")

    # Set seeds using value from config.py
    set_seeds(config.RANDOM_SEED)

    # --- 1. Load Data & Split ---
    df: pd.DataFrame
    train_df: pd.DataFrame
    val_df: Optional[pd.DataFrame]
    test_df: pd.DataFrame
    try:
        # Uses variables from config.py
        df = data_loader.load_and_prepare_data(
            file_path=config.CSV_FILE_PATH,
            date_column=config.DATE_COLUMN,
            value_column=config.VALUE_COLUMN,
            freq=config.TIME_SERIES_FREQUENCY
        )
        # Use the split function from data_loader.py
        train_df, val_df, test_df = data_loader.split_data_train_val_test(
            df, config.VALIDATION_SIZE, config.TEST_SIZE
        )
        # Check if val_df is None (if VALIDATION_SIZE was 0)
        if val_df is None and config.VALIDATION_SIZE > 0:
            warnings.warn("Validation size > 0 but val_df is None. Check split logic.")

    except (FileNotFoundError, KeyError, TypeError, ValueError, Exception) as e:
        print(f"\n--- Data Loading/Splitting Failed ---")
        print(f"Error: {e}")
        # Suggest checking specific config vars
        print(f"Please check configuration (e.g., CSV_FILE_PATH='{config.CSV_FILE_PATH}', "
              f"DATE_COLUMN='{config.DATE_COLUMN}', VALUE_COLUMN='{config.VALUE_COLUMN}') "
              f"in .env and the CSV file format/content.")
        sys.exit(1) # Exit if data loading fails

    # --- 2. Initialize Results Storage ---
    # Use a dictionary to store forecast results (DataFrames from models)
    forecast_results: Dict[str, pd.DataFrame] = {}
    evaluation_results: List[Dict[str, Any]] = []
    model_runtimes: Dict[str, float] = {}
    test_periods: int = len(test_df)
    test_index: pd.DatetimeIndex = test_df.index # Store the test index

    # --- 3. Run Models ---

    # SARIMA
    print("\n--- Running SARIMA ---")
    start_time: float = time.time()
    try:
        # run_sarima now returns a DataFrame ['yhat', 'yhat_lower', 'yhat_upper']
        sarima_forecast_df: pd.DataFrame = sarima_model.run_sarima(
            train_data=train_df.copy(),
            test_periods=test_periods,
            test_index=test_index
        )
        forecast_results['SARIMA'] = sarima_forecast_df # Store the DataFrame
    except Exception as e:
        print(f"Unexpected error running SARIMA: {e}")
        forecast_results['SARIMA'] = pd.DataFrame(np.nan, index=test_index, columns=['yhat', 'yhat_lower', 'yhat_upper'])
    model_runtimes['SARIMA'] = time.time() - start_time


    # Prophet
    print("\n--- Running Prophet ---")
    start_time = time.time()
    try:
        # run_prophet now returns a DataFrame ['yhat', 'yhat_lower', 'yhat_upper']
        prophet_forecast_df: pd.DataFrame = prophet_model.run_prophet(
            train_data=train_df.copy(),
            test_periods=test_periods,
            test_index=test_index
        )
        forecast_results['Prophet'] = prophet_forecast_df # Store the DataFrame
    except Exception as e:
        print(f"Unexpected error running Prophet: {e}")
        forecast_results['Prophet'] = pd.DataFrame(np.nan, index=test_index, columns=['yhat', 'yhat_lower', 'yhat_upper'])
    model_runtimes['Prophet'] = time.time() - start_time


    # RNN
    print("\n--- Running RNN ---")
    start_time = time.time()
    try:
        # run_rnn returns a Series (point forecast only)
        rnn_forecast_series: pd.Series = nn_models.run_rnn(
            train_data=train_df.copy(),
            val_data=val_df.copy() if val_df is not None else None,
            test_periods=test_periods,
            test_index=test_index
        )
        # Store as a DataFrame with 'yhat' column for consistency in plotting access
        forecast_results['RNN'] = pd.DataFrame({'yhat': rnn_forecast_series})
    except Exception as e:
        print(f"Unexpected error running RNN: {e}")
        forecast_results['RNN'] = pd.DataFrame(np.nan, index=test_index, columns=['yhat']) # Only yhat needed
    model_runtimes['RNN'] = time.time() - start_time


    # LSTM
    print("\n--- Running LSTM ---")
    start_time = time.time()
    try:
        # run_lstm returns a Series (point forecast only)
        lstm_forecast_series: pd.Series = nn_models.run_lstm(
            train_data=train_df.copy(),
            val_data=val_df.copy() if val_df is not None else None,
            test_periods=test_periods,
            test_index=test_index
        )
        # Store as a DataFrame with 'yhat' column
        forecast_results['LSTM'] = pd.DataFrame({'yhat': lstm_forecast_series})
    except Exception as e:
        print(f"Unexpected error running LSTM: {e}")
        forecast_results['LSTM'] = pd.DataFrame(np.nan, index=test_index, columns=['yhat']) # Only yhat needed
    model_runtimes['LSTM'] = time.time() - start_time


    # --- 4. Evaluate Models ---
    print("\n--- Evaluating Forecasts ---")
    # Prepare a DataFrame for easier access to point forecasts during evaluation
    point_forecasts_df = pd.DataFrame({
        model: results_df['yhat']
        for model, results_df in forecast_results.items()
        if isinstance(results_df, pd.DataFrame) and 'yhat' in results_df # Ensure it's a DF and has 'yhat'
    }, index=test_index)

    for model_name in point_forecasts_df.columns:
        # Check if forecasts were generated (not all NaN)
        if model_name in point_forecasts_df and not point_forecasts_df[model_name].isnull().all():
            try:
                eval_metrics: Dict[str, Any] = evaluation.evaluate_forecast(
                    y_true_series=test_df['y'],             # Actual test values
                    y_pred_series=point_forecasts_df[model_name], # Model's point predictions
                    model_name=model_name
                )
                # Add runtime to the evaluation results dictionary
                eval_metrics['Runtime (s)'] = model_runtimes.get(model_name, np.nan)
                evaluation_results.append(eval_metrics) # Append dict to list
            except Exception as e:
                 print(f"Could not evaluate {model_name}. Error: {e}")
        else:
            print(f"Skipping evaluation for {model_name}: All point forecasts were NaN or column missing.")

    # Convert evaluation results to DataFrame for easier access later
    evaluation_df: pd.DataFrame = pd.DataFrame(evaluation_results)
    if not evaluation_df.empty:
        evaluation_df = evaluation_df.set_index('Model')


    # --- 5. Display Results ---
    print("\n--- Forecast Values (Point Estimates) ---")
    # Use to_string() for potentially long dataframes to avoid truncation
    print(point_forecasts_df.to_string())

    print("\n--- Evaluation Metrics ---")
    if not evaluation_df.empty:
        # Format runtime column to 2 decimal places
        if 'Runtime (s)' in evaluation_df.columns:
            # Ensure the column is numeric before formatting, handle NaNs
            runtime_col = pd.to_numeric(evaluation_df['Runtime (s)'], errors='coerce')
            # Apply formatting only to non-NaN numeric values
            evaluation_df['Runtime (s)'] = runtime_col.apply(lambda x: f"{x:.2f}" if pd.notnull(x) else x)

        else:
             warnings.warn("Runtime (s) column not found in evaluation results.")

        # Sort by a chosen metric, e.g., MAE (optional)
        # evaluation_df = evaluation_df.sort_values(by='MAE')
        print(evaluation_df.to_string(float_format="%.4f")) # Use to_string for better formatting control

    else:
        print("No models were successfully evaluated.")


    # --- 5b. Save Results (Conditional) ---
    if config.SAVE_RESULTS:
        print(f"\n--- Saving Results to '{config.RESULTS_DIR}' ---")
        try:
            os.makedirs(config.RESULTS_DIR, exist_ok=True)

            # Save Evaluation Metrics
            if not evaluation_df.empty:
                eval_path = os.path.join(config.RESULTS_DIR, "evaluation_metrics.csv")
                evaluation_df.to_csv(eval_path)
                print(f"Saved evaluation metrics to {eval_path}")
            else:
                print("Skipping evaluation metrics save (no results).")

            # Save Point Forecasts
            if not point_forecasts_df.empty:
                points_path = os.path.join(config.RESULTS_DIR, "point_forecasts.csv")
                point_forecasts_df.to_csv(points_path)
                print(f"Saved point forecasts to {points_path}")
            else:
                print("Skipping point forecasts save (no results).")

            # Save Full Forecasts (including CIs if available) for each model
            all_forecasts_saved = False
            for model_name, forecast_df in forecast_results.items():
                 if isinstance(forecast_df, pd.DataFrame) and not forecast_df.empty:
                     # Check if it contains more than just NaNs
                     if not forecast_df.isnull().all().all():
                         model_forecast_path = os.path.join(config.RESULTS_DIR, f"forecast_{model_name}.csv")
                         forecast_df.to_csv(model_forecast_path)
                         print(f"Saved full forecast for {model_name} to {model_forecast_path}")
                         all_forecasts_saved = True
                     else:
                         print(f"Skipping saving full forecast for {model_name} (all NaNs).")
                 else:
                     print(f"Skipping saving full forecast for {model_name} (no valid DataFrame).")

            if not all_forecasts_saved:
                print("No full forecasts were saved.")

        except Exception as e:
            print(f"\nWarning: Could not save results to '{config.RESULTS_DIR}'. Error: {e}")
    else:
        print("\n--- Skipping Results Saving (SAVE_RESULTS=False) ---")


    # --- 6. Plot Results ---
    print("\n--- Plotting Results ---")
    # Check if there's anything to plot (evaluation results OR any valid forecast DataFrame)
    has_eval = not evaluation_df.empty
    has_forecasts = any(
        isinstance(df, pd.DataFrame) and not df.empty and not df.isnull().all().all()
        for df in forecast_results.values()
    )

    if has_eval or has_forecasts:
        try:
            # Combine train and validation data for plotting history if validation exists
            plot_train_df = pd.concat([train_df, val_df]) if val_df is not None else train_df
            plot_title = (f"Forecasting Comparison: {config.VALUE_COLUMN} over {config.DATE_COLUMN}\n"
                          f"(Val Size: {config.VALIDATION_SIZE if config.VALIDATION_SIZE > 0 else 'N/A'}, "
                          f"Test Size: {config.TEST_SIZE}, Freq: {df.index.freqstr or 'Inferred'})")

            plotting.plot_forecast_comparison(
                train_df=plot_train_df,
                test_df=test_df,
                forecast_dict=forecast_results, # Pass the dict containing forecast DataFrames
                evaluation_df=evaluation_df,    # Pass evaluation metrics
                date_col_name=config.DATE_COLUMN,
                value_col_name=config.VALUE_COLUMN,
                title=plot_title
            )
        except ImportError:
             print("Plotting skipped: matplotlib or its dependencies not found. Install with 'pip install matplotlib'")
        except Exception as e:
            print(f"Could not generate plot. Error: {e}")
    else:
        print("Skipping plotting as no models produced forecasts or evaluation results.")


    print("\n--- Forecasting Comparison Finished ---")


if __name__ == "__main__":
    # Ensures the script runs when executed directly
    run_comparison()
# --- END OF FILE main.py ---