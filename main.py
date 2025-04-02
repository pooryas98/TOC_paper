import pandas as pd
import numpy as np
import tensorflow as tf
import warnings
import sys
import time # For timing models
import os # For creating directories
import logging
import json # For saving parameters
import datetime # For timestamps
from typing import Dict, List, Tuple, Optional, Any

from tensorflow.keras.models import Model # Specific import for type hint

# Import project modules
try:
    from src import config
    from src import data_loader
    from src.models import sarima_model, prophet_model, nn_models
    from src import evaluation
    from src import plotting
    # Import specific types for hinting
    from statsmodels.tsa.statespace.sarimax import SARIMAXResultsWrapper
    from prophet import Prophet
    from sklearn.preprocessing import MinMaxScaler
    import joblib # For saving scaler

except ImportError as e:
     print(f"Error importing project modules: {e}")
     print("Ensure main.py is run from the correct directory or sys.path is configured.")
     sys.exit(1)


logger = logging.getLogger() # Use configured logger

def set_seeds(seed: int) -> None:
    """Sets random seeds for reproducibility."""
    np.random.seed(seed)
    tf.random.set_seed(seed)
    logger.info(f"Set random seeds to: {seed}")

# Custom JSON encoder for non-serializable types (models, numpy types etc.)
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, pd.Timestamp): return obj.isoformat()
        if isinstance(obj, (pd.Timedelta, pd.Period)): return str(obj)
        # Handle model objects and scaler object during serialization
        if isinstance(obj, (SARIMAXResultsWrapper, Prophet, Model, MinMaxScaler)):
             return f"<Object type: {type(obj).__name__}>" # Don't serialize models/scalers here
        try:
            return super(NpEncoder, self).default(obj)
        except TypeError:
             return f"<Unserializable type: {type(obj).__name__}>"

def save_run_parameters(params_dict: Dict[str, Any], file_path: str):
    """Saves the run parameters dictionary to a JSON file using the custom encoder."""
    try:
        # Directory creation handled upstream
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as f:
            json.dump(params_dict, f, indent=4, cls=NpEncoder)
        logger.info(f"Run parameters saved successfully to: {file_path}")
    except Exception as e:
        logger.error(f"Error saving run parameters to {file_path}: {e}", exc_info=True)


def run_comparison() -> None:
    """Runs the full forecasting model comparison workflow."""
    run_start_time = time.time()
    logger.info("--- Starting Forecasting Comparison Run ---")

    # --- Setup Timestamped Results Directory ---
    try:
         config_params = config.get_config_dict()
    except Exception as e:
         logger.error(f"Failed to retrieve configuration dictionary: {e}", exc_info=True)
         sys.exit(1)

    run_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    logger.info(f"Run Timestamp: {run_timestamp}")

    original_results_dir = config_params.get('RESULTS_DIR', 'results')
    original_tuner_dir_name = os.path.basename(config_params.get('KERAS_TUNER_DIR', 'keras_tuner_dir'))
    run_results_dir = os.path.join(original_results_dir, run_timestamp)
    run_tuner_dir = os.path.join(run_results_dir, original_tuner_dir_name)

    try:
        os.makedirs(run_results_dir, exist_ok=True)
        logger.info(f"Created run-specific results directory: {run_results_dir}")
    except OSError as e:
        logger.error(f"Fatal Error: Could not create run directory '{run_results_dir}'. Exiting. Error: {e}")
        sys.exit(1)

    # Update config_params in memory for this run
    config_params['RESULTS_DIR'] = run_results_dir
    config_params['KERAS_TUNER_DIR'] = run_tuner_dir
    config_params['RUN_IDENTIFIER'] = run_timestamp
    # --- End Directory Setup ---

    set_seeds(config_params['RANDOM_SEED'])

    # --- 1. Load Data, Impute & Split ---
    logger.info("--- Stage 1: Loading and Preparing Data ---")
    df: Optional[pd.DataFrame] = None
    data_load_params: Optional[Dict[str, Any]] = None
    train_df: Optional[pd.DataFrame] = None
    val_df: Optional[pd.DataFrame] = None
    test_df: Optional[pd.DataFrame] = None

    try:
        df, data_load_params = data_loader.load_and_prepare_data(
            file_path=config_params['CSV_FILE_PATH'],
            date_column=config_params['DATE_COLUMN'],
            value_column=config_params['VALUE_COLUMN'],
            config_params=config_params
        )
        if df is None or data_load_params.get('status') != 'Loaded Successfully':
             raise ValueError(f"Data loading failed. Status: {data_load_params.get('status', 'Unknown')}")

        train_df, val_df, test_df = data_loader.split_data_train_val_test(
            df, config_params['VALIDATION_SIZE'], config_params['TEST_SIZE']
        )
        logger.info("Data loading, preparation, and splitting successful.")
    except (FileNotFoundError, KeyError, TypeError, ValueError, Exception) as e:
        logger.error("--- Fatal Error: Data Loading/Splitting Failed ---", exc_info=True)
        logger.error(f"Error details: {e}")
        logger.error("Check config (CSV_FILE_PATH, DATE_COLUMN, VALUE_COLUMN, sizes) and CSV file.")
        sys.exit(1)

    # --- 2. Initialize Results Storage ---
    logger.info("--- Stage 2: Initializing Results Storage ---")
    all_run_parameters: Dict[str, Any] = {
        'config_settings': config_params, # Store modified config
        'data_load_summary': data_load_params,
        'models': {}, # Results per model (evaluation run)
        'final_forecast_runs': {} # Results for final forecast run
    }
    forecast_results: Dict[str, pd.DataFrame] = {} # Holds EVALUATION forecasts [yhat, yhat_lower, yhat_upper]
    evaluation_results_list: List[Dict[str, Any]] = [] # List of metric dicts per model
    fitted_models: Dict[str, Any] = {} # Store fitted evaluation models if SAVE_TRAINED_MODELS
    future_forecast_results: Dict[str, pd.DataFrame] = {} # Holds FINAL forecasts

    test_periods: int = len(test_df)
    test_index: pd.DatetimeIndex = test_df.index
    logger.info(f"Test set covers {test_periods} periods from {test_index.min()} to {test_index.max()}.")

    # --- 3. Run Models (on Train/Val split for Evaluation) ---
    logger.info("--- Stage 3: Running Forecasting Models (Evaluation Run) ---")
    models_to_run = config_params.get('MODELS_TO_RUN', [])

    # SARIMA (Evaluation Run)
    if 'SARIMA' in models_to_run:
        model_name = 'SARIMA'
        start_time = time.time()
        logger.info(f"--- Running {model_name} (Evaluation Run) ---")
        sarima_forecast_df: Optional[pd.DataFrame] = None
        sarima_params_used: Optional[Dict[str, Any]] = None
        sarima_model_obj: Optional[Any] = None
        try:
            sarima_forecast_df, sarima_params_used, sarima_model_obj = sarima_model.run_sarima(
                train_data=train_df.copy(),
                test_periods=test_periods,
                test_index=test_index,
                config_params=config_params # Pass updated config
            )
            forecast_results[model_name] = sarima_forecast_df
            if sarima_model_obj and config_params.get('SAVE_TRAINED_MODELS'):
                 fitted_models[model_name] = sarima_model_obj
        except Exception as e:
            logger.error(f"Unexpected error running SARIMA in main loop (Eval): {e}", exc_info=True)
            forecast_results[model_name] = pd.DataFrame(np.nan, index=test_index, columns=['yhat', 'yhat_lower', 'yhat_upper'])
            if sarima_params_used is None: sarima_params_used = {'model_type': model_name}
            sarima_params_used['run_error'] = f"Main Loop Error: {e}"

        runtime = time.time() - start_time
        logger.info(f"--- {model_name} (Evaluation Run) Finished. Runtime: {runtime:.2f} seconds ---")
        if sarima_params_used:
             sarima_params_used['runtime_seconds'] = runtime
             all_run_parameters['models'][model_name] = sarima_params_used
        else:
             all_run_parameters['models'][model_name] = {'model_type': model_name, 'runtime_seconds': runtime, 'status': 'Params not captured'}


    # Prophet (Evaluation Run)
    if 'Prophet' in models_to_run:
        model_name = 'Prophet'
        start_time = time.time()
        logger.info(f"--- Running {model_name} (Evaluation Run) ---")
        prophet_forecast_df: Optional[pd.DataFrame] = None
        prophet_params_used: Optional[Dict[str, Any]] = None
        prophet_model_obj: Optional[Any] = None
        try:
            prophet_forecast_df, prophet_params_used, prophet_model_obj = prophet_model.run_prophet(
                train_data=train_df.copy(),
                test_periods=test_periods,
                test_index=test_index,
                config_params=config_params # Pass updated config
            )
            forecast_results[model_name] = prophet_forecast_df
            if prophet_model_obj and config_params.get('SAVE_TRAINED_MODELS'):
                fitted_models[model_name] = prophet_model_obj
        except Exception as e:
            logger.error(f"Unexpected error running Prophet in main loop (Eval): {e}", exc_info=True)
            forecast_results[model_name] = pd.DataFrame(np.nan, index=test_index, columns=['yhat', 'yhat_lower', 'yhat_upper'])
            if prophet_params_used is None: prophet_params_used = {'model_type': model_name}
            prophet_params_used['run_error'] = f"Main Loop Error: {e}"

        runtime = time.time() - start_time
        logger.info(f"--- {model_name} (Evaluation Run) Finished. Runtime: {runtime:.2f} seconds ---")
        if prophet_params_used:
             prophet_params_used['runtime_seconds'] = runtime
             all_run_parameters['models'][model_name] = prophet_params_used
        else:
            all_run_parameters['models'][model_name] = {'model_type': model_name, 'runtime_seconds': runtime, 'status': 'Params not captured'}


    # Neural Networks (RNN/LSTM) (Evaluation Run)
    nn_model_types = [m for m in ['RNN', 'LSTM'] if m in models_to_run]
    for model_name in nn_model_types:
        start_time = time.time()
        logger.info(f"--- Running {model_name} (Evaluation Run) ---")
        nn_forecast_series: Optional[pd.Series] = None
        nn_params_used: Optional[Dict[str, Any]] = None
        nn_model_obj: Optional[Model] = None
        try:
            # Unified runner handles tuning based on config_params
            nn_forecast_series, nn_params_used, nn_model_obj = nn_models.run_nn_model(
                model_type=model_name,
                train_data=train_df.copy(),
                val_data=val_df.copy() if val_df is not None else None,
                test_periods=test_periods,
                test_index=test_index,
                config_params=config_params # Pass updated config
            )
            # Store forecast as DataFrame, add NaN CI columns for consistency
            forecast_results[model_name] = pd.DataFrame({
                'yhat': nn_forecast_series,
                'yhat_lower': np.nan,
                'yhat_upper': np.nan
                })
            if nn_model_obj and config_params.get('SAVE_TRAINED_MODELS'):
                 fitted_models[model_name] = nn_model_obj # Includes scaler ref in params

        except Exception as e:
            logger.error(f"Unexpected error running {model_name} in main loop (Eval): {e}", exc_info=True)
            forecast_results[model_name] = pd.DataFrame(np.nan, index=test_index, columns=['yhat', 'yhat_lower', 'yhat_upper'])
            if nn_params_used is None: nn_params_used = {'model_type': model_name}
            nn_params_used['run_error'] = f"Main Loop Error: {e}"

        runtime = time.time() - start_time
        logger.info(f"--- {model_name} (Evaluation Run) Finished. Runtime: {runtime:.2f} seconds ---")
        if nn_params_used:
             nn_params_used['runtime_seconds'] = runtime
             all_run_parameters['models'][model_name] = nn_params_used
        else:
             all_run_parameters['models'][model_name] = {'model_type': model_name, 'runtime_seconds': runtime, 'status': 'Params not captured'}


    # --- 4. Evaluate Models (on Test Set) ---
    logger.info("--- Stage 4: Evaluating Forecasts ---")
    point_forecasts_df = pd.DataFrame(index=test_index)
    for model, results_df in forecast_results.items():
         if isinstance(results_df, pd.DataFrame) and 'yhat' in results_df:
             point_forecasts_df[model] = results_df['yhat']
         else:
             logger.warning(f"No valid 'yhat' column found for {model} in forecast_results. Skipping evaluation.")

    metrics_to_calc = config_params['EVALUATION_METRICS']
    evaluation_df: Optional[pd.DataFrame] = None # Initialize

    for model_name in point_forecasts_df.columns:
        if model_name not in all_run_parameters['models']:
             logger.warning(f"Model '{model_name}' found in forecasts but not in parameter store. Skipping evaluation linkage.")
             continue

        model_run_params = all_run_parameters['models'][model_name]

        if point_forecasts_df[model_name].isnull().all():
            logger.warning(f"Skipping evaluation for {model_name}: All point forecasts are NaN.")
            model_run_params['evaluation_metrics'] = {metric: np.nan for metric in metrics_to_calc}
            model_run_params['evaluation_status'] = 'Skipped - All NaN forecasts'
            nan_metrics = {'Model': model_name, **{metric: np.nan for metric in metrics_to_calc}}
            evaluation_results_list.append(nan_metrics)
        else:
            try:
                eval_metrics: Dict[str, Any] = evaluation.evaluate_forecast(
                    y_true_series=test_df['y'],
                    y_pred_series=point_forecasts_df[model_name],
                    model_name=model_name,
                    metrics_to_calculate=metrics_to_calc
                )
                model_run_params['evaluation_metrics'] = {k: v for k, v in eval_metrics.items() if k != 'Model'}
                model_run_params['evaluation_status'] = 'Success'
                evaluation_results_list.append(eval_metrics)
            except Exception as e:
                 logger.error(f"Could not evaluate {model_name}. Error: {e}", exc_info=True)
                 model_run_params['evaluation_metrics'] = {metric: np.nan for metric in metrics_to_calc}
                 model_run_params['evaluation_status'] = f'Error: {e}'
                 nan_metrics = {'Model': model_name, **{metric: np.nan for metric in metrics_to_calc}}
                 evaluation_results_list.append(nan_metrics)

    # Convert list of evaluation dicts to DataFrame
    if evaluation_results_list:
        evaluation_df = pd.DataFrame(evaluation_results_list)
        runtimes = {m: all_run_parameters['models'].get(m, {}).get('runtime_seconds', np.nan)
                    for m in evaluation_df['Model']}
        evaluation_df['Runtime (s)'] = evaluation_df['Model'].map(runtimes)
        evaluation_df = evaluation_df.set_index('Model')
    else:
         logger.warning("No models were evaluated, evaluation_df will be empty.")
         evaluation_df = pd.DataFrame() # Ensure empty DataFrame


    # --- 5. Display Results (Evaluation) ---
    logger.info("--- Stage 5: Displaying Evaluation Results ---")
    logger.info("\n" + "="*20 + " Point Forecasts (Test Set) " + "="*20)
    try:
        with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 1000):
            logger.info(f"\n{point_forecasts_df}")
    except Exception as display_err:
         logger.error(f"Error displaying point forecasts: {display_err}")
         logger.info(point_forecasts_df.to_string()) # Fallback

    logger.info("\n" + "="*20 + " Evaluation Metrics " + "="*20)
    if not evaluation_df.empty:
        if 'Runtime (s)' in evaluation_df.columns: # Format runtime if exists
            evaluation_df['Runtime (s)'] = pd.to_numeric(evaluation_df['Runtime (s)'], errors='coerce').map('{:.2f}'.format)
        try:
            with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 1000, 'display.float_format', '{:.4f}'.format):
                 logger.info(f"\n{evaluation_df}")
        except Exception as display_err:
             logger.error(f"Error displaying evaluation metrics: {display_err}")
             logger.info(evaluation_df.to_string(float_format="%.4f")) # Fallback
    else:
        logger.info("No models were successfully evaluated.")

    # --- 6. Save Results (Evaluation Run) ---
    logger.info("--- Stage 6: Saving Evaluation Run Results ---")
    results_dir = config_params['RESULTS_DIR'] # Timestamped path
    save_results_flag = config_params['SAVE_RESULTS']

    if save_results_flag:
        logger.info(f"Saving evaluation results to: '{results_dir}'")
        try:
            # Save Parameters (Includes evaluation run details)
            if config_params.get('SAVE_MODEL_PARAMETERS', False):
                 param_file = os.path.join(results_dir, "run_parameters.json")
                 save_run_parameters(all_run_parameters, param_file)

            # Save Evaluation Metrics
            if not evaluation_df.empty:
                eval_path = os.path.join(results_dir, "evaluation_metrics.csv")
                evaluation_df.to_csv(eval_path, float_format='%.6f')
                logger.info(f"Saved evaluation metrics to {eval_path}")
            else: logger.info("Skipping evaluation metrics save (no results).")

            # Save Point Forecasts (Test Set)
            if not point_forecasts_df.empty:
                points_path = os.path.join(results_dir, "point_forecasts.csv")
                point_forecasts_df.to_csv(points_path, float_format='%.6f')
                logger.info(f"Saved evaluation point forecasts to {points_path}")
            else: logger.info("Skipping evaluation point forecasts save (no results).")

            # Save Full Forecasts (Test Set, with CIs if available)
            all_eval_forecasts_saved = False
            for model_name, forecast_df in forecast_results.items():
                 if isinstance(forecast_df, pd.DataFrame) and not forecast_df.empty and not forecast_df.isnull().all().all():
                     model_forecast_path = os.path.join(results_dir, f"full_forecast_{model_name}.csv")
                     forecast_df.to_csv(model_forecast_path, float_format='%.6f')
                     logger.info(f"Saved evaluation full forecast for {model_name} to {model_forecast_path}")
                     all_eval_forecasts_saved = True
                 else: logger.debug(f"Skipping saving evaluation full forecast for {model_name} (empty/all NaN/not DF).")
            if not all_eval_forecasts_saved: logger.warning("No evaluation full forecasts were saved.")

            # Save Trained Models (From Evaluation Run)
            if config_params.get('SAVE_TRAINED_MODELS', False):
                 models_dir = os.path.join(results_dir, 'saved_models')
                 os.makedirs(models_dir, exist_ok=True)
                 logger.info(f"Saving trained evaluation models to: {models_dir}")
                 models_saved_count = 0
                 for model_name, model_obj in fitted_models.items():
                     if model_obj:
                         try:
                             save_path = os.path.join(models_dir, f"model_{model_name}")
                             if model_name == 'SARIMA':
                                 sarima_model.save_sarima_model(model_obj, save_path + ".pkl")
                             elif model_name == 'Prophet':
                                 prophet_model.save_prophet_model(model_obj, save_path + ".pkl")
                             elif model_name in ['RNN', 'LSTM']:
                                 nn_models.save_nn_model(model_obj, save_path) # Adds .keras or .h5
                                 # Also save the scaler
                                 eval_nn_params = all_run_parameters.get('models', {}).get(model_name, {})
                                 scaler_obj_eval = eval_nn_params.get('scaler_object_ref')
                                 if isinstance(scaler_obj_eval, MinMaxScaler):
                                     scaler_path_eval = os.path.join(models_dir, f"scaler_{model_name}.joblib")
                                     joblib.dump(scaler_obj_eval, scaler_path_eval)
                                     logger.info(f"Saved evaluation NN scaler for {model_name} to {scaler_path_eval}")
                                 else: logger.warning(f"Could not find/verify scaler object to save for eval NN model {model_name}")
                             else:
                                 logger.warning(f"Don't know how to save evaluation model type: {model_name}")
                                 continue
                             models_saved_count += 1
                         except Exception as save_model_err:
                             logger.error(f"Failed to save evaluation model {model_name}: {save_model_err}", exc_info=True)
                 logger.info(f"Finished saving evaluation run models ({models_saved_count}/{len(fitted_models)} saved).")

        except Exception as e:
            logger.error(f"Error during evaluation results saving: {e}", exc_info=True)
    else:
        logger.info("Skipping Evaluation Results Saving (SAVE_RESULTS=False in config).")

    # --- Stage 7: Final Forecasting (Train on Full Data, Predict Future) ---
    logger.info("--- Stage 7: Final Forecasting ---")
    if config_params['RUN_FINAL_FORECAST']:
        logger.info(f"Starting final forecast generation for {config_params['FORECAST_HORIZON']} periods.")

        if df is None: # Check if full df is available
            logger.error("Cannot run final forecast: Original DataFrame 'df' is not available.")
            config_params['RUN_FINAL_FORECAST'] = False # Mark as skipped
        else:
            full_train_df = df.copy() # Use original full dataframe
            forecast_horizon = config_params['FORECAST_HORIZON']
            fitted_final_models: Dict[str, Any] = {}

            # Determine Future Index
            future_index: Optional[pd.DatetimeIndex] = None
            last_date = full_train_df.index[-1]
            data_freq = full_train_df.index.freq
            if data_freq:
                try:
                    future_index = pd.date_range(start=last_date + data_freq, periods=forecast_horizon, freq=data_freq)
                    logger.info(f"Generated future index for {forecast_horizon} periods starting after {last_date}.")
                except Exception as freq_err:
                     logger.error(f"Error generating future date range: {freq_err}. Final forecasting skipped.", exc_info=True)
                     config_params['RUN_FINAL_FORECAST'] = False
            else:
                logger.error("Cannot generate future index: Frequency not found. Final forecasting skipped.")
                config_params['RUN_FINAL_FORECAST'] = False

            # Proceed only if future index was generated
            if future_index is not None:
                models_to_run_final = config_params.get('MODELS_TO_RUN', [])

                for model_name in models_to_run_final:
                    start_time = time.time()
                    logger.info(f"--- Running Final Forecast for {model_name} ---")
                    model_forecast_df: Optional[pd.DataFrame] = None
                    model_params_used: Optional[Dict[str, Any]] = None
                    model_obj: Optional[Any] = None
                    empty_forecast = pd.DataFrame(np.nan, index=future_index, columns=['yhat', 'yhat_lower', 'yhat_upper'])

                    try:
                        if model_name == 'SARIMA':
                            _, sarima_params, trained_sarima = sarima_model.run_sarima(
                                 train_data=full_train_df, # Train on full data
                                 test_periods=1, # Dummy
                                 test_index=pd.DatetimeIndex([future_index[0]]), # Dummy
                                 config_params=config_params
                            )
                            model_params_used = sarima_params
                            if trained_sarima:
                                fitted_final_models[model_name] = trained_sarima
                                logger.info(f"SARIMA retrained. Generating {forecast_horizon} step forecast...")
                                pred = trained_sarima.get_forecast(steps=forecast_horizon)
                                forecast_summary = pred.summary_frame(alpha=1.0 - config_params.get('PROPHET_INTERVAL_WIDTH', 0.95))
                                forecast_summary.index = future_index
                                model_forecast_df = forecast_summary[['mean', 'mean_ci_lower', 'mean_ci_upper']].rename(
                                    columns={'mean': 'yhat', 'mean_ci_lower': 'yhat_lower', 'mean_ci_upper': 'yhat_upper'}
                                )
                            else:
                                logger.error(f"SARIMA retraining failed.")
                                model_forecast_df = empty_forecast.copy()

                        elif model_name == 'Prophet':
                            model_forecast_df, model_params_used, model_obj = prophet_model.run_prophet(
                                train_data=full_train_df,
                                test_periods=forecast_horizon,
                                test_index=future_index,
                                config_params=config_params
                            )
                            if model_obj: fitted_final_models[model_name] = model_obj
                            if model_forecast_df is not None:
                                # Ensure standard columns exist
                                if 'yhat_lower' not in model_forecast_df: model_forecast_df['yhat_lower'] = np.nan
                                if 'yhat_upper' not in model_forecast_df: model_forecast_df['yhat_upper'] = np.nan
                            else: model_forecast_df = empty_forecast.copy()

                        elif model_name in ['RNN', 'LSTM']:
                            final_nn_config_params = config_params.copy()
                            final_nn_config_params['USE_KERAS_TUNER'] = False # Disable tuner for final run
                            logger.info(f"Temporarily disabling KerasTuner for final {model_name} retraining.")

                            nn_forecast_series, model_params_used, model_obj = nn_models.run_nn_model(
                                model_type=model_name,
                                train_data=full_train_df,
                                val_data=None, # No validation
                                test_periods=forecast_horizon,
                                test_index=future_index,
                                config_params=final_nn_config_params # Pass config with Tuner disabled
                             )
                            if isinstance(nn_forecast_series, pd.Series):
                                model_forecast_df = pd.DataFrame({
                                    'yhat': nn_forecast_series, 'yhat_lower': np.nan, 'yhat_upper': np.nan
                                    }, index=future_index)
                            else: model_forecast_df = empty_forecast.copy()
                            if model_obj: fitted_final_models[model_name] = model_obj

                        # Store results
                        if model_forecast_df is None: model_forecast_df = empty_forecast.copy()
                        future_forecast_results[model_name] = model_forecast_df.reindex(columns=['yhat', 'yhat_lower', 'yhat_upper'])

                    except Exception as e:
                        logger.error(f"Unexpected error running final forecast for {model_name}: {e}", exc_info=True)
                        future_forecast_results[model_name] = empty_forecast.copy()
                        if model_params_used is None: model_params_used = {'model_type': model_name, 'status': 'Failed'}
                        model_params_used['final_forecast_run_error'] = f"Main Loop Error: {e}"

                    runtime = time.time() - start_time
                    logger.info(f"--- {model_name} Final Forecast Finished. Runtime: {runtime:.2f} seconds ---")
                    if model_params_used:
                         model_params_used['final_forecast_runtime_seconds'] = runtime
                         all_run_parameters['final_forecast_runs'][model_name] = model_params_used

                # Display Future Forecasts
                logger.info("\n" + "="*20 + " Future Point Forecasts " + "="*20)
                future_point_forecasts_df = pd.DataFrame(index=future_index)
                for model, results_df in future_forecast_results.items():
                     if isinstance(results_df, pd.DataFrame) and 'yhat' in results_df:
                         future_point_forecasts_df[model] = results_df['yhat']
                try:
                    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 1000):
                        logger.info(f"\n{future_point_forecasts_df}")
                except Exception as display_err:
                     logger.error(f"Error displaying future point forecasts: {display_err}")
                     logger.info(future_point_forecasts_df.to_string()) # Fallback

                # Save Future Forecasts (Conditional)
                if save_results_flag:
                    logger.info(f"Saving future forecasts to: '{results_dir}'") # results_dir is timestamped
                    try:
                        # Save Future Point Forecasts
                        if not future_point_forecasts_df.empty:
                            points_path = os.path.join(results_dir, "future_point_forecasts.csv")
                            future_point_forecasts_df.to_csv(points_path, float_format='%.6f')
                            logger.info(f"Saved future point forecasts to {points_path}")

                        # Save Full Future Forecasts
                        for model_name, forecast_df in future_forecast_results.items():
                             if isinstance(forecast_df, pd.DataFrame) and not forecast_df.empty and not forecast_df.isnull().all().all():
                                 model_forecast_path = os.path.join(results_dir, f"future_full_forecast_{model_name}.csv")
                                 forecast_df.to_csv(model_forecast_path, float_format='%.6f')
                                 logger.info(f"Saved full future forecast for {model_name} to {model_forecast_path}")

                        # Optionally save models trained on full data
                        if config_params.get('SAVE_TRAINED_MODELS', False):
                            final_models_dir = os.path.join(results_dir, 'saved_final_models')
                            os.makedirs(final_models_dir, exist_ok=True)
                            logger.info(f"Saving final trained models (full data) to: {final_models_dir}")
                            models_saved_count = 0
                            for model_name, model_obj in fitted_final_models.items():
                                 if model_obj:
                                     try:
                                         save_path = os.path.join(final_models_dir, f"model_{model_name}")
                                         if model_name == 'SARIMA':
                                             sarima_model.save_sarima_model(model_obj, save_path + ".pkl")
                                         elif model_name == 'Prophet':
                                             prophet_model.save_prophet_model(model_obj, save_path + ".pkl")
                                         elif model_name in ['RNN', 'LSTM']:
                                             nn_models.save_nn_model(model_obj, save_path) # Adds extension
                                             # Retrieve scaler from the final run params dict
                                             final_nn_params = all_run_parameters.get('final_forecast_runs', {}).get(model_name, {})
                                             scaler_obj = final_nn_params.get('scaler_object_ref')
                                             if isinstance(scaler_obj, MinMaxScaler):
                                                 scaler_path = os.path.join(final_models_dir, f"scaler_{model_name}.joblib")
                                                 joblib.dump(scaler_obj, scaler_path)
                                                 logger.info(f"Saved final NN scaler for {model_name} to {scaler_path}")
                                             else: logger.warning(f"Could not find/verify scaler object for final NN model {model_name}")
                                         else: logger.warning(f"Don't know how to save final model type: {model_name}")
                                         models_saved_count += 1
                                     except Exception as save_model_err:
                                         logger.error(f"Failed to save final model {model_name}: {save_model_err}", exc_info=True)
                            logger.info(f"Finished saving final models ({models_saved_count}/{len(fitted_final_models)} saved).")

                    except Exception as e:
                        logger.error(f"Error during future results saving: {e}", exc_info=True)
    else:
        logger.info("Skipping Final Forecasting step (RUN_FINAL_FORECAST=False or frequency error).")

    # --- Stage 8: Generate Plots ---
    logger.info("--- Stage 8: Generating Plots ---")

    # Combine train and validation data for plotting evaluation history
    plot_train_val_df = pd.concat([train_df, val_df]) if val_df is not None else train_df.copy()

    # Prepare full dataset for future plot history
    full_dataset_df = df.copy() if df is not None else plot_train_val_df # Use train+val as fallback

    # Ensure evaluation_df is the DataFrame created in Stage 4
    eval_metrics_df = evaluation_df # Already created

    try:
        # Call the main orchestration function from plotting.py
        # Receives config_params with updated RESULTS_DIR
        plotting.generate_all_plots(
            train_df=plot_train_val_df,         # History for eval plots (Train+Val)
            test_df=test_df,                    # Actuals for eval plots
            full_df=full_dataset_df,            # History for future plot (Full dataset)
            eval_forecast_dict=forecast_results, # Forecasts matching test_df index
            future_forecast_dict=future_forecast_results, # Forecasts for future
            evaluation_metrics=eval_metrics_df, # DataFrame of metrics
            config_params=config_params        # Config dictionary (with timestamped paths)
        )
    except ImportError:
        logger.error("Plotting skipped: matplotlib or dependencies not found. Install with 'pip install matplotlib'")
    except Exception as e:
        logger.error(f"Could not generate plots. Error: {e}", exc_info=True)

    # --- End of Workflow ---
    run_end_time = time.time()
    total_runtime = run_end_time - run_start_time
    logger.info(f"--- Forecasting Comparison Finished ---")
    logger.info(f"Total Run Time: {total_runtime:.2f} seconds")
    logger.info(f"Results for this run saved in: {config_params['RESULTS_DIR']}") # Final path


    # Save parameters again (final version with runtime) to timestamped folder
    if config_params['SAVE_RESULTS'] and config_params.get('SAVE_MODEL_PARAMETERS', False):
         all_run_parameters['total_runtime_seconds'] = total_runtime
         param_file = os.path.join(config_params['RESULTS_DIR'], "run_parameters.json") # Use updated path
         save_run_parameters(all_run_parameters, param_file)


if __name__ == "__main__":
    run_comparison()