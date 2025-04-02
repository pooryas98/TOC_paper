import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, Input, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam, RMSprop, SGD, Adagrad # Import optimizers
from sklearn.preprocessing import MinMaxScaler
import warnings
import logging
import os
import json # For saving tuner HPs
from typing import Tuple, Optional, List, Union, Any, Dict

_keras_tuner_available = False
try:
    import keras_tuner as kt
    _keras_tuner_available = True
except ImportError: pass # Warning printed later if needed

logger = logging.getLogger(__name__)

# --- Helper: Create Sequences ---
def create_sequences_manual(data: Optional[np.ndarray], n_steps: int) -> Tuple[np.ndarray, np.ndarray]:
    """Creates input/output sequences from scaled data."""
    if data is None or len(data) <= n_steps:
        logger.debug(f"Not enough data ({len(data) if data is not None else 'None'}) for sequences (n_steps={n_steps})")
        return np.array([]), np.array([]) # Return empty
    X, y = [], []
    for i in range(len(data) - n_steps):
        end_ix = i + n_steps
        seq_x, seq_y = data[i:end_ix], data[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    if not X:
        logger.debug(f"Sequence creation yielded empty X (data len: {len(data)}, n_steps: {n_steps})")
        return np.array([]), np.array([])
    return np.array(X), np.array(y).flatten()

# --- Helper: Get Optimizer Instance ---
def get_optimizer(name: str, learning_rate: Optional[float] = None) -> keras.optimizers.Optimizer:
    """Gets a Keras optimizer instance by name, optionally setting LR."""
    name = name.lower()
    kwargs = {'learning_rate': learning_rate} if learning_rate is not None else {}

    if name == 'adam': return Adam(**kwargs)
    elif name == 'rmsprop': return RMSprop(**kwargs)
    elif name == 'sgd': return SGD(**kwargs)
    elif name == 'adagrad': return Adagrad(**kwargs)
    else:
        logger.warning(f"Unsupported optimizer '{name}'. Using Adam default.")
        return Adam()


# --- Hypermodel Builder Function ---
def build_hypermodel(hp: kt.HyperParameters, model_type: str, n_steps: int, config_params: Dict[str, Any], n_features: int = 1) -> Model:
    """Builds a Keras RNN/LSTM model with hyperparameters defined by KerasTuner."""
    model = Sequential()
    model.add(Input(shape=(n_steps, n_features)))

    # Tune Recurrent Layer
    hp_units: int = hp.Int('units', min_value=config_params['NN_TUNER_HP_UNITS_MIN'], max_value=config_params['NN_TUNER_HP_UNITS_MAX'], step=config_params['NN_TUNER_HP_UNITS_STEP'])
    hp_activation: str = hp.Choice('activation', values=config_params['NN_TUNER_HP_ACTIVATION_CHOICES'])

    if model_type == 'RNN': model.add(SimpleRNN(units=hp_units, activation=hp_activation, name=f"tuned_{model_type}_layer"))
    elif model_type == 'LSTM': model.add(LSTM(units=hp_units, activation=hp_activation, name=f"tuned_{model_type}_layer"))
    else: raise ValueError("Invalid model_type. Choose 'RNN' or 'LSTM'.")

    # Optionally force Dropout layer inclusion and tune rate (based on config)
    if config_params['NN_TUNER_HP_USE_DROPOUT']:
        hp_dropout_rate = hp.Float('dropout_rate', min_value=config_params['NN_TUNER_HP_DROPOUT_MIN'], max_value=config_params['NN_TUNER_HP_DROPOUT_MAX'], step=config_params['NN_TUNER_HP_DROPOUT_STEP'])
        model.add(Dropout(rate=hp_dropout_rate, name="tuned_dropout"))
        # logger.debug("Tuning: Added forced Dropout layer, tuning rate.")
    # else: logger.debug("Tuning: Dropout tuning disabled in config, skipping Dropout layer.")

    model.add(Dense(1, name="output_dense")) # Output Layer

    # Tune Optimizer and Learning Rate
    hp_learning_rate: float = hp.Float('learning_rate', min_value=config_params['NN_TUNER_HP_LR_MIN'], max_value=config_params['NN_TUNER_HP_LR_MAX'], sampling='log')
    hp_optimizer_choice: str = hp.Choice('optimizer', values=config_params['NN_TUNER_HP_OPTIMIZER_CHOICES'])

    optimizer = get_optimizer(hp_optimizer_choice, hp_learning_rate)
    model.compile(optimizer=optimizer, loss=config_params['NN_LOSS_FUNCTION'], metrics=['mae']) # Include MAE for monitoring
    return model


# --- Main Training Function ---
def train_nn(
    model_type: str,
    train_data: pd.DataFrame,
    val_data: Optional[pd.DataFrame],
    config_params: Dict[str, Any]
) -> Tuple[Optional[Model], Optional[MinMaxScaler], Optional[Dict[str, Any]]]:
    """
    Trains RNN/LSTM model, optionally using KerasTuner.
    Returns trained Keras model, scaler object, and parameters used.
    """
    logger.info(f"--- Starting {model_type} Training ---")
    model_params_used: Dict[str, Any] = {'model_type': model_type}
    n_steps = config_params['NN_STEPS']
    use_tuner = config_params['USE_KERAS_TUNER']
    best_model: Optional[Model] = None
    scaler: Optional[MinMaxScaler] = None
    history: Optional[keras.callbacks.History] = None
    tuner_results: Optional[Dict[str, Any]] = None

    if train_data is None or train_data.empty or train_data['y'].isnull().all():
        logger.warning(f"Skipping {model_type}: Training data empty or all NaN.")
        model_params_used['status'] = 'Skipped - No training data'
        return None, None, model_params_used

    # 1. Scaling
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_train: Optional[np.ndarray] = None
    scaled_val: Optional[np.ndarray] = None
    try:
        train_y = train_data[['y']].dropna() # Use double brackets, dropna
        if train_y.empty:
            logger.warning(f"Skipping {model_type}: No non-NaN training data.")
            model_params_used['status'] = 'Skipped - No non-NaN training data'
            return None, None, model_params_used

        scaled_train = scaler.fit_transform(train_y)
        model_params_used['scaler_type'] = 'MinMaxScaler'
        model_params_used['scaler_feature_range'] = (0, 1)
        model_params_used['scaler_object_ref'] = scaler # Store scaler ref
        logger.info(f"Training data scaled. Shape: {scaled_train.shape}")

        # Scale validation data if valid
        if val_data is not None and not val_data.empty and not val_data['y'].isnull().all():
             val_y = val_data[['y']].dropna()
             if not val_y.empty:
                scaled_val = scaler.transform(val_y) # Use transform
                logger.info(f"Validation data scaled. Shape: {scaled_val.shape}")
             else:
                logger.warning("Val data contains only NaNs after dropna(). Val disabled.")
                scaled_val = None
        else: # Log reason if val data invalid
            if val_data is None: logger.info("No validation data provided.")
            elif val_data.empty: logger.warning("Validation data provided is empty.")
            else: logger.warning("Validation data provided is all NaN.")
            scaled_val = None

        # Check tuner objective vs. available validation data
        tuner_objective = config_params.get('NN_TUNER_OBJECTIVE', 'val_loss')
        if use_tuner and tuner_objective.startswith('val_') and scaled_val is None:
             logger.error(f"Tuner objective '{tuner_objective}' requires valid val data, none available. Aborting {model_type}.")
             model_params_used['status'] = 'Error - Tuner needs valid validation data'
             return None, scaler, model_params_used # Return scaler

    except Exception as e:
        logger.error(f"Error during scaling for {model_type}: {e}. Skipping.", exc_info=True)
        model_params_used['status'] = f'Error - Scaling failed: {e}'
        return None, scaler, model_params_used

    # 2. Create Sequences
    X_train, y_train = create_sequences_manual(scaled_train, n_steps)
    X_val, y_val = None, None
    validation_data_for_fit: Optional[Tuple[np.ndarray, np.ndarray]] = None

    if scaled_val is not None:
        X_val_temp, y_val_temp = create_sequences_manual(scaled_val, n_steps)
        if X_val_temp.size > 0 and y_val_temp.size > 0:
            X_val, y_val = X_val_temp, y_val_temp
            logger.info(f"Created val sequences: X_val {X_val.shape}, y_val {y_val.shape}")
            validation_data_for_fit = (X_val, y_val) # Prepare tuple for Keras
        else: logger.warning(f"Could not create val sequences (Val len {len(scaled_val)} <= n_steps {n_steps}?). Val disabled.")

    if X_train.size == 0:
        logger.error(f"Skipping {model_type}: Not enough train data ({len(scaled_train)}) for sequences (n_steps={n_steps}).")
        model_params_used['status'] = 'Error - Not enough data for sequences'
        return None, scaler, model_params_used

    logger.info(f"Created train sequences: X_train {X_train.shape}, y_train {y_train.shape}")
    model_params_used['n_steps'] = n_steps

    # Reshape for Keras (samples, timesteps, features=1)
    n_features: int = 1
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], n_features))
    if validation_data_for_fit: # Reshape X_val if created
        X_val = X_val.reshape((X_val.shape[0], X_val.shape[1], n_features))
        validation_data_for_fit = (X_val, y_val) # Update tuple

    # 3. Early Stopping Callback Setup
    early_stopping_patience = config_params['NN_EARLY_STOPPING_PATIENCE']
    final_fit_callbacks: List[keras.callbacks.Callback] = []
    early_stopping_callback = None
    if early_stopping_patience > 0:
        monitor_metric = 'val_loss' if validation_data_for_fit else 'loss'
        early_stopping_callback = EarlyStopping(monitor=monitor_metric, patience=early_stopping_patience,
                                                verbose=1, mode='min', restore_best_weights=True)
        final_fit_callbacks.append(early_stopping_callback)
        logger.info(f"Early stopping enabled for final fit ({monitor_metric}, patience {early_stopping_patience}).")
        model_params_used.update({'early_stopping_patience': early_stopping_patience, 'early_stopping_monitor': monitor_metric})
    else:
        logger.info("Early stopping disabled for final fit.")
        model_params_used['early_stopping_patience'] = 0

    # 4. Model Definition and Training
    tf.random.set_seed(config_params['RANDOM_SEED'])
    epochs = config_params['RNN_EPOCHS'] if model_type == 'RNN' else config_params['LSTM_EPOCHS']
    batch_size = config_params['NN_BATCH_SIZE']
    verbose = config_params['NN_VERBOSE']
    loss_function = config_params['NN_LOSS_FUNCTION']
    model_params_used.update({'batch_size': batch_size, 'loss_function': loss_function})

    if use_tuner:
        if not _keras_tuner_available:
            logger.error(f"USE_KERAS_TUNER=True, but KerasTuner not installed. Aborting {model_type}.")
            model_params_used['status'] = 'Error - KerasTuner not installed'
            return None, scaler, model_params_used

        logger.info(f"--- Initiating KerasTuner Search for {model_type} ---")
        model_params_used['tuning_attempted'] = True

        tuner_objective = config_params['NN_TUNER_OBJECTIVE']
        if tuner_objective.startswith('val_') and not validation_data_for_fit:
             logger.error(f"FATAL: Tuner objective '{tuner_objective}' needs val data, none available. Aborting.")
             model_params_used['status'] = 'Error - Tuner objective needs valid val data'
             return None, scaler, model_params_used

        model_builder = lambda hp: build_hypermodel(hp, model_type=model_type, n_steps=n_steps, config_params=config_params, n_features=n_features)

        # Configure and Run Tuner
        tuner_type = config_params['NN_TUNER_TYPE']
        tuner_class_map = {'RandomSearch': kt.RandomSearch, 'Hyperband': kt.Hyperband, 'BayesianOptimization': kt.BayesianOptimization}
        tuner_class = tuner_class_map.get(tuner_type)
        if not tuner_class:
            logger.warning(f"Unsupported NN_TUNER_TYPE '{tuner_type}'. Defaulting to RandomSearch.")
            tuner_class, tuner_type = kt.RandomSearch, 'RandomSearch'

        project_name = f"{config_params['NN_TUNER_PROJECT_NAME_PREFIX']}_{model_type}"
        tuner_dir = config_params['KERAS_TUNER_DIR'] # Timestamped path from main
        tuner_overwrite = config_params['KERAS_TUNER_OVERWRITE']
        max_trials = config_params['NN_TUNER_MAX_TRIALS']
        executions_per_trial = config_params['NN_TUNER_EXECUTIONS_PER_TRIAL']
        tuner_epochs = config_params['NN_TUNER_EPOCHS']

        # Store tuner config
        model_params_used['tuner_config'] = { 'type': tuner_type, 'objective': tuner_objective, 'max_trials': max_trials,
            'executions_per_trial': executions_per_trial, 'epochs_per_trial': tuner_epochs, 'directory': tuner_dir,
            'project_name': project_name, 'overwrite': tuner_overwrite, 'seed': config_params['RANDOM_SEED'] }

        tuner = tuner_class(model_builder, objective=kt.Objective(tuner_objective, direction="min"),
                            max_trials=max_trials, executions_per_trial=executions_per_trial,
                            directory=tuner_dir, project_name=project_name, seed=config_params['RANDOM_SEED'],
                            overwrite=tuner_overwrite) # Add Hyperband args if needed

        logger.info(f"--- Tuner ({tuner_type}) Search Space ---"); tuner.search_space_summary() # Log summary
        logger.info(f"--- Starting Tuner Search (Max Trials: {max_trials}, Epochs/Trial: {tuner_epochs}) ---")
        logger.info(f"Tuner results in: {os.path.join(tuner_dir, project_name)}")

        search_callbacks: List[keras.callbacks.Callback] = []
        if early_stopping_patience > 0: # Optional early stopping during search
            search_early_stopping = EarlyStopping(monitor=tuner_objective, patience=early_stopping_patience,
                                                  verbose=1, mode='min', restore_best_weights=False)
            search_callbacks.append(search_early_stopping)
            logger.info(f"Early stopping during search active ({tuner_objective}, patience {early_stopping_patience})")

        try:
             tuner.search(X_train, y_train, epochs=tuner_epochs, validation_data=validation_data_for_fit,
                          batch_size=batch_size, callbacks=search_callbacks, verbose=verbose)
             model_params_used['tuning_status'] = 'Completed'
        except Exception as e:
             logger.error(f"KerasTuner search failed for {model_type}: {e}", exc_info=True)
             model_params_used.update({'tuning_status': f'Failed: {e}', 'status': 'Error - Tuner search failed'})
             return None, scaler, model_params_used

        logger.info(f"\n--- KerasTuner Search Complete for {model_type} ---")
        try: # Get best HPs and build best model
            best_hps_list = tuner.get_best_hyperparameters(num_trials=1)
            if not best_hps_list:
                logger.error(f"Error: Tuner found no best hyperparameters for {model_type}.")
                model_params_used.update({'tuning_status': 'Completed - No best HPs', 'status': 'Error - No best HPs from tuner'})
                return None, scaler, model_params_used

            best_hps = best_hps_list[0]
            model_params_used['best_hyperparameters'] = best_hps.values
            logger.info(f"\n--- Best Hyperparameters Found ({model_type}) ---")
            for param, value in sorted(best_hps.values.items()): logger.info(f"  - {param}: {value}")

            logger.info(f"\n--- Building Best {model_type} Model from Tuner ---")
            best_model = tuner.hypermodel.build(best_hps) # Build final model with best HPs

        except Exception as e:
             logger.error(f"Error retrieving/building from tuner results: {e}", exc_info=True)
             model_params_used.update({'tuning_status': f'Completed - Error processing: {e}', 'status': 'Error - Tuner result processing failed'})
             return None, scaler, model_params_used

    else: # Manual Model Building (Tuner Disabled)
        logger.info(f"--- Building {model_type} Manually (Tuner Disabled) ---")
        model_params_used['tuning_attempted'] = False
        manual_model = Sequential(name=f"Manual_{model_type}")
        manual_model.add(Input(shape=(n_steps, n_features)))

        units = config_params['NN_UNITS']; activation = config_params['NN_ACTIVATION']
        optimizer_name = config_params['NN_OPTIMIZER']
        add_dropout = config_params['NN_ADD_DROPOUT']; dropout_rate = config_params['NN_DROPOUT_RATE']

        layer_name = f"manual_{model_type}_layer"
        if model_type == 'RNN': manual_model.add(SimpleRNN(units=units, activation=activation, name=layer_name))
        elif model_type == 'LSTM': manual_model.add(LSTM(units=units, activation=activation, name=layer_name))
        if add_dropout: manual_model.add(Dropout(rate=dropout_rate, name="manual_dropout"))
        manual_model.add(Dense(1, name="output_dense"))

        optimizer_instance = get_optimizer(optimizer_name)
        manual_model.compile(optimizer=optimizer_instance, loss=loss_function, metrics=['mae'])
        best_model = manual_model

        model_params_used['manual_config'] = {'units': units, 'activation': activation, 'optimizer': optimizer_name,
                                              'add_dropout': add_dropout, 'dropout_rate': dropout_rate if add_dropout else None}

    # Final Training (Best Tuned Model or Manual Model)
    if best_model is not None:
        logger.info(f"\n--- Final Training Phase ({model_type}, Max Epochs: {epochs}) ---")
        logger.info(f"Final Model Architecture Summary ({model_type}):"); best_model.summary(print_fn=logger.info)
        model_params_used['final_epochs_configured'] = epochs

        try:
            history = best_model.fit(X_train, y_train, epochs=epochs, validation_data=validation_data_for_fit,
                                     batch_size=batch_size, callbacks=final_fit_callbacks, verbose=verbose)
            logger.info(f"--- {model_type} Final Training Complete ---")
            model_params_used['status'] = 'Trained Successfully'

            # Record actual epochs run and final metrics
            actual_epochs_run = len(history.epoch)
            model_params_used['final_epochs_run'] = actual_epochs_run
            final_metrics = history.history
            if early_stopping_callback and early_stopping_callback.stopped_epoch > 0: # Stopped early
                 best_epoch_val = early_stopping_callback.best
                 monitor = early_stopping_callback.monitor
                 best_epoch_num = early_stopping_callback.best_epoch + 1
                 logger.info(f"Final train stopped early at epoch {actual_epochs_run}. Best weights from epoch {best_epoch_num}.")
                 logger.info(f"Best monitored value ({monitor}): {best_epoch_val:.4f}")
                 model_params_used.update({'early_stopping_triggered': True, 'best_monitored_value': float(best_epoch_val), 'best_epoch_number': best_epoch_num})
            else: # Ran full epochs or didn't trigger stop
                 logger.info(f"Final training ran for {actual_epochs_run} epochs.")
                 model_params_used['early_stopping_triggered'] = False
                 if 'loss' in final_metrics and final_metrics['loss']: model_params_used['final_train_loss'] = float(final_metrics['loss'][-1])
                 if 'val_loss' in final_metrics and final_metrics['val_loss']: model_params_used['final_val_loss'] = float(final_metrics['val_loss'][-1])
                 if 'mae' in final_metrics and final_metrics['mae']: model_params_used['final_train_mae'] = float(final_metrics['mae'][-1])
                 if 'val_mae' in final_metrics and final_metrics['val_mae']: model_params_used['final_val_mae'] = float(final_metrics['val_mae'][-1])

            return best_model, scaler, model_params_used # Success

        except Exception as e:
             logger.error(f"Error during {model_type} final fit: {e}", exc_info=True)
             model_params_used['status'] = f'Error - Final fit failed: {e}'
             return None, scaler, model_params_used # Return scaler even if fit fails
    else:
         logger.error(f"Error: No {model_type} model available for final training.")
         model_params_used['status'] = 'Error - No model for final training'
         return None, scaler, model_params_used


# --- Prediction Function ---
def forecast_nn(
    model: Optional[Model], scaler: Optional[MinMaxScaler], train_data: pd.DataFrame,
    test_periods: int, n_steps: int, model_type: str
) -> np.ndarray:
    """Generates forecasts using a trained NN model iteratively."""
    logger.info(f"--- Generating {model_type} forecast ({test_periods} periods) ---")
    n_features: int = 1
    forecasts_final = np.full(test_periods, np.nan) # Initialize with NaNs

    if model is None or scaler is None:
        logger.error(f"{model_type} Forecast Error: Model or scaler unavailable (training failed?).")
        return forecasts_final

    # Get tail of input data for initialization
    train_y_for_init = train_data[['y']]
    if train_y_for_init.isnull().any().any():
         logger.warning(f"{model_type} Forecast Warn: Input data for init contains NaNs. Dropping NaNs from tail.")
         train_y_for_init = train_y_for_init.dropna()
    if len(train_y_for_init) < n_steps:
         logger.error(f"{model_type} Forecast Error: Not enough valid data ({len(train_y_for_init)}) in tail for init (need {n_steps}).")
         return forecasts_final

    try: # Scale the exact last n_steps points
        scaled_train_tail = scaler.transform(train_y_for_init.iloc[-n_steps:])
    except Exception as e:
        logger.error(f"{model_type} Forecast Error: Scaling train tail failed: {e}", exc_info=True)
        return forecasts_final

    forecasts_scaled: List[float] = []
    current_input_list: List[List[float]] = scaled_train_tail.tolist() # Start with last known scaled values

    logger.info(f"Starting iterative prediction for {test_periods} steps...")
    for i in range(test_periods):
        try:
            current_batch: np.ndarray = np.array(current_input_list).reshape((1, n_steps, n_features))
            current_pred_scaled: np.ndarray = model.predict(current_batch, verbose=0) # Predict step i+1
            pred_value: float = current_pred_scaled[0, 0] # Extract scalar prediction
            forecasts_scaled.append(pred_value)

            # Update input for next step: drop oldest, append prediction
            current_input_list.pop(0)
            current_input_list.append([pred_value])
        except Exception as e:
             logger.error(f"Error during {model_type} forecast step {i+1}/{test_periods}: {e}", exc_info=True)
             break # Stop forecasting on error

    num_forecasts_generated = len(forecasts_scaled)
    logger.info(f"Generated {num_forecasts_generated} scaled forecast(s).")

    if forecasts_scaled: # Inverse transform if forecasts were generated
        forecasts_scaled_arr = np.array(forecasts_scaled).reshape(-1, 1)
        try:
            forecasts_unscaled = scaler.inverse_transform(forecasts_scaled_arr).flatten()
            forecasts_final[:num_forecasts_generated] = forecasts_unscaled # Fill final array
            logger.info("Inverse scaling successful.")
        except Exception as e:
            logger.error(f"Error during inverse scaling for {model_type}: {e}. Forecasts incomplete/NaN.", exc_info=True)

    logger.info(f"--- {model_type} Forecasting Finished ---")
    return forecasts_final


def save_nn_model(model: Model, file_path: str):
    """Saves a Keras model (using .keras or .h5 based on TF version)."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        # Use recommended .keras format if TF >= 2.7, else HDF5
        save_format = '.keras' if hasattr(tf, '__version__') and tf.__version__ >= '2.7' else '.h5'
        # Ensure correct extension
        base_path, _ = os.path.splitext(file_path)
        file_path_with_ext = base_path + save_format

        model.save(file_path_with_ext)
        logger.info(f"Keras model saved successfully to: {file_path_with_ext} (format: {save_format})")
    except Exception as e:
        logger.error(f"Error saving Keras model to {file_path_with_ext}: {e}", exc_info=True)


# --- Unified Runner Function ---
def run_nn_model(
    model_type: str, # 'RNN' or 'LSTM'
    train_data: pd.DataFrame,
    val_data: Optional[pd.DataFrame],
    test_periods: int, # Test set size OR forecast horizon
    test_index: pd.DatetimeIndex, # Test set index OR future index
    config_params: Dict[str, Any]
) -> Tuple[pd.Series, Optional[Dict[str, Any]], Optional[Model]]:
    """
    Orchestrates NN training and forecasting for evaluation or final prediction.
    Returns: Forecast Series, Parameters Dict (incl. scaler ref), Fitted Model object.
    """
    model: Optional[Model] = None
    scaler: Optional[MinMaxScaler] = None
    model_params: Optional[Dict[str, Any]] = {'model_type': model_type, 'status': 'Not Run', 'scaler_object_ref': None}
    forecast_values: np.ndarray = np.full(test_periods, np.nan)

    try:
        # Train model (handles scaling, sequences, tuning, fitting)
        model, scaler, model_params = train_nn(model_type, train_data, val_data, config_params)

        # Forecast only if training succeeded
        if model and scaler and model_params and model_params.get('status') == 'Trained Successfully':
            forecast_values = forecast_nn(model=model, scaler=scaler, train_data=train_data, # Use data model was trained on
                                          test_periods=test_periods, n_steps=config_params['NN_STEPS'], model_type=model_type)
        else:
            status_msg = model_params.get('status', 'Unknown Failure') if model_params else 'Init Failed'
            logger.warning(f"{model_type} training failed ({status_msg}), skipping forecast.")

    except Exception as e:
        logger.error(f"{model_type} Error during run_nn_model: {e}", exc_info=True)
        if model_params: model_params.update({'status': 'Error - Overall Run Failed', 'run_error': str(e)})

    # Return forecast Series with correct index, params dict, and model
    forecast_series = pd.Series(forecast_values, index=test_index, name=model_type)
    if model_params is None: model_params = {'model_type': model_type, 'status': 'Failed before param capture'} # Safety net

    return forecast_series, model_params, model