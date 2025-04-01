import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, Input, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam, RMSprop, SGD, Adagrad # Import more optimizers
from sklearn.preprocessing import MinMaxScaler
import warnings
import logging
import os # For saving models
import json # For saving tuner HPs
from typing import Tuple, Optional, List, Union, Any, Dict

# KerasTuner handling
_keras_tuner_available = False
try:
    import keras_tuner as kt
    _keras_tuner_available = True
except ImportError:
    # Warning printed only if USE_KERAS_TUNER is True and kt is missing
    pass

logger = logging.getLogger(__name__)

# --- Helper: Create Sequences ---
def create_sequences_manual(data: Optional[np.ndarray], n_steps: int) -> Tuple[np.ndarray, np.ndarray]:
    """Creates sequences manually from scaled data, handling edge cases."""
    if data is None or len(data) <= n_steps:
        logger.debug(f"Not enough data ({len(data) if data is not None else 'None'}) to create sequences with n_steps={n_steps}")
        return np.array([]), np.array([]) # Return empty arrays
    X, y = [], []
    for i in range(len(data) - n_steps):
        end_ix = i + n_steps
        seq_x, seq_y = data[i:end_ix], data[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    if not X:
        logger.debug(f"Sequence creation resulted in empty X list (data length: {len(data)}, n_steps: {n_steps})")
        return np.array([]), np.array([])
    return np.array(X), np.array(y).flatten()

# --- Helper: Get Optimizer Instance ---
def get_optimizer(name: str, learning_rate: Optional[float] = None) -> keras.optimizers.Optimizer:
    """Gets a Keras optimizer instance by name, optionally setting the learning rate."""
    name = name.lower()
    kwargs = {}
    if learning_rate is not None:
        kwargs['learning_rate'] = learning_rate

    if name == 'adam': return Adam(**kwargs)
    elif name == 'rmsprop': return RMSprop(**kwargs)
    elif name == 'sgd': return SGD(**kwargs)
    elif name == 'adagrad': return Adagrad(**kwargs)
    # Add more optimizers if needed
    else:
        logger.warning(f"Unsupported optimizer name '{name}'. Using Adam with default settings.")
        return Adam() # Default safe choice


# --- Hypermodel Builder Function (Updated) ---
def build_hypermodel(hp: kt.HyperParameters, model_type: str, n_steps: int, config_params: Dict[str, Any], n_features: int = 1) -> Model:
    """Builds a Keras RNN or LSTM model with hyperparameters defined by KerasTuner."""
    model = Sequential()
    model.add(Input(shape=(n_steps, n_features)))

    # Tune Recurrent Layer
    hp_units: int = hp.Int('units',
                           min_value=config_params['NN_TUNER_HP_UNITS_MIN'],
                           max_value=config_params['NN_TUNER_HP_UNITS_MAX'],
                           step=config_params['NN_TUNER_HP_UNITS_STEP'])
    hp_activation: str = hp.Choice('activation', values=config_params['NN_TUNER_HP_ACTIVATION_CHOICES'])

    if model_type == 'RNN':
        model.add(SimpleRNN(units=hp_units, activation=hp_activation, name=f"tuned_{model_type}_layer"))
    elif model_type == 'LSTM':
        model.add(LSTM(units=hp_units, activation=hp_activation, name=f"tuned_{model_type}_layer"))
    else:
        raise ValueError("Invalid model_type. Choose 'RNN' or 'LSTM'.")

    # Tune Optional Dropout
    use_dropout = False
    if config_params['NN_TUNER_HP_USE_DROPOUT']:
         # Let tuner decide if dropout should be used, even if globally enabled for tuning space
         use_dropout = hp.Boolean("use_dropout", default=True)

    if use_dropout:
         hp_dropout_rate = hp.Float('dropout_rate',
                                    min_value=config_params['NN_TUNER_HP_DROPOUT_MIN'],
                                    max_value=config_params['NN_TUNER_HP_DROPOUT_MAX'],
                                    step=config_params['NN_TUNER_HP_DROPOUT_STEP'])
         model.add(Dropout(rate=hp_dropout_rate, name="tuned_dropout"))

    # Output Layer
    model.add(Dense(1, name="output_dense"))

    # Tune Optimizer and Learning Rate
    hp_learning_rate: float = hp.Float('learning_rate',
                                      min_value=config_params['NN_TUNER_HP_LR_MIN'],
                                      max_value=config_params['NN_TUNER_HP_LR_MAX'],
                                      sampling='log')
    hp_optimizer_choice: str = hp.Choice('optimizer', values=config_params['NN_TUNER_HP_OPTIMIZER_CHOICES'])

    optimizer = get_optimizer(hp_optimizer_choice, hp_learning_rate)

    model.compile(optimizer=optimizer, loss=config_params['NN_LOSS_FUNCTION'], metrics=['mae']) # Always include MAE for potential monitoring
    return model


# --- Main Training Function (Updated) ---
def train_nn(
    model_type: str,
    train_data: pd.DataFrame,
    val_data: Optional[pd.DataFrame],
    config_params: Dict[str, Any] # Pass relevant config
) -> Tuple[Optional[Model], Optional[MinMaxScaler], Optional[Dict[str, Any]]]:
    """
    Trains an RNN or LSTM model, potentially using KerasTuner.
    Returns the trained Keras model, the scaler object, and a dict of parameters used.
    """
    logger.info(f"--- Starting {model_type} Training ---")
    model_params_used: Dict[str, Any] = {'model_type': model_type}
    n_steps = config_params['NN_STEPS']
    use_tuner = config_params['USE_KERAS_TUNER']
    best_model: Optional[Model] = None
    scaler: Optional[MinMaxScaler] = None
    history: Optional[keras.callbacks.History] = None
    tuner_results: Optional[Dict[str, Any]] = None # To store best HPs if tuning

    if train_data is None or train_data.empty or train_data['y'].isnull().all():
        logger.warning(f"Skipping {model_type} training: Training data is empty or all NaN.")
        model_params_used['status'] = 'Skipped - No training data'
        return None, None, model_params_used

    # --- 1. Scaling ---
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_train: Optional[np.ndarray] = None
    scaled_val: Optional[np.ndarray] = None
    try:
        # Always use double brackets to keep DataFrame structure for scaler
        train_y = train_data[['y']].dropna()
        if train_y.empty:
            logger.warning(f"Skipping {model_type} training: No non-NaN training data after dropna().")
            model_params_used['status'] = 'Skipped - No non-NaN training data'
            return None, None, model_params_used

        scaled_train = scaler.fit_transform(train_y)
        model_params_used['scaler_type'] = 'MinMaxScaler'
        model_params_used['scaler_feature_range'] = (0, 1)
        # Store the scaler reference in the parameters dictionary
        model_params_used['scaler_object_ref'] = scaler
        logger.info(f"Training data scaled using MinMaxScaler. Shape: {scaled_train.shape}")

        # Scale validation data if it exists and is valid
        if val_data is not None and not val_data.empty and not val_data['y'].isnull().all():
             val_y = val_data[['y']].dropna()
             if not val_y.empty:
                scaled_val = scaler.transform(val_y) # Use transform, not fit_transform
                logger.info(f"Validation data scaled using the same scaler. Shape: {scaled_val.shape}")
             else:
                logger.warning("Validation data contains only NaNs after dropna(). Validation disabled for NN training/tuning.")
                scaled_val = None # Ensure it's None if invalid
        else:
            # Log if validation data is missing or invalid
            if val_data is None:
                logger.info("No validation data provided.")
            elif val_data.empty:
                logger.warning("Validation data provided is empty.")
            else: # Must be all NaN
                logger.warning("Validation data provided contains only NaN values.")
            scaled_val = None # Ensure it's None

        # Check tuner objective requirements against available validation data
        tuner_objective = config_params.get('NN_TUNER_OBJECTIVE', 'val_loss') # Default if key missing
        if use_tuner and tuner_objective.startswith('val_') and scaled_val is None:
             logger.error(f"KerasTuner objective '{tuner_objective}' requires valid validation data, but none is available after scaling. Aborting {model_type} tuning/training.")
             model_params_used['status'] = 'Error - Tuner needs valid validation data'
             return None, scaler, model_params_used # Return scaler even if tuning fails

    except Exception as e:
        logger.error(f"Error during scaling for {model_type}: {e}. Skipping training.", exc_info=True)
        model_params_used['status'] = f'Error - Scaling failed: {e}'
        return None, scaler, model_params_used # Return scaler maybe

    # --- 2. Create Sequences ---
    X_train, y_train = create_sequences_manual(scaled_train, n_steps)
    X_val, y_val = None, None
    validation_data_for_fit: Optional[Tuple[np.ndarray, np.ndarray]] = None

    if scaled_val is not None:
        X_val_temp, y_val_temp = create_sequences_manual(scaled_val, n_steps)
        if X_val_temp.size > 0 and y_val_temp.size > 0:
            X_val, y_val = X_val_temp, y_val_temp
            logger.info(f"Created validation sequences: X_val {X_val.shape}, y_val {y_val.shape}")
            validation_data_for_fit = (X_val, y_val) # Prepare tuple for Keras fit/search
        else:
            logger.warning(f"Could not create validation sequences (Val data length {len(scaled_val)} <= n_steps {n_steps}?). Validation disabled for fit/search.")
            # If tuner needed val data, we already errored out above.
            # If it didn't (e.g., objective='loss'), we just proceed without validation.

    if X_train.size == 0:
        logger.error(f"Skipping {model_type} training: Not enough training data ({len(scaled_train)}) to create sequences with n_steps={n_steps}.")
        model_params_used['status'] = 'Error - Not enough data for sequences'
        return None, scaler, model_params_used

    logger.info(f"Created training sequences: X_train {X_train.shape}, y_train {y_train.shape}")
    model_params_used['n_steps'] = n_steps

    # Reshape for Keras (samples, timesteps, features)
    n_features: int = 1
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], n_features))
    if validation_data_for_fit: # If validation sequences were created
        X_val = X_val.reshape((X_val.shape[0], X_val.shape[1], n_features))
        # Re-assign the reshaped X_val back into the tuple
        validation_data_for_fit = (X_val, y_val)


    # --- 3. Early Stopping Callback ---
    early_stopping_patience = config_params['NN_EARLY_STOPPING_PATIENCE']
    final_fit_callbacks: List[keras.callbacks.Callback] = []
    early_stopping_callback = None # Initialize
    if early_stopping_patience > 0:
        monitor_metric = 'val_loss' if validation_data_for_fit else 'loss'
        early_stopping_callback = EarlyStopping(
            monitor=monitor_metric,
            patience=early_stopping_patience,
            verbose=1,
            mode='min',
            restore_best_weights=True # Restore best weights based on monitored metric
        )
        final_fit_callbacks.append(early_stopping_callback)
        logger.info(f"Early stopping enabled for final fit, monitoring '{monitor_metric}' with patience {early_stopping_patience}.")
        model_params_used['early_stopping_patience'] = early_stopping_patience
        model_params_used['early_stopping_monitor'] = monitor_metric
    else:
        logger.info("Early stopping disabled for final fit.")
        model_params_used['early_stopping_patience'] = 0


    # --- 4. Model Definition and Training ---
    tf.random.set_seed(config_params['RANDOM_SEED'])
    epochs = config_params['RNN_EPOCHS'] if model_type == 'RNN' else config_params['LSTM_EPOCHS']
    batch_size = config_params['NN_BATCH_SIZE']
    verbose = config_params['NN_VERBOSE']
    loss_function = config_params['NN_LOSS_FUNCTION']

    model_params_used['batch_size'] = batch_size
    model_params_used['loss_function'] = loss_function

    if use_tuner:
        if not _keras_tuner_available:
            logger.error(f"USE_KERAS_TUNER is True, but KerasTuner is not installed. Aborting {model_type}.")
            model_params_used['status'] = 'Error - KerasTuner requested but not installed'
            return None, scaler, model_params_used

        logger.info(f"--- Initiating KerasTuner Search for {model_type} ---")
        model_params_used['tuning_attempted'] = True

        # Double-check validation data requirement (already checked during scaling, but belt-and-suspenders)
        tuner_objective = config_params['NN_TUNER_OBJECTIVE']
        if tuner_objective.startswith('val_') and not validation_data_for_fit:
             logger.error(f"FATAL ERROR: Tuner objective '{tuner_objective}' requires validation data, but none available. Aborting {model_type}.")
             model_params_used['status'] = 'Error - Tuner objective needs valid validation data'
             return None, scaler, model_params_used

        # Define the hypermodel builder lambda
        model_builder = lambda hp: build_hypermodel(hp, model_type=model_type, n_steps=n_steps, config_params=config_params, n_features=n_features)

        # --- Configure and Run the Tuner ---
        tuner_type = config_params['NN_TUNER_TYPE']
        tuner_class: Optional[type[kt.Tuner]] = None
        if tuner_type == 'RandomSearch': tuner_class = kt.RandomSearch
        elif tuner_type == 'Hyperband': tuner_class = kt.Hyperband # Note: Hyperband uses objective differently
        elif tuner_type == 'BayesianOptimization': tuner_class = kt.BayesianOptimization
        else:
            logger.warning(f"Unsupported NN_TUNER_TYPE '{tuner_type}'. Defaulting to RandomSearch.")
            tuner_class = kt.RandomSearch
            tuner_type = 'RandomSearch' # Update type for logging/params

        project_name = f"{config_params['NN_TUNER_PROJECT_NAME_PREFIX']}_{model_type}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
        tuner_dir = config_params['KERAS_TUNER_DIR']
        tuner_overwrite = config_params['KERAS_TUNER_OVERWRITE']
        max_trials = config_params['NN_TUNER_MAX_TRIALS']
        executions_per_trial = config_params['NN_TUNER_EXECUTIONS_PER_TRIAL']
        tuner_epochs = config_params['NN_TUNER_EPOCHS']


        # Store tuner config
        model_params_used['tuner_config'] = {
            'type': tuner_type, 'objective': tuner_objective, 'max_trials': max_trials,
            'executions_per_trial': executions_per_trial, 'epochs_per_trial': tuner_epochs,
            'directory': tuner_dir, 'project_name': project_name, 'overwrite': tuner_overwrite,
            'seed': config_params['RANDOM_SEED']
            }

        # Initialize tuner
        tuner = tuner_class(
            model_builder,
            objective=kt.Objective(tuner_objective, direction="min"), # Assume minimization objective
            max_trials=max_trials,
            executions_per_trial=executions_per_trial,
            directory=tuner_dir,
            project_name=project_name,
            seed=config_params['RANDOM_SEED'],
            overwrite=tuner_overwrite
            # Hyperband needs max_epochs if used, factor hyperband specific args here
        )

        logger.info(f"--- Tuner ({tuner_type}) Search Space Summary ---")
        # --- MODIFICATION START ---
        # tuner.search_space_summary(print_fn=logger.info) # Log summary via logger <-- OLD
        tuner.search_space_summary() # <-- NEW (Prints to stdout/stderr, check console or log file)
        # --- MODIFICATION END ---
        logger.info(f"--- Starting Tuner Search (Max Trials: {max_trials}, Epochs/Trial: {tuner_epochs}) ---")

        search_callbacks: List[keras.callbacks.Callback] = []
        if early_stopping_patience > 0:
            # Early stopping during search monitors the tuner's objective
            search_early_stopping = EarlyStopping(
                monitor=tuner_objective, # Monitor the actual tuner objective
                patience=early_stopping_patience, # Could use a shorter patience here if desired
                verbose=1, mode='min', restore_best_weights=False # Tuner finds best HPs, not weights
            )
            search_callbacks.append(search_early_stopping)
            logger.info(f"Early stopping during search monitors '{tuner_objective}' with patience {early_stopping_patience}")

        try:
             tuner.search(X_train, y_train,
                          epochs=tuner_epochs, # Epochs for each trial
                          validation_data=validation_data_for_fit, # Pass tuple directly
                          batch_size=batch_size, # Use configured batch size during search
                          callbacks=search_callbacks,
                          verbose=verbose)
             model_params_used['tuning_status'] = 'Completed'
        except Exception as e:
             logger.error(f"KerasTuner search failed for {model_type}: {e}", exc_info=True)
             model_params_used['tuning_status'] = f'Failed: {e}'
             model_params_used['status'] = 'Error - Tuner search failed'
             return None, scaler, model_params_used

        logger.info(f"\n--- KerasTuner Search Complete for {model_type} ---")
        try:
            # Get best hyperparameters
            best_hps_list = tuner.get_best_hyperparameters(num_trials=1)
            if not best_hps_list:
                logger.error(f"Error: KerasTuner could not find any best hyperparameters for {model_type}.")
                model_params_used['tuning_status'] = 'Completed - No best HPs found'
                model_params_used['status'] = 'Error - No best HPs from tuner'
                return None, scaler, model_params_used

            best_hps = best_hps_list[0]
            tuner_results = {'best_hyperparameters': best_hps.values}
            model_params_used['best_hyperparameters'] = best_hps.values # Store in main params dict
            logger.info(f"\n--- Best Hyperparameters Found for {model_type} ---")
            # Log hyperparameters cleanly
            best_hps_dict = best_hps.values
            for param, value in sorted(best_hps_dict.items()): # Sort for consistent logging
                logger.info(f"  - {param}: {value}")

            # Build the final model using the best hyperparameters
            logger.info(f"\n--- Building Best {model_type} Model from Tuner Results ---")
            best_model = tuner.hypermodel.build(best_hps) # Build the definitive best model

        except Exception as e:
             logger.error(f"Error retrieving results or building best model from tuner: {e}", exc_info=True)
             model_params_used['tuning_status'] = f'Completed - Error processing results: {e}'
             model_params_used['status'] = 'Error - Tuner result processing failed'
             return None, scaler, model_params_used

    else:
        # --- Manual Model Building ---
        logger.info(f"--- Building {model_type} Manually (Tuner Disabled) ---")
        model_params_used['tuning_attempted'] = False
        manual_model = Sequential(name=f"Manual_{model_type}")
        manual_model.add(Input(shape=(n_steps, n_features)))

        activation = config_params['NN_ACTIVATION']
        units = config_params['NN_UNITS']
        optimizer_name = config_params['NN_OPTIMIZER']
        add_dropout = config_params['NN_ADD_DROPOUT']
        dropout_rate = config_params['NN_DROPOUT_RATE']

        layer_name = f"manual_{model_type}_layer"
        if model_type == 'RNN':
            manual_model.add(SimpleRNN(units=units, activation=activation, name=layer_name))
        elif model_type == 'LSTM':
            manual_model.add(LSTM(units=units, activation=activation, name=layer_name))

        if add_dropout:
             manual_model.add(Dropout(rate=dropout_rate, name="manual_dropout"))

        manual_model.add(Dense(1, name="output_dense"))

        optimizer_instance = get_optimizer(optimizer_name) # Gets default LR here
        manual_model.compile(optimizer=optimizer_instance, loss=loss_function, metrics=['mae'])

        best_model = manual_model # Use this manually defined model

        # Log manual parameters
        model_params_used['manual_config'] = {
            'units': units, 'activation': activation, 'optimizer': optimizer_name,
            'add_dropout': add_dropout, 'dropout_rate': dropout_rate if add_dropout else None
            }

    # --- Final Training (Best Tuned Model or Manual Model) ---
    if best_model is not None:
        logger.info(f"\n--- Final Training Phase for {model_type} (Max Epochs: {epochs}) ---")
        logger.info(f"Final Model Architecture Summary ({model_type}):")
        best_model.summary(print_fn=logger.info) # Log summary using logger

        model_params_used['final_epochs_configured'] = epochs

        try:
            history = best_model.fit(
                X_train, y_train,
                epochs=epochs, # Max epochs for final training
                validation_data=validation_data_for_fit, # Pass tuple directly (can be None)
                batch_size=batch_size,
                callbacks=final_fit_callbacks, # Use the early stopping callback here (can be empty list)
                verbose=verbose
            )
            logger.info(f"--- {model_type} Final Training Complete ---")
            model_params_used['status'] = 'Trained Successfully'

            # Record actual epochs run and final metrics
            actual_epochs_run = len(history.epoch)
            model_params_used['final_epochs_run'] = actual_epochs_run
            final_metrics = history.history
            # Store last value of monitored metric (or best if early stopping restored weights)
            if early_stopping_callback and early_stopping_callback.stopped_epoch > 0:
                 # Early stopping restored best weights, so metrics from history are from the last epoch run, not necessarily the best
                 # Use the 'best' attribute from the callback which stores the best monitored value
                 best_epoch_val = early_stopping_callback.best
                 monitor = early_stopping_callback.monitor
                 best_epoch_num = early_stopping_callback.best_epoch + 1 # epoch numbers are 0-based in callback
                 logger.info(f"Final training stopped early at epoch {actual_epochs_run}. Best weights restored from epoch {best_epoch_num}.")
                 logger.info(f"Best monitored value ({monitor}): {best_epoch_val:.4f}")
                 model_params_used['early_stopping_triggered'] = True
                 model_params_used['best_monitored_value'] = float(best_epoch_val) # Ensure serializable
                 model_params_used['best_epoch_number'] = best_epoch_num
            else:
                 logger.info(f"Final training ran for all {actual_epochs_run} epochs (or early stopping didn't trigger).")
                 model_params_used['early_stopping_triggered'] = False
                 # Store final epoch's metrics
                 if 'loss' in final_metrics and final_metrics['loss']: model_params_used['final_train_loss'] = float(final_metrics['loss'][-1])
                 if 'val_loss' in final_metrics and final_metrics['val_loss']: model_params_used['final_val_loss'] = float(final_metrics['val_loss'][-1])
                 if 'mae' in final_metrics and final_metrics['mae']: model_params_used['final_train_mae'] = float(final_metrics['mae'][-1])
                 if 'val_mae' in final_metrics and final_metrics['val_mae']: model_params_used['final_val_mae'] = float(final_metrics['val_mae'][-1])


            return best_model, scaler, model_params_used # Return the trained model, scaler, and params

        except Exception as e:
             logger.error(f"Error during {model_type} final model.fit: {e}", exc_info=True)
             model_params_used['status'] = f'Error - Final fit failed: {e}'
             return None, scaler, model_params_used # Return scaler even if fit fails, model is None
    else:
         logger.error(f"Error: No {model_type} model (tuned or manual) was available for final training.")
         model_params_used['status'] = 'Error - No model available for final training'
         return None, scaler, model_params_used # Return scaler maybe, model is None


# --- Prediction Function (Updated) ---
def forecast_nn(
    model: Optional[Model],
    scaler: Optional[MinMaxScaler],
    train_data: pd.DataFrame, # Original UNscaled training data (or full data for final forecast)
    test_periods: int, # Number of periods to forecast (horizon)
    n_steps: int,
    model_type: str # For logging
) -> np.ndarray:
    """Generates forecasts using a trained NN model iteratively."""
    logger.info(f"--- Generating {model_type} forecast (Periods: {test_periods}) ---")
    n_features: int = 1
    forecasts_final = np.full(test_periods, np.nan) # Initialize with NaNs

    if model is None or scaler is None:
        logger.error(f"{model_type} Forecasting Error: Model or scaler not available (likely due to training error).")
        return forecasts_final

    # Get the tail end of the *input* training data (which could be full data for final run)
    # Data should already be imputed if needed before calling train_nn/run_nn_model
    train_y_for_init = train_data[['y']]

    if train_y_for_init.isnull().any().any():
         logger.warning(f"{model_type} Forecasting Warning: Input data for prediction initialization contains NaNs. This might indicate an issue upstream (imputation/data loading). Attempting to proceed after dropping NaNs from tail.")
         train_y_for_init = train_y_for_init.dropna()

    if len(train_y_for_init) < n_steps:
         logger.error(f"{model_type} Forecasting Error: Not enough valid data points ({len(train_y_for_init)}) in the provided training data tail for forecast initialization (need {n_steps}).")
         return forecasts_final

    try:
        # Scale the exact last n_steps points from the potentially filtered tail
        scaled_train_tail = scaler.transform(train_y_for_init.iloc[-n_steps:])
    except Exception as e:
        logger.error(f"{model_type} Forecasting Error: Scaling train_data tail for prediction failed: {e}", exc_info=True)
        return forecasts_final

    forecasts_scaled: List[float] = []
    current_input_list: List[List[float]] = scaled_train_tail.tolist() # Start with the last known scaled values

    logger.info(f"Starting iterative prediction for {test_periods} steps using last {n_steps} known points...")
    for i in range(test_periods):
        try:
            # Ensure the input shape is correct (1 sample, n_steps time steps, n_features features)
            current_batch: np.ndarray = np.array(current_input_list).reshape((1, n_steps, n_features))

            # Use verbose=0 for predict loop to avoid excessive logging
            current_pred_scaled: np.ndarray = model.predict(current_batch, verbose=0)
            # The prediction is typically [[value]], extract the scalar
            pred_value: float = current_pred_scaled[0, 0]

            # Store the scaled prediction
            forecasts_scaled.append(pred_value)

            # Update the input sequence for the next prediction:
            # Remove the oldest value and append the new prediction
            current_input_list.pop(0)
            current_input_list.append([pred_value]) # Append as a list containing the value

        except Exception as e:
             logger.error(f"Error during {model_type} forecast step {i+1}/{test_periods}: {e}", exc_info=True)
             # Stop forecasting if an error occurs at a step
             break # Exit forecast loop, keep forecasts generated so far

    num_forecasts_generated = len(forecasts_scaled)
    logger.info(f"Generated {num_forecasts_generated} scaled forecast(s).")

    if forecasts_scaled:
        # Reshape for inverse_transform expects (n_samples, n_features)
        forecasts_scaled_arr = np.array(forecasts_scaled).reshape(-1, 1)
        try:
            # Inverse transform the scaled predictions
            forecasts_unscaled = scaler.inverse_transform(forecasts_scaled_arr).flatten()
            # Fill the final array up to the number of successful predictions
            forecasts_final[:num_forecasts_generated] = forecasts_unscaled
            logger.info("Inverse scaling successful.")
        except Exception as e:
            logger.error(f"Error during inverse scaling for {model_type}: {e}. Forecasts might be incomplete or NaN.", exc_info=True)

    logger.info(f"--- {model_type} Forecasting Finished ---")
    return forecasts_final


def save_nn_model(model: Model, file_path: str):
    """Saves a Keras model."""
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # Determine format and ensure correct extension
        # Use the recommended .keras format if TF >= 2.7, otherwise use HDF5
        save_format = '.keras' # Default to modern format
        if hasattr(tf, '__version__') and tf.__version__ < '2.7':
            save_format = '.h5'

        # Ensure file path has the correct extension
        if not file_path.endswith(save_format):
            # Remove existing extension if present (e.g., .pkl from generic save path)
            base_path = os.path.splitext(file_path)[0]
            file_path = base_path + save_format

        model.save(file_path) # Saves architecture, weights, optimizer state
        logger.info(f"Keras model saved successfully to: {file_path} (format: {save_format})")

    except Exception as e:
        logger.error(f"Error saving Keras model to {file_path}: {e}", exc_info=True)


# --- Specific Model Runners (Updated Signatures) ---
# These functions now only orchestrate calls to train_nn and forecast_nn

def run_nn_model(
    model_type: str, # 'RNN' or 'LSTM'
    train_data: pd.DataFrame,
    val_data: Optional[pd.DataFrame],
    test_periods: int, # Corresponds to test set size OR forecast horizon
    test_index: pd.DatetimeIndex, # Corresponds to test set index OR future index
    config_params: Dict[str, Any]
) -> Tuple[pd.Series, Optional[Dict[str, Any]], Optional[Model]]:
    """
    Orchestrates NN training and forecasting for either evaluation or final prediction.

    Args:
        model_type: 'RNN' or 'LSTM'.
        train_data: Data to train on (can be train set or full dataset).
        val_data: Validation data (optional, used for evaluation run tuning/early stopping).
        test_periods: Number of periods to forecast (test set size or future horizon).
        test_index: The index for the forecast series (test set index or future index).
        config_params: Configuration dictionary.

    Returns:
        Tuple containing:
        - pd.Series: The forecast values indexed by test_index.
        - Optional[Dict[str, Any]]: Dictionary of parameters used, including scaler reference.
        - Optional[Model]: The trained Keras model object.
    """

    model: Optional[Model] = None
    scaler: Optional[MinMaxScaler] = None
    # Initialize params dict, ensure scaler_object_ref is handled if train_nn returns it early
    model_params: Optional[Dict[str, Any]] = {'model_type': model_type, 'status': 'Not Run', 'scaler_object_ref': None}
    forecast_values: np.ndarray = np.full(test_periods, np.nan)

    try:
        # Train the model. train_nn handles scaling, sequence creation, tuning (optional), and fitting.
        # It returns the fitted model, the scaler object, and the parameters used (including scaler ref).
        model, scaler, model_params = train_nn(model_type, train_data, val_data, config_params)

        # Proceed to forecast only if training was successful
        if model and scaler and model_params.get('status') == 'Trained Successfully':
            # Generate forecast using the trained model, scaler, and the *original* train_data
            # for initialization. forecast_nn handles the iterative prediction.
            # Pass the *entire* train_data used for this run (could be train split or full data)
            forecast_values = forecast_nn(
                model=model,
                scaler=scaler,
                train_data=train_data, # Use the data the model was just trained on for init
                test_periods=test_periods, # Predict this many steps ahead
                n_steps=config_params['NN_STEPS'],
                model_type=model_type
            )
        else:
            status_msg = model_params.get('status', 'Unknown Failure') if model_params else 'Initialization Failed'
            logger.warning(f"{model_type} training did not complete successfully ({status_msg}), skipping forecast.")
            # model_params should already contain the status/error from train_nn

    except Exception as e:
        logger.error(f"{model_type} Error during overall run_nn_model execution: {e}", exc_info=True)
        if model_params: # Add error to params if possible
             model_params['status'] = 'Error - Overall Run Failed'
             model_params['run_error'] = str(e)
        # forecast_values remains array of NaNs

    # Return forecast Series, the parameters dictionary (which includes scaler ref), and the fitted model
    # Ensure the returned Series uses the correct index (test_index or future_index)
    forecast_series = pd.Series(forecast_values, index=test_index, name=model_type)

    return forecast_series, model_params, model