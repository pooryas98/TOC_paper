# --- START OF FILE src/models/nn_models.py ---

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model # Added Model
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, Input, Dropout # Added Input, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam, RMSprop # Import specific optimizers
from sklearn.preprocessing import MinMaxScaler
import warnings
from typing import Tuple, Optional, List, Union, Any # Added type hints
from .. import config # Import config from parent directory's config.py
import kerastuner as kt # Import KerasTuner alias

# --- Helper: Create Sequences (Manual) ---
def create_sequences_manual(data: Optional[np.ndarray], n_steps: int) -> Tuple[np.ndarray, np.ndarray]:
    """Creates sequences manually from scaled data, handling edge cases."""
    # Expects data as a numpy array (e.g., from scaler)
    if data is None or len(data) <= n_steps:
        # Needs > n_steps to create at least one sequence (input + target)
        return np.array([]), np.array([]) # Return empty arrays if not enough data

    X, y = [], []
    # Loop stops such that the last index `end_ix` is `len(data) - 1`
    # This means the target `y` goes up to index `len(data) - 1`
    for i in range(len(data) - n_steps):
        end_ix = i + n_steps
        seq_x, seq_y = data[i:end_ix], data[end_ix] # data[end_ix] is the target
        X.append(seq_x)
        y.append(seq_y)

    if not X: # If loop didn't run
         return np.array([]), np.array([])

    # Ensure outputs are numpy arrays
    return np.array(X), np.array(y).flatten() # Flatten y to ensure it's 1D


# --- Hypermodel Builder Function ---
def build_hypermodel(hp: kt.HyperParameters, model_type: str, n_steps: int, n_features: int = 1) -> Model:
    """
    Builds a Keras RNN or LSTM model with hyperparameters defined by KerasTuner.

    Args:
        hp: KerasTuner HyperParameters object.
        model_type (str): 'RNN' or 'LSTM'.
        n_steps (int): Number of time steps in input sequences.
        n_features (int): Number of features per time step (usually 1).

    Returns:
        A compiled Keras model.
    """
    model = Sequential()
    model.add(Input(shape=(n_steps, n_features))) # Use Input layer for clarity

    # --- Tune Recurrent Layer ---
    # Tune the number of units
    hp_units: int = hp.Int('units', min_value=32, max_value=128, step=32) # Simplified steps
    # Tune activation for recurrent layer
    hp_activation: str = hp.Choice('activation', ['relu', 'tanh'])

    if model_type == 'RNN':
        # Consider return_sequences=True if stacking RNN/LSTM layers
        model.add(SimpleRNN(units=hp_units, activation=hp_activation))
    elif model_type == 'LSTM':
        # Consider return_sequences=True if stacking RNN/LSTM layers
        model.add(LSTM(units=hp_units, activation=hp_activation))
    else:
        raise ValueError("Invalid model_type. Choose 'RNN' or 'LSTM'.")

    # --- Tune Optional Dropout ---
    # Tune whether to include dropout and its rate
    if hp.Boolean("use_dropout"):
         model.add(Dropout(rate=hp.Float('dropout_rate', min_value=0.1, max_value=0.4, step=0.1))) # Smaller range/step

    # --- Output Layer ---
    model.add(Dense(1))  # Output layer for regression

    # --- Tune Optimizer and Learning Rate ---
    # Tune the learning rate
    hp_learning_rate: float = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log')
    # Tune optimizer choice (Simplified: Adam vs RMSprop)
    hp_optimizer_choice: str = hp.Choice('optimizer', ['adam', 'rmsprop'])

    optimizer: keras.optimizers.Optimizer
    if hp_optimizer_choice == 'adam':
        optimizer = Adam(learning_rate=hp_learning_rate)
    else: # 'rmsprop'
        optimizer = RMSprop(learning_rate=hp_learning_rate)

    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae']) # Add MAE metric
    return model


# --- Main Training Function (Modified for KerasTuner) ---
def train_nn(
    model_type: str,
    train_data: pd.DataFrame,
    val_data: Optional[pd.DataFrame],
    n_steps: int,
    epochs: int,
    verbose: int
) -> Tuple[Optional[Model], Optional[MinMaxScaler]]:
    """
    Trains an RNN or LSTM model, potentially using KerasTuner for hyperparameter optimization.
    Uses config values imported from src.config.

    Returns:
        Tuple containing the trained Keras model (or None on failure) and the scaler object (or None).
    """
    print(f"--- Starting {model_type} Training ---")
    use_tuner: bool = config.USE_KERAS_TUNER

    if train_data is None or train_data.empty or train_data['y'].isnull().all():
        warnings.warn(f"Skipping {model_type} training: Training data is empty or all NaN.")
        return None, None

    # --- 1. Scaling ---
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_train: Optional[np.ndarray] = None
    scaled_val: Optional[np.ndarray] = None
    try:
        # Ensure we only scale non-NaN values if imputation wasn't done
        train_y = train_data[['y']].dropna()
        if train_y.empty:
            warnings.warn(f"Skipping {model_type} training: No non-NaN training data after dropna().")
            return None, None
        scaled_train = scaler.fit_transform(train_y)

        if val_data is not None and not val_data.empty and not val_data['y'].isnull().all():
             val_y = val_data[['y']].dropna()
             if not val_y.empty:
                scaled_val = scaler.transform(val_y) # Use the SAME scaler
                print(f"Validation data shape (scaled): {scaled_val.shape}")
             else:
                warnings.warn("Validation data contains only NaNs after dropna(). Validation disabled.")
        # else: # Cases where val_data is None, empty, or all NaN initially
        #     if use_tuner and config.NN_TUNER_OBJECTIVE.startswith('val_'):
        #          # This should be caught by config.py, but good to double-check contextually
        #          warnings.warn(f"Tuner objective '{config.NN_TUNER_OBJECTIVE}' requires validation data, but none was provided or it's invalid. Tuning might fail or use training loss.")
        #     print("No valid validation data provided for scaling.")

    except Exception as e: # Catch potential errors during scaling
        warnings.warn(f"Error during scaling for {model_type}: {e}. Skipping training.")
        return None, None

    # --- 2. Create Sequences ---
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: Optional[np.ndarray] = None
    y_val: Optional[np.ndarray] = None
    validation_data_for_fit: Optional[Tuple[np.ndarray, np.ndarray]] = None # Tuple (X_val, y_val) for model.fit/tuner.search

    X_train, y_train = create_sequences_manual(scaled_train, n_steps)

    if scaled_val is not None:
        X_val_temp, y_val_temp = create_sequences_manual(scaled_val, n_steps)
        if X_val_temp.size > 0 and y_val_temp.size > 0: # Check if sequences were created
            X_val, y_val = X_val_temp, y_val_temp
            validation_data_for_fit = (X_val, y_val)
            print(f"Created validation sequences: X_val {X_val.shape}, y_val {y_val.shape}")
        else:
            warnings.warn(f"Could not create validation sequences (Val data length {len(scaled_val)} <= n_steps {n_steps}?) Validation disabled for fit/search.")

    if X_train.size == 0:
        warnings.warn(f"Skipping {model_type} training: Not enough training data ({len(scaled_train)}) to create sequences with n_steps={n_steps}.")
        return None, None

    # Reshape input to be [samples, time steps, features]
    n_features: int = 1 # Univariate
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], n_features))
    if validation_data_for_fit and X_val is not None:
        X_val = X_val.reshape((X_val.shape[0], X_val.shape[1], n_features))
        validation_data_for_fit = (X_val, validation_data_for_fit[1]) # Update tuple with reshaped X_val


    # --- 3. Early Stopping Callback ---
    early_stopping_callback: Optional[EarlyStopping] = None
    monitor_metric: str = 'loss' # Default if no validation data
    if validation_data_for_fit and config.NN_EARLY_STOPPING_PATIENCE > 0:
        # Determine monitor based on whether tuning is active
        monitor_metric = config.NN_TUNER_OBJECTIVE if use_tuner else 'val_loss'
        # Safety check: if tuner objective is not val_something, default monitor to val_loss for early stopping
        if not monitor_metric.startswith('val_'):
             warnings.warn(f"Early stopping monitor '{monitor_metric}' (from Tuner objective) doesn't start with 'val_'. Monitoring 'val_loss' instead during training.")
             monitor_metric = 'val_loss' # Fallback safely for the callback

        early_stopping_callback = EarlyStopping(
            monitor=monitor_metric,
            patience=config.NN_EARLY_STOPPING_PATIENCE,
            verbose=1,
            mode='min', # Assume lower is better for loss/MAE
            restore_best_weights=True
        )
        print(f"Early stopping enabled, monitoring '{monitor_metric}' with patience {config.NN_EARLY_STOPPING_PATIENCE}.")
    elif config.NN_EARLY_STOPPING_PATIENCE > 0:
        # Still allow monitoring training loss if requested and no validation data
        monitor_metric = 'loss'
        early_stopping_callback = EarlyStopping(
            monitor=monitor_metric,
            patience=config.NN_EARLY_STOPPING_PATIENCE,
            verbose=1,
            mode='min',
            restore_best_weights=True
            )
        warnings.warn(f"Early stopping configured (patience={config.NN_EARLY_STOPPING_PATIENCE}), but no valid validation data. Monitoring training '{monitor_metric}'.")


    # --- 4. Model Definition and Training ---
    tf.random.set_seed(config.RANDOM_SEED) # Set seed for reproducibility before model build/train
    best_model: Optional[Model] = None # This will hold the final trained model

    if use_tuner:
        print(f"--- Initiating KerasTuner Search for {model_type} ---")
        # Check again if objective needs validation data that isn't available
        tuner_objective_needs_val: bool = config.NN_TUNER_OBJECTIVE.startswith('val_')
        if tuner_objective_needs_val and not validation_data_for_fit:
             # This should have been caught by config.py, but is a critical failure point
             print(f"FATAL ERROR: Tuner objective '{config.NN_TUNER_OBJECTIVE}' requires validation data, but none could be prepared. Aborting {model_type} run.")
             return None, None # Stop this model's run

        # Define the hypermodel builder function with necessary arguments fixed
        # Need to use a lambda or functools.partial to pass fixed args
        model_builder = lambda hp: build_hypermodel(hp, model_type=model_type, n_steps=n_steps, n_features=n_features)

        # --- Choose and Configure the Tuner ---
        # Using RandomSearch here. Could switch to kt.Hyperband or kt.BayesianOptimization
        tuner = kt.RandomSearch(
            model_builder,
            objective=kt.Objective(config.NN_TUNER_OBJECTIVE, direction="min"), # Explicit objective direction
            max_trials=config.NN_TUNER_MAX_TRIALS,
            executions_per_trial=config.NN_TUNER_EXECUTIONS_PER_TRIAL,
            directory='keras_tuner_dir', # Directory to store results
            project_name=f'{model_type}_tuning_{pd.Timestamp.now().strftime("%Y%m%d_%H%M")}', # Simpler timestamp
            seed=config.RANDOM_SEED, # Seed the tuner's sampling
            overwrite=True # Overwrite previous tuning results in the project dir
        )

        print("--- Tuner Search Space Summary ---")
        tuner.search_space_summary()
        print(f"--- Starting Tuner Search (Max Trials: {config.NN_TUNER_MAX_TRIALS}, Epochs/Trial: {config.NN_TUNER_EPOCHS}) ---")

        # Prepare callbacks for the search
        search_callbacks: List[keras.callbacks.Callback] = []
        if early_stopping_callback:
            # Need to monitor the *actual* objective the tuner is using for early stopping during search
            search_early_stopping = EarlyStopping(
                monitor=config.NN_TUNER_OBJECTIVE, # Use the objective directly
                patience=config.NN_EARLY_STOPPING_PATIENCE,
                verbose=1,
                mode='min',
                restore_best_weights=False # Tuner handles best HPs, not weights during search
            )
            search_callbacks.append(search_early_stopping)
            print(f"Early stopping during search monitors '{config.NN_TUNER_OBJECTIVE}'")

        # Run the search
        try:
             tuner.search(X_train, y_train,
                          epochs=config.NN_TUNER_EPOCHS, # Epochs for *each trial*
                          validation_data=validation_data_for_fit, # Crucial for tuner objective
                          callbacks=search_callbacks,
                          verbose=verbose)
        except Exception as e:
             print(f"KerasTuner search failed for {model_type}: {e}")
             return None, None # Abort if search fails

        print("\n--- KerasTuner Search Complete ---")
        try:
            # Show top N results (e.g., top 3)
            print("\n--- KerasTuner Top Trial(s) Summary ---")
            tuner.results_summary(num_trials=min(3, config.NN_TUNER_MAX_TRIALS))

            # Get the optimal hyperparameters
            best_hps_list = tuner.get_best_hyperparameters(num_trials=1)
            if not best_hps_list:
                print(f"Error: KerasTuner could not find any best hyperparameters for {model_type}.")
                return None, None
            best_hps = best_hps_list[0]

            print(f"\n--- Best Hyperparameters Found for {model_type} ---")
            for param, value in best_hps.values.items():
                print(f"- {param}: {value}")

            # Build the best model using the optimal hyperparameters
            print(f"\n--- Building Best {model_type} Model from Tuner ---")
            # Need to rebuild the model with the best HPs
            best_model = tuner.hypermodel.build(best_hps)
            best_model.summary() # Print summary of the chosen model

        except Exception as e:
             print(f"Error retrieving results or building best model from tuner: {e}")
             return None, None # Abort if results processing fails

    else:
        # --- Manual Model Building (Original Logic Path) ---
        print(f"--- Building {model_type} Manually (Tuner Disabled) ---")
        manual_model = Sequential(name=f"Manual_{model_type}")
        manual_model.add(Input(shape=(n_steps, n_features)))

        activation = config.NN_ACTIVATION
        units = config.NN_UNITS
        optimizer_name = config.NN_OPTIMIZER

        if model_type == 'RNN':
            manual_model.add(SimpleRNN(units=units, activation=activation))
        elif model_type == 'LSTM':
            manual_model.add(LSTM(units=units, activation=activation))

        manual_model.add(Dense(1))

        # Compile with configured optimizer (gets default settings if just name)
        try:
            # Use lowercase for optimizer name consistency with Keras
            optimizer_instance = tf.keras.optimizers.get(optimizer_name.lower())
        except ValueError:
            warnings.warn(f"Unknown optimizer '{optimizer_name}'. Using 'adam'.")
            optimizer_instance = Adam() # Default safe choice

        manual_model.compile(optimizer=optimizer_instance, loss='mean_squared_error', metrics=['mae'])
        print(f"Manual {model_type} Model Summary:")
        manual_model.summary()
        best_model = manual_model # Use this model for final training


    # --- Final Training (Either Tuned Best Model or Manual Model) ---
    if best_model is not None:
        print(f"\n--- Final Training Phase for {model_type} (Epochs: {epochs}) ---")
        final_callbacks: List[keras.callbacks.Callback] = []
        final_early_stopping = None # Separate instance for final training

        # Re-create early stopping for the *final fit*, monitoring val_loss or loss
        if validation_data_for_fit and config.NN_EARLY_STOPPING_PATIENCE > 0:
            final_monitor = 'val_loss' # Monitor validation loss during final fit
            final_early_stopping = EarlyStopping(
                monitor=final_monitor,
                patience=config.NN_EARLY_STOPPING_PATIENCE,
                verbose=1, mode='min', restore_best_weights=True
            )
            final_callbacks.append(final_early_stopping)
            print(f"Final training early stopping monitors '{final_monitor}'")
        elif config.NN_EARLY_STOPPING_PATIENCE > 0:
             final_monitor = 'loss' # Monitor training loss if no validation
             final_early_stopping = EarlyStopping(
                 monitor=final_monitor,
                 patience=config.NN_EARLY_STOPPING_PATIENCE,
                 verbose=1, mode='min', restore_best_weights=True
             )
             final_callbacks.append(final_early_stopping)
             print(f"Final training early stopping monitors '{final_monitor}' (no validation data)")


        try:
            history: keras.callbacks.History = best_model.fit(
                X_train, y_train,
                epochs=epochs, # Use the full epochs specified for RNN/LSTM
                validation_data=validation_data_for_fit,
                callbacks=final_callbacks,
                verbose=verbose
            )
            print(f"--- {model_type} Final Training Complete ---")

            if final_early_stopping and final_early_stopping.stopped_epoch > 0:
                 actual_epochs_run = final_early_stopping.stopped_epoch + 1
                 monitor_val = final_early_stopping.best
                 print(f"Final training stopped early after {actual_epochs_run} epochs.")
                 print(f"Best monitored value ({final_early_stopping.monitor}): {monitor_val:.4f}")
            else:
                 final_epoch_count = len(history.epoch)
                 print(f"Final training ran for all {final_epoch_count} epochs.")


            return best_model, scaler # Return the final trained model and scaler

        except Exception as e:
             print(f"Error during {model_type} final model.fit: {e}")
             return None, None # Return None if final fit fails
    else:
         print(f"Error: No model (tuned or manual) was available for final training for {model_type}.")
         return None, None # Should not happen if logic is correct, but safety check


# --- Prediction Function (Updated for Robustness) ---
def forecast_nn(
    model: Optional[Model],
    scaler: Optional[MinMaxScaler],
    train_data: pd.DataFrame, # Original unscaled training data
    test_periods: int,
    n_steps: int
) -> np.ndarray:
    """
    Generates forecasts using a trained NN model.

    Returns:
        A numpy array containing the forecasts for the test periods.
        Returns array of NaNs if forecasting fails.
    """
    print(f"Forecasting with NN (Periods: {test_periods})...")
    n_features: int = 1
    # Initialize forecast array with NaNs
    forecasts_final = np.full(test_periods, np.nan)

    if model is None or scaler is None:
        print("NN Forecasting Error: Model or scaler not available (likely due to training error).")
        return forecasts_final # Return array of NaNs

    # Use the *entire original training data* for getting the last sequence
    # Need to handle potential NaNs in the tail used for initialization
    train_y_for_init = train_data[['y']].dropna()
    if len(train_y_for_init) < n_steps:
         print(f"NN Forecasting Error: Not enough non-NaN training data ({len(train_y_for_init)}) for forecast initialization (need {n_steps}).")
         return forecasts_final # Return array of NaNs

    try:
        # Scale the tail end of the non-NaN training data
        scaled_train_tail = scaler.transform(train_y_for_init.iloc[-n_steps:])
    except Exception as e:
        print(f"NN Forecasting Error: Scaling train_data tail failed: {e}")
        return forecasts_final # Return array of NaNs

    forecasts_scaled: List[float] = []
    # Initialize input for the first prediction step
    # Ensure it's a list of lists/arrays for the loop logic
    current_input_list: List[List[float]] = scaled_train_tail.tolist()

    for i in range(test_periods):
        try:
            # Reshape current input for prediction
            current_batch: np.ndarray = np.array(current_input_list).reshape((1, n_steps, n_features))

            # Predict
            # verbose=0 suppresses prediction progress bars
            current_pred_scaled: np.ndarray = model.predict(current_batch, verbose=0)
            pred_value: float = current_pred_scaled[0, 0] # Extract scalar prediction

            # Store prediction (scaled)
            forecasts_scaled.append(pred_value)

            # Update input list for next step: remove oldest, append newest prediction
            current_input_list.pop(0)
            current_input_list.append([pred_value]) # Append as a list containing the float

        except Exception as e:
             print(f"Error during NN forecast step {i+1}/{test_periods}: {e}")
             # Keep already generated forecasts, remaining will be NaN
             break # Exit forecast loop

    # Inverse scale the successfully generated forecasts
    if forecasts_scaled: # Check if any forecasts were generated
        forecasts_scaled_arr = np.array(forecasts_scaled).reshape(-1, 1)
        try:
            # Inverse transform the scaled forecasts
            forecasts_unscaled = scaler.inverse_transform(forecasts_scaled_arr).flatten()
            # Place the unscaled forecasts into the final array
            num_forecasts = len(forecasts_unscaled)
            forecasts_final[:num_forecasts] = forecasts_unscaled
        except Exception as e:
            print(f"Error during inverse scaling: {e}. Some forecasts may remain NaN.")
            # `forecasts_final` will retain NaNs where inverse transform failed or loop broke early

    print("NN Forecasting Finished.")
    return forecasts_final # Return 1D array, possibly containing NaNs


# --- Specific Model Runners ---
# Updated signatures to accept val_data, use config directly
def run_rnn(
    train_data: pd.DataFrame,
    val_data: Optional[pd.DataFrame],
    test_periods: int,
    test_index: pd.DatetimeIndex
) -> pd.Series:
    """Orchestrates RNN training and forecasting using config."""
    model_type: str = 'RNN'
    n_steps: int = config.NN_STEPS
    epochs: int = config.RNN_EPOCHS # Epochs for final training
    verbose: int = config.NN_VERBOSE

    model: Optional[Model] = None
    scaler: Optional[MinMaxScaler] = None
    forecast_values: np.ndarray = np.full(test_periods, np.nan) # Initialize with NaNs

    try:
        # train_nn now handles both tuning and final training
        model, scaler = train_nn(model_type, train_data, val_data, n_steps, epochs, verbose)
        if model and scaler:
            forecast_values = forecast_nn(model, scaler, train_data, test_periods, n_steps)
        else:
            print(f"{model_type} training failed, skipping forecast.")
            # forecast_values remains array of NaNs

    except Exception as e:
        print(f"{model_type} Error during overall run: {e}")
        # forecast_values remains array of NaNs

    return pd.Series(forecast_values, index=test_index, name=model_type)


def run_lstm(
    train_data: pd.DataFrame,
    val_data: Optional[pd.DataFrame],
    test_periods: int,
    test_index: pd.DatetimeIndex
) -> pd.Series:
    """Orchestrates LSTM training and forecasting using config."""
    model_type: str = 'LSTM'
    n_steps: int = config.NN_STEPS
    epochs: int = config.LSTM_EPOCHS # Epochs for final training
    verbose: int = config.NN_VERBOSE

    model: Optional[Model] = None
    scaler: Optional[MinMaxScaler] = None
    forecast_values: np.ndarray = np.full(test_periods, np.nan) # Initialize with NaNs

    try:
        # train_nn now handles both tuning and final training
        model, scaler = train_nn(model_type, train_data, val_data, n_steps, epochs, verbose)
        if model and scaler:
             forecast_values = forecast_nn(model, scaler, train_data, test_periods, n_steps)
        else:
             print(f"{model_type} training failed, skipping forecast.")
             # forecast_values remains array of NaNs

    except Exception as e:
        print(f"{model_type} Error during overall run: {e}")
        # forecast_values remains array of NaNs

    return pd.Series(forecast_values, index=test_index, name=model_type)

# --- END OF FILE src/models/nn_models.py ---