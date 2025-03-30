# --- START OF FILE config.py ---

import os
from dotenv import load_dotenv
import warnings
from typing import Tuple, Optional # Added for type hinting

# Load environment variables from .env file located in the parent directory
# Assuming config.py is in src/, .env is one level up
dotenv_path: str = os.path.join(os.path.dirname(__file__), '..', '.env')

if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path)
    print("Loaded configuration from .env file.")
else:
    warnings.warn(".env file not found. Using default settings or environment variables.")

# --- Helper for boolean conversion ---
def str_to_bool(val: Optional[str]) -> bool:
    """Converts a string representation of truth to True."""
    # Ensure val is a string before calling lower()
    if isinstance(val, str):
        return val.lower() in ('true', '1', 't', 'y', 'yes')
    return bool(val) # Handle potential non-string inputs

# --- Data Settings ---
CSV_FILE_PATH: str = os.getenv('CSV_FILE_PATH', 'your_data.csv')
DATE_COLUMN: str = os.getenv('DATE_COLUMN', 'Date')
VALUE_COLUMN: str = os.getenv('VALUE_COLUMN', 'Value')
TIME_SERIES_FREQUENCY: Optional[str] = os.getenv('TIME_SERIES_FREQUENCY') # Keep as string or None

# --- Train/Validation/Test Split ---
try:
    TEST_SIZE: int = int(os.getenv('TEST_SIZE', '24'))
    VALIDATION_SIZE: int = int(os.getenv('VALIDATION_SIZE', '12')) # Default if not set
    if TEST_SIZE <= 0:
        warnings.warn("TEST_SIZE should be positive. Setting to 1.")
        TEST_SIZE = 1
    if VALIDATION_SIZE < 0:
        warnings.warn("VALIDATION_SIZE cannot be negative. Setting to 0 (disabled).")
        VALIDATION_SIZE = 0

except ValueError:
    warnings.warn("Invalid TEST_SIZE or VALIDATION_SIZE in .env. Using defaults (Test=24, Val=12).")
    TEST_SIZE = 24
    VALIDATION_SIZE = 12

# --- SARIMA Parameters ---
USE_AUTO_ARIMA: bool = str_to_bool(os.getenv('USE_AUTO_ARIMA', 'True')) # Changed default to True
SARIMA_AUTO_SEASONAL: bool = str_to_bool(os.getenv('SARIMA_AUTO_SEASONAL', 'True'))

try:
    SARIMA_ORDER: Tuple[int, int, int] = (
        int(os.getenv('SARIMA_P', '1')),
        int(os.getenv('SARIMA_D', '1')),
        int(os.getenv('SARIMA_Q', '1'))
    )
except ValueError:
     warnings.warn("Invalid SARIMA_P/D/Q in .env. Using defaults (1,1,1).")
     SARIMA_ORDER = (1, 1, 1)

try:
    manual_sarima_s: Optional[str] = os.getenv('SARIMA_S')
    SARIMA_SEASONAL_ORDER_NOS: Tuple[int, int, int] = (
        int(os.getenv('SARIMA_SP', '1')),
        int(os.getenv('SARIMA_SD', '1')),
        int(os.getenv('SARIMA_SQ', '0'))
    )
    # Handle blank SARIMA_S: None means infer, digit means use it
    SARIMA_MANUAL_S: Optional[int] = int(manual_sarima_s) if manual_sarima_s and manual_sarima_s.isdigit() else None
except ValueError:
    warnings.warn("Invalid SARIMA_SP/SD/SQ/S in .env. Using defaults (1,1,0) and inferring S.")
    SARIMA_SEASONAL_ORDER_NOS = (1, 1, 0)
    SARIMA_MANUAL_S = None


# --- Neural Network Parameters ---
try:
    NN_STEPS: int = int(os.getenv('NN_STEPS', '12'))
    NN_UNITS: int = int(os.getenv('NN_UNITS', '50'))
    NN_ACTIVATION: str = os.getenv('NN_ACTIVATION', 'relu')
    NN_OPTIMIZER: str = os.getenv('NN_OPTIMIZER', 'adam')
    RNN_EPOCHS: int = int(os.getenv('RNN_EPOCHS', '150'))
    LSTM_EPOCHS: int = int(os.getenv('LSTM_EPOCHS', '150'))
    NN_EARLY_STOPPING_PATIENCE: int = int(os.getenv('NN_EARLY_STOPPING_PATIENCE', '10'))
    NN_VERBOSE: int = int(os.getenv('NN_VERBOSE', '0'))
    if NN_STEPS <= 0:
        warnings.warn("NN_STEPS must be positive. Setting to 1.")
        NN_STEPS = 1
    if NN_UNITS <= 0: # Default units check
        warnings.warn("NN_UNITS must be positive. Setting to 50.")
        NN_UNITS = 50
    # Add checks for activation/optimizer validity if desired
except ValueError:
    warnings.warn("Invalid NN parameters (STEPS/UNITS/EPOCHS/PATIENCE/VERBOSE) in .env. Using defaults.")
    NN_STEPS = 12
    NN_UNITS = 50
    NN_ACTIVATION = 'relu'
    NN_OPTIMIZER = 'adam'
    RNN_EPOCHS = 150
    LSTM_EPOCHS = 150
    NN_EARLY_STOPPING_PATIENCE = 10
    NN_VERBOSE = 0

# --- KerasTuner Settings ---
USE_KERAS_TUNER: bool = str_to_bool(os.getenv('USE_KERAS_TUNER', 'False'))
try:
    NN_TUNER_MAX_TRIALS: int = int(os.getenv('NN_TUNER_MAX_TRIALS', '10'))
    NN_TUNER_EXECUTIONS_PER_TRIAL: int = int(os.getenv('NN_TUNER_EXECUTIONS_PER_TRIAL', '1'))
    NN_TUNER_EPOCHS: int = int(os.getenv('NN_TUNER_EPOCHS', '50'))
except ValueError:
    warnings.warn("Invalid KerasTuner parameters (TRIALS/EXECUTIONS/EPOCHS) in .env. Using defaults.")
    NN_TUNER_MAX_TRIALS = 10
    NN_TUNER_EXECUTIONS_PER_TRIAL = 1
    NN_TUNER_EPOCHS = 50
NN_TUNER_OBJECTIVE: str = os.getenv('NN_TUNER_OBJECTIVE', 'val_loss')


# --- Validation Check for Tuner ---
if USE_KERAS_TUNER:
    tuner_objective_needs_val: bool = NN_TUNER_OBJECTIVE is not None and NN_TUNER_OBJECTIVE.startswith('val_')
    if tuner_objective_needs_val and VALIDATION_SIZE <= NN_STEPS:
        # Fatal error because tuning cannot proceed as configured
        raise ValueError(f"KerasTuner is enabled (USE_KERAS_TUNER=True) with objective '{NN_TUNER_OBJECTIVE}', "
                         f"which requires validation data. However, VALIDATION_SIZE ({VALIDATION_SIZE}) "
                         f"is not strictly greater than NN_STEPS ({NN_STEPS}). "
                         "Please increase VALIDATION_SIZE in .env or change NN_TUNER_OBJECTIVE (e.g., to 'loss').")
    elif tuner_objective_needs_val and VALIDATION_SIZE == 0:
         raise ValueError(f"KerasTuner is enabled (USE_KERAS_TUNER=True) with objective '{NN_TUNER_OBJECTIVE}', "
                         f"which requires validation data, but VALIDATION_SIZE is 0. "
                          "Please set VALIDATION_SIZE > NN_STEPS in .env.")
    elif not tuner_objective_needs_val:
        warnings.warn(f"KerasTuner objective is '{NN_TUNER_OBJECTIVE}'. Tuning will proceed using training metrics only.")


# --- Other Settings ---
try:
    RANDOM_SEED: int = int(os.getenv('RANDOM_SEED', '42'))
except ValueError:
    warnings.warn("Invalid RANDOM_SEED in .env. Using default 42.")
    RANDOM_SEED = 42

SAVE_RESULTS: bool = str_to_bool(os.getenv('SAVE_RESULTS', 'True')) # Default to True for saving results
RESULTS_DIR: str = os.getenv('RESULTS_DIR', 'results') # Directory to save results

# --- Validate essential paths/values ---
if not CSV_FILE_PATH or CSV_FILE_PATH == 'your_data.csv':
     warnings.warn("CSV_FILE_PATH is not set or using default 'your_data.csv'. Please configure in .env.")


# --- Configuration Summary Print ---
print("-" * 30)
print("Configuration:")
print(f"  Data Path: {CSV_FILE_PATH}")
print(f"  Date Column: {DATE_COLUMN}")
print(f"  Value Column: {VALUE_COLUMN}")
print(f"  Frequency: {TIME_SERIES_FREQUENCY or 'Infer'}")
print(f"  Test Size: {TEST_SIZE}")
print(f"  Validation Size: {VALIDATION_SIZE if VALIDATION_SIZE > 0 else 'Disabled'}") # Improved print
print(f"  Use Auto ARIMA: {USE_AUTO_ARIMA}")
if USE_AUTO_ARIMA:
    print(f"    Auto Seasonal: {SARIMA_AUTO_SEASONAL}")
else:
    print(f"  SARIMA Order: {SARIMA_ORDER}")
    print(f"  SARIMA Seasonal (p,d,q): {SARIMA_SEASONAL_ORDER_NOS}")
    print(f"  SARIMA Manual S: {SARIMA_MANUAL_S or 'Infer'}")
print(f"  NN Steps: {NN_STEPS}")
# Only print default NN params if Tuner is OFF
if not USE_KERAS_TUNER:
    print(f"  NN Units: {NN_UNITS}")
    print(f"  NN Activation: {NN_ACTIVATION}")
    print(f"  NN Optimizer: {NN_OPTIMIZER}")
print(f"  RNN Epochs: {RNN_EPOCHS}")
print(f"  LSTM Epochs: {LSTM_EPOCHS}")
print(f"  NN Early Stopping Patience: {NN_EARLY_STOPPING_PATIENCE if NN_EARLY_STOPPING_PATIENCE > 0 else 'Disabled'}")
print(f"  NN Verbose: {NN_VERBOSE}")
# Print Tuner settings if enabled
print(f"  Use KerasTuner: {USE_KERAS_TUNER}")
if USE_KERAS_TUNER:
    print(f"    Tuner Max Trials: {NN_TUNER_MAX_TRIALS}")
    print(f"    Tuner Executions/Trial: {NN_TUNER_EXECUTIONS_PER_TRIAL}")
    print(f"    Tuner Epochs/Trial: {NN_TUNER_EPOCHS}")
    print(f"    Tuner Objective: {NN_TUNER_OBJECTIVE}")
print(f"  Random Seed: {RANDOM_SEED}")
print(f"  Save Results: {SAVE_RESULTS}") # Added
if SAVE_RESULTS:
    print(f"    Results Directory: {RESULTS_DIR}") # Added
print("-" * 30)

# --- END OF FILE config.py ---