import os
import logging
import warnings
from dotenv import load_dotenv
from typing import Tuple, Optional, List, Dict, Any
import datetime # For log file timestamping

# --- Helper Functions ---
def str_to_bool(val: Optional[str]) -> bool:
    """Converts various string representations of truth to True."""
    if isinstance(val, str):
        val = val.lower()
        return val in ('true', '1', 't', 'y', 'yes')
    return bool(val)

def parse_comma_sep_str(val: Optional[str], lower: bool = False) -> List[str]:
    """Parses a comma-separated string into a list of stripped strings."""
    if not val:
        return []
    items = [item.strip() for item in val.split(',') if item.strip()]
    if lower:
        items = [item.lower() for item in items]
    return items

def get_env_var(key: str, default: Any = None, required: bool = False, var_type: type = str) -> Any:
    """Gets env var, applies type conversion, checks requirement, returns default if missing/invalid."""
    value = os.getenv(key)
    if value is None:
        if required:
            raise ValueError(f"Missing required environment variable: {key}")
        return default

    original_value = value # Keep original for error messages
    try:
        if var_type == bool:
            return str_to_bool(value)
        elif var_type == int:
            return int(value)
        elif var_type == float:
            return float(value)
        elif var_type == list: # Assumes comma-separated string for list type
            return parse_comma_sep_str(value)
        elif var_type == dict: # Not typically used directly for env vars, maybe json string
             raise NotImplementedError("Direct dict parsing from env var not implemented. Use JSON string.")
        # Add more types if needed
        return var_type(value) # General case (string)
    except (ValueError, TypeError) as e:
        logging.warning(f"Invalid type for env var '{key}'. Value '{original_value}' cannot be converted to {var_type.__name__}. Using default: {default}. Error: {e}")
        return default


# --- Load .env File ---
dotenv_path: str = os.path.join(os.path.dirname(__file__), '..', '.env')
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path)
    print(f"Loaded configuration from .env file: {dotenv_path}")
else:
    print("Warning: .env file not found. Using environment variables or defaults.")

# --- Setup Logging ---
_log_level_str = get_env_var('LOG_LEVEL', 'INFO').upper()
_log_level = getattr(logging, _log_level_str, logging.INFO)
_log_file = get_env_var('LOG_FILE', None)

log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(module)s - %(message)s')
logger = logging.getLogger()
logger.setLevel(_log_level)

# Console Handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)
logger.addHandler(console_handler)

# File Handler (Optional)
if _log_file:
    # Simple timestamp replacement (replace with more robust logic if needed)
    if "${TIMESTAMP}" in _log_file:
         timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
         _log_file = _log_file.replace("${TIMESTAMP}", timestamp)
    # Ensure log directory exists
    log_dir = os.path.dirname(_log_file)
    if log_dir and not os.path.exists(log_dir):
        try:
            os.makedirs(log_dir)
            print(f"Created log directory: {log_dir}")
        except OSError as e:
            print(f"Warning: Could not create log directory '{log_dir}'. Log file might fail. Error: {e}")

    try:
        file_handler = logging.FileHandler(_log_file, encoding='utf-8')
        file_handler.setFormatter(log_formatter)
        logger.addHandler(file_handler)
        logging.info(f"Logging to file: {_log_file}")
    except Exception as e:
         logging.error(f"Failed to set up log file handler for '{_log_file}'. Error: {e}")
else:
    logging.info("Logging only to console (LOG_FILE not set).")


# --- Configuration Values ---
logging.info("Loading configuration settings...")

# Data Settings
CSV_FILE_PATH: str = get_env_var('CSV_FILE_PATH', required=True)
DATE_COLUMN: str = get_env_var('DATE_COLUMN', required=True)
VALUE_COLUMN: str = get_env_var('VALUE_COLUMN', required=True)
TIME_SERIES_FREQUENCY: Optional[str] = get_env_var('TIME_SERIES_FREQUENCY', None) # Keep as string or None
_imputation_method = get_env_var('DATA_IMPUTATION_METHOD', 'none').lower()
if _imputation_method not in ['none', 'ffill', 'bfill', 'mean', 'median', 'interpolate']:
    logging.warning(f"Invalid DATA_IMPUTATION_METHOD '{_imputation_method}'. Using 'none'.")
    DATA_IMPUTATION_METHOD: str = 'none'
else:
    DATA_IMPUTATION_METHOD: str = _imputation_method

# Train/Validation/Test Split
TEST_SIZE: int = get_env_var('TEST_SIZE', 24, var_type=int)
VALIDATION_SIZE: int = get_env_var('VALIDATION_SIZE', 12, var_type=int)
if TEST_SIZE <= 0:
    logging.warning("TEST_SIZE must be positive. Setting to 1.")
    TEST_SIZE = 1
if VALIDATION_SIZE < 0:
    logging.warning("VALIDATION_SIZE cannot be negative. Setting to 0 (disabled).")
    VALIDATION_SIZE = 0

# General Run Settings
RANDOM_SEED: int = get_env_var('RANDOM_SEED', 42, var_type=int)
_models_to_run_str = get_env_var('MODELS_TO_RUN', 'SARIMA,Prophet,RNN,LSTM')
MODELS_TO_RUN: List[str] = parse_comma_sep_str(_models_to_run_str)
# Validate model names maybe?
valid_models = {'SARIMA', 'Prophet', 'RNN', 'LSTM'}
MODELS_TO_RUN = [m for m in MODELS_TO_RUN if m in valid_models]
if not MODELS_TO_RUN:
     raise ValueError("No valid models specified in MODELS_TO_RUN. Check .env.")

# SARIMA Parameters
USE_AUTO_ARIMA: bool = get_env_var('USE_AUTO_ARIMA', True, var_type=bool)
SARIMA_AUTO_SEASONAL: bool = get_env_var('SARIMA_AUTO_SEASONAL', True, var_type=bool)
AUTO_ARIMA_START_P: int = get_env_var('AUTO_ARIMA_START_P', 1, var_type=int)
AUTO_ARIMA_MAX_P: int = get_env_var('AUTO_ARIMA_MAX_P', 3, var_type=int)
AUTO_ARIMA_START_Q: int = get_env_var('AUTO_ARIMA_START_Q', 1, var_type=int)
AUTO_ARIMA_MAX_Q: int = get_env_var('AUTO_ARIMA_MAX_Q', 3, var_type=int)
AUTO_ARIMA_MAX_D: int = get_env_var('AUTO_ARIMA_MAX_D', 2, var_type=int)
AUTO_ARIMA_START_SP: int = get_env_var('AUTO_ARIMA_START_SP', 0, var_type=int)
AUTO_ARIMA_MAX_SP: int = get_env_var('AUTO_ARIMA_MAX_SP', 2, var_type=int)
AUTO_ARIMA_START_SQ: int = get_env_var('AUTO_ARIMA_START_SQ', 0, var_type=int)
AUTO_ARIMA_MAX_SQ: int = get_env_var('AUTO_ARIMA_MAX_SQ', 2, var_type=int)
AUTO_ARIMA_MAX_SD: int = get_env_var('AUTO_ARIMA_MAX_SD', 1, var_type=int)
AUTO_ARIMA_TEST: str = get_env_var('AUTO_ARIMA_TEST', 'adf').lower()
AUTO_ARIMA_IC: str = get_env_var('AUTO_ARIMA_IC', 'aic').lower()
AUTO_ARIMA_STEPWISE: bool = get_env_var('AUTO_ARIMA_STEPWISE', True, var_type=bool)

SARIMA_P: int = get_env_var('SARIMA_P', 1, var_type=int)
SARIMA_D: int = get_env_var('SARIMA_D', 1, var_type=int)
SARIMA_Q: int = get_env_var('SARIMA_Q', 1, var_type=int)
SARIMA_SP: int = get_env_var('SARIMA_SP', 1, var_type=int)
SARIMA_SD: int = get_env_var('SARIMA_SD', 1, var_type=int)
SARIMA_SQ: int = get_env_var('SARIMA_SQ', 0, var_type=int)
_manual_s = get_env_var('SARIMA_MANUAL_S', None)
SARIMA_MANUAL_S: Optional[int] = int(_manual_s) if _manual_s and _manual_s.isdigit() else None

SARIMA_ENFORCE_STATIONARITY: bool = get_env_var('SARIMA_ENFORCE_STATIONARITY', False, var_type=bool)
SARIMA_ENFORCE_INVERTIBILITY: bool = get_env_var('SARIMA_ENFORCE_INVERTIBILITY', False, var_type=bool)
SARIMA_ORDER: Tuple[int, int, int] = (SARIMA_P, SARIMA_D, SARIMA_Q)
SARIMA_SEASONAL_ORDER_NOS: Tuple[int, int, int] = (SARIMA_SP, SARIMA_SD, SARIMA_SQ)


# Prophet Parameters
PROPHET_GROWTH: str = get_env_var('PROPHET_GROWTH', 'linear').lower()
_prophet_cap_str = get_env_var('PROPHET_CAP', None)
PROPHET_CAP: Optional[float] = float(_prophet_cap_str) if _prophet_cap_str else None
_prophet_floor_str = get_env_var('PROPHET_FLOOR', None)
PROPHET_FLOOR: Optional[float] = float(_prophet_floor_str) if _prophet_floor_str else None
# For 'auto'/True/False parameters, get_env_var handles bool conversion, need custom for 'auto'
_yas_str = get_env_var('PROPHET_ADD_YEARLY_SEASONALITY', 'auto').lower()
PROPHET_ADD_YEARLY_SEASONALITY: Any = _yas_str if _yas_str == 'auto' else str_to_bool(_yas_str)
_was_str = get_env_var('PROPHET_ADD_WEEKLY_SEASONALITY', 'auto').lower()
PROPHET_ADD_WEEKLY_SEASONALITY: Any = _was_str if _was_str == 'auto' else str_to_bool(_was_str)
_das_str = get_env_var('PROPHET_ADD_DAILY_SEASONALITY', 'auto').lower()
PROPHET_ADD_DAILY_SEASONALITY: Any = _das_str if _das_str == 'auto' else str_to_bool(_das_str)
PROPHET_SEASONALITY_MODE: str = get_env_var('PROPHET_SEASONALITY_MODE', 'additive').lower()
PROPHET_INTERVAL_WIDTH: float = get_env_var('PROPHET_INTERVAL_WIDTH', 0.95, var_type=float)
PROPHET_COUNTRY_HOLIDAYS: Optional[str] = get_env_var('PROPHET_COUNTRY_HOLIDAYS', None)


# Neural Network Parameters
NN_STEPS: int = get_env_var('NN_STEPS', 12, var_type=int)
if NN_STEPS <= 0:
    logging.warning("NN_STEPS must be positive. Setting to 1.")
    NN_STEPS = 1
NN_UNITS: int = get_env_var('NN_UNITS', 50, var_type=int)
if NN_UNITS <= 0:
    logging.warning("NN_UNITS must be positive. Setting to 50.")
    NN_UNITS = 50
NN_ACTIVATION: str = get_env_var('NN_ACTIVATION', 'relu').lower()
NN_OPTIMIZER: str = get_env_var('NN_OPTIMIZER', 'adam').lower()
NN_ADD_DROPOUT: bool = get_env_var('NN_ADD_DROPOUT', False, var_type=bool)
NN_DROPOUT_RATE: float = get_env_var('NN_DROPOUT_RATE', 0.2, var_type=float)
NN_LOSS_FUNCTION: str = get_env_var('NN_LOSS_FUNCTION', 'mean_squared_error').lower()
NN_BATCH_SIZE: int = get_env_var('NN_BATCH_SIZE', 32, var_type=int)
if NN_BATCH_SIZE <= 0:
    logging.warning("NN_BATCH_SIZE must be positive. Setting to 32.")
    NN_BATCH_SIZE = 32
RNN_EPOCHS: int = get_env_var('RNN_EPOCHS', 150, var_type=int)
LSTM_EPOCHS: int = get_env_var('LSTM_EPOCHS', 150, var_type=int)
NN_EARLY_STOPPING_PATIENCE: int = get_env_var('NN_EARLY_STOPPING_PATIENCE', 15, var_type=int)
NN_VERBOSE: int = get_env_var('NN_VERBOSE', 1, var_type=int)


# KerasTuner Settings
USE_KERAS_TUNER: bool = get_env_var('USE_KERAS_TUNER', False, var_type=bool)
NN_TUNER_TYPE: str = get_env_var('NN_TUNER_TYPE', 'RandomSearch')
NN_TUNER_MAX_TRIALS: int = get_env_var('NN_TUNER_MAX_TRIALS', 10, var_type=int)
NN_TUNER_EXECUTIONS_PER_TRIAL: int = get_env_var('NN_TUNER_EXECUTIONS_PER_TRIAL', 1, var_type=int)
NN_TUNER_EPOCHS: int = get_env_var('NN_TUNER_EPOCHS', 50, var_type=int)
NN_TUNER_OBJECTIVE: str = get_env_var('NN_TUNER_OBJECTIVE', 'val_loss').lower()
KERAS_TUNER_DIR: str = get_env_var('KERAS_TUNER_DIR', 'keras_tuner_dir')
NN_TUNER_PROJECT_NAME_PREFIX: str = get_env_var('NN_TUNER_PROJECT_NAME_PREFIX', 'tuning')
KERAS_TUNER_OVERWRITE: bool = get_env_var('KERAS_TUNER_OVERWRITE', True, var_type=bool)

NN_TUNER_HP_UNITS_MIN: int = get_env_var('NN_TUNER_HP_UNITS_MIN', 32, var_type=int)
NN_TUNER_HP_UNITS_MAX: int = get_env_var('NN_TUNER_HP_UNITS_MAX', 128, var_type=int)
NN_TUNER_HP_UNITS_STEP: int = get_env_var('NN_TUNER_HP_UNITS_STEP', 32, var_type=int)
NN_TUNER_HP_ACTIVATION_CHOICES: List[str] = parse_comma_sep_str(get_env_var('NN_TUNER_HP_ACTIVATION_CHOICES', 'relu,tanh'), lower=True)
NN_TUNER_HP_USE_DROPOUT: bool = get_env_var('NN_TUNER_HP_USE_DROPOUT', True, var_type=bool)
NN_TUNER_HP_DROPOUT_MIN: float = get_env_var('NN_TUNER_HP_DROPOUT_MIN', 0.1, var_type=float)
NN_TUNER_HP_DROPOUT_MAX: float = get_env_var('NN_TUNER_HP_DROPOUT_MAX', 0.4, var_type=float)
NN_TUNER_HP_DROPOUT_STEP: float = get_env_var('NN_TUNER_HP_DROPOUT_STEP', 0.1, var_type=float)
NN_TUNER_HP_LR_MIN: float = get_env_var('NN_TUNER_HP_LR_MIN', 1e-4, var_type=float)
NN_TUNER_HP_LR_MAX: float = get_env_var('NN_TUNER_HP_LR_MAX', 1e-2, var_type=float)
NN_TUNER_HP_OPTIMIZER_CHOICES: List[str] = parse_comma_sep_str(get_env_var('NN_TUNER_HP_OPTIMIZER_CHOICES', 'adam,rmsprop'), lower=True)

# Validation Check for Tuner Objective vs Validation Size
if USE_KERAS_TUNER:
    tuner_objective_needs_val: bool = NN_TUNER_OBJECTIVE.startswith('val_')
    if tuner_objective_needs_val and VALIDATION_SIZE <= NN_STEPS:
        raise ValueError(f"KerasTuner objective '{NN_TUNER_OBJECTIVE}' requires validation data, "
                         f"but VALIDATION_SIZE ({VALIDATION_SIZE}) must be strictly greater than NN_STEPS ({NN_STEPS}). "
                         "Increase VALIDATION_SIZE or change NN_TUNER_OBJECTIVE.")
    elif tuner_objective_needs_val and VALIDATION_SIZE == 0:
         raise ValueError(f"KerasTuner objective '{NN_TUNER_OBJECTIVE}' requires validation data, but VALIDATION_SIZE is 0. "
                          "Set VALIDATION_SIZE > NN_STEPS.")
    elif not tuner_objective_needs_val:
        logging.warning(f"KerasTuner objective is '{NN_TUNER_OBJECTIVE}'. Tuning will use training metrics only.")


# Evaluation Settings
_eval_metrics_str = get_env_var('EVALUATION_METRICS', 'MAE,RMSE,MAPE')
EVALUATION_METRICS: List[str] = parse_comma_sep_str(_eval_metrics_str, lower=False) # Keep case
valid_metrics = {'MAE', 'RMSE', 'MAPE'} # Add more if needed
EVALUATION_METRICS = [m for m in EVALUATION_METRICS if m in valid_metrics]
if not EVALUATION_METRICS:
     logging.warning("No valid metrics specified in EVALUATION_METRICS. Using default ['MAE', 'RMSE', 'MAPE'].")
     EVALUATION_METRICS = ['MAE', 'RMSE', 'MAPE']


# Output and Saving
SAVE_RESULTS: bool = get_env_var('SAVE_RESULTS', True, var_type=bool)
RESULTS_DIR: str = get_env_var('RESULTS_DIR', 'results')
SAVE_MODEL_PARAMETERS: bool = get_env_var('SAVE_MODEL_PARAMETERS', True, var_type=bool)
SAVE_TRAINED_MODELS: bool = get_env_var('SAVE_TRAINED_MODELS', False, var_type=bool)
SAVE_PLOTS: bool = get_env_var('SAVE_PLOTS', True, var_type=bool)
PLOT_OUTPUT_FORMAT: str = get_env_var('PLOT_OUTPUT_FORMAT', 'png').lower()
SHOW_PLOTS: bool = get_env_var('SHOW_PLOTS', True, var_type=bool)

# Final Forecasting Settings
RUN_FINAL_FORECAST: bool = get_env_var('RUN_FINAL_FORECAST', True, var_type=bool)
_forecast_horizon = get_env_var('FORECAST_HORIZON', 12, var_type=int)
if _forecast_horizon <= 0:
    logging.warning("FORECAST_HORIZON must be positive. Setting to 12.")
    FORECAST_HORIZON: int = 12
else:
    FORECAST_HORIZON: int = _forecast_horizon


# --- Log Final Configuration ---
def get_config_dict() -> Dict[str, Any]:
    """Returns a dictionary of the current configuration settings."""
    # Exclude internal/helper variables (starting with _) and functions/modules
    config_vars = {k: v for k, v in globals().items()
                   if not k.startswith('_') and k.isupper() and not callable(v) and not isinstance(v, type(os))}
    return config_vars

logging.info("--- Configuration Loaded ---")
config_summary = get_config_dict()
for key, value in sorted(config_summary.items()): # Sort for consistent logging
    logging.info(f"  {key}: {value}")
logging.info("--- End of Configuration ---")