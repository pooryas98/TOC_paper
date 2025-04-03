# ... (keep existing imports and helper functions like str_to_bool, parse_csv_str, get_env) ...
import os, logging, datetime, re # Added 're'
from dotenv import load_dotenv
from typing import Tuple, Optional, List, Dict, Any

# --- Helper function to parse tuple strings ---
def parse_tuple_str(val_str: Optional[str], expected_type=int, expected_len: Optional[int] = None) -> Optional[tuple]:
    """Parses a string like '(10, 5)' into a tuple."""
    if not isinstance(val_str, str):
        return None
    match = re.match(r'^\s*\(\s*([^,]+)\s*,\s*([^)]+)\s*\)\s*$', val_str)
    if not match:
        logging.warning(f"Could not parse tuple string: '{val_str}'. Expected format like '(val1, val2)'.")
        return None
    try:
        item1 = expected_type(match.group(1).strip())
        item2 = expected_type(match.group(2).strip())
        result = (item1, item2)
        if expected_len is not None and len(result) != expected_len:
             logging.warning(f"Parsed tuple '{result}' from '{val_str}' but length is not {expected_len}.")
             return None
        return result
    except (ValueError, TypeError) as e:
        logging.warning(f"Error converting tuple elements from '{val_str}' to {expected_type.__name__}: {e}")
        return None
# --- End Helper ---


def str_to_bool(val:Optional[str])->bool:
	if isinstance(val,str): return val.lower() in ('true','1','t','y','yes')
	return bool(val)

def parse_csv_str(val:Optional[str],lower:bool=False)->List[str]:
	if not val: return []
	items=[item.strip() for item in val.split(',') if item.strip()]
	return [item.lower() for item in items] if lower else items

def get_env(key:str,default:Any=None,required:bool=False,var_type:type=str)->Any:
	v=os.getenv(key)
	if v is None or v == '': # Treat empty string as None for optional vars
		if required: raise ValueError(f"Missing required env var: {key}")
		return default
	ov=v
	try:
		if var_type==bool: return str_to_bool(v)
		if var_type==int: return int(v)
		if var_type==float: return float(v)
		if var_type==list: return parse_csv_str(v)
		if var_type=='tuple_int_2': # Custom type for figure size
		    parsed = parse_tuple_str(v, expected_type=int, expected_len=2)
		    return parsed if parsed is not None else default
		if var_type==dict: raise NotImplementedError("Use JSON string for dict env var.")
		# Default to string if var_type is str or not specified
		return var_type(v) if var_type != str else v
	except (ValueError,TypeError) as e:
		logging.warning(f"Invalid type env var '{key}'. Val '{ov}' -> {var_type.__name__ if hasattr(var_type, '__name__') else var_type}'. Use default: {default}. Err: {e}")
		return default

dotenv_path:str=os.path.join(os.path.dirname(__file__),'..','.env')
if os.path.exists(dotenv_path): load_dotenv(dotenv_path); print(f"Loaded config: {dotenv_path}")
else: print("Warn: .env file not found. Using env vars/defaults.")

# --- Logging Setup (Keep as is) ---
log_level_str=get_env('LOG_LEVEL','INFO').upper()
log_level=getattr(logging,log_level_str,logging.INFO)
log_file=get_env('LOG_FILE',None)
log_fmt=logging.Formatter('%(asctime)s - %(levelname)s - %(module)s - %(message)s')
logger=logging.getLogger(); logger.setLevel(log_level)
ch=logging.StreamHandler(); ch.setFormatter(log_fmt); logger.addHandler(ch)
if log_file:
	if "${TIMESTAMP}" in log_file: log_file=log_file.replace("${TIMESTAMP}",datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
	log_dir=os.path.dirname(log_file)
	if log_dir and not os.path.exists(log_dir):
		try: os.makedirs(log_dir); print(f"Created log dir: {log_dir}")
		except OSError as e: print(f"Warn: Could not create log dir '{log_dir}'. Err: {e}")
	try:
		fh=logging.FileHandler(log_file,encoding='utf-8'); fh.setFormatter(log_fmt)
		logger.addHandler(fh); logging.info(f"Logging to file: {log_file}")
	except Exception as e: logging.error(f"Failed setup log file '{log_file}'. Err: {e}")
else: logging.info("Logging only to console (LOG_FILE not set).")

logging.info("Loading configuration...")

# --- Data Config (Keep as is) ---
CSV_FILE_PATH:str=get_env('CSV_FILE_PATH',required=True)
DATE_COLUMN:str=get_env('DATE_COLUMN',required=True)
VALUE_COLUMN:str=get_env('VALUE_COLUMN',required=True)
TIME_SERIES_FREQUENCY:Optional[str]=get_env('TIME_SERIES_FREQUENCY',None)
_imp_m=get_env('DATA_IMPUTATION_METHOD','none').lower()
_valid_imp=['none','ffill','bfill','mean','median','interpolate']
DATA_IMPUTATION_METHOD:str=_imp_m if _imp_m in _valid_imp else 'none'
if _imp_m not in _valid_imp: logging.warning(f"Invalid DATA_IMPUTATION_METHOD '{_imp_m}'. Using 'none'. Options: {_valid_imp}")
TEST_SIZE:int=get_env('TEST_SIZE',24,var_type=int); VALIDATION_SIZE:int=get_env('VALIDATION_SIZE',12,var_type=int)
if TEST_SIZE<=0: logging.warning("TEST_SIZE must be > 0. Setting to 1."); TEST_SIZE=1
if VALIDATION_SIZE<0: logging.warning("VALIDATION_SIZE must be >= 0. Setting to 0."); VALIDATION_SIZE=0
RANDOM_SEED:int=get_env('RANDOM_SEED',42,var_type=int)

# --- Model Selection (Keep as is) ---
_models_str=get_env('MODELS_TO_RUN','SARIMA,Prophet,RNN,LSTM')
MODELS_TO_RUN:List[str]=parse_csv_str(_models_str)
_valid_m={'SARIMA','Prophet','RNN','LSTM'}; MODELS_TO_RUN=[m for m in MODELS_TO_RUN if m in _valid_m]
if not MODELS_TO_RUN: raise ValueError(f"No valid models in MODELS_TO_RUN. Check .env. Valid: {_valid_m}")

# --- SARIMA Config (Keep as is) ---
USE_AUTO_ARIMA:bool=get_env('USE_AUTO_ARIMA',True,var_type=bool)
SARIMA_AUTO_SEASONAL:bool=get_env('SARIMA_AUTO_SEASONAL',True,var_type=bool)
AUTO_ARIMA_START_P:int=get_env('AUTO_ARIMA_START_P',1,var_type=int); AUTO_ARIMA_MAX_P:int=get_env('AUTO_ARIMA_MAX_P',3,var_type=int)
AUTO_ARIMA_START_Q:int=get_env('AUTO_ARIMA_START_Q',1,var_type=int); AUTO_ARIMA_MAX_Q:int=get_env('AUTO_ARIMA_MAX_Q',3,var_type=int)
AUTO_ARIMA_MAX_D:int=get_env('AUTO_ARIMA_MAX_D',2,var_type=int); AUTO_ARIMA_START_SP:int=get_env('AUTO_ARIMA_START_SP',0,var_type=int)
AUTO_ARIMA_MAX_SP:int=get_env('AUTO_ARIMA_MAX_SP',2,var_type=int); AUTO_ARIMA_START_SQ:int=get_env('AUTO_ARIMA_START_SQ',0,var_type=int)
AUTO_ARIMA_MAX_SQ:int=get_env('AUTO_ARIMA_MAX_SQ',2,var_type=int); AUTO_ARIMA_MAX_SD:int=get_env('AUTO_ARIMA_MAX_SD',1,var_type=int)
AUTO_ARIMA_TEST:str=get_env('AUTO_ARIMA_TEST','adf').lower(); AUTO_ARIMA_IC:str=get_env('AUTO_ARIMA_IC','aic').lower()
AUTO_ARIMA_STEPWISE:bool=get_env('AUTO_ARIMA_STEPWISE',True,var_type=bool)
SARIMA_P:int=get_env('SARIMA_P',1,var_type=int); SARIMA_D:int=get_env('SARIMA_D',1,var_type=int); SARIMA_Q:int=get_env('SARIMA_Q',1,var_type=int)
SARIMA_SP:int=get_env('SARIMA_SP',1,var_type=int); SARIMA_SD:int=get_env('SARIMA_SD',1,var_type=int); SARIMA_SQ:int=get_env('SARIMA_SQ',0,var_type=int)
_man_s=get_env('SARIMA_MANUAL_S',None); SARIMA_MANUAL_S:Optional[int]=int(_man_s) if _man_s and _man_s.isdigit() else None
SARIMA_ENFORCE_STATIONARITY:bool=get_env('SARIMA_ENFORCE_STATIONARITY',False,var_type=bool)
SARIMA_ENFORCE_INVERTIBILITY:bool=get_env('SARIMA_ENFORCE_INVERTIBILITY',False,var_type=bool)
SARIMA_ORDER:Tuple[int,int,int]=(SARIMA_P,SARIMA_D,SARIMA_Q)
SARIMA_SEASONAL_ORDER_NOS:Tuple[int,int,int]=(SARIMA_SP,SARIMA_SD,SARIMA_SQ)

# --- Prophet Config (Keep as is) ---
PROPHET_GROWTH:str=get_env('PROPHET_GROWTH','linear').lower()
_p_cap_s=get_env('PROPHET_CAP',None); PROPHET_CAP:Optional[float]=float(_p_cap_s) if _p_cap_s else None
_p_flr_s=get_env('PROPHET_FLOOR',None); PROPHET_FLOOR:Optional[float]=float(_p_flr_s) if _p_flr_s else None
_yas_s=get_env('PROPHET_ADD_YEARLY_SEASONALITY','auto').lower(); PROPHET_ADD_YEARLY_SEASONALITY:Any=_yas_s if _yas_s=='auto' else str_to_bool(_yas_s)
_was_s=get_env('PROPHET_ADD_WEEKLY_SEASONALITY','auto').lower(); PROPHET_ADD_WEEKLY_SEASONALITY:Any=_was_s if _was_s=='auto' else str_to_bool(_was_s)
_das_s=get_env('PROPHET_ADD_DAILY_SEASONALITY','auto').lower(); PROPHET_ADD_DAILY_SEASONALITY:Any=_das_s if _das_s=='auto' else str_to_bool(_das_s)
PROPHET_SEASONALITY_MODE:str=get_env('PROPHET_SEASONALITY_MODE','additive').lower()
PROPHET_INTERVAL_WIDTH:float=get_env('PROPHET_INTERVAL_WIDTH',0.95,var_type=float)
PROPHET_COUNTRY_HOLIDAYS:Optional[str]=get_env('PROPHET_COUNTRY_HOLIDAYS',None)

# --- NN Config (Keep as is) ---
NN_STEPS:int=get_env('NN_STEPS',12,var_type=int); NN_UNITS:int=get_env('NN_UNITS',50,var_type=int)
if NN_STEPS<=0: logging.warning("NN_STEPS must be > 0. Setting to 1."); NN_STEPS=1
if NN_UNITS<=0: logging.warning("NN_UNITS must be > 0. Setting to 50."); NN_UNITS=50
NN_ACTIVATION:str=get_env('NN_ACTIVATION','relu').lower(); NN_OPTIMIZER:str=get_env('NN_OPTIMIZER','adam').lower()
NN_ADD_DROPOUT:bool=get_env('NN_ADD_DROPOUT',False,var_type=bool)
NN_DROPOUT_RATE:float=get_env('NN_DROPOUT_RATE',0.2,var_type=float)
NN_LOSS_FUNCTION:str=get_env('NN_LOSS_FUNCTION','mean_squared_error').lower()
NN_BATCH_SIZE:int=get_env('NN_BATCH_SIZE',32,var_type=int)
if NN_BATCH_SIZE<=0: logging.warning("NN_BATCH_SIZE must be > 0. Setting to 32."); NN_BATCH_SIZE=32
RNN_EPOCHS:int=get_env('RNN_EPOCHS',150,var_type=int); LSTM_EPOCHS:int=get_env('LSTM_EPOCHS',150,var_type=int)
NN_EARLY_STOPPING_PATIENCE:int=get_env('NN_EARLY_STOPPING_PATIENCE',15,var_type=int)
NN_VERBOSE:int=get_env('NN_VERBOSE',1,var_type=int)

# --- Keras Tuner Config (Keep as is) ---
USE_KERAS_TUNER:bool=get_env('USE_KERAS_TUNER',False,var_type=bool)
NN_TUNER_TYPE:str=get_env('NN_TUNER_TYPE','RandomSearch')
NN_TUNER_MAX_TRIALS:int=get_env('NN_TUNER_MAX_TRIALS',10,var_type=int)
NN_TUNER_EXECUTIONS_PER_TRIAL:int=get_env('NN_TUNER_EXECUTIONS_PER_TRIAL',1,var_type=int)
NN_TUNER_EPOCHS:int=get_env('NN_TUNER_EPOCHS',50,var_type=int)
NN_TUNER_OBJECTIVE:str=get_env('NN_TUNER_OBJECTIVE','val_loss').lower()
KERAS_TUNER_DIR:str=get_env('KERAS_TUNER_DIR','keras_tuner_dir')
NN_TUNER_PROJECT_NAME_PREFIX:str=get_env('NN_TUNER_PROJECT_NAME_PREFIX','tuning')
KERAS_TUNER_OVERWRITE:bool=get_env('KERAS_TUNER_OVERWRITE',True,var_type=bool)
NN_TUNER_HP_UNITS_MIN:int=get_env('NN_TUNER_HP_UNITS_MIN',32,var_type=int)
NN_TUNER_HP_UNITS_MAX:int=get_env('NN_TUNER_HP_UNITS_MAX',128,var_type=int)
NN_TUNER_HP_UNITS_STEP:int=get_env('NN_TUNER_HP_UNITS_STEP',32,var_type=int)
NN_TUNER_HP_ACTIVATION_CHOICES:List[str]=parse_csv_str(get_env('NN_TUNER_HP_ACTIVATION_CHOICES','relu,tanh'),lower=True)
NN_TUNER_HP_USE_DROPOUT:bool=get_env('NN_TUNER_HP_USE_DROPOUT',True,var_type=bool)
NN_TUNER_HP_DROPOUT_MIN:float=get_env('NN_TUNER_HP_DROPOUT_MIN',0.1,var_type=float)
NN_TUNER_HP_DROPOUT_MAX:float=get_env('NN_TUNER_HP_DROPOUT_MAX',0.4,var_type=float)
NN_TUNER_HP_DROPOUT_STEP:float=get_env('NN_TUNER_HP_DROPOUT_STEP',0.1,var_type=float)
NN_TUNER_HP_LR_MIN:float=get_env('NN_TUNER_HP_LR_MIN',1e-4,var_type=float)
NN_TUNER_HP_LR_MAX:float=get_env('NN_TUNER_HP_LR_MAX',1e-2,var_type=float)
NN_TUNER_HP_OPTIMIZER_CHOICES:List[str]=parse_csv_str(get_env('NN_TUNER_HP_OPTIMIZER_CHOICES','adam,rmsprop'),lower=True)
# Validation check for tuner
if USE_KERAS_TUNER:
	tuner_obj_needs_val:bool=NN_TUNER_OBJECTIVE.startswith('val_')
	if tuner_obj_needs_val and VALIDATION_SIZE<=NN_STEPS: raise ValueError(f"KerasTuner obj '{NN_TUNER_OBJECTIVE}' needs val data, but VALIDATION_SIZE ({VALIDATION_SIZE}) must be > NN_STEPS ({NN_STEPS}).")
	if tuner_obj_needs_val and VALIDATION_SIZE==0: raise ValueError(f"KerasTuner obj '{NN_TUNER_OBJECTIVE}' needs val data, but VALIDATION_SIZE is 0.")
	elif not tuner_obj_needs_val: logging.warning(f"KerasTuner obj '{NN_TUNER_OBJECTIVE}'. Using training metrics only (ensure VALIDATION_SIZE=0 if this is intended).")


# --- Evaluation Config (Keep as is) ---
_eval_metrics_str=get_env('EVALUATION_METRICS','MAE,RMSE,MAPE'); EVALUATION_METRICS:List[str]=parse_csv_str(_eval_metrics_str,lower=False)
_valid_metrics={'MAE','RMSE','MAPE'}; EVALUATION_METRICS=[m for m in EVALUATION_METRICS if m in _valid_metrics]
if not EVALUATION_METRICS: logging.warning(f"No valid metrics in EVALUATION_METRICS. Using default: ['MAE','RMSE','MAPE']. Valid: {_valid_metrics}"); EVALUATION_METRICS=['MAE','RMSE','MAPE']

# --- Saving & Output Config (Keep most as is) ---
SAVE_RESULTS:bool=get_env('SAVE_RESULTS',True,var_type=bool); RESULTS_DIR:str=get_env('RESULTS_DIR','results')
SAVE_MODEL_PARAMETERS:bool=get_env('SAVE_MODEL_PARAMETERS',True,var_type=bool); SAVE_TRAINED_MODELS:bool=get_env('SAVE_TRAINED_MODELS',False,var_type=bool)
SAVE_PLOTS:bool=get_env('SAVE_PLOTS',True,var_type=bool) # Plot saving master switch
SHOW_PLOTS:bool=get_env('SHOW_PLOTS',True,var_type=bool) # Interactive plot display switch

# --- Final Forecast Config (Keep as is) ---
RUN_FINAL_FORECAST:bool=get_env('RUN_FINAL_FORECAST',True,var_type=bool)
_fc_h=get_env('FORECAST_HORIZON',12,var_type=int)
if _fc_h<=0: logging.warning("FORECAST_HORIZON must be > 0. Setting to 12."); _fc_h=12
FORECAST_HORIZON:int=_fc_h


# --- NEW: Plot Customization Configuration ---
logging.info("Loading Plot Customization Settings...")
# Figure Sizes (Parsed from string like "(10, 5)")
PLOT_FIG_SIZE_SINGLE: Optional[Tuple[int, int]] = get_env('PLOT_FIG_SIZE_SINGLE', default=(10, 5), var_type='tuple_int_2')
PLOT_FIG_SIZE_PAIRWISE: Optional[Tuple[int, int]] = get_env('PLOT_FIG_SIZE_PAIRWISE', default=(11, 6), var_type='tuple_int_2')
PLOT_FIG_SIZE_RESIDUAL: Optional[Tuple[int, int]] = get_env('PLOT_FIG_SIZE_RESIDUAL', default=(12, 5), var_type='tuple_int_2')
PLOT_FIG_SIZE_FUTURE: Optional[Tuple[int, int]] = get_env('PLOT_FIG_SIZE_FUTURE', default=(14, 7), var_type='tuple_int_2')

# Font Sizes
PLOT_FONTSIZE_TITLE: int = get_env('PLOT_FONTSIZE_TITLE', 16, var_type=int)
PLOT_FONTSIZE_LABEL: int = get_env('PLOT_FONTSIZE_LABEL', 13, var_type=int)
PLOT_FONTSIZE_TICKS: int = get_env('PLOT_FONTSIZE_TICKS', 10, var_type=int)
PLOT_FONTSIZE_LEGEND: int = get_env('PLOT_FONTSIZE_LEGEND', 10, var_type=int)

# Colors
PLOT_COLOR_MAP: str = get_env('PLOT_COLOR_MAP', 'tab10')
PLOT_COLOR_ACTUAL: str = get_env('PLOT_COLOR_ACTUAL', 'black')
PLOT_COLOR_HISTORICAL: str = get_env('PLOT_COLOR_HISTORICAL', 'darkgrey')
PLOT_COLOR_TREND: str = get_env('PLOT_COLOR_TREND', 'red')

# Line Styles
PLOT_LINESTYLE_TREND: str = get_env('PLOT_LINESTYLE_TREND', ':')

# Confidence Intervals
PLOT_CI_ALPHA: float = get_env('PLOT_CI_ALPHA', 0.15, var_type=float)
PLOT_SHOW_CI_PAIRWISE: bool = get_env('PLOT_SHOW_CI_PAIRWISE', True, var_type=bool)
PLOT_SHOW_CI_FUTURE: bool = get_env('PLOT_SHOW_CI_FUTURE', True, var_type=bool)

# Axis Labels
PLOT_LABEL_X: str = get_env('PLOT_LABEL_X', 'Date')
PLOT_LABEL_Y_VALUE: str = get_env('PLOT_LABEL_Y_VALUE', 'Value')
PLOT_LABEL_Y_RESIDUAL: str = get_env('PLOT_LABEL_Y_RESIDUAL', 'Residual')

# Grid
PLOT_GRID_VISIBLE: bool = get_env('PLOT_GRID_VISIBLE', True, var_type=bool)
PLOT_GRID_LINESTYLE: str = get_env('PLOT_GRID_LINESTYLE', ':')
PLOT_GRID_ALPHA: float = get_env('PLOT_GRID_ALPHA', 0.6, var_type=float)

# Legend
PLOT_LEGEND_LOCATION: str = get_env('PLOT_LEGEND_LOCATION', 'best') # Special handling for 'outside' needed in plotting.py
PLOT_LEGEND_FRAME: bool = get_env('PLOT_LEGEND_FRAME', True, var_type=bool)

# Output Format & DPI
PLOT_OUTPUT_FORMAT: str = get_env('PLOT_OUTPUT_FORMAT', 'png').lower()
PLOT_DPI: int = get_env('PLOT_DPI', 150, var_type=int)
# --- End Plot Config ---


def get_config_dict()->Dict[str,Any]:
	"""Returns dict of current configuration settings."""
	# Filter out modules, functions, private vars, and helper types
	exclude_keys = {'os', 'logging', 'datetime', 're', 'load_dotenv', 'sys', 'pd', 'np', 'tf'}
	exclude_types = (type(os), type(logging), type(datetime), type(re.compile('')), type(str_to_bool)) # Add types of imported modules/functions

	cfg_dict = {}
	for k, v in globals().items():
	    if not k.startswith('_') and k.isupper() and k not in exclude_keys and not callable(v) and not isinstance(v, exclude_types):
	        cfg_dict[k] = v
	return cfg_dict

logging.info("--- Config Loaded ---")
cfg_summary=get_config_dict()
# Optionally print only a subset or format differently if too long
# print_keys = sorted([k for k in cfg_summary.keys() if not k.startswith('AUTO_ARIMA_')]) # Example filter
print_keys = sorted(cfg_summary.keys())
for k in print_keys:
    v = cfg_summary[k]
    v_str = str(v)
    if len(v_str) > 100: v_str = v_str[:100] + '...' # Truncate long values for logging
    logging.info(f"  {k}: {v_str}")
# More selective logging:
# logging.info("Key config items loaded (see log file or run_parameters.json for full details).")
logging.info("--- End Config ---")