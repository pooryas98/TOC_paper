# Time Series Forecasting Framework

A flexible and configurable Python framework for comparing multiple time series forecasting models (SARIMA, Prophet, RNN, LSTM) on user-provided data. Features include data loading, preprocessing, imputation, automatic hyperparameter tuning (for SARIMA and NNs), detailed evaluation, result saving, and plotting capabilities.

This codebase has been optimized for reduced size while maintaining full functionality.

## Features

*   **Multiple Models:** Compare forecasts from:
    *   SARIMA (with optional automatic order selection via `pmdarima`)
    *   Prophet (handles holidays, seasonality modes)
    *   SimpleRNN
    *   LSTM
*   **Data Handling:**
    *   Load time series data from CSV files.
    *   Configurable date and value columns.
    *   Optional data imputation methods (ffill, bfill, mean, median, interpolate).
    *   Automatic frequency inference or manual specification.
*   **Hyperparameter Tuning:**
    *   Optional `auto_arima` for SARIMA order selection.
    *   Optional KerasTuner integration (`RandomSearch`, `Hyperband`, `BayesianOptimization`) for RNN/LSTM hyperparameter optimization (units, activation, dropout, learning rate, optimizer).
*   **Evaluation:**
    *   Splits data into Train, Validation (optional), and Test sets.
    *   Calculates standard metrics (MAE, RMSE, MAPE) on the test set.
*   **Visualization:**
    *   Generates plots comparing model forecasts against actuals.
    *   Individual model performance plots.
    *   Pairwise model comparison plots.
    *   Residual comparison plots.
    *   Future forecast plots combining historical data and predictions.
*   **Persistence:**
    *   Saves evaluation metrics, forecasts (point and full with CIs), and run parameters to a timestamped directory.
    *   Optionally saves trained model objects (`.pkl`, `.keras`, `.joblib` for scalers).
    *   Optionally saves generated plots.
*   **Configuration:** Highly configurable run behavior via a `.env` file.
*   **Final Forecasting:** Optionally retrains selected models on the full dataset to generate forecasts for a specified future horizon.

## Project Structure


.
├── .env # Local configuration (REQUIRED, create from example) - GITIGNORED
├── .gitignore # Files ignored by git
├── main.py # Main script to run the forecasting workflow
├── requirements.txt # Python dependencies
├── README.md # This file
│
└── src/ # Source code directory
├── init.py
├── config.py # Loads configuration from .env
├── data_loader.py # Data loading, cleaning, imputation, splitting
├── evaluation.py # Forecast evaluation metric calculations
├── plotting.py # Plot generation functions
│
└── models/ # Model-specific implementation files
├── init.py
├── nn_models.py # RNN and LSTM implementation (manual & KerasTuner)
├── prophet_model.py # Prophet implementation
└── sarima_model.py # SARIMA implementation (manual & auto_arima)

--- Generated during runtime ---
results/
└── <YYYYMMDD_HHMMSS>/ # Timestamped directory for each run
├── run_parameters.json
├── evaluation_metrics.csv
├── point_forecasts.csv
├── full_forecast_*.csv
├── future_point_forecasts.csv (if RUN_FINAL_FORECAST=True)
├── future_full_forecast_*.csv (if RUN_FINAL_FORECAST=True)
├── plots/ # Contains generated plots (if SAVE_PLOTS=True)
│ └── *.png | *.pdf | ...
├── saved_models/ # Contains saved evaluation models (if SAVE_TRAINED_MODELS=True)
│ └── model_.pkl | .keras | .h5 | scaler_.joblib
├── saved_final_models/# Contains saved final models (if SAVE_TRAINED_MODELS=True & RUN_FINAL_FORECAST=True)
│ └── model_.pkl | .keras | .h5 | scaler_.joblib
└── keras_tuner_dir/ # Contains KerasTuner trial data (if USE_KERAS_TUNER=True)
## Prerequisites

*   Python (>= 3.8 recommended)
*   `pip` (Python package installer)
*   Potentially system libraries required for building dependencies like `Prophet` or `TensorFlow` (refer to their respective installation guides if issues arise).

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd <repository-directory>
    ```

2.  **Create and activate a virtual environment (Recommended):**
    ```bash
    # Using venv (built-in)
    python -m venv .venv
    source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`

    # Or using conda
    # conda create -n forecastenv python=3.9
    # conda activate forecastenv
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Configuration (`.env` File)

The entire workflow is configured using a `.env` file in the project root directory.

1.  **Create the `.env` file:** Copy the example `.env` content provided below (or if one is included in the repo, copy `example.env` to `.env`).
2.  **Edit the `.env` file** to match your dataset, desired models, and preferences.

**Example `.env` content (refer to this structure):**

```dotenv
# --- Logging ---
# Levels: DEBUG, INFO, WARNING, ERROR, CRITICAL. Empty LOG_FILE logs to console only.
LOG_LEVEL=INFO
LOG_FILE=results/forecasting_run_${TIMESTAMP}.log # Use ${TIMESTAMP} for automatic timestamp

# --- Data Source (REQUIRED) ---
CSV_FILE_PATH=your_data.csv # IMPORTANT: Path to your CSV data file
DATE_COLUMN=DateColumnName  # IMPORTANT: Name of the date/datetime column in your CSV
VALUE_COLUMN=ValueColumnName # IMPORTANT: Name of the target value column in your CSV

# --- Time Series Properties ---
TIME_SERIES_FREQUENCY=MS  # Pandas frequency string ('MS', 'D', 'W', 'H', etc.). Empty ('') attempts inference.
DATA_IMPUTATION_METHOD=ffill # 'none', 'ffill', 'bfill', 'mean', 'median', 'interpolate'. 'none' requires data with no NaNs.

# --- Data Splitting ---
TEST_SIZE=24          # Number of periods for the final evaluation set (> 0).
VALIDATION_SIZE=12    # Number of periods for validation set (>= 0). Needed for KerasTuner & NN early stopping. Must be > NN_STEPS if used.

# --- General Settings ---
RANDOM_SEED=42
MODELS_TO_RUN=SARIMA,Prophet,LSTM # Comma-separated list of models to run: SARIMA, Prophet, RNN, LSTM

# --- SARIMA Specific Settings ---
USE_AUTO_ARIMA=True # True uses pmdarima.auto_arima to find orders, False uses manual SARIMA_ orders below.
SARIMA_AUTO_SEASONAL=True # If USE_AUTO_ARIMA=True, should it search for seasonal components (P,D,Q)?
# Auto-ARIMA Search Parameters (only used if USE_AUTO_ARIMA=True)
AUTO_ARIMA_START_P=0; AUTO_ARIMA_MAX_P=5
AUTO_ARIMA_START_Q=0; AUTO_ARIMA_MAX_Q=5
AUTO_ARIMA_MAX_D=2
AUTO_ARIMA_START_SP=0; AUTO_ARIMA_MAX_SP=2
AUTO_ARIMA_START_SQ=0; AUTO_ARIMA_MAX_SQ=2
AUTO_ARIMA_MAX_SD=1
AUTO_ARIMA_TEST=adf     # ('adf', 'kpss', 'pp')
AUTO_ARIMA_IC=aic       # ('aic', 'bic', 'hqic', 'oob')
AUTO_ARIMA_STEPWISE=True
# Manual SARIMA Orders (only used if USE_AUTO_ARIMA=False)
SARIMA_P=1; SARIMA_D=1; SARIMA_Q=1
SARIMA_SP=1; SARIMA_SD=1; SARIMA_SQ=0
SARIMA_MANUAL_S=         # Optional: Manually set seasonal period 's' (e.g., 12). Overrides inferred frequency if > 1.
# SARIMA Model Enforcement
SARIMA_ENFORCE_STATIONARITY=False
SARIMA_ENFORCE_INVERTIBILITY=False

# --- Prophet Specific Settings ---
PROPHET_GROWTH=linear    # 'linear' or 'logistic'. 'logistic' requires PROPHET_CAP.
# PROPHET_CAP=1000       # Required capacity if PROPHET_GROWTH=logistic.
# PROPHET_FLOOR=0        # Optional floor if PROPHET_GROWTH=logistic.
PROPHET_ADD_YEARLY_SEASONALITY=auto # 'auto', True, False
PROPHET_ADD_WEEKLY_SEASONALITY=auto # 'auto', True, False
PROPHET_ADD_DAILY_SEASONALITY=auto # 'auto', True, False
PROPHET_SEASONALITY_MODE=additive  # 'additive' or 'multiplicative'
PROPHET_INTERVAL_WIDTH=0.95        # Confidence interval width (e.g., 0.95 for 95%)
PROPHET_COUNTRY_HOLIDAYS= # Optional: Country code for holidays (e.g., 'US', 'DE'). Requires 'holidays' library.

# --- Neural Network (RNN/LSTM) Specific Settings ---
NN_STEPS=12 # Input sequence length (window size) for NNs. Must be < VALIDATION_SIZE if using validation.
# Manual NN Build Settings (used only if USE_KERAS_TUNER=False)
NN_UNITS=50
NN_ACTIVATION=relu # ('relu', 'tanh', etc.)
NN_OPTIMIZER=adam  # ('adam', 'rmsprop', 'sgd', etc.)
NN_ADD_DROPOUT=False
NN_DROPOUT_RATE=0.2 # Only used if NN_ADD_DROPOUT=True.
# General NN Training Settings
NN_LOSS_FUNCTION=mean_squared_error # Keras loss function identifier
NN_BATCH_SIZE=32
RNN_EPOCHS=100 # Max epochs for RNN final training/manual run
LSTM_EPOCHS=100 # Max epochs for LSTM final training/manual run
NN_EARLY_STOPPING_PATIENCE=10 # Epochs to wait for improvement before stopping. 0 disables. Monitors val_loss (if val set exists) or loss.
NN_VERBOSE=1 # Keras verbosity (0=silent, 1=progress bar, 2=one line per epoch)

# --- KerasTuner (NN Hyperparameter Optimization) Settings ---
USE_KERAS_TUNER=True # Enable KerasTuner? Overrides manual NN settings above. Requires VALIDATION_SIZE > NN_STEPS.
# Tuner Configuration (only used if USE_KERAS_TUNER=True)
NN_TUNER_TYPE=RandomSearch # Tuner class: 'RandomSearch', 'Hyperband', 'BayesianOptimization'
NN_TUNER_MAX_TRIALS=15 # Number of different hyperparameter combinations to try.
NN_TUNER_EXECUTIONS_PER_TRIAL=1 # How many times to train each model configuration.
NN_TUNER_EPOCHS=50 # Max epochs for *each tuner trial*. Can be less than final training epochs.
NN_TUNER_OBJECTIVE=val_loss # Metric the tuner tries to minimize ('val_loss', 'loss', 'val_mae', 'mae', etc.). Must start with 'val_' if VALIDATION_SIZE > 0.
KERAS_TUNER_DIR=keras_tuner_dir # Subdirectory name within results dir for tuner data.
KERAS_TUNER_PROJECT_NAME_PREFIX=tuning_project # Project name prefix for tuner.
KERAS_TUNER_OVERWRITE=True # Delete previous tuner results for the same project name?
# Tuner Hyperparameter Search Space (defines ranges for the tuner)
NN_TUNER_HP_UNITS_MIN=32; NN_TUNER_HP_UNITS_MAX=128; NN_TUNER_HP_UNITS_STEP=32
NN_TUNER_HP_ACTIVATION_CHOICES=relu,tanh # Comma-separated Keras activation functions
NN_TUNER_HP_USE_DROPOUT=True # Always include a Dropout layer? (Tuner will optimize the rate if True)
NN_TUNER_HP_DROPOUT_MIN=0.1; NN_TUNER_HP_DROPOUT_MAX=0.4; NN_TUNER_HP_DROPOUT_STEP=0.1
NN_TUNER_HP_LR_MIN=1e-4; NN_TUNER_HP_LR_MAX=1e-2 # Learning rate range (log scale)
NN_TUNER_HP_OPTIMIZER_CHOICES=adam,rmsprop # Comma-separated Keras optimizer names

# --- Evaluation Settings ---
EVALUATION_METRICS=MAE,RMSE,MAPE # Comma-separated metrics to calculate: 'MAE', 'RMSE', 'MAPE'.

# --- Output & Saving Settings ---
SAVE_RESULTS=True # Master switch to save any output files (metrics, forecasts, params).
RESULTS_DIR=results # Base directory where timestamped run folders will be created.
SAVE_MODEL_PARAMETERS=True # Save the detailed run_parameters.json file?
SAVE_TRAINED_MODELS=False # Save the actual trained model objects? Can consume significant disk space.
SAVE_PLOTS=True # Save the generated plots?
PLOT_OUTPUT_FORMAT=png # Format for saved plots ('png', 'pdf', 'svg', 'jpg').
SHOW_PLOTS=False # Display plots interactively after generation? (Requires a GUI environment).

# --- Final Forecasting (After Evaluation) ---
RUN_FINAL_FORECAST=True # Retrain models specified in MODELS_TO_RUN on full data and forecast future?
FORECAST_HORIZON=12     # Number of periods to forecast into the future (> 0).
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
IGNORE_WHEN_COPYING_END

Key Configuration Variables:

CSV_FILE_PATH: REQUIRED. Path to your input data file.

DATE_COLUMN: REQUIRED. Name of the column containing dates/timestamps.

VALUE_COLUMN: REQUIRED. Name of the column containing the time series values to forecast.

MODELS_TO_RUN: Comma-separated list of models you want to include in the comparison.

TEST_SIZE: Size of the hold-out set for final evaluation.

VALIDATION_SIZE: Size of the validation set used for NN hyperparameter tuning and early stopping. Set to 0 if not using NNs or KerasTuner. Must be greater than NN_STEPS if USE_KERAS_TUNER=True or NN_EARLY_STOPPING_PATIENCE > 0 and NN_TUNER_OBJECTIVE uses validation data.

USE_AUTO_ARIMA / USE_KERAS_TUNER: Switches to enable/disable automatic tuning for SARIMA / NNs.

SAVE_RESULTS, SAVE_MODEL_PARAMETERS, SAVE_TRAINED_MODELS, SAVE_PLOTS: Control what output gets saved.

RUN_FINAL_FORECAST / FORECAST_HORIZON: Control the final forecasting step.

Usage

Ensure you have installed the prerequisites and dependencies (see Installation).

Create and configure your .env file in the project root directory.

Run the main script from the project root directory:

python main.py
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Bash
IGNORE_WHEN_COPYING_END

The script will execute the following steps:

Load configuration from .env.

Set up logging (to console and optionally to file).

Load and prepare data (imputation, frequency setting, splitting).

Train and evaluate each model specified in MODELS_TO_RUN on the train/validation set, forecasting the test set period.

This includes optional hyperparameter tuning if enabled.

Calculate and display evaluation metrics on the test set.

Save evaluation results (metrics, forecasts, parameters, optionally models) to the timestamped results/ subdirectory.

Optionally retrain models on the full dataset and generate forecasts for the future FORECAST_HORIZON.

Save future forecasts (if generated).

Generate and save comparison plots (if enabled).

Output

Upon successful execution, a new subdirectory named with the run's timestamp (e.g., results/20231027_103000/) will be created within the RESULTS_DIR specified in .env. This directory will contain:

Log File: (If LOG_FILE is set) Contains detailed logs of the run.

run_parameters.json: (If SAVE_MODEL_PARAMETERS=True) A JSON file containing the configuration used for the run, data summary, model parameters (including tuned hyperparameters), runtimes, and evaluation metrics.

evaluation_metrics.csv: A CSV file summarizing the performance metrics (MAE, RMSE, MAPE, Runtime) for each model on the test set.

point_forecasts.csv: A CSV file containing the point forecasts (yhat) from each model for the test set period.

full_forecast_<model_name>.csv: Separate CSV files for each model containing the full forecast output for the test set, potentially including confidence intervals (yhat, yhat_lower, yhat_upper).

future_point_forecasts.csv: (If RUN_FINAL_FORECAST=True) Point forecasts for the future horizon.

future_full_forecast_<model_name>.csv: (If RUN_FINAL_FORECAST=True) Full future forecasts for each model, potentially including confidence intervals.

plots/ directory: (If SAVE_PLOTS=True) Contains generated plots in the format specified by PLOT_OUTPUT_FORMAT.

saved_models/ directory: (If SAVE_TRAINED_MODELS=True) Contains the saved model objects from the evaluation run.

saved_final_models/ directory: (If SAVE_TRAINED_MODELS=True and RUN_FINAL_FORECAST=True) Contains the saved model objects retrained on the full dataset.

keras_tuner_dir/ directory: (If USE_KERAS_TUNER=True) Contains data generated by KerasTuner during the hyperparameter search.

Contributing

Contributions are welcome! Feel free to fork the repository, submit issues, or create pull requests.

License

[Specify Your License Here - e.g., MIT License]

See the LICENSE file for details. (You should create a LICENSE file if you don't have one).

Potential Future Enhancements

Support for exogenous variables (regressors) in models that allow them (SARIMAX, Prophet, NNs).

Implementation of additional forecasting models (e.g., ETS, TBATS, N-BEATS).

Multivariate time series forecasting capabilities.

More advanced diagnostic plots (ACF/PACF of residuals, etc.).

Integration with experiment tracking platforms (e.g., MLflow).

Dockerization for easier deployment.

A simple web UI (e.g., using Streamlit or Flask) for interacting with the framework.

IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
IGNORE_WHEN_COPYING_END