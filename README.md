# Time Series Forecasting Model Comparison Framework

A Python framework designed to load time series data, run multiple forecasting models (SARIMA, Prophet, RNN, LSTM), evaluate their performance, and generate comparative results and plots.

## Table of Contents

- [Features](#features)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Input Data Format](#input-data-format)
- [Output](#output)
- [Getting Started (Quick Start)](#getting-started-quick-start)
- [Contributing](#contributing)
- [License](#license)
- [Support](#support)

## Features

*   **Data Handling:** Loads time series data from CSV files, handles date parsing, and offers optional imputation methods (ffill, bfill, mean, median, interpolate).
*   **Train/Validation/Test Splitting:** Splits data chronologically for model training, validation (optional), and testing.
*   **Multiple Model Implementations:** Includes implementations for:
    *   **SARIMA:** Supports both manual order specification and automatic order selection using `pmdarima` (auto\_arima).
    *   **Prophet:** Leverages Facebook's Prophet library, including options for growth models (linear/logistic), seasonality modes, and optional holiday integration (requires `holidays` package).
    *   **Recurrent Neural Networks (RNN & LSTM):** Implements SimpleRNN and LSTM models using TensorFlow/Keras. Includes data scaling (MinMaxScaler) and sequence generation.
*   **Hyperparameter Tuning:** Optional hyperparameter tuning for NN models (RNN/LSTM) using KerasTuner (RandomSearch, Hyperband, BayesianOptimization supported).
*   **Evaluation:** Compares model forecasts against the test set using configurable metrics (MAE, RMSE, MAPE).
*   **Visualization:** Generates plots using Matplotlib:
    *   Individual model forecasts vs actuals.
    *   Pairwise model forecast comparisons.
    *   Residual analysis plots.
    *   Combined future forecast plots.
*   **Future Forecasting:** Optionally retrains models on the full dataset and generates forecasts for a specified future horizon.
*   **Results Saving:** Saves evaluation metrics, point forecasts, full forecasts (including confidence intervals where applicable), run parameters, and trained models/scalers to a timestamped results directory.
*   **Configuration:** Highly configurable via environment variables or a `.env` file.

## Technologies Used

*   **Programming Language:** Python 3
*   **Core Libraries:**
    *   Pandas: Data manipulation and time series handling.
    *   NumPy: Numerical operations.
*   **Forecasting Models:**
    *   Statsmodels: SARIMA implementation.
    *   Pmdarima: Automatic ARIMA order selection.
    *   Prophet (fbprophet): Prophet model implementation.
    *   TensorFlow / Keras: RNN and LSTM model implementation.
*   **Machine Learning Utilities:**
    *   Scikit-learn: Data scaling (MinMaxScaler) and evaluation metrics.
    *   KerasTuner: Hyperparameter tuning for Keras models.
    *   Joblib: Saving/loading Scikit-learn objects (like scalers).
*   **Plotting:** Matplotlib
*   **Configuration:** python-dotenv
*   **Optional:**
    *   holidays: For adding country-specific holidays to Prophet.

## Installation

1.  **Clone the Repository:**
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2.  **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *Note: Depending on your system, installing `prophet` and `tensorflow` might require specific steps or dependencies. Refer to their official documentation if you encounter issues.*

4.  **Set up Configuration:** Create a `.env` file in the project's root directory (see [Configuration](#configuration) section below).

## Configuration

The framework is configured using environment variables, which can be conveniently managed using a `.env` file in the project root directory. Copy or create a `.env` file and set the required variables.

**Key Configuration Variables (.env):**

```dotenv
# --- Data ---
CSV_FILE_PATH=TOC_Data_Iran_2003.csv # Path to your input CSV file (REQUIRED)
DATE_COLUMN=Date                   # Name of the date/datetime column in the CSV (REQUIRED)
VALUE_COLUMN=TOC over Iran         # Name of the value column to forecast (REQUIRED)
TIME_SERIES_FREQUENCY=MS           # Optional: Pandas frequency string (e.g., 'D', 'W', 'MS'). If omitted, it tries to infer.
DATA_IMPUTATION_METHOD=none        # Imputation for missing values ('none', 'ffill', 'bfill', 'mean', 'median', 'interpolate')

# --- Data Splitting ---
TEST_SIZE=24                       # Number of periods for the test set
VALIDATION_SIZE=12                 # Number of periods for the validation set (used by NNs, especially with KerasTuner)

# --- Models ---
MODELS_TO_RUN=SARIMA,Prophet,RNN,LSTM # Comma-separated list of models to run

# --- SARIMA Specific ---
USE_AUTO_ARIMA=True                # Use pmdarima.auto_arima to find best order
SARIMA_AUTO_SEASONAL=True          # Allow auto_arima to search for seasonal orders (requires m > 1)
# SARIMA_P=1                       # Manual SARIMA order (p,d,q) - Used if USE_AUTO_ARIMA=False
# SARIMA_D=1
# SARIMA_Q=1
# SARIMA_SP=1                      # Manual SARIMA seasonal order (P,D,Q,s) - Used if USE_AUTO_ARIMA=False
# SARIMA_SD=1
# SARIMA_SQ=0
# SARIMA_MANUAL_S=12               # Manually specify seasonal period 's'. Overrides inferred frequency if set.

# --- Prophet Specific ---
# PROPHET_COUNTRY_HOLIDAYS=US      # Optional: Add country holidays (e.g., 'US', 'DE'). Requires 'holidays' package.
PROPHET_INTERVAL_WIDTH=0.95        # Confidence interval width for Prophet forecasts

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
Csv
IGNORE_WHEN_COPYING_END

Corresponding .env settings for this example:

DATE_COLUMN=Date
VALUE_COLUMN=TOC over Iran
TIME_SERIES_FREQUENCY=MS # Monthly Start frequency
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Dotenv
IGNORE_WHEN_COPYING_END
Output

If SAVE_RESULTS=True, the framework creates a timestamped subdirectory within the configured RESULTS_DIR (default: results/). This directory will contain:

run_parameters.json: A JSON file detailing the configuration used for the run, data summary, model parameters found (including tuned hyperparameters), runtimes, and evaluation metrics (if SAVE_MODEL_PARAMETERS=True).

evaluation_metrics.csv: A CSV file summarizing the performance metrics (MAE, RMSE, MAPE, etc.) and runtime for each model on the test set.

point_forecasts.csv: A CSV file containing the point forecasts (yhat) from each model for the test set period.

full_forecast_*.csv: Separate CSV files for each model containing the full forecast details for the test set, potentially including yhat, yhat_lower, and yhat_upper (confidence/prediction intervals).

future_point_forecasts.csv: A CSV file with point forecasts for the future horizon (if RUN_FINAL_FORECAST=True).

future_full_forecast_*.csv: Separate CSV files for each model's future forecast details (if RUN_FINAL_FORECAST=True).

plots/: A subdirectory containing generated plots (if SAVE_PLOTS=True).

eval_indiv_*.png: Individual model forecast plots vs actuals.

eval_pair_*.png: Pairwise model comparison plots.

eval_resid_comp.png: Comparison of residuals for all models.

future_fcst_combined.png: Combined plot of future forecasts from all models.

saved_models/: Subdirectory containing saved model objects from the evaluation run (if SAVE_TRAINED_MODELS=True). Includes model files (.pkl, .keras/.h5) and scalers (.joblib).

saved_final_models/: Subdirectory containing saved model objects trained on the full dataset for future forecasting (if SAVE_TRAINED_MODELS=True and RUN_FINAL_FORECAST=True).

keras_tuner_dir/: If USE_KERAS_TUNER=True, this directory (relative to the results directory) stores the detailed logs and checkpoints from the KerasTuner search process.

Logs are printed to the console and optionally saved to a file specified by LOG_FILE.

Getting Started (Quick Start)

Clone: git clone <repository-url> && cd <repository-directory>

Setup Env & Install:

python -m venv venv
source venv/bin/activate # or venv\Scripts\activate on Windows
pip install -r requirements.txt
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Bash
IGNORE_WHEN_COPYING_END

Configure: Create a .env file in the root directory with at least:

CSV_FILE_PATH=TOC_Data_Iran_2003.csv # Use the sample data or your own
DATE_COLUMN=Date
VALUE_COLUMN=TOC over Iran
MODELS_TO_RUN=SARIMA,Prophet # Run a subset for speed initially
TEST_SIZE=12
VALIDATION_SIZE=6
SAVE_RESULTS=True
SAVE_PLOTS=True
SHOW_PLOTS=False
RUN_FINAL_FORECAST=False # Disable final forecast for quicker first run
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Dotenv
IGNORE_WHEN_COPYING_END

Run:

python main.py
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Bash
IGNORE_WHEN_COPYING_END

Check Results: Examine the console output and the contents of the newly created results/<timestamp>/ directory.

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
Bash
IGNORE_WHEN_COPYING_END