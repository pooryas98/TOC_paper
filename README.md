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

--- Data ---
CSV_FILE_PATH=TOC_Data_Iran_2003.csv # Path to your input CSV file (REQUIRED)
DATE_COLUMN=Date # Name of the date/datetime column in the CSV (REQUIRED)
VALUE_COLUMN=TOC over Iran # Name of the value column to forecast (REQUIRED)
TIME_SERIES_FREQUENCY=MS # Optional: Pandas frequency string (e.g., 'D', 'W', 'MS'). If omitted, it tries to infer.
DATA_IMPUTATION_METHOD=none # Imputation for missing values ('none', 'ffill', 'bfill', 'mean', 'median', 'interpolate')

--- Data Splitting ---
TEST_SIZE=24 # Number of periods for the test set
VALIDATION_SIZE=12 # Number of periods for the validation set (used by NNs, especially with KerasTuner)

--- Models ---
MODELS_TO_RUN=SARIMA,Prophet,RNN,LSTM # Comma-separated list of models to run

--- SARIMA Specific ---
USE_AUTO_ARIMA=True # Use pmdarima.auto_arima to find best order
SARIMA_AUTO_SEASONAL=True # Allow auto_arima to search for seasonal orders (requires m > 1)

SARIMA_P=1 # Manual SARIMA order (p,d,q) - Used if USE_AUTO_ARIMA=False
SARIMA_D=1
SARIMA_Q=1
SARIMA_SP=1 # Manual SARIMA seasonal order (P,D,Q,s) - Used if USE_AUTO_ARIMA=False
SARIMA_SD=1
SARIMA_SQ=0
SARIMA_MANUAL_S=12 # Manually specify seasonal period 's'. Overrides inferred frequency if set.
--- Prophet Specific ---
PROPHET_COUNTRY_HOLIDAYS=US # Optional: Add country holidays (e.g., 'US', 'DE'). Requires 'holidays' package.
PROPHET_INTERVAL_WIDTH=0.95 # Confidence interval width for Prophet forecasts

--- NN (RNN/LSTM) Specific ---
NN_STEPS=12 # Number of time steps (lags) to use as input
NN_UNITS=50 # Number of units in RNN/LSTM layer (if not using KerasTuner)
RNN_EPOCHS=150 # Max epochs for RNN final training
LSTM_EPOCHS=150 # Max epochs for LSTM final training
NN_EARLY_STOPPING_PATIENCE=15 # Patience for early stopping (0 to disable)
USE_KERAS_TUNER=False # Enable KerasTuner for NN hyperparameter search

NN_TUNER_MAX_TRIALS=10 # KerasTuner: Max optimization trials
NN_TUNER_EPOCHS=50 # KerasTuner: Epochs per trial
KERAS_TUNER_DIR=keras_tuner_dir # Directory to store KerasTuner results
--- Evaluation & Results ---
EVALUATION_METRICS=MAE,RMSE,MAPE # Metrics to calculate (comma-separated)
SAVE_RESULTS=True # Save evaluation results, forecasts, plots, etc.
RESULTS_DIR=results # Directory to save results
SAVE_MODEL_PARAMETERS=True # Save the run configuration and model parameters to a JSON file
SAVE_TRAINED_MODELS=False # Save the trained model objects (.pkl, .keras/.h5, .joblib)
SAVE_PLOTS=True # Generate and save plots
SHOW_PLOTS=False # Display plots interactively using matplotlib (blocks execution)
PLOT_OUTPUT_FORMAT=png # Format for saved plots (e.g., png, jpg, pdf)

--- Final Forecast ---
RUN_FINAL_FORECAST=True # Retrain models on full data and forecast future periods
FORECAST_HORIZON=12 # Number of periods to forecast into the future

--- General ---
RANDOM_SEED=42 # Seed for reproducibility
LOG_LEVEL=INFO # Logging level (DEBUG, INFO, WARNING, ERROR)

LOG_FILE=logs/run_${TIMESTAMP}.log # Optional: Path to log file (supports ${TIMESTAMP})

Refer to `src/config.py` for the full list of configuration options and their default values.

Usage

Ensure you have installed the necessary dependencies (see Installation).

Prepare your input data CSV file (see Input Data Format).

Create and configure your .env file in the project root directory, pointing CSV_FILE_PATH, DATE_COLUMN, and VALUE_COLUMN to your data. Customize other settings as needed.

Run the main script from the project's root directory:

python main.py

The script will:

Load configuration from .env or environment variables.

Load and prepare the specified time series data.

Split the data into training, validation (if VALIDATION_SIZE > 0), and test sets.

Train the models specified in MODELS_TO_RUN on the training (and potentially validation) data.

If USE_KERAS_TUNER=True, it will perform hyperparameter search for NN models before final training.

Generate forecasts for the test period for each trained model.

Evaluate the forecasts using the specified metrics.

Print the evaluation results to the console.

If RUN_FINAL_FORECAST=True, retrain models on the full dataset and generate forecasts for the FORECAST_HORIZON.

Print the future forecasts to the console.

Save results (metrics, forecasts, parameters, plots, models) to the RESULTS_DIR/<timestamp>/ directory if enabled in the configuration.

Generate and save/show plots if enabled.

Input Data Format

The framework expects input data as a CSV file with at least two columns:

Date/Time Column: A column containing dates or timestamps that can be parsed by pandas.to_datetime. Specify the name of this column in the DATE_COLUMN configuration variable.

Value Column: A column containing the numerical time series values to be forecasted. Specify the name of this column in the VALUE_COLUMN configuration variable.

The script will use the date/time column as the index. Ensure the data is sorted chronologically if it isn't already in the CSV.

Example CSV (TOC_Data_Iran_2003.csv):

Date,TOC over Iran
1/1/2003 0:00,304.629822
2/1/2003 0:00,309.709167
3/1/2003 0:00,313.327026
...

Corresponding .env settings for this example:

DATE_COLUMN=Date
VALUE_COLUMN=TOC over Iran
TIME_SERIES_FREQUENCY=MS # Monthly Start frequency

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

Run:

python main.py

Check Results: Examine the console output and the contents of the newly created results/<timestamp>/ directory.

Contributing

Contributions are welcome! Please follow these steps:

Fork the repository on GitHub.

Create a new branch for your feature or bug fix:

git checkout -b feature/your-amazing-feature

Make your changes. Follow the existing code style and structure. Add comments where necessary.

Commit your changes:

git commit -m "Add some amazing feature"

Push to your branch:

git push origin feature/your-amazing-feature

Open a Pull Request against the main repository branch.

Please ensure your code is well-documented. While there isn't a formal test suite in the provided code, contributions that include tests for new functionality are highly encouraged. Be mindful of the files listed in .gitignore.

License

No LICENSE file was found in the repository. Please contact the project maintainers for information regarding licensing and usage terms.

Support

For issues, bug reports, or feature requests, please use the GitHub Issues tab of the repository.

The primary documentation is this README.md file. Refer to the code comments and docstrings within the src/ directory for more specific implementation details.