# Time Series Forecasting Framework 📊

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Not%20Specified-lightgrey.svg)](./LICENSE) <!-- Replace if a license file exists -->
<!-- Add other relevant badges like build status if applicable -->

A flexible and configurable Python framework designed to compare various time series forecasting models (SARIMA, Prophet, RNN, LSTM) on your data. It handles data loading, preprocessing, model training, evaluation, and visualization, providing a streamlined workflow for forecasting experiments.

---

## ✨ Key Features

*   **📈 Multiple Models:** Compare **SARIMA**, **Prophet**, **SimpleRNN**, and **LSTM** out-of-the-box.
*   **⚙️ Highly Configurable:** Easily control data sources, model parameters, evaluation metrics, and output options via a `.env` file.
*   **📊 Comprehensive Evaluation:** Automatically calculates standard metrics (MAE, RMSE, MAPE) and generates insightful comparison plots.
*   **🤖 Smart Defaults & Options:** Includes features like optional `auto_arima` for SARIMA and KerasTuner integration for NN hyperparameter optimization.
*   **💾 Systematic Output:** Saves forecasts, evaluation metrics, plots, run parameters, and trained models to timestamped directories for reproducibility.
*   **🔮 Future Forecasting:** Option to retrain models on the full dataset and forecast future periods beyond the test set.
*   **🧹 Data Handling:** Supports CSV input, date parsing, and optional missing value imputation.

---

## 🖼️ Example Output Plot

*(The framework automatically generates plots comparing model performance, residuals, and future forecasts. Below is a conceptual example of a comparison plot.)*

![Example Plot Placeholder](https://via.placeholder.com/800x400.png?text=Example+Model+Comparison+Plot)
*Conceptual plot showing Historical Data, Actual Test Data, and Forecasts from multiple models.*

---

## 🛠️ Technologies Used

*   **Language:** Python 3
*   **Core Libraries:**
    *   Pandas: Data manipulation & time series.
    *   NumPy: Numerical operations.
*   **Forecasting Models:**
    *   Statsmodels: SARIMA implementation.
    *   Pmdarima: Automatic ARIMA order selection.
    *   Prophet: Facebook's Prophet model.
    *   TensorFlow / Keras: RNN & LSTM implementations.
*   **Machine Learning Utilities:**
    *   Scikit-learn: Data scaling (MinMaxScaler) & evaluation metrics.
    *   KerasTuner: Hyperparameter tuning for Keras models.
    *   Joblib: Saving/loading Scikit-learn objects (scalers).
*   **Plotting:** Matplotlib
*   **Configuration:** python-dotenv

---

## 🚀 Installation

1.  **Clone the Repository:**
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2.  **Create and Activate a Virtual Environment** (Recommended):
    ```bash
    # Linux / macOS
    python3 -m venv venv
    source venv/bin/activate

    # Windows (Git Bash or Cmd)
    python -m venv venv
    venv\Scripts\activate
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *Note: `prophet` and `tensorflow` installation can sometimes have specific system requirements. Consult their official documentation if you encounter issues.*

4.  **Prepare Configuration:** Create a `.env` file in the project root directory (see [Configuration](#️-configuration) section below).

---

## ⚙️ Configuration

Control the framework's behavior by creating a `.env` file in the project's root directory. You can copy the structure below and modify the values.

**Example `.env` File:**

```env
# === Data Source ===
CSV_FILE_PATH=TOC_Data_Iran_2003.csv  # REQUIRED: Path to your time series CSV
DATE_COLUMN=Date                     # REQUIRED: Name of the date/datetime column
VALUE_COLUMN=TOC over Iran           # REQUIRED: Name of the column with values to forecast
TIME_SERIES_FREQUENCY=MS             # Optional: Pandas frequency string (e.g., 'D', 'W', 'MS'). Tries to infer if omitted.
DATA_IMPUTATION_METHOD=none          # Missing value handling: 'none', 'ffill', 'bfill', 'mean', 'median', 'interpolate'

# === Data Splitting ===
TEST_SIZE=24                         # Number of periods for the test set
VALIDATION_SIZE=12                   # Periods for validation set (used by NNs, especially tuning)

# === Model Selection ===
MODELS_TO_RUN=SARIMA,Prophet,RNN,LSTM # Comma-separated list of models to execute

# === SARIMA Specific ===
USE_AUTO_ARIMA=True                  # Use pmdarima.auto_arima? (Requires pmdarima)
SARIMA_AUTO_SEASONAL=True            # Allow auto_arima to find seasonal components?
# SARIMA_P=1                         # Manual order (p,d,q) - if USE_AUTO_ARIMA=False
# SARIMA_D=1
# SARIMA_Q=1
# SARIMA_SP=1                        # Manual seasonal order (P,D,Q,s) - if USE_AUTO_ARIMA=False & seasonal
# SARIMA_SD=1
# SARIMA_SQ=0
# SARIMA_MANUAL_S=12                 # Manually set seasonal period 's' (overrides inferred)

# === Prophet Specific ===
# PROPHET_COUNTRY_HOLIDAYS=US        # Optional: Add country holidays (e.g., 'US', 'DE'). Requires 'holidays' package.
PROPHET_INTERVAL_WIDTH=0.95          # Confidence interval width

# === NN (RNN/LSTM) Specific ===
NN_STEPS=12                          # Input sequence length (lags)
NN_UNITS=50                          # Units in RNN/LSTM layer (if not tuning)
RNN_EPOCHS=150                       # Max epochs for RNN final training
LSTM_EPOCHS=150                      # Max epochs for LSTM final training
NN_EARLY_STOPPING_PATIENCE=15        # Patience for early stopping (0 disables)
USE_KERAS_TUNER=False                # Enable KerasTuner? (Requires KerasTuner)
# NN_TUNER_MAX_TRIALS=10             # KerasTuner: Max tuning trials
# NN_TUNER_EPOCHS=50                 # KerasTuner: Epochs per trial
# KERAS_TUNER_DIR=keras_tuner_dir    # Dir for KerasTuner results (relative to results dir)

# === Evaluation & Output ===
EVALUATION_METRICS=MAE,RMSE,MAPE     # Metrics to calculate (comma-separated)
SAVE_RESULTS=True                    # Save all outputs?
RESULTS_DIR=results                  # Base directory for output runs
SAVE_MODEL_PARAMETERS=True           # Save run configuration and parameters to JSON?
SAVE_TRAINED_MODELS=False            # Save trained model files (.pkl, .h5/.keras, .joblib)?
SAVE_PLOTS=True                      # Generate and save plots?
SHOW_PLOTS=False                     # Display plots interactively via matplotlib? (Blocks execution)
PLOT_OUTPUT_FORMAT=png               # Image format for saved plots (png, jpg, pdf...)

# === Final Forecasting Step ===
RUN_FINAL_FORECAST=True              # Retrain on full data & predict future?
FORECAST_HORIZON=12                  # Number of future periods to forecast

# === General Settings ===
RANDOM_SEED=42                       # Seed for reproducibility
LOG_LEVEL=INFO                       # Logging verbosity: DEBUG, INFO, WARNING, ERROR
# LOG_FILE=logs/run_${TIMESTAMP}.log # Optional: Path for log file (supports ${TIMESTAMP} variable)


For a complete list of all configuration options and their default values, please refer to the src/config.py file.

▶️ Usage

Ensure all dependencies are installed (Installation).

Prepare your input time series data in CSV format.

Create and configure your .env file in the project root, pointing to your data file and customizing settings as desired.

Run the main execution script from the project's root directory:

python main.py
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Bash
IGNORE_WHEN_COPYING_END

Monitor the console output for progress and results.

If SAVE_RESULTS=True, check the newly created timestamped subdirectory inside your RESULTS_DIR (default: results/) for detailed outputs (metrics, forecasts, plots, etc.).

📁 Input Data Format

The framework expects a CSV file containing:

A Date/Time Column: Parsable by pandas.to_datetime. Specify its name via DATE_COLUMN in .env. This will become the time series index.

A Value Column: Containing the numerical data to forecast. Specify its name via VALUE_COLUMN in .env.

The data should ideally be sorted chronologically.

Example: (TOC_Data_Iran_2003.csv)

Date,TOC over Iran
1/1/2003 0:00,304.629822
2/1/2003 0:00,309.709167
3/1/2003 0:00,313.327026
...
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Csv
IGNORE_WHEN_COPYING_END
📄 Output Structure

When SAVE_RESULTS is enabled, a directory like results/YYYYMMDD_HHMMSS/ is created, containing:

run_parameters.json: Full configuration, data summary, model params, runtimes.

evaluation_metrics.csv: Performance metrics (MAE, RMSE, etc.) for each model on the test set.

point_forecasts.csv: Test set point forecasts (yhat) from all models.

full_forecast_[MODEL_NAME].csv: Test set forecast details (incl. intervals if available).

future_point_forecasts.csv: Future point forecasts (if RUN_FINAL_FORECAST=True).

future_full_forecast_[MODEL_NAME].csv: Future forecast details (if RUN_FINAL_FORECAST=True).

plots/: Subdirectory with generated .png (or other format) plots.

saved_models/: Saved model objects from the evaluation run (if SAVE_TRAINED_MODELS=True).

saved_final_models/: Saved model objects trained on full data (if applicable).

keras_tuner_dir/: KerasTuner detailed results (if USE_KERAS_TUNER=True).

Console logs provide real-time progress and summaries. An optional log file can capture this persistently.

🏁 Getting Started (Quick Start)

Clone & Install: Follow steps 1-3 in the Installation section.

Configure: Create a minimal .env file:

CSV_FILE_PATH=TOC_Data_Iran_2003.csv # Use provided sample or your path
DATE_COLUMN=Date
VALUE_COLUMN=TOC over Iran
MODELS_TO_RUN=SARIMA,Prophet # Start with fewer models
TEST_SIZE=12
VALIDATION_SIZE=6
SAVE_RESULTS=True
SAVE_PLOTS=True
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Env
IGNORE_WHEN_COPYING_END

Run:

python main.py
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Bash
IGNORE_WHEN_COPYING_END

Explore: Check the console and the results/<timestamp>/ folder.

🤝 Contributing

Contributions are welcome! Please follow standard GitHub practices:

Fork the repository.

Create a new branch (git checkout -b feature/my-new-feature).

Make your changes. Adhere to existing code style.

Commit your changes (git commit -am 'Add some feature').

Push to the branch (git push origin feature/my-new-feature).

Open a Pull Request.

Consider adding tests for new functionality if applicable. Please ensure your changes respect the .gitignore file.

📜 License

The codebase analysis did not find an explicit LICENSE file. Please assume the code is under the repository owner's copyright unless otherwise stated. Contact the maintainers for clarification on usage rights.

❓ Support

For bugs, questions, or feature requests, please use the GitHub Issues tab associated with this repository.

Primary documentation is this README.md. Consult code comments in the src/ directory for specific implementation details.