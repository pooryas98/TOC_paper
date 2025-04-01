# Time Series Forecasting Comparison Framework

This project provides a flexible framework for comparing multiple time series forecasting models on your data. It automates the process of data loading, preprocessing, training (including optional hyperparameter tuning), evaluation, result saving, plotting, and generating future forecasts.

Currently supported models:
*   SARIMA (including Auto-ARIMA via `pmdarima`)
*   Prophet
*   Simple RNN
*   LSTM (with optional KerasTuner integration)

The framework is designed to be easily configurable via a `.env` file and produces detailed outputs, including metrics, forecasts, parameters, and visualizations in timestamped directories for organized runs.

The example dataset used (`TOC_Data_Iran_2003.csv`) contains monthly Total Ozone Column (TOC) data over Iran from 2003 to 2022.

## Features

*   **Multiple Models:** Implements and compares SARIMA, Prophet, Simple RNN, and LSTM.
*   **Data Handling:**
    *   Loads data from standard CSV files.
    *   Handles configurable date and value columns.
    *   Supports various data imputation methods (ffill, bfill, mean, median, interpolate, or none).
    *   Infers or sets time series frequency based on configuration.
*   **Evaluation:**
    *   Splits data into Training, Validation (optional), and Test sets.
    *   Trains models on Train (+Val) data and evaluates performance on the unseen Test set.
    *   Calculates standard regression metrics (MAE, RMSE, MAPE).
*   **Hyperparameter Tuning:**
    *   Integrates **KerasTuner** (`RandomSearch`, `Hyperband`, `BayesianOptimization`) for automated hyperparameter optimization of RNN/LSTM models.
    *   Configurable search space for units, activation, dropout rate, learning rate, and optimizer.
    *   *Includes an option to force dropout regularization during tuning for RNN/LSTM.*
*   **Result Persistence & Organization:**
    *   **Timestamped Results:** Each run saves outputs to a unique subdirectory within `results/` (e.g., `results/20240115_103000/`) for easy tracking.
    *   Saves evaluation metrics, evaluation forecasts (point and full with CIs), and future forecasts to CSV files.
    *   Saves comprehensive run parameters (configuration, data summary, model details including selected hyperparameters, runtimes) to a JSON file (`run_parameters.json`).
    *   Optionally saves trained model objects from both the evaluation run (`saved_models/`) and the final forecast run (`saved_final_models/`) using appropriate formats (`.pkl`, `.keras`, `.joblib`).
*   **Visualization:**
    *   Generates and saves various plots to the run's `plots/` subdirectory:
        *   Individual model evaluation forecasts vs. actuals (with optional CIs).
        *   Pairwise model comparisons for evaluation forecasts.
        *   Residual plots comparing model errors during evaluation.
        *   Combined plot of future forecasts against historical data.
*   **Final Forecasting:**
    *   Optionally retrains the selected models on the full dataset.
    *   Generates forecasts for a user-specified future horizon.
*   **Configuration:** Easily configured via a `.env` file for most parameters.

## Project Structure

```
.
├── .env                  # Local configuration (sensitive, DO NOT COMMIT) - Create this file
├── .gitignore            # Specifies intentionally untracked files
├── Datasets.csv          # Example input time series data
├── main.py               # Main execution script
├── requirements.txt      # Project dependencies
├── README.md             # This file
└── src/                  # Source code directory
    ├── __init__.py
    ├── config.py         # Loads configuration from .env
    ├── data_loader.py    # Handles data loading, cleaning, splitting
    ├── evaluation.py     # Calculates forecast evaluation metrics
    ├── plotting.py       # Generates result plots
    └── models/           # Model implementations
        ├── __init__.py
        ├── nn_models.py    # RNN/LSTM implementation with KerasTuner
        ├── prophet_model.py # Prophet implementation
        └── sarima_model.py  # SARIMA implementation with Auto-ARIMA
```

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd <repository-directory>
    ```

2.  **Create and activate a virtual environment (Recommended):**
    ```bash
    # Using venv (Python 3 built-in)
    python -m venv .venv
    source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`

    # Or using conda
    # conda create -n forecast-env python=3.9 # Adjust Python version if needed
    # conda activate forecast-env
    ```
    *(Activate the environment before proceeding)*

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *   **Note:** `prophet` and `tensorflow` can sometimes have complex dependencies, especially on certain operating systems or with specific hardware (like GPUs). If you encounter installation errors, consult the official documentation for [Prophet](https://facebook.github.io/prophet/docs/installation.html) and [TensorFlow](https://www.tensorflow.org/install) for detailed instructions tailored to your system.

## Usage

1.  **Configure the run:**
    *   Create a file named `.env` in the project root directory (where `main.py` is).
    *   Copy the contents from `.env.example` (if provided) or add variables manually. **Review and set these essential variables for your data:**
        ```dotenv
        # --- Data Settings ---
        CSV_FILE_PATH=TOC_Data_Iran_2003.csv # Path to your CSV data
        DATE_COLUMN=Date                    # Name of the date column in your CSV
        VALUE_COLUMN=TOC over Iran          # Name of the value column in your CSV
        TIME_SERIES_FREQUENCY=MS            # Pandas frequency string (e.g., MS, M, D, W, H) or leave blank to infer

        # --- Models & Run Control ---
        MODELS_TO_RUN=SARIMA,Prophet,RNN,LSTM # Comma-separated list
        TEST_SIZE=24                        # Periods for the test set (e.g., 2 years for monthly data)
        VALIDATION_SIZE=12                  # Periods for validation (before test set). Set > 0 for NN tuning/early stopping.
        RUN_FINAL_FORECAST=True             # Retrain on full data and predict future? (True/False)
        FORECAST_HORIZON=24                 # Periods to forecast into the future if RUN_FINAL_FORECAST=True
        ```
    *   Review and adjust other settings in `.env` (model parameters, tuner settings, saving options) as needed. All available options are defined with defaults in `src/config.py`.

2.  **Run the main script:**
    ```bash
    python main.py
    ```

3.  **Check the output:**
    *   Follow the logs printed to the console (and optionally to the file specified by `LOG_FILE` in `.env`).
    *   If `SAVE_RESULTS=True`, find all outputs (metrics, forecasts, parameters, plots, saved models) organized in a timestamped subdirectory within `results/` (e.g., `results/20240115_103000/`).
    *   If `SHOW_PLOTS=True`, plots will be displayed interactively during the run.

## Configuration Details

The run behavior is controlled by environment variables defined in the `.env` file. Key categories include:

*   **Data:** `CSV_FILE_PATH`, `DATE_COLUMN`, `VALUE_COLUMN`, `TIME_SERIES_FREQUENCY`, `DATA_IMPUTATION_METHOD`.
*   **Splitting:** `TEST_SIZE`, `VALIDATION_SIZE`.
*   **Models:** `MODELS_TO_RUN`, specific parameters for SARIMA (`SARIMA_P`, `SARIMA_D`, etc., `USE_AUTO_ARIMA`), Prophet (`PROPHET_GROWTH`, `PROPHET_ADD_*_SEASONALITY`, etc.), and NNs (`NN_STEPS`, `NN_UNITS`, `NN_ACTIVATION`, `NN_OPTIMIZER`, `NN_ADD_DROPOUT`, `RNN_EPOCHS`, `LSTM_EPOCHS`).
*   **Tuning (KerasTuner):**
    *   `USE_KERAS_TUNER`: Enable/disable tuner (requires `VALIDATION_SIZE` > `NN_STEPS`). If `True`, it overrides manual NN settings (like `NN_UNITS`, `NN_ACTIVATION`, `NN_OPTIMIZER`, `NN_ADD_DROPOUT`, `NN_DROPOUT_RATE`).
    *   `NN_TUNER_*`: Control tuner type, trials, epochs per trial, objective, and hyperparameter search ranges (units, activation, dropout rate, learning rate, optimizer).
    *   `NN_TUNER_HP_USE_DROPOUT`: If `True`, *forces* a dropout layer to be included in *every* tuning trial (only the rate is tuned). If `False`, dropout is skipped during tuning.
*   **Evaluation:** `EVALUATION_METRICS`.
*   **Output:** `SAVE_RESULTS`, `RESULTS_DIR`, `SAVE_MODEL_PARAMETERS`, `SAVE_TRAINED_MODELS`, `SAVE_PLOTS`, `PLOT_OUTPUT_FORMAT`, `SHOW_PLOTS`.
*   **Final Forecast:** `RUN_FINAL_FORECAST`, `FORECAST_HORIZON`.
*   **General:** `RANDOM_SEED`, `LOG_LEVEL`, `LOG_FILE`.

*(Refer to `src/config.py` for a complete list, descriptions, and default values)*

## Output Structure (`results/<run_timestamp>/` directory)

If saving is enabled (`SAVE_RESULTS=True`), a typical run output directory will look like this:
```
results/
├── evaluation_metrics.csv       # Performance metrics on the test set
├── point_forecasts.csv          # Model point forecasts for the test set
├── full_forecast_SARIMA.csv     # Full SARIMA forecast (incl. CI) for test set
├── full_forecast_Prophet.csv    # Full Prophet forecast (incl. CI) for test set
├── full_forecast_RNN.csv        # Full RNN forecast for test set
├── full_forecast_LSTM.csv       # Full LSTM forecast for test set
├── future_point_forecasts.csv   # Point forecasts beyond the original data
├── future_full_forecast_*.csv   # Full future forecasts (incl. CI if available)
├── run_parameters.json          # Detailed parameters and results for the run
├── plots/                         # Directory for generated plots
│   ├── evaluation_combined_forecast.png
│   ├── evaluation_forecast_SARIMA.png
│   ├── evaluation_residuals_comparison.png
│   └── future_forecast_combined.png
├── saved_models/                  # Models trained during the evaluation run
│   ├── model_SARIMA.pkl
│   ├── model_Prophet.pkl
│   ├── model_RNN.keras           # or .h5 depending on TF version
│   ├── scaler_RNN.joblib
│   └── ...
└── saved_final_models/            # Models trained on the full dataset for final forecast
    ├── model_SARIMA.pkl
    ├── model_Prophet.pkl
    ├── model_RNN.keras
    ├── scaler_RNN.joblib
    └── ...
```

## Dependencies

Key libraries used:

*   pandas
*   numpy
*   statsmodels
*   prophet
*   pmdarima
*   tensorflow >= 2.0
*   keras-tuner
*   scikit-learn
*   matplotlib
*   python-dotenv
*   joblib

*(See `requirements.txt` for specific versions used during development)*

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues. For major changes, please open an issue first to discuss what you would like to change.

1.  Fork the repository.
2.  Create your feature branch (`git checkout -b feature/AmazingFeature`).
3.  Commit your changes (`git commit -m 'Add some AmazingFeature'`).
4.  Push to the branch (`git push origin feature/AmazingFeature`).
5.  Open a Pull Request.

## License

This project is licensed under the MIT License. See the LICENSE file for details (if one exists - assume MIT if not specified).