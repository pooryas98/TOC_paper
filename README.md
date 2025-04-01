# Time Series Forecasting Comparison Framework

This project provides a framework for comparing multiple time series forecasting models (SARIMA, Prophet, RNN, LSTM) on a given dataset. It includes data loading, preprocessing, model training, hyperparameter tuning (optional for NNs), evaluation, result saving, plotting, and final future forecasting.

The example dataset used (`Datasets.csv`) contains monthly Total Ozone Column (TOC) data over Iran from 2003 to 2022.

## Features

*   **Multiple Models:** Implements and compares:
    *   SARIMA (with optional Auto-ARIMA via `pmdarima`)
    *   Prophet
    *   Simple RNN (Recurrent Neural Network)
    *   LSTM (Long Short-Term Memory)
*   **Data Handling:**
    *   Loads data from CSV.
    *   Handles standard date/value columns.
    *   Supports various data imputation methods (ffill, bfill, mean, median, interpolate).
    *   Infers or sets time series frequency.
*   **Evaluation:**
    *   Splits data into Training, Validation (optional), and Test sets.
    *   Trains models on Train (+Val) data and evaluates on the Test set.
    *   Calculates standard regression metrics (MAE, RMSE, MAPE).
*   **Hyperparameter Tuning:**
    *   Optional integration with KerasTuner (`RandomSearch`, `Hyperband`, `BayesianOptimization`) for RNN/LSTM models.
*   **Result Persistence:**
    *   Saves evaluation metrics, point forecasts, and full forecasts (with CIs if available) to CSV files.
    *   Saves run parameters (configuration, data summary, model details, tuning results) to a JSON file.
    *   Optionally saves trained model objects (evaluation run and final run) using appropriate formats (`.pkl`, `.keras`, `.joblib`).
*   **Visualization:**
    *   Generates comparison plots for evaluation forecasts vs. actuals.
    *   Generates individual model evaluation plots with confidence intervals.
    *   Generates residual plots for evaluation forecasts.
    *   Generates plots for future forecasts against historical data.
*   **Final Forecasting:**
    *   Optionally retrains models on the full dataset.
    *   Generates forecasts for a specified future horizon.
*   **Configuration:** Easily configurable via a `.env` file.

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
    # conda create -n forecast-env python=3.9  # Adjust python version if needed
    # conda activate forecast-env
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *   **Note:** `prophet` and `tensorflow` can sometimes have complex dependencies. If you encounter issues, consult their official installation guides for your specific OS and hardware (e.g., GPU support for TensorFlow).

## Usage

1.  **Configure the run:**
    *   Create a file named `.env` in the project root directory.
    *   Copy the contents of `.env.example` (if provided) or manually add the required configuration variables. **At minimum, ensure these are set correctly for your data:**
        ```dotenv
        # --- Data Settings ---
        CSV_FILE_PATH=Datasets.csv
        DATE_COLUMN=Date           # Name of the date column in your CSV
        VALUE_COLUMN=TOC over Iran # Name of the value column in your CSV
        TIME_SERIES_FREQUENCY=MS   # Pandas frequency string (e.g., MS, M, D, W, H) or leave blank to infer

        # --- Models to Run ---
        MODELS_TO_RUN=SARIMA,Prophet,RNN,LSTM # Comma-separated list of models

        # --- Other Key Settings ---
        TEST_SIZE=24        # Number of periods for the test set
        VALIDATION_SIZE=12  # Number of periods for validation (set to 0 to disable)
        FORECAST_HORIZON=12 # Number of periods for the final future forecast
        # ... review other settings in src/config.py for defaults ...
        ```
    *   Adjust other settings in `.env` as needed (e.g., model parameters, tuner settings, saving options). Refer to `src/config.py` for all available options and their defaults.

2.  **Run the main script:**
    ```bash
    python main.py
    ```

3.  **Check the output:**
    *   Logs will be printed to the console (and optionally to a file specified in `.env`).
    *   Results (metrics, forecasts, parameters, plots, saved models) will be saved in the `results/` directory (if `SAVE_RESULTS=True`).
    *   Plots might be displayed interactively (if `SHOW_PLOTS=True`).

## Configuration Details

The run behavior is controlled by environment variables defined in the `.env` file. Key variables include:

*   **Data:** `CSV_FILE_PATH`, `DATE_COLUMN`, `VALUE_COLUMN`, `TIME_SERIES_FREQUENCY`, `DATA_IMPUTATION_METHOD`.
*   **Splitting:** `TEST_SIZE`, `VALIDATION_SIZE`.
*   **Models:** `MODELS_TO_RUN`, SARIMA parameters (`SARIMA_P`, `SARIMA_D`, etc., `USE_AUTO_ARIMA`), Prophet parameters (`PROPHET_GROWTH`, `PROPHET_ADD_*_SEASONALITY`, etc.), NN parameters (`NN_STEPS`, `NN_UNITS`, `NN_ACTIVATION`, etc., `RNN_EPOCHS`, `LSTM_EPOCHS`).
*   **Tuning:** `USE_KERAS_TUNER`, `NN_TUNER_*` variables (type, trials, objective, hyperparameter ranges).
*   **Evaluation:** `EVALUATION_METRICS`.
*   **Output:** `SAVE_RESULTS`, `RESULTS_DIR`, `SAVE_MODEL_PARAMETERS`, `SAVE_TRAINED_MODELS`, `SAVE_PLOTS`, `PLOT_OUTPUT_FORMAT`, `SHOW_PLOTS`.
*   **Final Forecast:** `RUN_FINAL_FORECAST`, `FORECAST_HORIZON`.
*   **General:** `RANDOM_SEED`, `LOG_LEVEL`, `LOG_FILE`.

*(Refer to `src/config.py` for a complete list and descriptions)*

## Output Structure (`results/` directory)

If saving is enabled, the following structure might be created:

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
*   tensorflow
*   keras-tuner
*   scikit-learn
*   matplotlib
*   python-dotenv
*   joblib

*(See `requirements.txt` for specific versions)*

## Contributing

Contributions are welcome! Please follow standard practices:

1.  Fork the repository.
2.  Create a new branch (`git checkout -b feature/your-feature-name`).
3.  Make your changes.
4.  Ensure code follows existing style and add tests if applicable.
5.  Commit your changes (`git commit -m 'Add some feature'`).
6.  Push to the branch (`git push origin feature/your-feature-name`).
7.  Open a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details (or choose another license if preferred).
