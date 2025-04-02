# Multi-Model Time Series Forecasting Framework

A Python framework for running, comparing, and evaluating multiple time series forecasting models (SARIMA, Prophet, RNN, LSTM) on univariate data.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
  - [Prerequisites](#prerequisites)
  - [Setup Steps](#setup-steps)
- [Configuration](#configuration)
- [Usage](#usage)
- [Getting Started (Quick Start)](#getting-started-quick-start)
- [Project Structure](#project-structure)
- [Output](#output)
- [Contributing](#contributing)
- [License](#license)
- [Support](#support)

## Overview

This project provides a structured way to apply and evaluate several popular time series forecasting models to your own univariate time series data. It handles data loading, preprocessing (imputation, frequency setting), model training (including optional hyperparameter tuning), evaluation on a hold-out test set, and generation of future forecasts. The results, including metrics, forecasts, plots, and model parameters, are systematically saved for analysis.

## Features

*   **Multiple Models:** Supports SARIMA, Facebook Prophet, Simple RNN, and LSTM models.
*   **Automated Workflow:** Runs the entire pipeline from data loading to evaluation and plotting with a single command.
*   **Configurable:** Easily configure data paths, model parameters, evaluation metrics, and output options via environment variables or a `.env` file.
*   **Data Preprocessing:** Includes options for handling missing values (imputation) and setting time series frequency.
*   **Hyperparameter Tuning:**
    *   Optional auto-ARIMA search for SARIMA orders using `pmdarima`.
    *   Optional hyperparameter tuning for RNN/LSTM models using `keras-tuner` (RandomSearch, Hyperband, BayesianOptimization).
*   **Evaluation:** Calculates standard forecasting metrics (MAE, RMSE, MAPE) on a test set.
*   **Visualization:** Generates plots comparing model forecasts against actual values, pairwise model comparisons, residual analysis, and future forecast visualizations using `matplotlib`.
*   **Results Saving:** Saves evaluation metrics, point forecasts, full forecasts (including confidence intervals where applicable), run parameters, and optionally trained models for reproducibility and further analysis.
*   **Future Forecasting:** Option to retrain models on the full dataset and generate forecasts for a specified future horizon.

## Technologies Used

*   **Programming Language:** Python 3
*   **Core Libraries:**
    *   `pandas`: Data manipulation and time series handling.
    *   `numpy`: Numerical operations.
    *   `statsmodels`: SARIMA model implementation.
    *   `prophet`: Facebook Prophet model implementation.
    *   `tensorflow` / `keras`: RNN and LSTM model implementation.
    *   `scikit-learn`: Data scaling (MinMaxScaler) and evaluation metrics (MAE, MSE).
    *   `matplotlib`: Plotting results.
*   **Optional Tuning Libraries:**
    *   `pmdarima` (auto-ARIMA): For automatic SARIMA order selection.
    *   `keras-tuner`: For RNN/LSTM hyperparameter optimization.
*   **Configuration:** `python-dotenv` (for loading `.env` files).
*   **Serialization:** `pickle`, `joblib` (for saving models/scalers).

## Installation

### Prerequisites

*   Python (>= 3.7 recommended, due to library dependencies)
*   `pip` (Python package installer)
*   `git` (for cloning the repository)

### Setup Steps

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2.  **Create and activate a virtual environment (Recommended):**
    ```bash
    # Linux/macOS
    python3 -m venv venv
    source venv/bin/activate

    # Windows
    python -m venv venv
    .\venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *Note: Installing `prophet` and `tensorflow` can sometimes have specific system dependencies. Refer to their official documentation if you encounter issues.*

4.  **Prepare Configuration:**
    Create a file named `.env` in the root directory of the project. This file defines how the script behaves. See the [Configuration](#configuration) section for details. A minimal `.env` requires:
    ```dotenv
    # --- Required ---
    CSV_FILE_PATH=path/to/your/data.csv
    DATE_COLUMN=YourDateColumnName
    VALUE_COLUMN=YourValueColumnName

    # --- Optional (Example using provided data) ---
    # CSV_FILE_PATH=TOC_Data_Iran_2003.csv
    # DATE_COLUMN=Date
    # VALUE_COLUMN='TOC over Iran'
    # TIME_SERIES_FREQUENCY=MS # Set to 'MS' (Month Start) for the example data
    ```

## Configuration

The framework is configured using environment variables, which can be conveniently managed using a `.env` file in the project's root directory. The `config.py` file loads these variables.

**Key Configuration Variables (in `.env`):**

*   `CSV_FILE_PATH`: (Required) Path to the input CSV data file.
*   `DATE_COLUMN`: (Required) Name of the column containing dates/timestamps.
*   `VALUE_COLUMN`: (Required) Name of the column containing the time series values.
*   `TIME_SERIES_FREQUENCY`: (Optional) Pandas frequency string (e.g., 'D', 'W', 'MS', 'H'). If omitted, the script attempts to infer it. Crucial for models like SARIMA and Prophet.
*   `MODELS_TO_RUN`: (Optional) Comma-separated list of models to run (e.g., `SARIMA,Prophet,RNN,LSTM`). Default includes all four.
*   `TEST_SIZE`: (Optional) Number of periods to use for the test set. Default: `24`.
*   `VALIDATION_SIZE`: (Optional) Number of periods *before* the test set to use for validation (primarily for NN early stopping/tuning). Default: `12`. Set to `0` to disable validation split.
*   `USE_AUTO_ARIMA`: (Optional) `True` or `False` to enable/disable pmdarima's auto_arima for SARIMA. Default: `True`.
*   `USE_KERAS_TUNER`: (Optional) `True` or `False` to enable/disable Keras Tuner for RNN/LSTM. Default: `False`.
*   `SAVE_RESULTS`: (Optional) `True` or `False` to save outputs (metrics, forecasts, plots). Default: `True`.
*   `SAVE_TRAINED_MODELS`: (Optional) `True` or `False` to save the trained model objects (after evaluation run and/or final run). Default: `False`.
*   `RUN_FINAL_FORECAST`: (Optional) `True` or `False` to retrain models on full data and predict future periods. Default: `True`.
*   `FORECAST_HORIZON`: (Optional) Number of periods to forecast into the future if `RUN_FINAL_FORECAST` is `True`. Default: `12`.

*Refer to `config.py` for a complete list of configuration options and their default values, covering model-specific parameters, tuning settings, imputation methods, logging, and plot customization.*

## Usage

1.  Ensure you have installed the dependencies and created your `.env` configuration file as described in [Installation](#installation) and [Configuration](#configuration).
2.  Run the main script from the project's root directory:
    ```bash
    python main.py
    ```
3.  The script will:
    *   Load the configuration.
    *   Load and prepare the data according to the specified CSV path, columns, frequency, and imputation method.
    *   Split the data into training (+ optional validation) and test sets.
    *   Train each model specified in `MODELS_TO_RUN` on the training (+ validation) data.
    *   Generate forecasts for the test period.
    *   Evaluate the forecasts using the specified metrics and print results to the console.
    *   If `SAVE_RESULTS=True`, save evaluation metrics, forecasts, plots, and run parameters to a timestamped subdirectory within `results/` (or the directory specified by `RESULTS_DIR` in `.env`).
    *   If `SAVE_TRAINED_MODELS=True`, save the trained model objects.
    *   If `RUN_FINAL_FORECAST=True`, retrain the models on the entire dataset and generate forecasts for the specified `FORECAST_HORIZON`. These future forecasts and potentially the retrained models are also saved if saving is enabled.

## Getting Started (Quick Start)

This project includes sample data (`TOC_Data_Iran_2003.csv`). To run a quick demo:

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Create a `.env` file** in the project root with the following content:
    ```dotenv
    CSV_FILE_PATH=TOC_Data_Iran_2003.csv
    DATE_COLUMN=Date
    VALUE_COLUMN='TOC over Iran'
    TIME_SERIES_FREQUENCY=MS  # Monthly Start frequency for this dataset
    # Optional: Reduce test/validation size for a quicker run
    TEST_SIZE=12
    VALIDATION_SIZE=6
    # Optional: Disable slower features for a very quick test
    # USE_AUTO_ARIMA=False
    # USE_KERAS_TUNER=False
    # RUN_FINAL_FORECAST=False
    # SAVE_TRAINED_MODELS=False
    # MODELS_TO_RUN=Prophet # Run only one model
    ```
4.  **Run the script:**
    ```bash
    python main.py
    ```
5.  **Check the results:** A new directory will be created inside `results/` (e.g., `results/20231027_103000/`). This directory will contain:
    *   `evaluation_metrics.csv`: Performance metrics for each model.
    *   `point_forecasts.csv`: Model forecasts for the test period.
    *   `run_parameters.json`: Configuration and parameters used for the run.
    *   `plots/`: Directory containing generated plots (if `SAVE_PLOTS=True`).
    *   `full_forecast_*.csv`: Detailed forecasts from the evaluation run (including CI if available).
    *   Potentially `future_point_forecasts.csv`, `future_full_forecast_*.csv` if `RUN_FINAL_FORECAST=True`.
    *   Potentially `saved_models/` and `saved_final_models/` if `SAVE_TRAINED_MODELS=True`.


## Project Structure

The project is organized as follows:

.
├── .env                # Local environment configuration file (Needs to be created)
├── .gitignore          # Specifies intentionally untracked files that Git should ignore
├── main.py             # Main entry point script to run the forecasting comparison
├── requirements.txt    # Lists Python package dependencies for installation
├── TOC_Data_Iran_2003.csv # Example time series data file
│
├── src/                # Source code directory
│   ├── __init__.py
│   ├── config.py         # Handles loading configuration settings
│   ├── data_loader.py    # Functions for loading, cleaning, and splitting data
│   ├── evaluation.py     # Functions for calculating forecast evaluation metrics
│   ├── plotting.py       # Functions for generating forecast visualizations
│   └── models/           # Directory containing different forecasting model implementations
│       ├── __init__.py
│       ├── nn_models.py      # Implementation for RNN and LSTM models (using TensorFlow/Keras)
│       ├── prophet_model.py  # Implementation for the Prophet model
│       └── sarima_model.py   # Implementation for the SARIMA model (using statsmodels/pmdarima)
│
├── results/            # Default directory for output files (created during runtime)
│   └── <run_timestamp>/  # Subdirectory for each specific run, named with a timestamp
│       ├── evaluation_metrics.csv   # CSV with performance metrics
│       ├── point_forecasts.csv      # CSV with test set point forecasts
│       ├── run_parameters.json      # JSON dump of run configuration and model parameters
│       ├── plots/                   # Directory containing generated plots (e.g., .png files)
│       ├── saved_models/            # Optional: Saved models trained on the evaluation split
│       ├── saved_final_models/      # Optional: Saved models trained on the full dataset
│       ├── *.csv                    # Other CSV files (e.g., full forecasts, future forecasts)
│       └── ...
│
└── keras_tuner_dir/    # Default directory for Keras Tuner trial results (created if tuner is used)
    └── ...


## Output

If `SAVE_RESULTS` is enabled (default), the script generates the following outputs in a timestamped subdirectory under the `results` directory:

*   **Console Logs:** Detailed progress, warnings, errors, and evaluation summaries are printed to the console. Log level and file output can be configured via `.env`.
*   `run_parameters.json`: A JSON file containing the configuration settings used for the run, data loading summary, model parameters used/found (including tuning results), and evaluation metrics.
*   `evaluation_metrics.csv`: A CSV file summarizing the performance metrics (MAE, RMSE, MAPE, Runtime) for each model on the test set.
*   `point_forecasts.csv`: A CSV file containing the point forecasts (`yhat`) from each model for the test set period.
*   `full_forecast_[MODEL_NAME].csv`: Separate CSV files for each model containing the detailed forecast for the test period, including confidence/prediction intervals (`yhat_lower`, `yhat_upper`) if available (Prophet, SARIMA).
*   `future_point_forecasts.csv`: (If `RUN_FINAL_FORECAST=True`) CSV file with point forecasts for the future horizon.
*   `future_full_forecast_[MODEL_NAME].csv`: (If `RUN_FINAL_FORECAST=True`) Detailed future forecasts for each model.
*   `plots/`: A subdirectory containing various plots in the specified format (`PLOT_OUTPUT_FORMAT`, default 'png'), including individual model forecasts, pairwise comparisons, residuals, and future forecasts.
*   `saved_models/`: (If `SAVE_TRAINED_MODELS=True`) Subdirectory containing the saved model objects trained during the evaluation phase (using train/validation data). Models are saved as `.pkl` (SARIMA, Prophet), `.keras`/`.h5` (NNs), or `.joblib` (NN scalers).
*   `saved_final_models/`: (If `SAVE_TRAINED_MODELS=True` and `RUN_FINAL_FORECAST=True`) Subdirectory containing the saved model objects trained on the full dataset for final forecasting.

## Contributing

Contributions are welcome! If you'd like to contribute, please follow these steps:

1.  Fork the repository on GitHub.
2.  Create a new branch for your feature or bug fix:
    ```bash
    git checkout -b feature/your-feature-name
    ```
3.  Make your changes and commit them with clear, concise messages:
    ```bash
    git commit -m "Add feature: Describe your changes"
    ```
4.  Push your changes to your forked repository:
    ```bash
    git push origin feature/your-feature-name
    ```
5.  Open a Pull Request (PR) from your branch to the `main` branch of the original repository.
6.  Please ensure your code follows the existing style and includes relevant comments or documentation updates.

## License

The license for this project is not explicitly specified in the provided codebase. Please assume it is under a private license unless otherwise stated by the repository owner.

## Support

*   For issues, bugs, or feature requests, please use the GitHub Issue Tracker associated with this repository.
*   Refer to the code comments and `config.py` for more detailed explanations of specific functions and configuration options.
