# Time Series Forecasting Model Comparison

This project provides a framework for comparing the performance of popular time series forecasting models, including SARIMA, Prophet, RNN, and LSTM, on your own data. It streamlines the process from data loading and preparation to model training, evaluation, and visualization.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) <!-- Optional: Replace MIT if you chose another license -->

## Key Features

*   **Multiple Models:** Implements and compares:
    *   SARIMA (Statistical model with optional automatic order selection via `pmdarima`)
    *   Prophet (Facebook's robust forecasting tool)
    *   SimpleRNN (Recurrent Neural Network)
    *   LSTM (Long Short-Term Memory Network)
*   **Hyperparameter Tuning:** Integrates `KerasTuner` for automated optimization of RNN/LSTM hyperparameters (optional).
*   **Comprehensive Evaluation:** Calculates standard metrics (MAE, RMSE, MAPE) and model runtimes.
*   **Clear Visualizations:**
    *   Generates individual plots for each model's forecast against actual values, including confidence intervals where available (SARIMA, Prophet).
    *   Creates a comparative plot of forecast residuals across all models.
*   **Flexible Configuration:** Easily configure data paths, model parameters, train/validation/test splits, and execution options via a `.env` file.
*   **Reproducibility:** Supports setting random seeds for consistent results.
*   **Organized Output:** Optionally saves evaluation metrics and detailed forecast results to CSV files.

## Getting Started

Follow these steps to set up and run the project:

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/YourUsername/YourRepositoryName.git # Replace with your repo URL
    cd YourRepositoryName
    ```

2.  **Create and Activate a Virtual Environment:** (Highly Recommended)
    ```bash
    # Linux/macOS
    python3 -m venv .venv
    source .venv/bin/activate

    # Windows (cmd/powershell)
    python -m venv .venv
    .\.venv\Scripts\activate
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *   *Note:* Installing `prophet` and `tensorflow` can sometimes require specific system libraries or versions. Consult their official documentation if you encounter installation errors.

4.  **Prepare Configuration:**
    *   Copy the example environment file:
        ```bash
        cp .env.example .env
        ```
    *   **Edit the `.env` file:** Open the newly created `.env` file in a text editor.
    *   **Crucially, update `CSV_FILE_PATH`, `DATE_COLUMN`, and `VALUE_COLUMN`** to match your dataset.
    *   Adjust other parameters (model settings, split sizes, tuning options) as needed. See the comments within `.env.example` for details on each variable.
    *   **Important:** The `.env` file is ignored by Git (via `.gitignore`) and should **never** be committed to version control, especially if it contains sensitive information.

## How to Run

Ensure your virtual environment is activated and you are in the project's root directory. Execute the main script:

```bash
python main.py
