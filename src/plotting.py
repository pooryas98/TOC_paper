# --- START OF FILE src/plotting.py ---

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
from typing import Dict, Optional, Any # Corrected import
import warnings # Added import

def plot_individual_forecasts(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    forecast_dict: Dict[str, pd.DataFrame],
    evaluation_df: pd.DataFrame,
    date_col_name: str,
    value_col_name: str,
    main_title_prefix: str = "Forecast"
) -> None:
    """
    Generates a separate plot figure for each model's forecast results.

    Args:
        train_df (pd.DataFrame): Training data (including validation if used). Must have 'y' column.
        test_df (pd.DataFrame): Actual test data. Must have 'y' column.
        forecast_dict (dict): Dictionary where keys are model names and values are
                              DataFrames containing forecasts. Expected columns in DataFrames:
                              'yhat' (point forecast), and optionally 'yhat_lower', 'yhat_upper'.
        evaluation_df (pd.DataFrame): DataFrame containing evaluation metrics (MAE, RMSE, MAPE),
                                      indexed by model name.
        date_col_name (str): Name of the original date column for axis labels.
        value_col_name (str): Name of the original value column for axis labels.
        main_title_prefix (str): Prefix for the individual plot titles.
    """
    # Filter models that produced valid (non-empty, non-all-NaN 'yhat') forecasts
    active_models: Dict[str, pd.DataFrame] = {
        k: v for k, v in forecast_dict.items()
        if isinstance(v, pd.DataFrame) and not v.empty and 'yhat' in v.columns and not v['yhat'].isnull().all()
    }
    num_models: int = len(active_models)

    if num_models == 0:
        print("Plotting skipped: No valid forecast data provided.")
        return

    actual_color: str = 'black'
    historical_color: str = 'grey'
    forecast_color: str = 'tab:blue' # Consistent forecast color within each plot
    ci_color: str = 'tab:cyan'     # Consistent CI color

    # --- Generate one plot per model ---
    for model_name, forecast_df_orig in active_models.items():

        # --- Create a New Figure for Each Model ---
        fig, ax = plt.subplots(figsize=(12, 6))

        # *** Diagnostic Print for Prophet ***
        # if model_name == 'Prophet':
        #     print(f"\n--- Plotting Data for Prophet ---")
        #     print("Forecast DataFrame Head:")
        #     print(forecast_df_orig.head())
        #     print("\nTest DataFrame Head:")
        #     print(test_df.head())
        #     print(f"Forecast Index Type: {type(forecast_df_orig.index)}")
        #     print(f"Test Index Type: {type(test_df.index)}")
        #     print(f"Forecast Index Freq: {forecast_df_orig.index.freqstr}")
        #     print(f"Test Index Freq: {test_df.index.freqstr}")
        #     print("-" * 30)


        # Ensure index alignment before plotting
        # Reindex forecast_df to match test_df's index strictly for plotting this specific model
        forecast_df = forecast_df_orig.reindex(test_df.index)

        # Check for NaNs introduced by reindexing (if forecast index was slightly off)
        if forecast_df['yhat'].isnull().all():
            warnings.warn(f"Plotting Warning ({model_name}): All 'yhat' values became NaN after reindexing to test_df index. Skipping plot for this model.")
            plt.close(fig) # Close the empty figure
            continue

        # Get MAE for title, handle potential KeyError or NaN
        mae_metric: float = np.nan
        title_suffix: str = ""
        if not evaluation_df.empty and model_name in evaluation_df.index:
            try:
                mae_metric = evaluation_df.loc[model_name, 'MAE']
                if pd.notna(mae_metric):
                    title_suffix = f" (MAE: {mae_metric:.2f})"
            except KeyError:
                pass # MAE not found for this model

        plot_title: str = f"{main_title_prefix}: {model_name}{title_suffix}"
        ax.set_title(plot_title, fontsize=14)

        # Plot recent historical data (e.g., last N periods before test)
        history_periods: int = len(test_df) * 3 # Show history 3x the test length
        # Ensure train_df has 'y' column
        if 'y' in train_df.columns:
            recent_train_df: pd.DataFrame = train_df.iloc[-history_periods:]
            ax.plot(recent_train_df.index, recent_train_df['y'], label='Historical', color=historical_color, alpha=0.7, linewidth=1.5)
        else:
             warnings.warn("Plotting Warning: 'y' column not found in train_df for historical plot.")


        # Plot actual test data
        if 'y' in test_df.columns:
            ax.plot(test_df.index, test_df['y'], label='Actual Test Data', color=actual_color, linewidth=2, marker='.', markersize=8)
        else:
            warnings.warn("Plotting Warning: 'y' column not found in test_df for actuals plot.")


        # Plot point forecast (use the reindexed df)
        ax.plot(forecast_df.index, forecast_df['yhat'], label=f'{model_name} Forecast', color=forecast_color, linestyle='--', linewidth=2)

        # Plot confidence intervals if available and valid in the reindexed df
        plot_ci = False
        if 'yhat_lower' in forecast_df.columns and 'yhat_upper' in forecast_df.columns:
             # Check if CI columns are numeric and not all NaN after reindex
             lower_valid = pd.api.types.is_numeric_dtype(forecast_df['yhat_lower']) and not forecast_df['yhat_lower'].isnull().all()
             upper_valid = pd.api.types.is_numeric_dtype(forecast_df['yhat_upper']) and not forecast_df['yhat_upper'].isnull().all()
             if lower_valid and upper_valid:
                 plot_ci = True

        if plot_ci:
            try:
                ax.fill_between(forecast_df.index,
                                forecast_df['yhat_lower'],
                                forecast_df['yhat_upper'],
                                color=ci_color, alpha=0.3, label='95% Confidence Interval')
            except Exception as fill_err:
                 # This might happen if CI data is badly formed after reindex
                 warnings.warn(f"Plotting Warning ({model_name}): Could not plot confidence interval. Error: {fill_err}")
        elif 'yhat_lower' in forecast_df_orig.columns: # Check original df if reindexed failed CI
             # Warn if CIs existed but couldn't be plotted
             warnings.warn(f"Plotting Warning ({model_name}): Confidence interval columns exist but could not be plotted (possibly due to NaNs after reindex or non-numeric type).")


        ax.set_xlabel(date_col_name, fontsize=10)
        ax.set_ylabel(value_col_name, fontsize=10)
        ax.grid(True, linestyle=':', alpha=0.7)

        # Adjust legend location if CI is plotted vs not
        legend_loc = 'upper left'
        ax.legend(loc=legend_loc, fontsize=9)

        # Improve date formatting
        try:
            locator = mdates.AutoDateLocator(minticks=5, maxticks=12)
            formatter = mdates.ConciseDateFormatter(locator)
            ax.xaxis.set_major_locator(locator)
            ax.xaxis.set_major_formatter(formatter)
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=15, ha='right') # Rotate labels
        except Exception as date_fmt_err:
            warnings.warn(f"Plotting Warning ({model_name}): Could not apply date formatting. Error: {date_fmt_err}")


        plt.tight_layout()
        # Only show plot if it wasn't skipped
        plt.show()


def plot_residuals_comparison(
    test_df: pd.DataFrame,
    forecast_dict: Dict[str, pd.DataFrame],
    date_col_name: str
) -> None:
    """
    Generates a single separate plot comparing the residuals of all models.

    Args:
        test_df (pd.DataFrame): Actual test data. Must have 'y' column.
        forecast_dict (dict): Dictionary where keys are model names and values are
                              DataFrames containing forecasts ('yhat' column required).
        date_col_name (str): Name of the original date column for axis labels.
    """
    # Filter models that produced valid (non-empty, non-all-NaN 'yhat') forecasts
    active_models: Dict[str, pd.DataFrame] = {
        k: v for k, v in forecast_dict.items()
        if isinstance(v, pd.DataFrame) and not v.empty and 'yhat' in v.columns and not v['yhat'].isnull().all()
    }
    num_models: int = len(active_models)

    if num_models == 0:
        print("Residual plot skipped: No valid forecast data.")
        return

    # --- Create a Single Figure for Residuals ---
    fig_res, ax_res = plt.subplots(figsize=(12, 5))
    ax_res.set_title("Comparison of Forecast Residuals (Actual - Forecast)", fontsize=14)
    ax_res.set_ylabel("Residual", fontsize=10)
    ax_res.set_xlabel(date_col_name, fontsize=10)

    # Define colors and markers
    # Use a colormap that provides distinct colors
    colors = plt.cm.get_cmap('tab10', num_models) if num_models <= 10 else plt.cm.get_cmap('viridis', num_models)
    model_colors: Dict[str, Any] = {model_name: colors(i) for i, model_name in enumerate(active_models.keys())}
    residual_styles: list[str] = ['o', 's', '^', 'd', 'v', '<', '>'] # Different markers

    # Horizontal line at zero
    ax_res.axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.7)

    # Plot residuals for each model
    model_items = list(active_models.items()) # Ensure consistent order
    plotted_something = False
    for i in range(num_models):
        model_name, forecast_df_orig = model_items[i]
        color = model_colors[model_name]
        marker = residual_styles[i % len(residual_styles)]

        # Align forecast 'yhat' with test data 'y' and calculate residuals
        # Ensure test_df has 'y'
        if 'y' not in test_df.columns:
             warnings.warn(f"Residual Plot Warning ({model_name}): 'y' column not found in test_df. Skipping residual calculation.")
             continue

        y_pred_aligned: pd.Series = forecast_df_orig['yhat'].reindex(test_df.index)
        residuals: pd.Series = test_df['y'] - y_pred_aligned

        # Plot only non-NaN residuals
        valid_residuals: pd.Series = residuals.dropna()
        if not valid_residuals.empty:
             ax_res.plot(valid_residuals.index, valid_residuals, label=f'{model_name}',
                         color=color, linestyle='None', marker=marker, markersize=5, alpha=0.8) # Markers only
             plotted_something = True
        else:
             warnings.warn(f"Residual Plot Warning ({model_name}): No valid residuals to plot after alignment/dropna.")


    if not plotted_something:
        print("Residual plot skipped: No valid residuals found for any model.")
        plt.close(fig_res)
        return

    ax_res.grid(True, linestyle=':', alpha=0.6)
    # Only add legend if there are labels to show
    handles, labels = ax_res.get_legend_handles_labels()
    if handles:
        ax_res.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=min(num_models, 4), fontsize=9, frameon=False) # Legend below plot

    # Improve date formatting
    try:
        locator = mdates.AutoDateLocator(minticks=5, maxticks=12)
        formatter = mdates.ConciseDateFormatter(locator)
        ax_res.xaxis.set_major_locator(locator)
        ax_res.xaxis.set_major_formatter(formatter)
        plt.setp(ax_res.xaxis.get_majorticklabels(), rotation=15, ha='right') # Rotate labels
    except Exception as date_fmt_err:
            warnings.warn(f"Residual Plot Warning: Could not apply date formatting. Error: {date_fmt_err}")


    plt.tight_layout(rect=[0, 0.1, 1, 0.95]) # Adjust layout for bottom legend
    plt.show() # Display the residuals comparison plot


# --- Main plotting function called by main.py ---
def plot_forecast_comparison(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    forecast_dict: Dict[str, pd.DataFrame],
    evaluation_df: pd.DataFrame,
    date_col_name: str,
    value_col_name: str,
    title: str = "Time Series Forecasting"
) -> None:
    """
    Orchestrates the plotting by calling functions for individual forecast plots
    and a combined residuals plot.

    Args:
        train_df (pd.DataFrame): Training data (including validation if used).
        test_df (pd.DataFrame): Actual test data.
        forecast_dict (dict): Dictionary containing forecast DataFrames for each model.
        evaluation_df (pd.DataFrame): DataFrame containing evaluation metrics.
        date_col_name (str): Name of the original date column.
        value_col_name (str): Name of the original value column.
        title (str): Base title prefix for individual plots.
    """

    # 1. Generate individual forecast plots
    print("Generating individual forecast plots...")
    plot_individual_forecasts(
        train_df=train_df.copy(), # Pass copies to avoid modifying originals
        test_df=test_df.copy(),
        forecast_dict=forecast_dict, # Pass original dict, plotting func handles reindexing
        evaluation_df=evaluation_df,
        date_col_name=date_col_name,
        value_col_name=value_col_name,
        main_title_prefix=title # Use the passed title as prefix
    )

    # 2. Generate the combined residuals plot
    print("Generating residuals comparison plot...")
    plot_residuals_comparison(
        test_df=test_df.copy(),
        forecast_dict=forecast_dict,
        date_col_name=date_col_name
    )
    print("Plotting finished.")
# --- END OF FILE src/plotting.py ---