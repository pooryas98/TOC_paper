import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
import os # For saving plots
import logging # Use logging
from typing import Dict, Optional, Any, List # Added List

logger = logging.getLogger(__name__)

# --- MODIFICATION START ---
# Add config_params argument definition
def _plot_single_model(
    ax: plt.Axes,
    model_name: str,
    train_df: pd.DataFrame, # Combined train+val
    test_df: pd.DataFrame,
    forecast_df_orig: pd.DataFrame, # Original forecast DF
    evaluation_df: pd.DataFrame,
    config_params: Dict[str, Any], # <<< ADD THIS ARGUMENT
    date_col_name: str,
    value_col_name: str,
    ci_alpha: float = 0.3,
    history_multiplier: int = 3
):
# --- MODIFICATION END ---
    """Helper function to plot a single model's forecast on a given Axes object (for evaluation plots)."""
    actual_color: str = 'black'
    historical_color: str = 'grey'
    forecast_color: str = 'tab:blue' # Consistent forecast color within each plot
    ci_color: str = 'tab:cyan'     # Consistent CI color

    # Reindex forecast_df to match test_df's index strictly for plotting this specific model
    forecast_df = forecast_df_orig.reindex(test_df.index)

    if forecast_df['yhat'].isnull().all():
        logger.warning(f"Plotting Warning ({model_name}): All 'yhat' values became NaN after reindexing to test_df index. Skipping plot content for this model.")
        ax.text(0.5, 0.5, f"{model_name}\nForecast Incompatible\nwith Test Index",
                horizontalalignment='center', verticalalignment='center', transform=ax.transAxes,
                fontsize=12, color='red')
        return # Don't plot anything else for this model

    # Get MAE for title, handle potential errors
    mae_metric: float = np.nan
    title_suffix: str = ""
    if not evaluation_df.empty and model_name in evaluation_df.index:
        try:
            if 'MAE' in evaluation_df.columns:
                mae_metric = evaluation_df.loc[model_name, 'MAE']
                if pd.notna(mae_metric):
                    title_suffix = f" (MAE: {mae_metric:.2f})"
            else:
                 logger.debug(f"MAE column not found in evaluation_df for model {model_name}")
        except KeyError:
            logger.debug(f"Model {model_name} not found in evaluation_df index")
        except Exception as e:
            logger.warning(f"Error retrieving MAE for {model_name}: {e}")


    plot_title: str = f"{model_name} Forecast{title_suffix}"
    ax.set_title(plot_title, fontsize=12) # Smaller title for subplots

    # Plot recent historical data (from train_df, which includes train+val)
    history_periods: int = len(test_df) * history_multiplier
    if 'y' in train_df.columns:
        recent_train_df: pd.DataFrame = train_df.iloc[-history_periods:]
        ax.plot(recent_train_df.index, recent_train_df['y'], label='Historical', color=historical_color, alpha=0.7, linewidth=1.5)
    else:
         logger.warning("Plotting Warning: 'y' column not found in train_df for historical plot.")

    # Plot actual test data
    if 'y' in test_df.columns:
        ax.plot(test_df.index, test_df['y'], label='Actual Test Data', color=actual_color, linewidth=2, marker='.', markersize=6)
    else:
        logger.warning("Plotting Warning: 'y' column not found in test_df for actuals plot.")

    # Plot point forecast (use the reindexed df)
    ax.plot(forecast_df.index, forecast_df['yhat'], label=f'Forecast', color=forecast_color, linestyle='--', linewidth=2)

    # Plot confidence intervals if available and valid
    plot_ci = False
    if 'yhat_lower' in forecast_df.columns and 'yhat_upper' in forecast_df.columns:
         lower_valid = pd.api.types.is_numeric_dtype(forecast_df['yhat_lower']) and not forecast_df['yhat_lower'].isnull().all()
         upper_valid = pd.api.types.is_numeric_dtype(forecast_df['yhat_upper']) and not forecast_df['yhat_upper'].isnull().all()
         if lower_valid and upper_valid:
             plot_ci = True

    if plot_ci:
        try:
            # Calculate CI width dynamically for label (assuming symmetric alpha)
            # This assumes the interval width matches the alpha used for summary_frame/predict
            # Defaulting to 95% if Prophet's interval_width isn't directly available here
            # --- MODIFICATION START ---
            # Use passed config_params
            ci_width_pct = config_params.get('PROPHET_INTERVAL_WIDTH', 0.95) * 100
            # --- MODIFICATION END ---
            ax.fill_between(forecast_df.index, forecast_df['yhat_lower'], forecast_df['yhat_upper'],
                            color=ci_color, alpha=ci_alpha, label=f'{ci_width_pct:.0f}% CI')
        except Exception as fill_err:
             logger.warning(f"Plotting Warning ({model_name}): Could not plot confidence interval. Error: {fill_err}")
    elif 'yhat_lower' in forecast_df_orig.columns: # Check original DF
         logger.debug(f"Plotting Debug ({model_name}): Confidence interval columns exist but could not be plotted (NaNs after reindex or non-numeric?).")

    ax.set_xlabel(date_col_name, fontsize=9)
    ax.set_ylabel(value_col_name, fontsize=9)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='upper left', fontsize=8)

    # Date formatting
    try:
        locator = mdates.AutoDateLocator(minticks=4, maxticks=8)
        formatter = mdates.ConciseDateFormatter(locator)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=15, ha='right', fontsize=8)
        plt.setp(ax.yaxis.get_majorticklabels(), fontsize=8)
    except Exception as date_fmt_err:
        logger.warning(f"Plotting Warning ({model_name}): Could not apply date formatting. Error: {date_fmt_err}")


def plot_forecast_comparison(
    train_df: pd.DataFrame, # Combined train+val for history
    test_df: pd.DataFrame, # Test set for actuals
    forecast_dict: Dict[str, pd.DataFrame], # Forecasts corresponding to test_df
    evaluation_df: pd.DataFrame, # Assumes index is Model name
    config_params: Dict[str, Any], # Pass config
    main_title: str = "Forecast Comparison (Evaluation Run)"
) -> None:
    """
    Generates combined and individual plots for forecast evaluation comparison.
    Handles saving and showing plots based on config.
    """
    logger.info("--- Generating Evaluation Forecast Plots ---")

    save_plots = config_params.get('SAVE_PLOTS', False)
    show_plots = config_params.get('SHOW_PLOTS', True)
    plot_format = config_params.get('PLOT_OUTPUT_FORMAT', 'png')
    results_dir = config_params.get('RESULTS_DIR', 'results')
    plots_dir = os.path.join(results_dir, 'plots')
    date_col_name = config_params.get('DATE_COLUMN', 'Date')
    value_col_name = config_params.get('VALUE_COLUMN', 'Value')


    if save_plots:
        try:
            os.makedirs(plots_dir, exist_ok=True)
            logger.info(f"Saving evaluation plots to directory: {plots_dir}")
        except OSError as e:
            logger.error(f"Could not create plots directory '{plots_dir}'. Saving disabled. Error: {e}")
            save_plots = False # Disable saving if dir creation fails

    # Filter models with valid forecast DataFrames containing 'yhat' for the evaluation period
    active_models: Dict[str, pd.DataFrame] = {
        k: v for k, v in forecast_dict.items()
        if isinstance(v, pd.DataFrame) and not v.empty and 'yhat' in v.columns and not v['yhat'].isnull().all()
    }
    num_models: int = len(active_models)

    if num_models == 0:
        logger.warning("Evaluation plotting skipped: No valid forecast data found in forecast_dict.")
        return

    # Ensure test data has 'y' column
    if 'y' not in test_df.columns:
         logger.error("Evaluation plotting error: 'y' column not found in test_df. Cannot plot actuals.")
         # Decide if we should proceed without actuals or stop
         # For now, let's stop if actuals are missing for evaluation plots
         return


    # --- 1. Combined Overview Plot (All Forecasts vs Actual Test) ---
    logger.info("Generating combined evaluation forecast overview plot...")
    fig_ov, ax_ov = plt.subplots(figsize=(14, 7))
    fig_ov.suptitle(main_title, fontsize=16) # Use passed title

    # Use a perceptually uniform colormap
    colors = plt.cm.get_cmap('viridis', num_models) if num_models > 1 else ['tab:blue']
    model_colors: Dict[str, Any] = {name: colors(i) for i, name in enumerate(active_models.keys())}


    # Plot History (Train+Val) and Actuals (Test)
    history_periods: int = len(test_df) * 3 # Show history relative to test period length
    if 'y' in train_df.columns:
        recent_train_df = train_df.iloc[-history_periods:]
        ax_ov.plot(recent_train_df.index, recent_train_df['y'], label='Historical', color='grey', alpha=0.7, linewidth=1.5)
    if 'y' in test_df.columns:
        ax_ov.plot(test_df.index, test_df['y'], label='Actual Test Data', color='black', linewidth=2.5, marker='o', markersize=4, zorder=num_models + 2) # Ensure actuals are on top

    # Plot each model's forecast for the test period
    for i, (model_name, forecast_df_orig) in enumerate(active_models.items()):
        forecast_df = forecast_df_orig.reindex(test_df.index) # Align strictly to test index
        if not forecast_df['yhat'].isnull().all():
            ax_ov.plot(forecast_df.index, forecast_df['yhat'], label=f'{model_name}', color=model_colors[model_name], linestyle='--', linewidth=1.8, zorder=i+1)
        else:
             logger.warning(f"Combined evaluation plot: Skipping {model_name} due to NaN forecast after reindex.")

    ax_ov.set_xlabel(date_col_name, fontsize=10)
    ax_ov.set_ylabel(value_col_name, fontsize=10)
    ax_ov.grid(True, linestyle=':', alpha=0.6)
    ax_ov.legend(loc='upper left', fontsize=9)
    ax_ov.tick_params(axis='x', rotation=15, labelsize=9)
    ax_ov.tick_params(axis='y', labelsize=9)
    # Adjust x-axis limits to focus on recent history and test period
    combined_index = recent_train_df.index.union(test_df.index)
    ax_ov.set_xlim([combined_index.min(), combined_index.max()])
    # Date formatting
    try:
        locator = mdates.AutoDateLocator(minticks=6, maxticks=12); formatter = mdates.ConciseDateFormatter(locator)
        ax_ov.xaxis.set_major_locator(locator); ax_ov.xaxis.set_major_formatter(formatter)
    except Exception as e:
         logger.warning(f"Evaluation Overview Plot: Could not apply date formatting: {e}")


    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout for suptitle

    if save_plots:
        figname = os.path.join(plots_dir, f"evaluation_combined_forecast.{plot_format}")
        try:
            fig_ov.savefig(figname, dpi=150) # Increase DPI slightly for better quality
            logger.info(f"Saved combined evaluation overview plot to {figname}")
        except Exception as e:
            logger.error(f"Failed to save combined evaluation overview plot: {e}")
    if show_plots: plt.show()
    plt.close(fig_ov) # Close figure


    # --- 2. Individual Model Plots (Evaluation) ---
    logger.info("Generating individual evaluation forecast plots...")
    for model_name, forecast_df_orig in active_models.items():
        fig_ind, ax_ind = plt.subplots(figsize=(10, 5)) # Smaller figure for individual plots
        # --- MODIFICATION START ---
        # Pass config_params to the helper function
        _plot_single_model(
            ax=ax_ind, model_name=model_name, train_df=train_df, test_df=test_df,
            forecast_df_orig=forecast_df_orig, evaluation_df=evaluation_df,
            config_params=config_params, # <<< PASS THE ARGUMENT HERE
            date_col_name=date_col_name, value_col_name=value_col_name
            )
        # --- MODIFICATION END ---
        # Adjust x-axis limits for individual plots as well
        ind_combined_index = train_df.iloc[-len(test_df)*3:].index.union(test_df.index)
        ax_ind.set_xlim([ind_combined_index.min(), ind_combined_index.max()])
        plt.tight_layout()

        if save_plots:
            figname = os.path.join(plots_dir, f"evaluation_forecast_{model_name}.{plot_format}")
            try:
                fig_ind.savefig(figname, dpi=120)
                logger.info(f"Saved individual evaluation plot for {model_name} to {figname}")
            except Exception as e:
                 logger.error(f"Failed to save individual evaluation plot for {model_name}: {e}")
        if show_plots: plt.show()
        plt.close(fig_ind) # Close figure


    # --- 3. Residuals Plot (Evaluation) ---
    logger.info("Generating evaluation residuals comparison plot...")
    fig_res, ax_res = plt.subplots(figsize=(12, 5))
    ax_res.set_title("Evaluation Forecast Residuals (Actual Test - Forecast)", fontsize=14)
    ax_res.set_ylabel("Residual", fontsize=10)
    ax_res.set_xlabel(date_col_name, fontsize=10)
    ax_res.axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.7)

    # Use consistent colors from overview plot
    residual_styles = ['o', 's', '^', 'd', 'v', '<', '>','*','+','x'] # Markers

    plotted_residuals = False
    if 'y' not in test_df.columns:
        logger.error("Residual plot error: 'y' column not found in test_df. Skipping.")
    else:
        for i, (model_name, forecast_df_orig) in enumerate(active_models.items()):
            color = model_colors[model_name]
            marker = residual_styles[i % len(residual_styles)]

            y_pred_aligned: pd.Series = forecast_df_orig['yhat'].reindex(test_df.index)
            residuals: pd.Series = test_df['y'] - y_pred_aligned
            valid_residuals: pd.Series = residuals.dropna()

            if not valid_residuals.empty:
                 ax_res.plot(valid_residuals.index, valid_residuals, label=f'{model_name}',
                             color=color, linestyle='None', marker=marker, markersize=5, alpha=0.8)
                 plotted_residuals = True
            else:
                 logger.warning(f"Evaluation Residual Plot Warning ({model_name}): No valid residuals to plot after alignment/dropna.")

    if plotted_residuals:
        ax_res.grid(True, linestyle=':', alpha=0.6)
        handles, labels = ax_res.get_legend_handles_labels()
        if handles:
            # Adjust legend position for residuals plot
            ax_res.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=min(num_models, 5), fontsize=9, frameon=False)

        try:
            locator = mdates.AutoDateLocator(minticks=5, maxticks=12)
            formatter = mdates.ConciseDateFormatter(locator)
            ax_res.xaxis.set_major_locator(locator)
            ax_res.xaxis.set_major_formatter(formatter)
            plt.setp(ax_res.xaxis.get_majorticklabels(), rotation=15, ha='right', fontsize=9)
            plt.setp(ax_res.yaxis.get_majorticklabels(), fontsize=9)
        except Exception as date_fmt_err:
             logger.warning(f"Evaluation Residual Plot Warning: Could not apply date formatting. Error: {date_fmt_err}")

        plt.tight_layout(rect=[0, 0.1, 1, 0.95]) # Adjust layout for legend

        if save_plots:
             figname = os.path.join(plots_dir, f"evaluation_residuals_comparison.{plot_format}")
             try:
                 fig_res.savefig(figname, dpi=120)
                 logger.info(f"Saved evaluation residuals comparison plot to {figname}")
             except Exception as e:
                  logger.error(f"Failed to save evaluation residuals comparison plot: {e}")
        if show_plots: plt.show()
    else:
         logger.warning("Evaluation residual plot skipped: No models had valid residuals to plot.")

    plt.close(fig_res) # Close figure
    logger.info("--- Evaluation Plotting Finished ---")


def plot_future_forecasts(
    historical_df: pd.DataFrame, # Full historical data (df)
    future_forecast_dict: Dict[str, pd.DataFrame],
    config_params: Dict[str, Any],
    main_title: str = "Future Forecast"
) -> None:
    """
    Generates plots showing historical data and future forecasts.
    """
    logger.info("--- Generating Future Forecast Plots ---")

    save_plots = config_params.get('SAVE_PLOTS', False)
    show_plots = config_params.get('SHOW_PLOTS', True)
    plot_format = config_params.get('PLOT_OUTPUT_FORMAT', 'png')
    results_dir = config_params.get('RESULTS_DIR', 'results')
    plots_dir = os.path.join(results_dir, 'plots') # Save in the same plots dir
    date_col_name = config_params.get('DATE_COLUMN', 'Date')
    value_col_name = config_params.get('VALUE_COLUMN', 'Value')
    history_multiplier = 5 # Show N times the forecast horizon as history

    if save_plots:
        try:
            os.makedirs(plots_dir, exist_ok=True)
            logger.info(f"Saving future forecast plots to directory: {plots_dir}")
        except OSError as e:
            logger.error(f"Could not create plots directory '{plots_dir}'. Saving disabled. Error: {e}")
            save_plots = False

    # Filter models with valid future forecast DataFrames
    active_models: Dict[str, pd.DataFrame] = {
        k: v for k, v in future_forecast_dict.items()
        if isinstance(v, pd.DataFrame) and not v.empty and 'yhat' in v.columns and not v['yhat'].isnull().all()
    }
    num_models: int = len(active_models)

    if num_models == 0:
        logger.warning("Future forecast plotting skipped: No valid forecast data found.")
        return

    if historical_df.empty or 'y' not in historical_df.columns:
        logger.warning("Future forecast plotting skipped: Historical data is empty or missing 'y' column.")
        return

    # Determine forecast horizon from the first valid forecast df
    first_valid_forecast = next(iter(active_models.values()))
    forecast_horizon = len(first_valid_forecast)
    history_periods = forecast_horizon * history_multiplier

    # --- Combined Future Forecast Plot ---
    logger.info("Generating combined future forecast plot...")
    fig_future, ax_future = plt.subplots(figsize=(14, 7))
    # Add dataset info to title
    dataset_name = os.path.basename(config_params.get('CSV_FILE_PATH', 'Unknown Data'))
    fig_future.suptitle(f"{main_title}\n({dataset_name} - {value_col_name})", fontsize=16)

    # Use consistent colors (can reuse colormap generation)
    colors = plt.cm.get_cmap('viridis', num_models) if num_models > 1 else ['tab:blue']
    model_colors: Dict[str, Any] = {name: colors(i) for i, name in enumerate(active_models.keys())}

    # Plot Recent History (tail of the full dataset)
    recent_history_df = historical_df.iloc[-history_periods:]
    ax_future.plot(recent_history_df.index, recent_history_df['y'], label='Historical Data', color='black', linewidth=2, alpha=0.8)

    # Plot each model's future forecast
    for i, (model_name, forecast_df) in enumerate(active_models.items()):
        # Ensure index is datetime if needed (should be from main.py)
        ax_future.plot(forecast_df.index, forecast_df['yhat'], label=f'{model_name} Forecast', color=model_colors[model_name], linestyle='--', linewidth=1.8, zorder=i+1)

        # Optionally plot confidence intervals if available and not all NaN
        if 'yhat_lower' in forecast_df.columns and 'yhat_upper' in forecast_df.columns:
             if not forecast_df['yhat_lower'].isnull().all() and not forecast_df['yhat_upper'].isnull().all():
                 try:
                     # Use a consistent alpha for CIs
                     ci_alpha_fill = 0.2
                     # Label CI simply or get width if possible (e.g., from prophet params if stored)
                     ci_label = f'{model_name} CI'
                     ax_future.fill_between(forecast_df.index, forecast_df['yhat_lower'], forecast_df['yhat_upper'],
                                            color=model_colors[model_name], alpha=ci_alpha_fill, label=ci_label)
                 except Exception as fill_err:
                     logger.warning(f"Future Plot Warning ({model_name}): Could not plot confidence interval. Error: {fill_err}")


    ax_future.set_xlabel(date_col_name, fontsize=10)
    ax_future.set_ylabel(value_col_name, fontsize=10)
    ax_future.grid(True, linestyle=':', alpha=0.6)
    # Adjust legend - might need more space if CIs are plotted per model
    ax_future.legend(loc='upper left', fontsize=9)
    ax_future.tick_params(axis='x', rotation=15, labelsize=9)
    ax_future.tick_params(axis='y', labelsize=9)

    # Adjust x-axis limits to show history and forecast smoothly
    full_index = recent_history_df.index.union(first_valid_forecast.index)
    ax_future.set_xlim([full_index.min(), full_index.max()])

    # Date formatting
    try:
        locator = mdates.AutoDateLocator(minticks=6, maxticks=12)
        formatter = mdates.ConciseDateFormatter(locator)
        ax_future.xaxis.set_major_locator(locator)
        ax_future.xaxis.set_major_formatter(formatter)
    except Exception as date_fmt_err:
        logger.warning(f"Future Plot Warning: Could not apply date formatting. Error: {date_fmt_err}")


    plt.tight_layout(rect=[0, 0.03, 1, 0.93]) # Adjust layout for suptitle

    if save_plots:
        figname = os.path.join(plots_dir, f"future_forecast_combined.{plot_format}")
        try:
            fig_future.savefig(figname, dpi=150)
            logger.info(f"Saved combined future forecast plot to {figname}")
        except Exception as e:
            logger.error(f"Failed to save combined future forecast plot: {e}")
    if show_plots: plt.show()
    plt.close(fig_future)

    logger.info("--- Future Forecast Plotting Finished ---")