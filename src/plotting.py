# src/plotting.py

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.colors as mcolors
import pandas as pd
import numpy as np
import os
import logging
import itertools
from typing import Dict, Optional, Any, List, Tuple

logger = logging.getLogger(__name__)

# --- Configuration Defaults (can be overridden by config_params) ---
DEFAULT_FIG_SIZE_SINGLE = (10, 5)
DEFAULT_FIG_SIZE_PAIRWISE = (11, 6) # Slightly wider for potentially more legend items
DEFAULT_FIG_SIZE_RESIDUAL = (12, 5)
DEFAULT_FIG_SIZE_FUTURE = (14, 7)
DEFAULT_DPI = 150
DEFAULT_HISTORY_MULTIPLIER = 3
DEFAULT_FONTSIZE_TITLE = 14
DEFAULT_FONTSIZE_LABEL = 11
DEFAULT_FONTSIZE_TICKS = 9
DEFAULT_FONTSIZE_LEGEND = 9
DEFAULT_COLOR_MAP = 'tab10' # Good for up to 10 distinct colors
DEFAULT_ACTUAL_COLOR = 'black'
DEFAULT_HISTORICAL_COLOR = 'darkgrey'
DEFAULT_CI_ALPHA = 0.2

# --- Helper Functions ---

def _setup_plot_environment(config_params: Dict[str, Any]) -> Tuple[str, bool, bool, str, str, str]:
    """Creates plot directory and retrieves common plot settings."""
    results_dir = config_params.get('RESULTS_DIR', 'results')
    plots_dir = os.path.join(results_dir, 'plots')
    save_plots = config_params.get('SAVE_PLOTS', True)
    show_plots = config_params.get('SHOW_PLOTS', False) # Default to False for non-interactive runs
    plot_format = config_params.get('PLOT_OUTPUT_FORMAT', 'png').lower()
    date_col_name = config_params.get('DATE_COLUMN', 'Date')
    value_col_name = config_params.get('VALUE_COLUMN', 'Value')

    if save_plots:
        try:
            os.makedirs(plots_dir, exist_ok=True)
            logger.info(f"Plots will be saved to: {plots_dir}")
        except OSError as e:
            logger.error(f"Could not create plots directory '{plots_dir}'. Saving disabled. Error: {e}")
            save_plots = False
    return plots_dir, save_plots, show_plots, plot_format, date_col_name, value_col_name

def _get_plot_colors(model_names: List[str], cmap_name: str = DEFAULT_COLOR_MAP) -> Dict[str, str]:
    """Generates a dictionary mapping model names to distinct colors."""
    try:
        cmap = plt.get_cmap(cmap_name)
        # Ensure we handle cases where number of models exceeds colormap entries gracefully
        num_colors_needed = len(model_names)
        if isinstance(cmap, mcolors.ListedColormap):
             num_cmap_colors = len(cmap.colors)
             colors = [cmap(i % num_cmap_colors) for i in range(num_colors_needed)]
        else: # Continuous colormap
             colors = [cmap(i / max(1, num_colors_needed - 1)) for i in range(num_colors_needed)]

        return {name: mcolors.to_hex(colors[i]) for i, name in enumerate(model_names)}
    except Exception as e:
        logger.error(f"Could not get colormap '{cmap_name}'. Falling back to default colors. Error: {e}")
        # Fallback to basic colors
        basic_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        return {name: basic_colors[i % len(basic_colors)] for i, name in enumerate(model_names)}


def _format_axis(ax: plt.Axes, date_col_name: str, value_col_name: str, config_params: Dict[str, Any], title: str = ""):
    """Applies standard formatting to plot axes."""
    fontsize_label = config_params.get('PLOT_FONTSIZE_LABEL', DEFAULT_FONTSIZE_LABEL)
    fontsize_ticks = config_params.get('PLOT_FONTSIZE_TICKS', DEFAULT_FONTSIZE_TICKS)
    fontsize_title = config_params.get('PLOT_FONTSIZE_TITLE', DEFAULT_FONTSIZE_TITLE)
    fontsize_legend = config_params.get('PLOT_FONTSIZE_LEGEND', DEFAULT_FONTSIZE_LEGEND)

    ax.set_title(title, fontsize=fontsize_title)
    ax.set_xlabel(date_col_name, fontsize=fontsize_label)
    ax.set_ylabel(value_col_name, fontsize=fontsize_label)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.tick_params(axis='both', which='major', labelsize=fontsize_ticks)

    # Date formatting
    try:
        locator = mdates.AutoDateLocator(minticks=5, maxticks=10)
        formatter = mdates.ConciseDateFormatter(locator)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=15, ha='right')
    except Exception as date_fmt_err:
        logger.warning(f"Could not apply auto date formatting. Error: {date_fmt_err}")

    # Improve legend handling
    handles, labels = ax.get_legend_handles_labels()
    if handles: # Only add legend if there are labeled items
         # Try placing automatically first, otherwise place outside if too many items
         if len(handles) <= 5:
             ax.legend(loc='best', fontsize=fontsize_legend, frameon=True, framealpha=0.8)
         else:
             # Place legend outside the plot area to avoid obscuring data
             ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0., fontsize=fontsize_legend, frameon=True)
             # Adjust layout to make space for the legend outside
             plt.gcf().subplots_adjust(right=0.75) # Adjust this value as needed

def _save_and_show_plot(fig: plt.Figure, file_path: str, save_plots: bool, show_plots: bool, dpi: int = DEFAULT_DPI):
    """Saves and optionally shows a plot, then closes the figure."""
    if save_plots:
        try:
            fig.savefig(file_path, dpi=dpi, bbox_inches='tight')
            logger.info(f"Plot saved to {file_path}")
        except Exception as e:
            logger.error(f"Failed to save plot to {file_path}: {e}")
    if show_plots:
        plt.show()
    plt.close(fig)


# Inside src/plotting.py

def _plot_base(
    ax: plt.Axes,
    train_df: pd.DataFrame,
    test_df: Optional[pd.DataFrame], # Optional for future plots
    value_col_name: str,
    config_params: Dict[str, Any],
    forecasts: List[Tuple[str, pd.DataFrame, str, str]], # List of (name, df, color, linestyle)
    metrics: Optional[Dict[str, Dict[str, float]]] = None, # Dict[model_name, Dict[metric, value]]
    show_ci: bool = True,
    is_future_plot: bool = False # Flag to adjust behavior for future plots
):
    """Core plotting logic for history, actuals, forecasts, and CIs."""
    historical_color = config_params.get('PLOT_HISTORICAL_COLOR', DEFAULT_HISTORICAL_COLOR)
    actual_color = config_params.get('PLOT_ACTUAL_COLOR', DEFAULT_ACTUAL_COLOR)
    history_multiplier = config_params.get('PLOT_HISTORY_MULTIPLIER', DEFAULT_HISTORY_MULTIPLIER)
    ci_alpha = config_params.get('PLOT_CI_ALPHA', DEFAULT_CI_ALPHA)
    ci_level_pct = config_params.get('PROPHET_INTERVAL_WIDTH', 0.95) * 100 # Assumes Prophet CI width for label

    # 1. Plot Recent Historical Data (from train_df)
    history_end_date = test_df.index.min() if test_df is not None and not is_future_plot else train_df.index.max()
    history_periods_to_show = len(test_df) * history_multiplier if test_df is not None and not is_future_plot else len(train_df) // 4 # Show recent quarter for future

    # --- MODIFICATION START ---
    # Calculate history start date using offset arithmetic
    try:
        # Use the frequency object directly if available
        base_freq = train_df.index.freq
        if base_freq:
            # Multiply the offset by the number of periods
            history_offset = history_periods_to_show * base_freq
            history_start_date = history_end_date - history_offset
        else:
            # Fallback if frequency is None (less precise)
            logger.warning("Frequency is None, falling back to day-based offset for history plot.")
            history_start_date = history_end_date - pd.Timedelta(days=history_periods_to_show)
    except Exception as e:
         logger.error(f"Error calculating history start date offset: {e}. Falling back to simpler calculation.")
         # Fallback if offset arithmetic fails for some reason
         history_start_date = history_end_date - pd.Timedelta(days=history_periods_to_show)
    # --- MODIFICATION END ---

    recent_train_df = train_df.loc[history_start_date:history_end_date]
    if not recent_train_df.empty and 'y' in recent_train_df.columns:
        ax.plot(recent_train_df.index, recent_train_df['y'], label='Historical Data', color=historical_color, linewidth=1.5, zorder=1)

    # 2. Plot Actual Test Data (if available and not a future plot)
    if test_df is not None and not is_future_plot and 'y' in test_df.columns:
        ax.plot(test_df.index, test_df['y'], label='Actual Data', color=actual_color, linewidth=2, marker='.', markersize=5, zorder=len(forecasts) + 2)
        plot_index = recent_train_df.index.union(test_df.index)
    elif is_future_plot and forecasts: # Future plot needs index from forecasts
         # Combine historical index with the index of the first forecast for future plots
         first_forecast_index = forecasts[0][1].index
         plot_index = recent_train_df.index.union(first_forecast_index)
    else: # Only historical data
        plot_index = recent_train_df.index

    # 3. Plot Forecasts and Confidence Intervals
    all_forecast_indices = []
    for i, (model_name, forecast_df_orig, color, line_style) in enumerate(forecasts):
        target_index = test_df.index if test_df is not None and not is_future_plot else forecast_df_orig.index
        # Align forecast strictly to the target index for plotting
        forecast_df = forecast_df_orig.reindex(target_index)
        all_forecast_indices.append(forecast_df.index)

        if forecast_df.empty or forecast_df['yhat'].isnull().all():
            logger.warning(f"Plotting Warning ({model_name}): No valid forecast data for the target period. Skipping forecast plot.")
            continue

        # Construct label with metric if available
        metric_str = ""
        if metrics and model_name in metrics:
            # Prioritize RMSE then MAE for the label
            if 'RMSE' in metrics[model_name] and pd.notna(metrics[model_name]['RMSE']):
                metric_str = f" (RMSE: {metrics[model_name]['RMSE']:.2f})"
            elif 'MAE' in metrics[model_name] and pd.notna(metrics[model_name]['MAE']):
                metric_str = f" (MAE: {metrics[model_name]['MAE']:.2f})"

        label = f'{model_name} Forecast{metric_str}'

        # Plot point forecast
        ax.plot(forecast_df.index, forecast_df['yhat'], label=label, color=color, linestyle=line_style, linewidth=2, zorder=i + 2)

        # Plot confidence interval
        if show_ci and 'yhat_lower' in forecast_df.columns and 'yhat_upper' in forecast_df.columns:
             lower_valid = pd.api.types.is_numeric_dtype(forecast_df['yhat_lower']) and not forecast_df['yhat_lower'].isnull().all()
             upper_valid = pd.api.types.is_numeric_dtype(forecast_df['yhat_upper']) and not forecast_df['yhat_upper'].isnull().all()
             if lower_valid and upper_valid:
                 try:
                     # Make CI label unique if multiple forecasts shown, otherwise simpler
                     ci_label_suffix = f' {ci_level_pct:.0f}% CI' if len(forecasts) > 1 else f'{ci_level_pct:.0f}% CI'
                     ax.fill_between(forecast_df.index, forecast_df['yhat_lower'], forecast_df['yhat_upper'],
                                     color=color, alpha=ci_alpha, label=f'{model_name}{ci_label_suffix}', zorder=i + 1)
                 except Exception as fill_err:
                     logger.warning(f"Plotting Warning ({model_name}): Could not plot confidence interval. Error: {fill_err}")

    # 4. Set Axis Limits
    if all_forecast_indices:
         # Ensure the plot covers the full range of historical and forecasted data shown
         full_forecast_index = all_forecast_indices[0]
         for idx in all_forecast_indices[1:]:
             # Handle potential NaT in index during union
             if pd.isna(full_forecast_index).all():
                 full_forecast_index = idx
             elif not pd.isna(idx).all():
                 full_forecast_index = full_forecast_index.union(idx)

         # Also union with recent_train_df index
         if not pd.isna(recent_train_df.index).all():
              plot_index = recent_train_df.index.union(full_forecast_index)
         else:
              plot_index = full_forecast_index

    if plot_index is not None and not plot_index.empty and not pd.isna(plot_index).all():
        try:
             ax.set_xlim(plot_index.min(), plot_index.max())
        except TypeError as e:
             logger.warning(f"Could not set xlim automatically due to index issue: {e}. Plot limits might be incorrect.")
    # Let y-limits adjust automatically, or set manually if needed based on data range

# --- Main Plotting Functions ---

def plot_individual_model_evaluations(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    forecast_dict: Dict[str, pd.DataFrame],
    evaluation_metrics: Optional[pd.DataFrame], # DataFrame with models as index, metrics as columns
    config_params: Dict[str, Any],
    file_prefix: str = "eval_individual"
):
    """Generates and saves individual plots for each model's evaluation forecast."""
    logger.info("--- Generating Individual Model Evaluation Plots ---")
    plots_dir, save_plots, show_plots, plot_format, date_col, value_col = _setup_plot_environment(config_params)

    active_models = [k for k, v in forecast_dict.items() if isinstance(v, pd.DataFrame) and not v.empty and 'yhat' in v.columns]
    if not active_models:
        logger.warning("Skipping individual evaluation plots: No valid forecast data found.")
        return

    model_colors = _get_plot_colors(active_models, config_params.get('PLOT_COLOR_MAP', DEFAULT_COLOR_MAP))
    metrics_dict = evaluation_metrics.to_dict('index') if evaluation_metrics is not None else None

    for model_name in active_models:
        forecast_df = forecast_dict[model_name]
        fig, ax = plt.subplots(figsize=config_params.get('PLOT_FIG_SIZE_SINGLE', DEFAULT_FIG_SIZE_SINGLE))

        title = f"{model_name} Forecast vs Actuals (Evaluation Period)"

        _plot_base(
            ax=ax,
            train_df=train_df,
            test_df=test_df,
            value_col_name=value_col,
            config_params=config_params,
            forecasts=[(model_name, forecast_df, model_colors[model_name], '--')],
            metrics=metrics_dict,
            show_ci=True,
            is_future_plot=False
        )

        _format_axis(ax, date_col, value_col, config_params, title)
        fig_path = os.path.join(plots_dir, f"{file_prefix}_{model_name}.{plot_format}")
        _save_and_show_plot(fig, fig_path, save_plots, show_plots, config_params.get('PLOT_DPI', DEFAULT_DPI))

    logger.info("--- Finished Individual Model Evaluation Plots ---")


def plot_pairwise_model_comparisons(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    forecast_dict: Dict[str, pd.DataFrame],
    evaluation_metrics: Optional[pd.DataFrame],
    config_params: Dict[str, Any],
    file_prefix: str = "eval_pairwise"
):
    """Generates and saves pairwise comparison plots for model evaluation forecasts."""
    logger.info("--- Generating Pairwise Model Comparison Plots ---")
    plots_dir, save_plots, show_plots, plot_format, date_col, value_col = _setup_plot_environment(config_params)

    active_models = [k for k, v in forecast_dict.items() if isinstance(v, pd.DataFrame) and not v.empty and 'yhat' in v.columns]
    if len(active_models) < 2:
        logger.warning("Skipping pairwise comparison plots: Fewer than two models with valid forecasts.")
        return

    model_colors = _get_plot_colors(active_models, config_params.get('PLOT_COLOR_MAP', DEFAULT_COLOR_MAP))
    metrics_dict = evaluation_metrics.to_dict('index') if evaluation_metrics is not None else None

    # Define distinct line styles for pairwise plots
    line_styles = ['--', ':']

    for i, (model1_name, model2_name) in enumerate(itertools.combinations(active_models, 2)):
        forecast1_df = forecast_dict[model1_name]
        forecast2_df = forecast_dict[model2_name]

        fig, ax = plt.subplots(figsize=config_params.get('PLOT_FIG_SIZE_PAIRWISE', DEFAULT_FIG_SIZE_PAIRWISE))

        title = f"Comparison: {model1_name} vs {model2_name} (Evaluation Period)"

        _plot_base(
            ax=ax,
            train_df=train_df,
            test_df=test_df,
            value_col_name=value_col,
            config_params=config_params,
            forecasts=[
                (model1_name, forecast1_df, model_colors[model1_name], line_styles[0]),
                (model2_name, forecast2_df, model_colors[model2_name], line_styles[1])
            ],
            metrics=metrics_dict,
            show_ci=config_params.get('PLOT_SHOW_CI_PAIRWISE', True), # Option to disable CI in pairwise
            is_future_plot=False
        )

        _format_axis(ax, date_col, value_col, config_params, title)
        fig_path = os.path.join(plots_dir, f"{file_prefix}_{model1_name}_vs_{model2_name}.{plot_format}")
        _save_and_show_plot(fig, fig_path, save_plots, show_plots, config_params.get('PLOT_DPI', DEFAULT_DPI))

    logger.info("--- Finished Pairwise Model Comparison Plots ---")


def plot_residuals_comparison(
    test_df: pd.DataFrame,
    forecast_dict: Dict[str, pd.DataFrame],
    config_params: Dict[str, Any],
    file_prefix: str = "eval_residuals"
):
    """Generates a plot comparing the residuals of different models."""
    logger.info("--- Generating Evaluation Residuals Comparison Plot ---")
    plots_dir, save_plots, show_plots, plot_format, date_col, value_col = _setup_plot_environment(config_params)

    if 'y' not in test_df.columns:
        logger.error("Residual plot error: 'y' column not found in test_df. Skipping.")
        return

    active_models = [k for k, v in forecast_dict.items() if isinstance(v, pd.DataFrame) and not v.empty and 'yhat' in v.columns]
    if not active_models:
        logger.warning("Skipping residuals plot: No valid forecast data found.")
        return

    model_colors = _get_plot_colors(active_models, config_params.get('PLOT_COLOR_MAP', DEFAULT_COLOR_MAP))
    # Use distinct markers for residuals
    markers = ['o', 's', '^', 'd', 'v', '<', '>', 'p', '*', 'X']

    fig, ax = plt.subplots(figsize=config_params.get('PLOT_FIG_SIZE_RESIDUAL', DEFAULT_FIG_SIZE_RESIDUAL))
    title = "Forecast Residuals (Actual - Forecast) during Evaluation"
    ax.axhline(0, color='black', linestyle='-', linewidth=1.5, alpha=0.8, zorder=1)

    plotted_residuals = False
    for i, model_name in enumerate(active_models):
        forecast_df = forecast_dict[model_name]
        # Align forecast to test data index for residual calculation
        y_pred_aligned = forecast_df['yhat'].reindex(test_df.index)
        residuals = test_df['y'] - y_pred_aligned
        valid_residuals = residuals.dropna() # Drop NaNs resulting from subtraction or alignment

        if not valid_residuals.empty:
            ax.plot(valid_residuals.index, valid_residuals,
                    label=f'{model_name}',
                    color=model_colors[model_name],
                    linestyle='None', # No lines for residuals
                    marker=markers[i % len(markers)],
                    markersize=6,
                    alpha=0.7,
                    zorder=i + 2)
            plotted_residuals = True
        else:
            logger.warning(f"Residual Plot Warning ({model_name}): No valid residuals to plot after alignment/dropna.")

    if not plotted_residuals:
        logger.warning("Residual plot skipped: No models had valid residuals.")
        plt.close(fig)
        return

    # Special formatting for residuals plot
    fontsize_label = config_params.get('PLOT_FONTSIZE_LABEL', DEFAULT_FONTSIZE_LABEL)
    fontsize_ticks = config_params.get('PLOT_FONTSIZE_TICKS', DEFAULT_FONTSIZE_TICKS)
    fontsize_title = config_params.get('PLOT_FONTSIZE_TITLE', DEFAULT_FONTSIZE_TITLE)
    fontsize_legend = config_params.get('PLOT_FONTSIZE_LEGEND', DEFAULT_FONTSIZE_LEGEND)

    ax.set_title(title, fontsize=fontsize_title)
    ax.set_xlabel(date_col, fontsize=fontsize_label)
    ax.set_ylabel("Residual (Actual - Forecast)", fontsize=fontsize_label)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.tick_params(axis='both', which='major', labelsize=fontsize_ticks)

    # Date formatting
    try:
        locator = mdates.AutoDateLocator(minticks=5, maxticks=10)
        formatter = mdates.ConciseDateFormatter(locator)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=15, ha='right')
    except Exception as date_fmt_err:
        logger.warning(f"Residual Plot: Could not apply auto date formatting. Error: {date_fmt_err}")

    # Legend below the plot for residuals
    handles, labels = ax.get_legend_handles_labels()
    if handles:
         fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.05),
                    ncol=min(len(handles), 5), fontsize=fontsize_legend, frameon=False)
         # Adjust layout to make space for the legend
         fig.subplots_adjust(bottom=0.2) # Increase bottom margin


    fig_path = os.path.join(plots_dir, f"{file_prefix}_comparison.{plot_format}")
    _save_and_show_plot(fig, fig_path, save_plots, show_plots, config_params.get('PLOT_DPI', DEFAULT_DPI))
    logger.info("--- Finished Residuals Comparison Plot ---")


def plot_future_forecasts(
    historical_df: pd.DataFrame, # Full historical data (df)
    future_forecast_dict: Dict[str, pd.DataFrame],
    config_params: Dict[str, Any],
    file_prefix: str = "future_forecast"
):
    """Generates a combined plot showing historical data and future forecasts."""
    logger.info("--- Generating Future Forecast Plot ---")
    plots_dir, save_plots, show_plots, plot_format, date_col, value_col = _setup_plot_environment(config_params)

    active_models = [k for k, v in future_forecast_dict.items() if isinstance(v, pd.DataFrame) and not v.empty and 'yhat' in v.columns]
    if not active_models:
        logger.warning("Skipping future forecast plot: No valid forecast data found.")
        return

    if historical_df.empty or 'y' not in historical_df.columns:
        logger.warning("Skipping future forecast plot: Historical data is empty or missing 'y' column.")
        return

    model_colors = _get_plot_colors(active_models, config_params.get('PLOT_COLOR_MAP', DEFAULT_COLOR_MAP))

    fig, ax = plt.subplots(figsize=config_params.get('PLOT_FIG_SIZE_FUTURE', DEFAULT_FIG_SIZE_FUTURE))

    # Construct title with dataset info
    dataset_name = os.path.basename(config_params.get('CSV_FILE_PATH', 'Unknown Data'))
    forecast_horizon = len(next(iter(future_forecast_dict.values()))) # Get length from first forecast
    title = f"Future Forecast ({forecast_horizon} Periods) - {value_col}\nData: {dataset_name}"

    # Prepare forecast list for _plot_base
    forecasts_to_plot = []
    for model_name in active_models:
        forecasts_to_plot.append(
            (model_name, future_forecast_dict[model_name], model_colors[model_name], '--')
        )

    # Use _plot_base, indicating it's a future plot (no test_df actuals)
    _plot_base(
        ax=ax,
        train_df=historical_df, # Use full history
        test_df=None,           # No test actuals in the future
        value_col_name=value_col,
        config_params=config_params,
        forecasts=forecasts_to_plot,
        metrics=None,           # No evaluation metrics for future plots
        show_ci=config_params.get('PLOT_SHOW_CI_FUTURE', True),
        is_future_plot=True     # Set flag
    )

    # Final formatting
    _format_axis(ax, date_col, value_col, config_params, title)

    fig_path = os.path.join(plots_dir, f"{file_prefix}_combined.{plot_format}")
    _save_and_show_plot(fig, fig_path, save_plots, show_plots, config_params.get('PLOT_DPI', DEFAULT_DPI))
    logger.info("--- Finished Future Forecast Plot ---")

# --- Main Orchestration Function ---

def generate_all_plots(
    train_df: pd.DataFrame, # Includes training and validation data
    test_df: pd.DataFrame,
    full_df: pd.DataFrame, # Full dataset for future plot history
    eval_forecast_dict: Dict[str, pd.DataFrame],
    future_forecast_dict: Dict[str, pd.DataFrame],
    evaluation_metrics: Optional[pd.DataFrame],
    config_params: Dict[str, Any]
):
    """Calls all relevant plotting functions."""
    logger.info("--- Generating All Plots ---")

    # Check if plotting is globally disabled
    if not config_params.get('SAVE_PLOTS', False) and not config_params.get('SHOW_PLOTS', False):
        logger.info("Plotting is disabled (SAVE_PLOTS=False and SHOW_PLOTS=False). Skipping.")
        return

    # 1. Individual Evaluation Plots
    if eval_forecast_dict:
        plot_individual_model_evaluations(
            train_df=train_df,
            test_df=test_df,
            forecast_dict=eval_forecast_dict,
            evaluation_metrics=evaluation_metrics,
            config_params=config_params
        )
    else:
        logger.info("Skipping individual evaluation plots (no evaluation forecasts provided).")


    # 2. Pairwise Evaluation Plots
    if eval_forecast_dict:
        plot_pairwise_model_comparisons(
            train_df=train_df,
            test_df=test_df,
            forecast_dict=eval_forecast_dict,
            evaluation_metrics=evaluation_metrics,
            config_params=config_params
        )
    else:
        logger.info("Skipping pairwise evaluation plots (no evaluation forecasts provided).")

    # 3. Residuals Plot
    if eval_forecast_dict and not test_df.empty:
        plot_residuals_comparison(
            test_df=test_df,
            forecast_dict=eval_forecast_dict,
            config_params=config_params
        )
    else:
         logger.info("Skipping residuals plot (no evaluation forecasts or test data provided).")


    # 4. Future Forecast Plot
    if config_params.get('RUN_FINAL_FORECAST', False) and future_forecast_dict and not full_df.empty:
        plot_future_forecasts(
            historical_df=full_df,
            future_forecast_dict=future_forecast_dict,
            config_params=config_params
        )
    elif not config_params.get('RUN_FINAL_FORECAST', False):
         logger.info("Skipping future forecast plot (RUN_FINAL_FORECAST=False).")
    else:
        logger.info("Skipping future forecast plot (no future forecasts or full dataset provided).")

    logger.info("--- Finished Generating All Plots ---")