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

# Import numpy for polyfit/polyval
# from scipy import stats # Alternative for linear regression

logger = logging.getLogger(__name__)

# --- Configuration Defaults (can be overridden by config_params) ---
DEFAULT_FIG_SIZE_SINGLE = (10, 5)
DEFAULT_FIG_SIZE_PAIRWISE = (11, 6) # Slightly wider for potentially more legend items
DEFAULT_FIG_SIZE_RESIDUAL = (12, 5)
DEFAULT_FIG_SIZE_FUTURE = (14, 7)
DEFAULT_DPI = 150
# DEFAULT_HISTORY_MULTIPLIER = 3 # No longer needed
DEFAULT_FONTSIZE_TITLE = 14
DEFAULT_FONTSIZE_LABEL = 11
DEFAULT_FONTSIZE_TICKS = 9
DEFAULT_FONTSIZE_LEGEND = 9
DEFAULT_COLOR_MAP = 'tab10' # Good for up to 10 distinct colors
DEFAULT_ACTUAL_COLOR = 'black'
DEFAULT_HISTORICAL_COLOR = 'darkgrey'
DEFAULT_TREND_COLOR = 'red' # Color for the trend line
DEFAULT_TREND_LINESTYLE = ':' # Linestyle for the trend line
DEFAULT_CI_ALPHA = 0.2

# --- Helper Functions ---

def _setup_plot_environment(config_params: Dict[str, Any]) -> Tuple[str, bool, bool, str]:
    """Creates plot directory and retrieves common plot settings."""
    # Removed date_col_name, value_col_name return as they are overridden now
    results_dir = config_params.get('RESULTS_DIR', 'results')
    plots_dir = os.path.join(results_dir, 'plots')
    save_plots = config_params.get('SAVE_PLOTS', True)
    show_plots = config_params.get('SHOW_PLOTS', False) # Default to False for non-interactive runs
    plot_format = config_params.get('PLOT_OUTPUT_FORMAT', 'png').lower()


    if save_plots:
        try:
            os.makedirs(plots_dir, exist_ok=True)
            logger.info(f"Plots will be saved to: {plots_dir}")
        except OSError as e:
            logger.error(f"Could not create plots directory '{plots_dir}'. Saving disabled. Error: {e}")
            save_plots = False
    return plots_dir, save_plots, show_plots, plot_format

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


def _format_axis(ax: plt.Axes, config_params: Dict[str, Any], title: str = "", is_residual_plot: bool = False):
    """Applies standard formatting to plot axes, using predefined labels."""
    # MODIFICATION: Use hardcoded labels for academic paper
    x_label = "Year"
    y_label = "Residual (Actual - Forecast)" if is_residual_plot else "Total Ozone Column (DU)"

    fontsize_label = config_params.get('PLOT_FONTSIZE_LABEL', DEFAULT_FONTSIZE_LABEL)
    fontsize_ticks = config_params.get('PLOT_FONTSIZE_TICKS', DEFAULT_FONTSIZE_TICKS)
    fontsize_title = config_params.get('PLOT_FONTSIZE_TITLE', DEFAULT_FONTSIZE_TITLE)
    fontsize_legend = config_params.get('PLOT_FONTSIZE_LEGEND', DEFAULT_FONTSIZE_LEGEND)

    ax.set_title(title, fontsize=fontsize_title)
    ax.set_xlabel(x_label, fontsize=fontsize_label)
    ax.set_ylabel(y_label, fontsize=fontsize_label)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.tick_params(axis='both', which='major', labelsize=fontsize_ticks)

    # Date formatting for the x-axis (which represents 'Year')
    try:
        # AutoLocator works well for years, ConciseFormatter keeps it clean
        locator = mdates.AutoDateLocator(minticks=5, maxticks=12) # Adjust maxticks for density
        formatter = mdates.ConciseDateFormatter(locator)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)
        # Rotation might still be needed depending on date range density
        # plt.setp(ax.xaxis.get_majorticklabels(), rotation=0, ha='center') # Less rotation for just years
    except Exception as date_fmt_err:
        logger.warning(f"Could not apply auto date formatting. Error: {date_fmt_err}")

    # Improve legend handling
    handles, labels = ax.get_legend_handles_labels()
    if handles: # Only add legend if there are labeled items
        # Increase threshold for placing outside slightly, as trend line is added
        if len(handles) <= 6:
            ax.legend(loc='best', fontsize=fontsize_legend, frameon=True, framealpha=0.8)
        else:
            # Place legend outside the plot area to avoid obscuring data
            ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0., fontsize=fontsize_legend, frameon=True)
            # Adjust layout to make space for the legend outside
            try:
                # Adjust based on figure content
                plt.gcf().tight_layout(rect=[0, 0, 0.85, 1]) # Leave space on the right
            except ValueError as layout_err:
                 logger.warning(f"Could not apply tight_layout with legend adjustment: {layout_err}")
                 # Fallback adjustment
                 plt.gcf().subplots_adjust(right=0.75 if not is_residual_plot else 1.0) # Only adjust if legend is outside

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


def _plot_base(
    ax: plt.Axes,
    historical_df: pd.DataFrame, # Changed name from train_df for clarity
    test_df: Optional[pd.DataFrame], # Optional for future plots
    trend_line_series: Optional[pd.Series], # NEW: Trend line calculated over history
    config_params: Dict[str, Any],
    forecasts: List[Tuple[str, pd.DataFrame, str, str]], # List of (name, df, color, linestyle)
    metrics: Optional[Dict[str, Dict[str, float]]] = None, # Dict[model_name, Dict[metric, value]]
    show_ci: bool = True,
    is_future_plot: bool = False # Flag to adjust behavior for future plots
):
    """Core plotting logic for history, actuals, forecasts, CIs, and trend line."""
    historical_color = config_params.get('PLOT_HISTORICAL_COLOR', DEFAULT_HISTORICAL_COLOR)
    actual_color = config_params.get('PLOT_ACTUAL_COLOR', DEFAULT_ACTUAL_COLOR)
    trend_color = config_params.get('PLOT_TREND_COLOR', DEFAULT_TREND_COLOR)
    trend_linestyle = config_params.get('PLOT_TREND_LINESTYLE', DEFAULT_TREND_LINESTYLE)
    ci_alpha = config_params.get('PLOT_CI_ALPHA', DEFAULT_CI_ALPHA)
    ci_level_pct = config_params.get('PROPHET_INTERVAL_WIDTH', 0.95) * 100

    plot_start_date = historical_df.index.min()
    plot_end_date = historical_df.index.max() # Initialize with historical end

    # 1. Plot *Full* Historical Data
    # MODIFICATION: Plot entire historical_df, not just recent part
    if not historical_df.empty and 'y' in historical_df.columns:
        ax.plot(historical_df.index, historical_df['y'], label='Historical Data (Train+Val)', color=historical_color, linewidth=1.5, zorder=1)
    else:
        logger.warning("Historical data is empty or missing 'y' column. Skipping plot.")
        return # Cant plot anything without history

    # 2. Plot Overall Trend Line (if provided)
    # MODIFICATION: Add trend line plot
    if trend_line_series is not None and not trend_line_series.empty:
        # Trend should align with the historical_df index
        trend_to_plot = trend_line_series.reindex(historical_df.index)
        ax.plot(trend_to_plot.index, trend_to_plot.values,
                label='Overall Trend (Train+Val)', # Indicate period used for calculation
                color=trend_color,
                linestyle=trend_linestyle,
                linewidth=2, zorder=2) # Plot trend above history, below actuals/forecasts

    # 3. Plot Actual Test Data (if available and not a future plot)
    if test_df is not None and not is_future_plot and 'y' in test_df.columns:
        ax.plot(test_df.index, test_df['y'], label='Actual Data (Test)', color=actual_color, linewidth=2, marker='.', markersize=5, zorder=len(forecasts) + 3)
        plot_end_date = test_df.index.max() # Update plot end date
    elif forecasts: # Extend plot end date based on forecasts if no test data
         try:
             forecast_end_dates = [f[1].index.max() for f in forecasts if f[1] is not None and not f[1].empty]
             if forecast_end_dates:
                 plot_end_date = max(plot_end_date, max(forecast_end_dates))
         except Exception: # Handle cases where index might be non-comparable briefly
             pass


    # 4. Plot Forecasts and Confidence Intervals
    all_forecast_indices = []
    for i, (model_name, forecast_df_orig, color, line_style) in enumerate(forecasts):
        target_index = test_df.index if test_df is not None and not is_future_plot else forecast_df_orig.index
        if target_index is None or target_index.empty:
             logger.warning(f"Plotting Warning ({model_name}): Target index is invalid. Skipping forecast.")
             continue

        # Align forecast strictly to the target index for plotting
        forecast_df = forecast_df_orig.reindex(target_index)
        all_forecast_indices.append(forecast_df.index)
        if not is_future_plot: # Update plot end date if forecast extends beyond test data
             plot_end_date = max(plot_end_date, forecast_df.index.max())

        if forecast_df.empty or forecast_df['yhat'].isnull().all():
            logger.warning(f"Plotting Warning ({model_name}): No valid forecast data for the target period. Skipping forecast plot.")
            continue

        # Construct label with metric if available (only for evaluation plots)
        metric_str = ""
        if not is_future_plot and metrics and model_name in metrics:
            # Prioritize RMSE then MAE for the label
            if 'RMSE' in metrics[model_name] and pd.notna(metrics[model_name]['RMSE']):
                metric_str = f" (RMSE: {metrics[model_name]['RMSE']:.2f})"
            elif 'MAE' in metrics[model_name] and pd.notna(metrics[model_name]['MAE']):
                metric_str = f" (MAE: {metrics[model_name]['MAE']:.2f})"

        label = f'{model_name} Forecast{metric_str}'

        # Plot point forecast
        ax.plot(forecast_df.index, forecast_df['yhat'], label=label, color=color, linestyle=line_style, linewidth=2, zorder=i + 4) # Increase zorder

        # Plot confidence interval
        if show_ci and 'yhat_lower' in forecast_df.columns and 'yhat_upper' in forecast_df.columns:
             lower_valid = pd.api.types.is_numeric_dtype(forecast_df['yhat_lower']) and not forecast_df['yhat_lower'].isnull().all()
             upper_valid = pd.api.types.is_numeric_dtype(forecast_df['yhat_upper']) and not forecast_df['yhat_upper'].isnull().all()
             if lower_valid and upper_valid:
                 try:
                     # Make CI label unique if multiple forecasts shown, otherwise simpler
                     ci_label_suffix = f' {ci_level_pct:.0f}% CI' if len(forecasts) > 1 else f'{ci_level_pct:.0f}% CI'
                     ax.fill_between(forecast_df.index, forecast_df['yhat_lower'], forecast_df['yhat_upper'],
                                     color=color, alpha=ci_alpha, label=f'{model_name}{ci_label_suffix}', zorder=i + 3) # CI below forecast line
                 except Exception as fill_err:
                     logger.warning(f"Plotting Warning ({model_name}): Could not plot confidence interval. Error: {fill_err}")

    # 5. Set Axis Limits
    # MODIFICATION: Set x-limits from start of historical data to end of plotted data
    if plot_start_date is not None and plot_end_date is not None:
        try:
             # Add a small padding to the end date if possible
             padding = pd.Timedelta(days=30) # Approx 1 month padding
             if hasattr(plot_end_date, 'freq') and plot_end_date.freq:
                  padding = plot_end_date.freq * 1 # Pad by one frequency unit
             ax.set_xlim(plot_start_date, plot_end_date + padding)
        except TypeError as e:
             logger.warning(f"Could not set xlim automatically due to index/padding issue: {e}. Plot limits might be incorrect.")
        except Exception as e: # Catch other potential errors
             logger.warning(f"Unexpected error setting xlim: {e}. Plot limits might be incorrect.")


# --- Main Plotting Functions ---

def plot_individual_model_evaluations(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    trend_line_series: Optional[pd.Series], # ADDED
    forecast_dict: Dict[str, pd.DataFrame],
    evaluation_metrics: Optional[pd.DataFrame],
    config_params: Dict[str, Any],
    file_prefix: str = "eval_individual"
):
    """Generates and saves individual plots for each model's evaluation forecast."""
    logger.info("--- Generating Individual Model Evaluation Plots ---")
    # MODIFICATION: Removed date_col, value_col from setup return
    plots_dir, save_plots, show_plots, plot_format = _setup_plot_environment(config_params)

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
            historical_df=train_df, # Pass train+val data as history
            test_df=test_df,
            trend_line_series=trend_line_series, # Pass trend line
            config_params=config_params,
            forecasts=[(model_name, forecast_df, model_colors[model_name], '--')],
            metrics=metrics_dict,
            show_ci=True,
            is_future_plot=False
        )

        _format_axis(ax, config_params, title) # No need to pass labels anymore
        fig_path = os.path.join(plots_dir, f"{file_prefix}_{model_name}.{plot_format}")
        _save_and_show_plot(fig, fig_path, save_plots, show_plots, config_params.get('PLOT_DPI', DEFAULT_DPI))

    logger.info("--- Finished Individual Model Evaluation Plots ---")


def plot_pairwise_model_comparisons(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    trend_line_series: Optional[pd.Series], # ADDED
    forecast_dict: Dict[str, pd.DataFrame],
    evaluation_metrics: Optional[pd.DataFrame],
    config_params: Dict[str, Any],
    file_prefix: str = "eval_pairwise"
):
    """Generates and saves pairwise comparison plots for model evaluation forecasts."""
    logger.info("--- Generating Pairwise Model Comparison Plots ---")
    # MODIFICATION: Removed date_col, value_col from setup return
    plots_dir, save_plots, show_plots, plot_format = _setup_plot_environment(config_params)

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
            historical_df=train_df, # Pass train+val data as history
            test_df=test_df,
            trend_line_series=trend_line_series, # Pass trend line
            config_params=config_params,
            forecasts=[
                (model1_name, forecast1_df, model_colors[model1_name], line_styles[0]),
                (model2_name, forecast2_df, model_colors[model2_name], line_styles[1])
            ],
            metrics=metrics_dict,
            show_ci=config_params.get('PLOT_SHOW_CI_PAIRWISE', True), # Option to disable CI in pairwise
            is_future_plot=False
        )

        _format_axis(ax, config_params, title) # No need to pass labels anymore
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
    # MODIFICATION: Removed date_col, value_col from setup return
    plots_dir, save_plots, show_plots, plot_format = _setup_plot_environment(config_params)

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

    # MODIFICATION: Apply formatting with specific residual label flag
    _format_axis(ax, config_params, title, is_residual_plot=True)

    # Legend below the plot for residuals
    handles, labels = ax.get_legend_handles_labels()
    if handles:
         try:
             # Adjust bottom dynamically if legend is added
             fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.05),
                        ncol=min(len(handles), 5), fontsize=config_params.get('PLOT_FONTSIZE_LEGEND', DEFAULT_FONTSIZE_LEGEND), frameon=False)
             fig.tight_layout(rect=[0, 0.05, 1, 1]) # Adjust bottom slightly more reliably
         except Exception as legend_err:
             logger.warning(f"Could not place legend below residual plot: {legend_err}")
             # Fallback adjustment if tight_layout fails
             fig.subplots_adjust(bottom=0.2)


    fig_path = os.path.join(plots_dir, f"{file_prefix}_comparison.{plot_format}")
    _save_and_show_plot(fig, fig_path, save_plots, show_plots, config_params.get('PLOT_DPI', DEFAULT_DPI))
    logger.info("--- Finished Residuals Comparison Plot ---")


def plot_future_forecasts(
    historical_df: pd.DataFrame, # Full historical data (df)
    trend_line_series: Optional[pd.Series], # ADDED (Calculated on Train+Val part)
    future_forecast_dict: Dict[str, pd.DataFrame],
    config_params: Dict[str, Any],
    file_prefix: str = "future_forecast"
):
    """Generates a combined plot showing historical data and future forecasts."""
    logger.info("--- Generating Future Forecast Plot ---")
    # MODIFICATION: Removed date_col, value_col from setup return
    plots_dir, save_plots, show_plots, plot_format = _setup_plot_environment(config_params)

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
    # Safely get forecast horizon
    forecast_horizon = 0
    if future_forecast_dict:
         first_key = next(iter(future_forecast_dict))
         if future_forecast_dict[first_key] is not None:
             forecast_horizon = len(future_forecast_dict[first_key])

    title = f"Future Forecast ({forecast_horizon} Periods)"
    # Add subtitle for data source if desired
    # ax.text(0.5, 1.02, f"Data: {dataset_name}", transform=ax.transAxes, ha='center', fontsize=10)


    # Prepare forecast list for _plot_base
    forecasts_to_plot = []
    for model_name in active_models:
        forecasts_to_plot.append(
            (model_name, future_forecast_dict[model_name], model_colors[model_name], '--')
        )

    # Use _plot_base, indicating it's a future plot (no test_df actuals)
    _plot_base(
        ax=ax,
        historical_df=historical_df, # Use full history
        test_df=None,           # No test actuals in the future
        trend_line_series=trend_line_series, # Pass trend line (calculated on train+val)
        config_params=config_params,
        forecasts=forecasts_to_plot,
        metrics=None,           # No evaluation metrics for future plots
        show_ci=config_params.get('PLOT_SHOW_CI_FUTURE', True),
        is_future_plot=True     # Set flag
    )

    # Final formatting
    _format_axis(ax, config_params, title) # No need to pass labels anymore

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
    """Calls all relevant plotting functions, calculating trend line first."""
    logger.info("--- Generating All Plots ---")

    # Check if plotting is globally disabled
    if not config_params.get('SAVE_PLOTS', False) and not config_params.get('SHOW_PLOTS', False):
        logger.info("Plotting is disabled (SAVE_PLOTS=False and SHOW_PLOTS=False). Skipping.")
        return

    # --- MODIFICATION: Calculate Trend Line on Train+Val Data ---
    trend_line_series: Optional[pd.Series] = None
    if not train_df.empty and 'y' in train_df.columns:
        try:
            y_values = train_df['y'].dropna() # Use only non-NaN values for trend calculation
            if len(y_values) >= 2: # Need at least two points for a line
                # Create numerical representation for x-axis (simple integer sequence)
                x_numeric = np.arange(len(y_values))
                # Calculate linear trend (degree 1 polynomial)
                coeffs = np.polyfit(x_numeric, y_values.values, 1)
                trend_values = np.polyval(coeffs, x_numeric)
                # Create a Series with the original DatetimeIndex
                trend_line_series = pd.Series(trend_values, index=y_values.index, name='Trend')
                logger.info(f"Calculated linear trend line (slope={coeffs[0]:.4f}, intercept={coeffs[1]:.4f}) over train+val period.")
            else:
                 logger.warning("Skipping trend line calculation: Not enough non-NaN data points in train_df.")
        except Exception as e:
            logger.error(f"Error calculating trend line: {e}", exc_info=True)
            trend_line_series = None # Ensure it's None if calculation fails
    else:
         logger.warning("Skipping trend line calculation: train_df is empty or missing 'y' column.")
    # --- End Trend Line Calculation ---


    # 1. Individual Evaluation Plots
    if eval_forecast_dict:
        plot_individual_model_evaluations(
            train_df=train_df,
            test_df=test_df,
            trend_line_series=trend_line_series, # Pass trend line
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
            trend_line_series=trend_line_series, # Pass trend line
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
            # Trend line is NOT passed here
        )
    else:
         logger.info("Skipping residuals plot (no evaluation forecasts or test data provided).")


    # 4. Future Forecast Plot
    if config_params.get('RUN_FINAL_FORECAST', False) and future_forecast_dict and not full_df.empty:
        plot_future_forecasts(
            historical_df=full_df,
            trend_line_series=trend_line_series, # Pass trend line (calculated on train+val)
            future_forecast_dict=future_forecast_dict,
            config_params=config_params
        )
    elif not config_params.get('RUN_FINAL_FORECAST', False):
         logger.info("Skipping future forecast plot (RUN_FINAL_FORECAST=False).")
    else:
        logger.info("Skipping future forecast plot (no future forecasts or full dataset provided).")

    logger.info("--- Finished Generating All Plots ---")