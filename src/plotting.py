import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.colors as mcolors
import pandas as pd
import numpy as np
import os
import logging
import itertools
from typing import Dict, Optional, Any, List, Tuple

# from scipy import stats # Alternative for linear regression trend

logger = logging.getLogger(__name__)

# --- Configuration Defaults ---
DEFAULT_FIG_SIZE_SINGLE = (10, 5)
DEFAULT_FIG_SIZE_PAIRWISE = (11, 6)
DEFAULT_FIG_SIZE_RESIDUAL = (12, 5)
DEFAULT_FIG_SIZE_FUTURE = (14, 7)
DEFAULT_DPI = 150
DEFAULT_FONTSIZE_TITLE = 14
DEFAULT_FONTSIZE_LABEL = 11
DEFAULT_FONTSIZE_TICKS = 9
DEFAULT_FONTSIZE_LEGEND = 9
DEFAULT_COLOR_MAP = 'tab10'
DEFAULT_ACTUAL_COLOR = 'black'
DEFAULT_HISTORICAL_COLOR = 'darkgrey'
DEFAULT_TREND_COLOR = 'red'
DEFAULT_TREND_LINESTYLE = ':'
DEFAULT_CI_ALPHA = 0.2

# --- Helper Functions ---

def _setup_plot_environment(config_params: Dict[str, Any]) -> Tuple[str, bool, bool, str]:
    """Creates plot directory and gets common plot settings."""
    results_dir = config_params.get('RESULTS_DIR', 'results')
    plots_dir = os.path.join(results_dir, 'plots')
    save_plots = config_params.get('SAVE_PLOTS', True)
    show_plots = config_params.get('SHOW_PLOTS', False)
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
    """Maps model names to distinct colors, handling cmap limits."""
    try:
        cmap = plt.get_cmap(cmap_name)
        num_colors_needed = len(model_names)
        if isinstance(cmap, mcolors.ListedColormap):
             num_cmap_colors = len(cmap.colors)
             colors = [cmap(i % num_cmap_colors) for i in range(num_colors_needed)]
        else: # Continuous cmap
             colors = [cmap(i / max(1, num_colors_needed - 1)) for i in range(num_colors_needed)]
        return {name: mcolors.to_hex(colors[i]) for i, name in enumerate(model_names)}
    except Exception as e:
        logger.error(f"Could not get colormap '{cmap_name}'. Falling back. Error: {e}")
        basic_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        return {name: basic_colors[i % len(basic_colors)] for i, name in enumerate(model_names)}


def _format_axis(ax: plt.Axes, config_params: Dict[str, Any], title: str = "", is_residual_plot: bool = False):
    """Applies standard formatting to plot axes (using hardcoded labels)."""
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

    # Date formatting for the x-axis
    try:
        locator = mdates.AutoDateLocator(minticks=5, maxticks=12)
        formatter = mdates.ConciseDateFormatter(locator)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)
    except Exception as date_fmt_err:
        logger.warning(f"Could not apply auto date formatting. Error: {date_fmt_err}")

    # Legend handling (place outside if too many items)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        if len(handles) <= 6:
            ax.legend(loc='best', fontsize=fontsize_legend, frameon=True, framealpha=0.8)
        else:
            ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0., fontsize=fontsize_legend, frameon=True)
            try: # Adjust layout for outside legend
                plt.gcf().tight_layout(rect=[0, 0, 0.85, 1])
            except ValueError as layout_err:
                 logger.warning(f"Could not tight_layout with legend adjustment: {layout_err}")
                 plt.gcf().subplots_adjust(right=0.75) # Fallback

def _save_and_show_plot(fig: plt.Figure, file_path: str, save_plots: bool, show_plots: bool, dpi: int = DEFAULT_DPI):
    """Saves and optionally shows a plot, then closes it."""
    if save_plots:
        try:
            fig.savefig(file_path, dpi=dpi, bbox_inches='tight')
            logger.info(f"Plot saved to {file_path}")
        except Exception as e: logger.error(f"Failed to save plot to {file_path}: {e}")
    if show_plots: plt.show()
    plt.close(fig)


def _plot_base(
    ax: plt.Axes,
    historical_df: pd.DataFrame,
    test_df: Optional[pd.DataFrame],
    trend_line_series: Optional[pd.Series],
    config_params: Dict[str, Any],
    forecasts: List[Tuple[str, pd.DataFrame, str, str]],
    metrics: Optional[Dict[str, Dict[str, float]]] = None,
    show_ci: bool = True,
    is_future_plot: bool = False
):
    """Core plotting logic: history, actuals, forecasts, CIs, trend line."""
    historical_color = config_params.get('PLOT_HISTORICAL_COLOR', DEFAULT_HISTORICAL_COLOR)
    actual_color = config_params.get('PLOT_ACTUAL_COLOR', DEFAULT_ACTUAL_COLOR)
    trend_color = config_params.get('PLOT_TREND_COLOR', DEFAULT_TREND_COLOR)
    trend_linestyle = config_params.get('PLOT_TREND_LINESTYLE', DEFAULT_TREND_LINESTYLE)
    ci_alpha = config_params.get('PLOT_CI_ALPHA', DEFAULT_CI_ALPHA)
    ci_level_pct = config_params.get('PROPHET_INTERVAL_WIDTH', 0.95) * 100

    if historical_df.empty or 'y' not in historical_df.columns:
        logger.warning("Historical data empty/missing 'y'. Skipping base plot.")
        return
    plot_start_date = historical_df.index.min()
    plot_end_date = historical_df.index.max()

    # 1. Plot Full Historical Data
    ax.plot(historical_df.index, historical_df['y'], label='Historical Data (Train+Val)', color=historical_color, linewidth=1.5, zorder=1)

    # 2. Plot Overall Trend Line
    if trend_line_series is not None and not trend_line_series.empty:
        trend_to_plot = trend_line_series.reindex(historical_df.index)
        ax.plot(trend_to_plot.index, trend_to_plot.values, label='Overall Trend (Train+Val)',
                color=trend_color, linestyle=trend_linestyle, linewidth=2, zorder=2)

    # 3. Plot Actual Test Data (if available & not future plot)
    if test_df is not None and not is_future_plot and 'y' in test_df.columns:
        ax.plot(test_df.index, test_df['y'], label='Actual Data (Test)', color=actual_color, linewidth=2, marker='.', markersize=5, zorder=len(forecasts) + 3)
        plot_end_date = test_df.index.max()
    elif forecasts: # Extend plot end based on forecasts if no test data
         try:
             forecast_end_dates = [f[1].index.max() for f in forecasts if f[1] is not None and not f[1].empty]
             if forecast_end_dates: plot_end_date = max(plot_end_date, max(forecast_end_dates))
         except Exception: pass # Ignore potential index comparison errors

    # 4. Plot Forecasts and Confidence Intervals
    for i, (model_name, forecast_df_orig, color, line_style) in enumerate(forecasts):
        target_index = test_df.index if test_df is not None and not is_future_plot else forecast_df_orig.index
        if target_index is None or target_index.empty:
             logger.warning(f"Plot Warn ({model_name}): Invalid target index. Skipping forecast.")
             continue

        forecast_df = forecast_df_orig.reindex(target_index)
        if not is_future_plot and not forecast_df.empty: plot_end_date = max(plot_end_date, forecast_df.index.max())

        if forecast_df.empty or forecast_df['yhat'].isnull().all():
            logger.warning(f"Plot Warn ({model_name}): No valid forecast data for target period. Skipping.")
            continue

        metric_str = "" # Add metric to label if eval plot
        if not is_future_plot and metrics and model_name in metrics:
            if 'RMSE' in metrics[model_name] and pd.notna(metrics[model_name]['RMSE']): metric_str = f" (RMSE: {metrics[model_name]['RMSE']:.2f})"
            elif 'MAE' in metrics[model_name] and pd.notna(metrics[model_name]['MAE']): metric_str = f" (MAE: {metrics[model_name]['MAE']:.2f})"

        label = f'{model_name} Forecast{metric_str}'
        ax.plot(forecast_df.index, forecast_df['yhat'], label=label, color=color, linestyle=line_style, linewidth=2, zorder=i + 4)

        # Plot CI if available and valid
        if show_ci and 'yhat_lower' in forecast_df.columns and 'yhat_upper' in forecast_df.columns:
             lower_valid = pd.api.types.is_numeric_dtype(forecast_df['yhat_lower']) and not forecast_df['yhat_lower'].isnull().all()
             upper_valid = pd.api.types.is_numeric_dtype(forecast_df['yhat_upper']) and not forecast_df['yhat_upper'].isnull().all()
             if lower_valid and upper_valid:
                 try:
                     ci_label_suffix = f' {ci_level_pct:.0f}% CI' if len(forecasts) > 1 else f'{ci_level_pct:.0f}% CI'
                     ax.fill_between(forecast_df.index, forecast_df['yhat_lower'], forecast_df['yhat_upper'],
                                     color=color, alpha=ci_alpha, label=f'{model_name}{ci_label_suffix}', zorder=i + 3)
                 except Exception as fill_err:
                     logger.warning(f"Plot Warn ({model_name}): Could not plot CI. Error: {fill_err}")

    # 5. Set Axis Limits (full history to end of plot)
    if plot_start_date is not None and plot_end_date is not None:
        try:
             padding = pd.Timedelta(days=30) # Default padding
             if hasattr(plot_end_date, 'freq') and plot_end_date.freq: padding = plot_end_date.freq * 1
             ax.set_xlim(plot_start_date, plot_end_date + padding)
        except Exception as e: logger.warning(f"Could not set xlim automatically: {e}.")


# --- Main Plotting Functions ---

def plot_individual_model_evaluations(
    train_df: pd.DataFrame, test_df: pd.DataFrame, trend_line_series: Optional[pd.Series],
    forecast_dict: Dict[str, pd.DataFrame], evaluation_metrics: Optional[pd.DataFrame],
    config_params: Dict[str, Any], file_prefix: str = "eval_individual"
):
    """Generates individual plots for each model's evaluation forecast."""
    logger.info("--- Generating Individual Model Evaluation Plots ---")
    plots_dir, save_plots, show_plots, plot_format = _setup_plot_environment(config_params)

    active_models = [k for k, v in forecast_dict.items() if isinstance(v, pd.DataFrame) and not v.empty and 'yhat' in v.columns]
    if not active_models: logger.warning("Skipping individual eval plots: No valid forecasts."); return

    model_colors = _get_plot_colors(active_models, config_params.get('PLOT_COLOR_MAP', DEFAULT_COLOR_MAP))
    metrics_dict = evaluation_metrics.to_dict('index') if evaluation_metrics is not None else None

    for model_name in active_models:
        forecast_df = forecast_dict[model_name]
        fig, ax = plt.subplots(figsize=config_params.get('PLOT_FIG_SIZE_SINGLE', DEFAULT_FIG_SIZE_SINGLE))
        title = f"{model_name} Forecast vs Actuals (Evaluation Period)"
        _plot_base(ax=ax, historical_df=train_df, test_df=test_df, trend_line_series=trend_line_series,
                   config_params=config_params, forecasts=[(model_name, forecast_df, model_colors[model_name], '--')],
                   metrics=metrics_dict, show_ci=True, is_future_plot=False)
        _format_axis(ax, config_params, title)
        fig_path = os.path.join(plots_dir, f"{file_prefix}_{model_name}.{plot_format}")
        _save_and_show_plot(fig, fig_path, save_plots, show_plots, config_params.get('PLOT_DPI', DEFAULT_DPI))
    logger.info("--- Finished Individual Model Evaluation Plots ---")


def plot_pairwise_model_comparisons(
    train_df: pd.DataFrame, test_df: pd.DataFrame, trend_line_series: Optional[pd.Series],
    forecast_dict: Dict[str, pd.DataFrame], evaluation_metrics: Optional[pd.DataFrame],
    config_params: Dict[str, Any], file_prefix: str = "eval_pairwise"
):
    """Generates pairwise comparison plots for model evaluation forecasts."""
    logger.info("--- Generating Pairwise Model Comparison Plots ---")
    plots_dir, save_plots, show_plots, plot_format = _setup_plot_environment(config_params)

    active_models = [k for k, v in forecast_dict.items() if isinstance(v, pd.DataFrame) and not v.empty and 'yhat' in v.columns]
    if len(active_models) < 2: logger.warning("Skipping pairwise plots: < 2 models."); return

    model_colors = _get_plot_colors(active_models, config_params.get('PLOT_COLOR_MAP', DEFAULT_COLOR_MAP))
    metrics_dict = evaluation_metrics.to_dict('index') if evaluation_metrics is not None else None
    line_styles = ['--', ':']

    for i, (model1_name, model2_name) in enumerate(itertools.combinations(active_models, 2)):
        forecast1_df = forecast_dict[model1_name]
        forecast2_df = forecast_dict[model2_name]
        fig, ax = plt.subplots(figsize=config_params.get('PLOT_FIG_SIZE_PAIRWISE', DEFAULT_FIG_SIZE_PAIRWISE))
        title = f"Comparison: {model1_name} vs {model2_name} (Evaluation Period)"
        _plot_base(ax=ax, historical_df=train_df, test_df=test_df, trend_line_series=trend_line_series,
                   config_params=config_params,
                   forecasts=[(model1_name, forecast1_df, model_colors[model1_name], line_styles[0]),
                              (model2_name, forecast2_df, model_colors[model2_name], line_styles[1])],
                   metrics=metrics_dict, show_ci=config_params.get('PLOT_SHOW_CI_PAIRWISE', True), is_future_plot=False)
        _format_axis(ax, config_params, title)
        fig_path = os.path.join(plots_dir, f"{file_prefix}_{model1_name}_vs_{model2_name}.{plot_format}")
        _save_and_show_plot(fig, fig_path, save_plots, show_plots, config_params.get('PLOT_DPI', DEFAULT_DPI))
    logger.info("--- Finished Pairwise Model Comparison Plots ---")


def plot_residuals_comparison(
    test_df: pd.DataFrame, forecast_dict: Dict[str, pd.DataFrame], config_params: Dict[str, Any],
    file_prefix: str = "eval_residuals"
):
    """Generates a plot comparing the residuals (Actual - Forecast) of models."""
    logger.info("--- Generating Evaluation Residuals Comparison Plot ---")
    plots_dir, save_plots, show_plots, plot_format = _setup_plot_environment(config_params)

    if 'y' not in test_df.columns: logger.error("Residual plot error: 'y' not in test_df."); return

    active_models = [k for k, v in forecast_dict.items() if isinstance(v, pd.DataFrame) and not v.empty and 'yhat' in v.columns]
    if not active_models: logger.warning("Skipping residuals plot: No valid forecasts."); return

    model_colors = _get_plot_colors(active_models, config_params.get('PLOT_COLOR_MAP', DEFAULT_COLOR_MAP))
    markers = ['o', 's', '^', 'd', 'v', '<', '>', 'p', '*', 'X']

    fig, ax = plt.subplots(figsize=config_params.get('PLOT_FIG_SIZE_RESIDUAL', DEFAULT_FIG_SIZE_RESIDUAL))
    title = "Forecast Residuals (Actual - Forecast) during Evaluation"
    ax.axhline(0, color='black', linestyle='-', linewidth=1.5, alpha=0.8, zorder=1) # Zero line

    plotted_residuals = False
    for i, model_name in enumerate(active_models):
        forecast_df = forecast_dict[model_name]
        y_pred_aligned = forecast_df['yhat'].reindex(test_df.index)
        residuals = test_df['y'] - y_pred_aligned
        valid_residuals = residuals.dropna()

        if not valid_residuals.empty:
            ax.plot(valid_residuals.index, valid_residuals, label=f'{model_name}', color=model_colors[model_name],
                    linestyle='None', marker=markers[i % len(markers)], markersize=6, alpha=0.7, zorder=i + 2)
            plotted_residuals = True
        else: logger.warning(f"Residual Plot Warn ({model_name}): No valid residuals to plot.")

    if not plotted_residuals: logger.warning("Residual plot skipped: No models had valid residuals."); plt.close(fig); return

    _format_axis(ax, config_params, title, is_residual_plot=True)

    # Legend below plot for residuals
    handles, labels = ax.get_legend_handles_labels()
    if handles:
         try:
             fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.05),
                        ncol=min(len(handles), 5), fontsize=config_params.get('PLOT_FONTSIZE_LEGEND', DEFAULT_FONTSIZE_LEGEND), frameon=False)
             fig.tight_layout(rect=[0, 0.05, 1, 1]) # Adjust bottom for legend
         except Exception as legend_err:
             logger.warning(f"Could not place legend below residual plot: {legend_err}")
             fig.subplots_adjust(bottom=0.2) # Fallback

    fig_path = os.path.join(plots_dir, f"{file_prefix}_comparison.{plot_format}")
    _save_and_show_plot(fig, fig_path, save_plots, show_plots, config_params.get('PLOT_DPI', DEFAULT_DPI))
    logger.info("--- Finished Residuals Comparison Plot ---")


def plot_future_forecasts(
    historical_df: pd.DataFrame, trend_line_series: Optional[pd.Series],
    future_forecast_dict: Dict[str, pd.DataFrame], config_params: Dict[str, Any],
    file_prefix: str = "future_forecast"
):
    """Generates a combined plot showing historical data and future forecasts."""
    logger.info("--- Generating Future Forecast Plot ---")
    plots_dir, save_plots, show_plots, plot_format = _setup_plot_environment(config_params)

    active_models = [k for k, v in future_forecast_dict.items() if isinstance(v, pd.DataFrame) and not v.empty and 'yhat' in v.columns]
    if not active_models: logger.warning("Skipping future plot: No valid forecasts."); return
    if historical_df.empty or 'y' not in historical_df.columns: logger.warning("Skipping future plot: Historical data empty/missing 'y'."); return

    model_colors = _get_plot_colors(active_models, config_params.get('PLOT_COLOR_MAP', DEFAULT_COLOR_MAP))
    fig, ax = plt.subplots(figsize=config_params.get('PLOT_FIG_SIZE_FUTURE', DEFAULT_FIG_SIZE_FUTURE))

    # Determine forecast horizon from data
    forecast_horizon = 0
    first_fc = future_forecast_dict.get(active_models[0])
    if first_fc is not None: forecast_horizon = len(first_fc)
    title = f"Future Forecast ({forecast_horizon} Periods)"

    # Prepare forecasts for plotting
    forecasts_to_plot = [(m, future_forecast_dict[m], model_colors[m], '--') for m in active_models]

    # Use base plotting function
    _plot_base(ax=ax, historical_df=historical_df, test_df=None, trend_line_series=trend_line_series,
               config_params=config_params, forecasts=forecasts_to_plot, metrics=None,
               show_ci=config_params.get('PLOT_SHOW_CI_FUTURE', True), is_future_plot=True)

    _format_axis(ax, config_params, title) # Final formatting
    fig_path = os.path.join(plots_dir, f"{file_prefix}_combined.{plot_format}")
    _save_and_show_plot(fig, fig_path, save_plots, show_plots, config_params.get('PLOT_DPI', DEFAULT_DPI))
    logger.info("--- Finished Future Forecast Plot ---")

# --- Main Orchestration Function ---

def generate_all_plots(
    train_df: pd.DataFrame, test_df: pd.DataFrame, full_df: pd.DataFrame,
    eval_forecast_dict: Dict[str, pd.DataFrame], future_forecast_dict: Dict[str, pd.DataFrame],
    evaluation_metrics: Optional[pd.DataFrame], config_params: Dict[str, Any]
):
    """Calls all relevant plotting functions after calculating trend line."""
    logger.info("--- Generating All Plots ---")

    if not config_params.get('SAVE_PLOTS', False) and not config_params.get('SHOW_PLOTS', False):
        logger.info("Plotting disabled (SAVE_PLOTS=False and SHOW_PLOTS=False). Skipping.")
        return

    # Calculate Trend Line on Train+Val Data
    trend_line_series: Optional[pd.Series] = None
    if not train_df.empty and 'y' in train_df.columns:
        try:
            y_values = train_df['y'].dropna() # Use non-NaN for trend calc
            if len(y_values) >= 2: # Need >= 2 points
                x_numeric = np.arange(len(y_values))
                coeffs = np.polyfit(x_numeric, y_values.values, 1) # Linear trend
                trend_values = np.polyval(coeffs, x_numeric)
                trend_line_series = pd.Series(trend_values, index=y_values.index, name='Trend')
                logger.info(f"Calculated linear trend (slope={coeffs[0]:.4f}, intercept={coeffs[1]:.4f}) over train+val.")
            else: logger.warning("Skipping trend line: < 2 non-NaN data points in train_df.")
        except Exception as e: logger.error(f"Error calculating trend line: {e}", exc_info=True); trend_line_series = None
    else: logger.warning("Skipping trend line: train_df empty or missing 'y'.")

    # Call specific plot functions based on available data
    if eval_forecast_dict:
        plot_individual_model_evaluations(train_df=train_df, test_df=test_df, trend_line_series=trend_line_series,
                                          forecast_dict=eval_forecast_dict, evaluation_metrics=evaluation_metrics, config_params=config_params)
        plot_pairwise_model_comparisons(train_df=train_df, test_df=test_df, trend_line_series=trend_line_series,
                                        forecast_dict=eval_forecast_dict, evaluation_metrics=evaluation_metrics, config_params=config_params)
    else: logger.info("Skipping individual & pairwise eval plots (no eval forecasts).")

    if eval_forecast_dict and not test_df.empty:
        plot_residuals_comparison(test_df=test_df, forecast_dict=eval_forecast_dict, config_params=config_params)
    else: logger.info("Skipping residuals plot (no eval forecasts or test data).")

    # Only plot future if run and data available
    if config_params.get('RUN_FINAL_FORECAST', False) and future_forecast_dict and not full_df.empty:
        plot_future_forecasts(historical_df=full_df, trend_line_series=trend_line_series,
                              future_forecast_dict=future_forecast_dict, config_params=config_params)
    elif not config_params.get('RUN_FINAL_FORECAST', False): logger.info("Skipping future plot (RUN_FINAL_FORECAST=False).")
    else: logger.info("Skipping future plot (no future forecasts or full dataset).")

    logger.info("--- Finished Generating All Plots ---")