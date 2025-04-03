# src/plotting.py
import matplotlib.pyplot as plt, matplotlib.dates as mdates, matplotlib.colors as mcolors
import pandas as pd, numpy as np, os, logging, itertools
from typing import Dict, Optional, Any, List, Tuple

logger=logging.getLogger(__name__)

def _setup_plot(cfg:Dict[str,Any])->Tuple[str,bool,bool,str]:
	"""Prepares directories and gets save/show flags from config."""
	res_dir=cfg.get('RESULTS_DIR','results'); plots_dir=os.path.join(res_dir,'plots')
	save=cfg.get('SAVE_PLOTS',True); show=cfg.get('SHOW_PLOTS',False)
	fmt=cfg.get('PLOT_OUTPUT_FORMAT','png').lower() # Already read from config
	if save:
		try: os.makedirs(plots_dir,exist_ok=True); logger.info(f"Plots will be saved to: {plots_dir}")
		except OSError as e: logger.error(f"Could not create plots dir '{plots_dir}'. Saving disabled. Err: {e}"); save=False
	return plots_dir,save,show,fmt

def _get_colors(names:List[str], cfg:Dict[str,Any])->Dict[str,str]:
	"""Gets colors for model lines using the configured colormap."""
	cmap_name = cfg.get('PLOT_COLOR_MAP', 'tab10') # Use configured cmap
	try:
		cmap=plt.get_cmap(cmap_name); n_needed=len(names)
		if n_needed == 0: return {}
		if isinstance(cmap,mcolors.ListedColormap):
			n_cmap=len(cmap.colors); colors=[cmap(i%n_cmap) for i in range(n_needed)]
		else: colors=[cmap(i/max(1,n_needed-1)) for i in range(n_needed)]
		return {name:mcolors.to_hex(colors[i]) for i,name in enumerate(names)}
	except Exception as e:
		logger.error(f"Could not get cmap '{cmap_name}'. Falling back to basic colors. Err: {e}")
		basic=['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd','#8c564b','#e377c2','#7f7f7f','#bcbd22','#17becf']
		return {name:basic[i%len(basic)] for i,name in enumerate(names)}

def _format_ax(ax:plt.Axes,cfg:Dict[str,Any],title:str="",is_resid:bool=False):
	"""Applies configured formatting to the plot axes."""
	# Get labels from config
	x_lbl = cfg.get('PLOT_LABEL_X', "Date")
	y_lbl_val = cfg.get('PLOT_LABEL_Y_VALUE', "Value")
	y_lbl_res = cfg.get('PLOT_LABEL_Y_RESIDUAL', "Residual")
	y_lbl = y_lbl_res if is_resid else y_lbl_val

    # Get font sizes from config
	ft_lbl=cfg.get('PLOT_FONTSIZE_LABEL', 13)
	ft_ticks=cfg.get('PLOT_FONTSIZE_TICKS', 10)
	ft_title=cfg.get('PLOT_FONTSIZE_TITLE', 16)
	ft_legend=cfg.get('PLOT_FONTSIZE_LEGEND', 10)

	ax.set_title(title,fontsize=ft_title);
	ax.set_xlabel(x_lbl,fontsize=ft_lbl);
	ax.set_ylabel(y_lbl,fontsize=ft_lbl)

    # Grid settings from config
	if cfg.get('PLOT_GRID_VISIBLE', True):
		grid_ls = cfg.get('PLOT_GRID_LINESTYLE', ':')
		grid_alpha = cfg.get('PLOT_GRID_ALPHA', 0.6)
		ax.grid(True,linestyle=grid_ls,alpha=grid_alpha)
	else:
		ax.grid(False)

	ax.tick_params(axis='both',which='major',labelsize=ft_ticks)

	# Date formatting (keep simple AutoDateLocator for now)
	try:
		locator=mdates.AutoDateLocator(minticks=5,maxticks=12); formatter=mdates.ConciseDateFormatter(locator)
		ax.xaxis.set_major_locator(locator); ax.xaxis.set_major_formatter(formatter)
		# Improve rotation for potentially dense labels
		plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
	except Exception as date_fmt_err: logger.warning(f"Could not apply auto date format. Err: {date_fmt_err}")

	# Legend settings from config
	handles,labels=ax.get_legend_handles_labels()
	if handles:
		legend_loc = cfg.get('PLOT_LEGEND_LOCATION', 'best')
		legend_frame = cfg.get('PLOT_LEGEND_FRAME', True)
		ncol = 1 # Default columns

		# Special handling for 'outside' placement (useful for busy plots)
		if legend_loc == 'outside':
			if len(handles) > 6: # Place outside if many items
				ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0., fontsize=ft_legend, frameon=legend_frame)
				try: plt.gcf().tight_layout(rect=[0,0,0.85,1]) # Adjust layout to make space
				except ValueError as layout_err: logger.warning(f"Tight_layout fail for outside legend: {layout_err}"); plt.gcf().subplots_adjust(right=0.75)
			else: # If few items, put inside even if 'outside' was requested
				ax.legend(loc='best', fontsize=ft_legend, frameon=legend_frame)

		# Handle placement below plot (often for residuals)
		elif legend_loc == 'below':
		    ncol = min(len(handles), 5) # Auto columns based on number of items
		    # Place legend below axes
		    ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.15), # Adjust y position as needed
		              ncol=ncol, fontsize=ft_legend, frameon=legend_frame)
		    try:
		        plt.gcf().tight_layout(rect=[0, 0.05, 1, 0.95]) # Adjust layout to prevent overlap
		    except ValueError as layout_err: logger.warning(f"Tight_layout fail for below legend: {layout_err}"); plt.gcf().subplots_adjust(bottom=0.25)

		# Standard locations
		else:
		    ax.legend(loc=legend_loc, fontsize=ft_legend, frameon=legend_frame)


def _save_show(fig:plt.Figure,fpath:str,save:bool,show:bool,cfg:Dict[str,Any]):
	"""Saves and/or shows the plot based on config."""
	dpi = cfg.get('PLOT_DPI', 150) # Get DPI from config
	if save:
		try: fig.savefig(fpath,dpi=dpi,bbox_inches='tight'); logger.info(f"Plot saved: {fpath}")
		except Exception as e: logger.error(f"Failed save plot {fpath}: {e}")
	if show: plt.show()
	plt.close(fig) # Close figure to free memory

def _plot_base(ax:plt.Axes,hist_df:pd.DataFrame,test_df:Optional[pd.DataFrame],trend_ser:Optional[pd.Series],cfg:Dict[str,Any],fcsts:List[Tuple[str,pd.DataFrame,str,str]],metrics:Optional[Dict[str,Dict[str,float]]]=None,show_ci:bool=True,is_future:bool=False):
	"""Internal function to plot the base data (hist, actual, trend) and forecasts."""
    # Get colors and styles from config
	c_hist=cfg.get('PLOT_COLOR_HISTORICAL', 'darkgrey')
	c_act=cfg.get('PLOT_COLOR_ACTUAL', 'black')
	c_tr=cfg.get('PLOT_COLOR_TREND', 'red')
	ls_tr=cfg.get('PLOT_LINESTYLE_TREND', ':')
	ci_a=cfg.get('PLOT_CI_ALPHA', 0.15)

	# Plot historical data
	if hist_df is not None and not hist_df.empty and 'y' in hist_df.columns:
		start_dt=hist_df.index.min(); end_dt=hist_df.index.max()
		ax.plot(hist_df.index,hist_df['y'],label='Historical Data',color=c_hist,lw=1.5,zorder=1)
	else:
		logger.warning("Hist data empty/missing 'y'. Base plot might be limited."); start_dt=None; end_dt=None

    # Plot trend line
	if trend_ser is not None and not trend_ser.empty:
		# Ensure trend is plotted only over the range it was calculated for (usually hist_df)
		plot_trend_index = hist_df.index if hist_df is not None else trend_ser.index
		trend_plot=trend_ser.reindex(plot_trend_index) # Reindex to match historical data span
		if not trend_plot.isnull().all():
			ax.plot(trend_plot.index,trend_plot.values,label='Overall Trend',color=c_tr,ls=ls_tr,lw=2,zorder=2)

    # Plot actual test data
	if test_df is not None and not test_df.empty and not is_future and 'y' in test_df.columns:
		ax.plot(test_df.index,test_df['y'],label='Actual Data (Test)',color=c_act,lw=2,marker='.',ms=5,zorder=len(fcsts)+3)
		if end_dt is None: end_dt = test_df.index.max()
		else: end_dt=max(end_dt, test_df.index.max())

	# Determine end date from forecasts if needed
	if end_dt is None and fcsts:
	    try:
	        fcst_end_dts = [f[1].index.max() for f in fcsts if f[1] is not None and not f[1].empty]
	        if fcst_end_dts: end_dt = max(fcst_end_dts)
	    except Exception: pass # Ignore errors finding max date


	# Plot forecasts
	for i,(name,fcst_df_orig,color,ls) in enumerate(fcsts):
		if fcst_df_orig is None or fcst_df_orig.empty:
			logger.warning(f"Plot Warn ({name}): Fcst data is None or empty. Skip.")
			continue

        # Determine target index for plotting this forecast
		if is_future:
		    target_idx = fcst_df_orig.index # Use forecast's own index for future plots
		    if end_dt is None: end_dt = target_idx.max()
		    else: end_dt = max(end_dt, target_idx.max())
		elif test_df is not None and not test_df.empty:
		    target_idx = test_df.index # Align with test data index for eval plots
		else:
		    target_idx = fcst_df_orig.index # Fallback if no test data
		    if end_dt is None: end_dt = target_idx.max()
		    else: end_dt = max(end_dt, target_idx.max())


		if target_idx is None or target_idx.empty: logger.warning(f"Plot Warn ({name}): Invalid target index. Skip fcst."); continue
		fcst_df=fcst_df_orig.reindex(target_idx) # Reindex to ensure alignment

		if fcst_df.empty or fcst_df['yhat'].isnull().all(): logger.warning(f"Plot Warn ({name}): No valid fcst data after reindex. Skip."); continue

		met_str="" # Add metric to label if available for eval plots
		if not is_future and metrics and name in metrics:
			# Prioritize RMSE, then MAE
			if 'RMSE' in metrics[name] and pd.notna(metrics[name]['RMSE']): met_str=f" (RMSE: {metrics[name]['RMSE']:.2f})"
			elif 'MAE' in metrics[name] and pd.notna(metrics[name]['MAE']): met_str=f" (MAE: {metrics[name]['MAE']:.2f})"
		label=f'{name} Forecast{met_str}'

		# Plot the main forecast line
		ax.plot(fcst_df.index,fcst_df['yhat'],label=label,color=color,ls=ls,lw=2,zorder=i+4)

		# Plot confidence intervals if requested and available
		if show_ci and 'yhat_lower' in fcst_df.columns and 'yhat_upper' in fcst_df.columns:
			# Check if CI columns are numeric and not all NaN
			low_ok = pd.api.types.is_numeric_dtype(fcst_df['yhat_lower']) and not fcst_df['yhat_lower'].isnull().all()
			up_ok = pd.api.types.is_numeric_dtype(fcst_df['yhat_upper']) and not fcst_df['yhat_upper'].isnull().all()
			if low_ok and up_ok:
				try:
					# Get interval width from Prophet config for label (fallback if not Prophet)
					ci_pct = cfg.get('PROPHET_INTERVAL_WIDTH', 0.95) * 100
					ci_lbl_suf=f' {ci_pct:.0f}% CI'
					ax.fill_between(fcst_df.index,fcst_df['yhat_lower'],fcst_df['yhat_upper'],color=color,alpha=ci_a,label=f'_{name}_CI', zorder=i+3) # Use _nolabel_ format if label gets redundant
				except Exception as fill_err: logger.warning(f"Plot Warn ({name}): Could not plot CI. Err: {fill_err}")

	# Set plot limits with some padding
	if start_dt is not None and end_dt is not None:
		try:
		    time_range = end_dt - start_dt
		    # Add padding relative to the time range (e.g., 5%)
		    pad = time_range * 0.05
		    if pad < pd.Timedelta(days=1): pad = pd.Timedelta(days=30) # Min padding
		    ax.set_xlim(start_dt - pad*0.1, end_dt + pad) # Add padding to both ends
		except Exception as e: logger.warning(f"Could not set xlim: {e}.")
	elif end_dt is not None: # Handle case where start_dt wasn't set
	    ax.set_xlim(right=end_dt + pd.Timedelta(days=30))


def plot_individual_model_evals(train_df:pd.DataFrame,test_df:pd.DataFrame,trend_ser:Optional[pd.Series],fcst_dict:Dict[str,pd.DataFrame],metrics_df:Optional[pd.DataFrame],cfg:Dict[str,Any],prefix:str="eval_indiv"):
	logger.info("--- Generating Individual Model Eval Plots ---")
	plots_dir,save,show,fmt=_setup_plot(cfg)
	active=[k for k,v in fcst_dict.items() if isinstance(v,pd.DataFrame) and not v.empty and 'yhat' in v.columns]
	if not active: logger.warning("Skipping individual eval plots: No valid forecasts found."); return

	colors=_get_colors(active,cfg); # Get colors based on config
	metrics_dict=metrics_df.to_dict('index') if metrics_df is not None else None

	for name in active:
		fcst_df=fcst_dict[name];
		# Get fig size from config
		fig_size = cfg.get('PLOT_FIG_SIZE_SINGLE', (10, 5))
		fig,ax=plt.subplots(figsize=fig_size)
		title=f"{name} Forecast vs Actuals (Evaluation Period)"

		_plot_base(ax=ax,hist_df=train_df,test_df=test_df,trend_ser=trend_ser,cfg=cfg,fcsts=[(name,fcst_df,colors[name],'--')],metrics=metrics_dict,show_ci=True,is_future=False) # CI always shown for individual plots
		_format_ax(ax,cfg,title); fpath=os.path.join(plots_dir,f"{prefix}_{name}.{fmt}")
		_save_show(fig,fpath,save,show,cfg) # Pass full cfg
	logger.info("--- Finished Individual Model Eval Plots ---")

def plot_pairwise_model_comps(train_df:pd.DataFrame,test_df:pd.DataFrame,trend_ser:Optional[pd.Series],fcst_dict:Dict[str,pd.DataFrame],metrics_df:Optional[pd.DataFrame],cfg:Dict[str,Any],prefix:str="eval_pair"):
	logger.info("--- Generating Pairwise Model Comparison Plots ---")
	plots_dir,save,show,fmt=_setup_plot(cfg)
	active=[k for k,v in fcst_dict.items() if isinstance(v,pd.DataFrame) and not v.empty and 'yhat' in v.columns]
	if len(active)<2: logger.warning("Skipping pairwise plots: Fewer than 2 models have valid forecasts."); return

	colors=_get_colors(active,cfg); # Get colors based on config
	metrics_dict=metrics_df.to_dict('index') if metrics_df is not None else None;
	ls=['--',':'] # Simple linestyles for pairs

	show_ci_pairwise = cfg.get('PLOT_SHOW_CI_PAIRWISE', True) # Get flag from config

	for i,(m1,m2) in enumerate(itertools.combinations(active,2)):
		fcst1=fcst_dict[m1]; fcst2=fcst_dict[m2]
		# Get fig size from config
		fig_size = cfg.get('PLOT_FIG_SIZE_PAIRWISE', (11, 6))
		fig,ax=plt.subplots(figsize=fig_size);
		title=f"Comparison: {m1} vs {m2} (Evaluation Period)"

		_plot_base(ax=ax,hist_df=train_df,test_df=test_df,trend_ser=trend_ser,cfg=cfg,fcsts=[(m1,fcst1,colors[m1],ls[0]),(m2,fcst2,colors[m2],ls[1])],metrics=metrics_dict,show_ci=show_ci_pairwise,is_future=False) # Use configured CI flag
		_format_ax(ax,cfg,title); fpath=os.path.join(plots_dir,f"{prefix}_{m1}_vs_{m2}.{fmt}")
		_save_show(fig,fpath,save,show,cfg) # Pass full cfg
	logger.info("--- Finished Pairwise Model Comparison Plots ---")

def plot_residuals_comp(test_df:pd.DataFrame,fcst_dict:Dict[str,pd.DataFrame],cfg:Dict[str,Any],prefix:str="eval_resid"):
	logger.info("--- Generating Evaluation Residuals Comparison Plot ---")
	plots_dir,save,show,fmt=_setup_plot(cfg)
	if test_df is None or test_df.empty or 'y' not in test_df.columns: logger.error("Residual plot error: test_df is empty or missing 'y' column."); return

	active=[k for k,v in fcst_dict.items() if isinstance(v,pd.DataFrame) and not v.empty and 'yhat' in v.columns]
	if not active: logger.warning("Skipping residuals plot: No valid forecasts found."); return

	colors=_get_colors(active,cfg); # Get colors based on config
	markers=['o','s','^','d','v','<','>','p','*','X'] # Standard markers cycle

	# Get fig size from config
	fig_size = cfg.get('PLOT_FIG_SIZE_RESIDUAL', (12, 5))
	fig,ax=plt.subplots(figsize=fig_size)
	title="Forecast Residuals (Actual - Forecast) during Evaluation";
	ax.axhline(0,color='black',ls='-',lw=1.5,alpha=0.8,zorder=1) # Horizontal line at zero

	plotted_any=False
	for i,name in enumerate(active):
		fcst_df=fcst_dict[name];
		# Align forecast to test data index and calculate residuals
		y_pred_aligned=fcst_df['yhat'].reindex(test_df.index)
		resids=test_df['y']-y_pred_aligned;
		valid_resids=resids.dropna() # Remove NaNs where prediction or actual was missing

		if not valid_resids.empty:
			ax.plot(valid_resids.index,valid_resids,label=f'{name}',color=colors[name],ls='None',marker=markers[i%len(markers)],ms=6,alpha=0.7,zorder=i+2)
			plotted_any=True
		else: logger.warning(f"Residual Plot Warn ({name}): No valid residuals to plot.")

	if not plotted_any: logger.warning("Residual plot skipped: No models had valid residuals."); plt.close(fig); return

	# Format axes, potentially placing legend below
	cfg_copy = cfg.copy() # Create copy to potentially modify legend location for this plot type
	if cfg.get('PLOT_LEGEND_LOCATION') == 'best': # If default 'best', suggest placing below for residuals
	    cfg_copy['PLOT_LEGEND_LOCATION'] = 'below'
	_format_ax(ax,cfg_copy,title,is_resid=True) # Pass modified config, indicate it's residual plot

	fpath=os.path.join(plots_dir,f"{prefix}_comp.{fmt}")
	_save_show(fig,fpath,save,show,cfg) # Pass original cfg for saving
	logger.info("--- Finished Residuals Comparison Plot ---")

def plot_future_forecasts(hist_df:pd.DataFrame,trend_ser:Optional[pd.Series],future_fcsts:Dict[str,pd.DataFrame],cfg:Dict[str,Any],prefix:str="future_fcst"):
	logger.info("--- Generating Future Forecast Plot ---")
	plots_dir,save,show,fmt=_setup_plot(cfg)
	active=[k for k,v in future_fcsts.items() if isinstance(v,pd.DataFrame) and not v.empty and 'yhat' in v.columns]
	if not active: logger.warning("Skipping future forecast plot: No valid future forecasts found."); return
	if hist_df is None or hist_df.empty or 'y' not in hist_df.columns: logger.warning("Skipping future forecast plot: Historical data is empty or missing 'y'."); return

	colors=_get_colors(active,cfg); # Get colors based on config
	# Get fig size from config
	fig_size = cfg.get('PLOT_FIG_SIZE_FUTURE', (14, 7))
	fig,ax=plt.subplots(figsize=fig_size)

	# Try to get horizon from the first forecast dataframe
	horizon=0; first_fc=future_fcsts.get(active[0])
	if first_fc is not None: horizon=len(first_fc)
	title=f"Future Forecast ({horizon} Periods ahead)" if horizon > 0 else "Future Forecast"

	fcsts_plot_data=[(m,future_fcsts[m],colors[m],'--') for m in active] # Use dashed lines for future
	show_ci_future = cfg.get('PLOT_SHOW_CI_FUTURE', True) # Get flag from config

	_plot_base(ax=ax,hist_df=hist_df,test_df=None,trend_ser=trend_ser,cfg=cfg,fcsts=fcsts_plot_data,metrics=None,show_ci=show_ci_future,is_future=True) # Use configured CI flag, mark as future

    # Format axes, potentially placing legend outside
	cfg_copy = cfg.copy() # Create copy to potentially modify legend location
	if cfg.get('PLOT_LEGEND_LOCATION') == 'best': # If default 'best', suggest placing outside for future
	    cfg_copy['PLOT_LEGEND_LOCATION'] = 'outside'
	_format_ax(ax,cfg_copy,title); # Pass modified config

	fpath=os.path.join(plots_dir,f"{prefix}_combined.{fmt}")
	_save_show(fig,fpath,save,show,cfg) # Pass original cfg for saving
	logger.info("--- Finished Future Forecast Plot ---")


def generate_all_plots(train_df:pd.DataFrame,test_df:pd.DataFrame,full_df:pd.DataFrame,eval_fcsts:Dict[str,pd.DataFrame],future_fcsts:Dict[str,pd.DataFrame],metrics_df:Optional[pd.DataFrame],cfg:Dict[str,Any]):
	logger.info("--- Generating All Plots (Using Configured Settings) ---")
	if not cfg.get('SAVE_PLOTS',False) and not cfg.get('SHOW_PLOTS',False):
		logger.info("Plotting disabled (SAVE_PLOTS=False and SHOW_PLOTS=False). Skipping plot generation.")
		return

    # Calculate Trend (using historical part: train + val)
	hist_df = pd.concat([train_df, test_df.iloc[0:0]]) # Combine train and potential validation (if val was split before train_df)
	trend_ser:Optional[pd.Series]=None
	if hist_df is not None and not hist_df.empty and 'y' in hist_df.columns:
		try:
			y_vals=hist_df['y'].dropna()
			if len(y_vals)>=2:
				# Use time index converted to numerical representation for polyfit
				x_num = mdates.date2num(y_vals.index.to_pydatetime())
				coeffs=np.polyfit(x_num - x_num[0], y_vals.values, 1) # Fit linear trend (degree 1) relative to start
				trend_vals=np.polyval(coeffs, x_num - x_num[0]);
				trend_ser=pd.Series(trend_vals,index=y_vals.index,name='Trend')
				logger.info(f"Calculated linear trend (Slope: {coeffs[0]:.4f}, Intercept: {coeffs[1]:.4f}) over historical period.")
			else: logger.warning("Skipping trend line calculation: Fewer than 2 non-NaN points in historical data.")
		except Exception as e: logger.error(f"Error calculating trend line: {e}",exc_info=True); trend_ser=None
	else: logger.warning("Skipping trend line calculation: Historical data (train+val) is empty or missing 'y'.")

    # Generate Evaluation Plots
	if eval_fcsts:
		plot_individual_model_evals(train_df=hist_df,test_df=test_df,trend_ser=trend_ser,fcst_dict=eval_fcsts,metrics_df=metrics_df,cfg=cfg)
		plot_pairwise_model_comps(train_df=hist_df,test_df=test_df,trend_ser=trend_ser,fcst_dict=eval_fcsts,metrics_df=metrics_df,cfg=cfg)
	else: logger.info("Skipping individual & pairwise evaluation plots (no evaluation forecasts available).")

    # Generate Residual Plot
	if eval_fcsts and test_df is not None and not test_df.empty:
	    plot_residuals_comp(test_df=test_df,fcst_dict=eval_fcsts,cfg=cfg)
	else: logger.info("Skipping residuals plot (no evaluation forecasts or test data available).")

    # Generate Future Forecast Plot
	if cfg.get('RUN_FINAL_FORECAST',False) and future_fcsts and full_df is not None and not full_df.empty:
		# Recalculate trend over the *full* dataset for the future plot context
		full_trend_ser:Optional[pd.Series]=None
		try:
		    y_vals_full = full_df['y'].dropna()
		    if len(y_vals_full) >= 2:
		        x_num_full = mdates.date2num(y_vals_full.index.to_pydatetime())
		        coeffs_full = np.polyfit(x_num_full - x_num_full[0], y_vals_full.values, 1)
		        trend_vals_full = np.polyval(coeffs_full, x_num_full - x_num_full[0])
		        full_trend_ser = pd.Series(trend_vals_full, index=y_vals_full.index, name='Overall Trend')
		        logger.info(f"Recalculated trend for future plot using full data (Slope: {coeffs_full[0]:.4f}).")
		    else: logger.warning("Skipping full trend line for future plot: < 2 non-NaN points in full data.")
		except Exception as e: logger.error(f"Error calculating full trend line for future plot: {e}"); full_trend_ser = None

		plot_future_forecasts(hist_df=full_df,trend_ser=full_trend_ser,future_fcsts=future_fcsts,cfg=cfg)

	elif not cfg.get('RUN_FINAL_FORECAST',False): logger.info("Skipping future forecast plot (RUN_FINAL_FORECAST=False).")
	else: logger.info("Skipping future forecast plot (no future forecasts generated or full dataset missing).")

	logger.info("--- Finished Generating All Plots ---")