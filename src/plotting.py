import matplotlib.pyplot as plt, matplotlib.dates as mdates, matplotlib.colors as mcolors
import pandas as pd, numpy as np, os, logging, itertools
from typing import Dict, Optional, Any, List, Tuple
logger=logging.getLogger(__name__)

DEF_FS_S=(10,5); DEF_FS_P=(11,6); DEF_FS_R=(12,5); DEF_FS_F=(14,7); DEF_DPI=150
DEF_FT_T=14; DEF_FT_L=11; DEF_FT_K=9; DEF_FT_G=9; DEF_CMAP='tab10'
DEF_C_ACT='black'; DEF_C_HIST='darkgrey'; DEF_C_TR='red'; DEF_LS_TR=':'; DEF_CI_A=0.2

def _setup_plot(cfg:Dict[str,Any])->Tuple[str,bool,bool,str]:
	res_dir=cfg.get('RESULTS_DIR','results'); plots_dir=os.path.join(res_dir,'plots')
	save=cfg.get('SAVE_PLOTS',True); show=cfg.get('SHOW_PLOTS',False)
	fmt=cfg.get('PLOT_OUTPUT_FORMAT','png').lower()
	if save:
		try: os.makedirs(plots_dir,exist_ok=True); logger.info(f"Plots saved to: {plots_dir}")
		except OSError as e: logger.error(f"Could not create plots dir '{plots_dir}'. Save disabled. Err: {e}"); save=False
	return plots_dir,save,show,fmt

def _get_colors(names:List[str],cmap_name:str=DEF_CMAP)->Dict[str,str]:
	try:
		cmap=plt.get_cmap(cmap_name); n_needed=len(names)
		if isinstance(cmap,mcolors.ListedColormap):
			n_cmap=len(cmap.colors); colors=[cmap(i%n_cmap) for i in range(n_needed)]
		else: colors=[cmap(i/max(1,n_needed-1)) for i in range(n_needed)]
		return {name:mcolors.to_hex(colors[i]) for i,name in enumerate(names)}
	except Exception as e:
		logger.error(f"Could not get cmap '{cmap_name}'. Fallback. Err: {e}")
		basic=['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd','#8c564b','#e377c2','#7f7f7f','#bcbd22','#17becf']
		return {name:basic[i%len(basic)] for i,name in enumerate(names)}

def _format_ax(ax:plt.Axes,cfg:Dict[str,Any],title:str="",is_resid:bool=False):
	x_lbl="Year"; y_lbl="Residual (Actual-Fcst)" if is_resid else "Value" # Generic Y-label
	ft_lbl=cfg.get('PLOT_FONTSIZE_LABEL',DEF_FT_L); ft_ticks=cfg.get('PLOT_FONTSIZE_TICKS',DEF_FT_K)
	ft_title=cfg.get('PLOT_FONTSIZE_TITLE',DEF_FT_T); ft_legend=cfg.get('PLOT_FONTSIZE_LEGEND',DEF_FT_G)
	ax.set_title(title,fontsize=ft_title); ax.set_xlabel(x_lbl,fontsize=ft_lbl); ax.set_ylabel(y_lbl,fontsize=ft_lbl)
	ax.grid(True,linestyle=':',alpha=0.6); ax.tick_params(axis='both',which='major',labelsize=ft_ticks)
	try:
		locator=mdates.AutoDateLocator(minticks=5,maxticks=12); formatter=mdates.ConciseDateFormatter(locator)
		ax.xaxis.set_major_locator(locator); ax.xaxis.set_major_formatter(formatter)
	except Exception as date_fmt_err: logger.warning(f"Could not apply auto date format. Err: {date_fmt_err}")
	handles,labels=ax.get_legend_handles_labels()
	if handles:
		if len(handles)<=6: ax.legend(loc='best',fontsize=ft_legend,frameon=True,framealpha=0.8)
		else:
			ax.legend(loc='upper left',bbox_to_anchor=(1.02,1),borderaxespad=0.,fontsize=ft_legend,frameon=True)
			try: plt.gcf().tight_layout(rect=[0,0,0.85,1])
			except ValueError as layout_err: logger.warning(f"Tight_layout fail: {layout_err}"); plt.gcf().subplots_adjust(right=0.75)

def _save_show(fig:plt.Figure,fpath:str,save:bool,show:bool,dpi:int=DEF_DPI):
	if save:
		try: fig.savefig(fpath,dpi=dpi,bbox_inches='tight'); logger.info(f"Plot saved: {fpath}")
		except Exception as e: logger.error(f"Failed save plot {fpath}: {e}")
	if show: plt.show()
	plt.close(fig)

def _plot_base(ax:plt.Axes,hist_df:pd.DataFrame,test_df:Optional[pd.DataFrame],trend_ser:Optional[pd.Series],cfg:Dict[str,Any],fcsts:List[Tuple[str,pd.DataFrame,str,str]],metrics:Optional[Dict[str,Dict[str,float]]]=None,show_ci:bool=True,is_future:bool=False):
	c_hist=cfg.get('PLOT_HISTORICAL_COLOR',DEF_C_HIST); c_act=cfg.get('PLOT_ACTUAL_COLOR',DEF_C_ACT)
	c_tr=cfg.get('PLOT_TREND_COLOR',DEF_C_TR); ls_tr=cfg.get('PLOT_TREND_LINESTYLE',DEF_LS_TR)
	ci_a=cfg.get('PLOT_CI_ALPHA',DEF_CI_A); ci_pct=cfg.get('PROPHET_INTERVAL_WIDTH',0.95)*100
	if hist_df.empty or 'y' not in hist_df.columns: logger.warning("Hist data empty/missing 'y'. Skip base plot."); return
	start_dt=hist_df.index.min(); end_dt=hist_df.index.max()
	ax.plot(hist_df.index,hist_df['y'],label='Historical Data',color=c_hist,lw=1.5,zorder=1)
	if trend_ser is not None and not trend_ser.empty:
		trend_plot=trend_ser.reindex(hist_df.index)
		ax.plot(trend_plot.index,trend_plot.values,label='Overall Trend',color=c_tr,ls=ls_tr,lw=2,zorder=2)
	if test_df is not None and not is_future and 'y' in test_df.columns:
		ax.plot(test_df.index,test_df['y'],label='Actual Data (Test)',color=c_act,lw=2,marker='.',ms=5,zorder=len(fcsts)+3)
		end_dt=test_df.index.max()
	elif fcsts:
		try:
			fcst_end_dts=[f[1].index.max() for f in fcsts if f[1] is not None and not f[1].empty]
			if fcst_end_dts: end_dt=max(end_dt,max(fcst_end_dts))
		except Exception: pass
	for i,(name,fcst_df_orig,color,ls) in enumerate(fcsts):
		target_idx=test_df.index if test_df is not None and not is_future else fcst_df_orig.index
		if target_idx is None or target_idx.empty: logger.warning(f"Plot Warn ({name}): Invalid target idx. Skip fcst."); continue
		fcst_df=fcst_df_orig.reindex(target_idx)
		if not is_future and not fcst_df.empty: end_dt=max(end_dt,fcst_df.index.max())
		if fcst_df.empty or fcst_df['yhat'].isnull().all(): logger.warning(f"Plot Warn ({name}): No valid fcst data. Skip."); continue
		met_str=""
		if not is_future and metrics and name in metrics:
			if 'RMSE' in metrics[name] and pd.notna(metrics[name]['RMSE']): met_str=f" (RMSE: {metrics[name]['RMSE']:.2f})"
			elif 'MAE' in metrics[name] and pd.notna(metrics[name]['MAE']): met_str=f" (MAE: {metrics[name]['MAE']:.2f})"
		label=f'{name} Fcst{met_str}'
		ax.plot(fcst_df.index,fcst_df['yhat'],label=label,color=color,ls=ls,lw=2,zorder=i+4)
		if show_ci and 'yhat_lower' in fcst_df.columns and 'yhat_upper' in fcst_df.columns:
			low_ok=pd.api.types.is_numeric_dtype(fcst_df['yhat_lower']) and not fcst_df['yhat_lower'].isnull().all()
			up_ok=pd.api.types.is_numeric_dtype(fcst_df['yhat_upper']) and not fcst_df['yhat_upper'].isnull().all()
			if low_ok and up_ok:
				try:
					ci_lbl_suf=f' {ci_pct:.0f}% CI' if len(fcsts)>1 else f'{ci_pct:.0f}% CI'
					ax.fill_between(fcst_df.index,fcst_df['yhat_lower'],fcst_df['yhat_upper'],color=color,alpha=ci_a,label=f'{name}{ci_lbl_suf}',zorder=i+3)
				except Exception as fill_err: logger.warning(f"Plot Warn ({name}): Could not plot CI. Err: {fill_err}")
	if start_dt is not None and end_dt is not None:
		try: pad=pd.Timedelta(days=30); ax.set_xlim(start_dt,end_dt+pad)
		except Exception as e: logger.warning(f"Could not set xlim: {e}.")

def plot_individual_model_evals(train_df:pd.DataFrame,test_df:pd.DataFrame,trend_ser:Optional[pd.Series],fcst_dict:Dict[str,pd.DataFrame],metrics_df:Optional[pd.DataFrame],cfg:Dict[str,Any],prefix:str="eval_indiv"):
	logger.info("--- Gen Individual Model Eval Plots ---")
	plots_dir,save,show,fmt=_setup_plot(cfg)
	active=[k for k,v in fcst_dict.items() if isinstance(v,pd.DataFrame) and not v.empty and 'yhat' in v.columns]
	if not active: logger.warning("Skip indiv eval plots: No valid fcsts."); return
	colors=_get_colors(active,cfg.get('PLOT_COLOR_MAP',DEF_CMAP)); metrics_dict=metrics_df.to_dict('index') if metrics_df is not None else None
	for name in active:
		fcst_df=fcst_dict[name]; fig,ax=plt.subplots(figsize=cfg.get('PLOT_FIG_SIZE_SINGLE',DEF_FS_S))
		title=f"{name} Fcst vs Actuals (Eval Period)"
		_plot_base(ax=ax,hist_df=train_df,test_df=test_df,trend_ser=trend_ser,cfg=cfg,fcsts=[(name,fcst_df,colors[name],'--')],metrics=metrics_dict,show_ci=True,is_future=False)
		_format_ax(ax,cfg,title); fpath=os.path.join(plots_dir,f"{prefix}_{name}.{fmt}")
		_save_show(fig,fpath,save,show,cfg.get('PLOT_DPI',DEF_DPI))
	logger.info("--- Finished Individual Model Eval Plots ---")

def plot_pairwise_model_comps(train_df:pd.DataFrame,test_df:pd.DataFrame,trend_ser:Optional[pd.Series],fcst_dict:Dict[str,pd.DataFrame],metrics_df:Optional[pd.DataFrame],cfg:Dict[str,Any],prefix:str="eval_pair"):
	logger.info("--- Gen Pairwise Model Comp Plots ---")
	plots_dir,save,show,fmt=_setup_plot(cfg)
	active=[k for k,v in fcst_dict.items() if isinstance(v,pd.DataFrame) and not v.empty and 'yhat' in v.columns]
	if len(active)<2: logger.warning("Skip pairwise plots: < 2 models."); return
	colors=_get_colors(active,cfg.get('PLOT_COLOR_MAP',DEF_CMAP)); metrics_dict=metrics_df.to_dict('index') if metrics_df is not None else None; ls=['--',':']
	for i,(m1,m2) in enumerate(itertools.combinations(active,2)):
		fcst1=fcst_dict[m1]; fcst2=fcst_dict[m2]
		fig,ax=plt.subplots(figsize=cfg.get('PLOT_FIG_SIZE_PAIRWISE',DEF_FS_P)); title=f"Compare: {m1} vs {m2} (Eval Period)"
		_plot_base(ax=ax,hist_df=train_df,test_df=test_df,trend_ser=trend_ser,cfg=cfg,fcsts=[(m1,fcst1,colors[m1],ls[0]),(m2,fcst2,colors[m2],ls[1])],metrics=metrics_dict,show_ci=cfg.get('PLOT_SHOW_CI_PAIRWISE',True),is_future=False)
		_format_ax(ax,cfg,title); fpath=os.path.join(plots_dir,f"{prefix}_{m1}_vs_{m2}.{fmt}")
		_save_show(fig,fpath,save,show,cfg.get('PLOT_DPI',DEF_DPI))
	logger.info("--- Finished Pairwise Model Comp Plots ---")

def plot_residuals_comp(test_df:pd.DataFrame,fcst_dict:Dict[str,pd.DataFrame],cfg:Dict[str,Any],prefix:str="eval_resid"):
	logger.info("--- Gen Eval Residuals Comp Plot ---")
	plots_dir,save,show,fmt=_setup_plot(cfg)
	if 'y' not in test_df.columns: logger.error("Resid plot err: 'y' not in test_df."); return
	active=[k for k,v in fcst_dict.items() if isinstance(v,pd.DataFrame) and not v.empty and 'yhat' in v.columns]
	if not active: logger.warning("Skip resid plot: No valid fcsts."); return
	colors=_get_colors(active,cfg.get('PLOT_COLOR_MAP',DEF_CMAP)); markers=['o','s','^','d','v','<','>','p','*','X']
	fig,ax=plt.subplots(figsize=cfg.get('PLOT_FIG_SIZE_RESIDUAL',DEF_FS_R))
	title="Fcst Residuals (Actual - Fcst) during Eval"; ax.axhline(0,color='black',ls='-',lw=1.5,alpha=0.8,zorder=1)
	plotted=False
	for i,name in enumerate(active):
		fcst_df=fcst_dict[name]; y_pred_a=fcst_df['yhat'].reindex(test_df.index)
		resids=test_df['y']-y_pred_a; valid_resids=resids.dropna()
		if not valid_resids.empty:
			ax.plot(valid_resids.index,valid_resids,label=f'{name}',color=colors[name],ls='None',marker=markers[i%len(markers)],ms=6,alpha=0.7,zorder=i+2)
			plotted=True
		else: logger.warning(f"Resid Plot Warn ({name}): No valid resids.")
	if not plotted: logger.warning("Resid plot skipped: No models had valid resids."); plt.close(fig); return
	_format_ax(ax,cfg,title,is_resid=True)
	handles,labels=ax.get_legend_handles_labels()
	if handles:
		try:
			fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(0.5,-0.05),ncol=min(len(handles),5),fontsize=cfg.get('PLOT_FONTSIZE_LEGEND',DEF_FT_G),frameon=False)
			fig.tight_layout(rect=[0,0.05,1,1])
		except Exception as legend_err: logger.warning(f"Could not place legend below resid plot: {legend_err}"); fig.subplots_adjust(bottom=0.2)
	fpath=os.path.join(plots_dir,f"{prefix}_comp.{fmt}")
	_save_show(fig,fpath,save,show,cfg.get('PLOT_DPI',DEF_DPI))
	logger.info("--- Finished Residuals Comp Plot ---")

def plot_future_forecasts(hist_df:pd.DataFrame,trend_ser:Optional[pd.Series],future_fcsts:Dict[str,pd.DataFrame],cfg:Dict[str,Any],prefix:str="future_fcst"):
	logger.info("--- Gen Future Forecast Plot ---")
	plots_dir,save,show,fmt=_setup_plot(cfg)
	active=[k for k,v in future_fcsts.items() if isinstance(v,pd.DataFrame) and not v.empty and 'yhat' in v.columns]
	if not active: logger.warning("Skip future plot: No valid fcsts."); return
	if hist_df.empty or 'y' not in hist_df.columns: logger.warning("Skip future plot: Hist data empty/missing 'y'."); return
	colors=_get_colors(active,cfg.get('PLOT_COLOR_MAP',DEF_CMAP))
	fig,ax=plt.subplots(figsize=cfg.get('PLOT_FIG_SIZE_FUTURE',DEF_FS_F))
	horizon=0; first_fc=future_fcsts.get(active[0])
	if first_fc is not None: horizon=len(first_fc)
	title=f"Future Forecast ({horizon} Periods)"
	fcsts_plot=[(m,future_fcsts[m],colors[m],'--') for m in active]
	_plot_base(ax=ax,hist_df=hist_df,test_df=None,trend_ser=trend_ser,cfg=cfg,fcsts=fcsts_plot,metrics=None,show_ci=cfg.get('PLOT_SHOW_CI_FUTURE',True),is_future=True)
	_format_ax(ax,cfg,title); fpath=os.path.join(plots_dir,f"{prefix}_combined.{fmt}")
	_save_show(fig,fpath,save,show,cfg.get('PLOT_DPI',DEF_DPI))
	logger.info("--- Finished Future Forecast Plot ---")

def generate_all_plots(train_df:pd.DataFrame,test_df:pd.DataFrame,full_df:pd.DataFrame,eval_fcsts:Dict[str,pd.DataFrame],future_fcsts:Dict[str,pd.DataFrame],metrics_df:Optional[pd.DataFrame],cfg:Dict[str,Any]):
	logger.info("--- Generating All Plots ---")
	if not cfg.get('SAVE_PLOTS',False) and not cfg.get('SHOW_PLOTS',False): logger.info("Plotting disabled. Skip."); return
	trend_ser:Optional[pd.Series]=None
	if not train_df.empty and 'y' in train_df.columns:
		try:
			y_vals=train_df['y'].dropna()
			if len(y_vals)>=2:
				x_num=np.arange(len(y_vals)); coeffs=np.polyfit(x_num,y_vals.values,1)
				trend_vals=np.polyval(coeffs,x_num); trend_ser=pd.Series(trend_vals,index=y_vals.index,name='Trend')
				logger.info(f"Calc linear trend (slope={coeffs[0]:.4f}, intercept={coeffs[1]:.4f}) over train+val.")
			else: logger.warning("Skip trend line: < 2 non-NaN points in train_df.")
		except Exception as e: logger.error(f"Error calc trend line: {e}",exc_info=True); trend_ser=None
	else: logger.warning("Skip trend line: train_df empty or no 'y'.")
	if eval_fcsts:
		plot_individual_model_evals(train_df=train_df,test_df=test_df,trend_ser=trend_ser,fcst_dict=eval_fcsts,metrics_df=metrics_df,cfg=cfg)
		plot_pairwise_model_comps(train_df=train_df,test_df=test_df,trend_ser=trend_ser,fcst_dict=eval_fcsts,metrics_df=metrics_df,cfg=cfg)
	else: logger.info("Skip individual & pairwise eval plots (no eval fcsts).")
	if eval_fcsts and not test_df.empty: plot_residuals_comp(test_df=test_df,fcst_dict=eval_fcsts,cfg=cfg)
	else: logger.info("Skip residuals plot (no eval fcsts or test data).")
	if cfg.get('RUN_FINAL_FORECAST',False) and future_fcsts and not full_df.empty:
		plot_future_forecasts(hist_df=full_df,trend_ser=trend_ser,future_fcsts=future_fcsts,cfg=cfg)
	elif not cfg.get('RUN_FINAL_FORECAST',False): logger.info("Skip future plot (RUN_FINAL_FORECAST=False).")
	else: logger.info("Skip future plot (no future fcsts or full dataset).")
	logger.info("--- Finished Generating All Plots ---")