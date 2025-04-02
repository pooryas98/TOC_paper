import pandas as pd
import numpy as np
from prophet import Prophet
import logging, pickle, os
from typing import Optional, Dict, Any, Tuple
logger=logging.getLogger(__name__)
_holidays_available=False
try: import holidays; _holidays_available=True
except ImportError: pass

def run_prophet(train_data:pd.DataFrame,test_periods:int,test_index:pd.DatetimeIndex,cfg:Dict[str,Any])->Tuple[pd.DataFrame,Optional[Dict[str,Any]],Optional[Prophet]]:
	logger.info("--- Running Prophet ---")
	params:Dict[str,Any]={'model_type':'Prophet'}
	fitted_model:Optional[Prophet]=None
	nan_df=pd.DataFrame(np.nan,index=test_index,columns=['yhat','yhat_lower','yhat_upper'])
	if train_data['y'].isnull().any(): logger.warning("Prophet: Train data has NaNs.")
	train_df_p:pd.DataFrame=train_data.reset_index().rename(columns={'index':'ds',train_data.index.name:'ds','y':'y'})
	if not pd.api.types.is_datetime64_any_dtype(train_df_p['ds']):
		try: train_df_p['ds']=pd.to_datetime(train_df_p['ds'])
		except Exception as e: logger.error(f"Prophet Err: ds->datetime: {e}"); return nan_df, params, None
	growth=cfg['PROPHET_GROWTH']
	cap=cfg.get('PROPHET_CAP')
	floor=cfg.get('PROPHET_FLOOR')
	if growth=='logistic':
		if cap is None: logger.error("Prophet Err: Growth 'logistic' but PROPHET_CAP missing."); return nan_df,params,None
		train_df_p['cap']=cap
		if floor is not None: train_df_p['floor']=floor
		params.update({'cap':cap,'floor':floor})
	params.update({'growth':growth,'seasonality_mode':cfg['PROPHET_SEASONALITY_MODE'],'interval_width':cfg['PROPHET_INTERVAL_WIDTH'],'yearly_seasonality':cfg['PROPHET_ADD_YEARLY_SEASONALITY'],'weekly_seasonality':cfg['PROPHET_ADD_WEEKLY_SEASONALITY'],'daily_seasonality':cfg['PROPHET_ADD_DAILY_SEASONALITY']})
	country_holidays=cfg.get('PROPHET_COUNTRY_HOLIDAYS')
	prophet_holidays=None
	if country_holidays:
		if _holidays_available:
			min_dt=train_df_p['ds'].min(); max_dt_fcst=test_index.max()
			year_list=list(range(min_dt.year,max_dt_fcst.year+1))
			try:
				h_obj=holidays.country_holidays(country_holidays,years=year_list)
				if h_obj:
					prophet_holidays=pd.DataFrame(list(h_obj.items()),columns=['ds','holiday'])
					prophet_holidays['ds']=pd.to_datetime(prophet_holidays['ds'])
					prophet_holidays=prophet_holidays[(prophet_holidays['ds']>=min_dt)&(prophet_holidays['ds']<=max_dt_fcst)]
					logger.info(f"Using holidays for: {country_holidays} ({len(prophet_holidays)} found)")
					params['country_holidays_used']=country_holidays
				else: logger.warning(f"No holidays for country: {country_holidays}"); params['country_holidays_used']=f"{country_holidays} (Not Found)"
			except Exception as holiday_err: logger.error(f"Holiday err {country_holidays}: {holiday_err}"); params['country_holidays_used']=f"{country_holidays} (Error)"
		else: logger.warning("'holidays' lib missing. Skipping."); params['country_holidays_used']=f"{country_holidays} (Lib Missing)"
	else: params['country_holidays_used']=None
	try:
		model=Prophet(growth=growth,yearly_seasonality=cfg['PROPHET_ADD_YEARLY_SEASONALITY'],weekly_seasonality=cfg['PROPHET_ADD_WEEKLY_SEASONALITY'],daily_seasonality=cfg['PROPHET_ADD_DAILY_SEASONALITY'],seasonality_mode=cfg['PROPHET_SEASONALITY_MODE'],interval_width=cfg['PROPHET_INTERVAL_WIDTH'],holidays=prophet_holidays)
		logger.info("Fitting Prophet...")
		fitted_model=model.fit(train_df_p); logger.info("Prophet fit complete.")
		freq_str:Optional[str]=train_data.index.freqstr or pd.infer_freq(train_data.index)
		if freq_str: logger.info(f"Prophet freq: {freq_str}")
		else: logger.error("Prophet Err: Freq undetermined."); params['fit_error']="Freq Undetermined"; return nan_df,params,fitted_model
		future:pd.DataFrame=model.make_future_dataframe(periods=test_periods,freq=freq_str)
		if growth=='logistic':
			future['cap']=cap
			if floor is not None: future['floor']=floor
		if future.empty or len(future)!=len(train_df_p)+test_periods: raise ValueError(f"Bad future df. Expected {len(train_df_p)+test_periods}, got {len(future)}.")
		logger.info("Generating Prophet forecast...")
		fcst:pd.DataFrame=model.predict(future); logger.info("Prophet forecast done.")
		fcst_subset:pd.DataFrame=fcst.set_index('ds')[['yhat','yhat_lower','yhat_upper']].iloc[-test_periods:]
		if len(fcst_subset)!=test_periods:
			logger.warning(f"Prophet Warn: Fcst len ({len(fcst_subset)})!=test ({test_periods}). Reindexing.")
			fcst_subset=fcst_subset.reindex(test_index)
		else: fcst_subset.index=test_index
		if fcst_subset['yhat'].isnull().any(): logger.warning("Prophet fcst has NaNs.")
		logger.info("Prophet Finished.")
		return fcst_subset,params,fitted_model
	except Exception as e:
		logger.error(f"Prophet Err fit/pred: {type(e).__name__}: {e}",exc_info=True)
		params['fit_error']=f"{type(e).__name__}: {e}"
		return nan_df,params,fitted_model

def save_prophet_model(model:Prophet,file_path:str):
	"""Saves fitted Prophet model via pickle."""
	try:
		os.makedirs(os.path.dirname(file_path),exist_ok=True)
		with open(file_path,'wb') as f: pickle.dump(model,f)
		logger.info(f"Prophet model saved: {file_path}")
	except Exception as e: logger.error(f"Err saving Prophet model {file_path}: {e}",exc_info=True)