import pandas as pd, numpy as np, logging, pickle, os
from statsmodels.tsa.statespace.sarimax import SARIMAX, SARIMAXResultsWrapper
from typing import Tuple, Optional, Any, Dict
try: import pmdarima as pm; _pm_avail = True
except ImportError: _pm_avail = False
logger=logging.getLogger(__name__)

def get_s_period(freq_str:Optional[str],manual_s:Optional[int]=None)->int:
	if manual_s is not None and manual_s>1: logger.info(f"Using manual SARIMA S = {manual_s}"); return manual_s
	if not freq_str: logger.warning("SARIMA: Freq undetermined. Assuming s=1."); return 1
	freq_base=freq_str.upper().split('-')[0]
	s_map={'A':1,'Y':1,'Q':4,'M':12,'W':52,'D':7,'B':5,'H':24,'T':1440,'S':86400}
	s=s_map.get(freq_base[0],1); logger.info(f"Inferred SARIMA S = {s} from freq '{freq_str}'")
	return s

def run_sarima(train_data:pd.DataFrame,test_periods:int,test_index:pd.DatetimeIndex,cfg:Dict[str,Any])->Tuple[pd.DataFrame,Optional[Dict[str,Any]],Optional[SARIMAXResultsWrapper]]:
	logger.info("--- Running SARIMA ---")
	params:Dict[str,Any]={'model_type':'SARIMA'}
	fitted_model:Optional[SARIMAXResultsWrapper]=None
	nan_df=pd.DataFrame(np.nan,index=test_index,columns=['yhat','yhat_lower','yhat_upper'])
	if train_data['y'].isnull().any(): logger.error("SARIMA Err: Train data has NaNs."); return nan_df,params,None
	s_period:int=get_s_period(train_data.index.freqstr,cfg.get('SARIMA_MANUAL_S'))
	params['seasonal_period_used']=s_period
	order:Tuple[int,int,int]=cfg['SARIMA_ORDER']
	s_order=cfg['SARIMA_SEASONAL_ORDER_NOS']+(s_period,) if s_period>1 else (0,0,0,0)
	use_auto=cfg['USE_AUTO_ARIMA']; params['auto_arima_attempted']=use_auto
	if use_auto:
		if not _pm_avail: logger.error("SARIMA Err: USE_AUTO_ARIMA=True, but pmdarima missing."); return nan_df,params,None
		logger.info("Attempting auto_arima...")
		is_seasonal:bool=cfg['SARIMA_AUTO_SEASONAL'] and s_period>1
		try:
			min_samples=s_period*2 if is_seasonal else 2
			if len(train_data)<min_samples:
				if is_seasonal: logger.warning(f"Not enough samples ({len(train_data)}) for auto seasonal search (m={s_period}). Disabling.")
				is_seasonal=False
			m=s_period if is_seasonal else 1; D_param=None if is_seasonal else 0
			auto_model=pm.auto_arima(train_data['y'],
				start_p=cfg['AUTO_ARIMA_START_P'],max_p=cfg['AUTO_ARIMA_MAX_P'],
				start_q=cfg['AUTO_ARIMA_START_Q'],max_q=cfg['AUTO_ARIMA_MAX_Q'],
				max_d=cfg['AUTO_ARIMA_MAX_D'],m=m,seasonal=is_seasonal,
				start_P=cfg['AUTO_ARIMA_START_SP'],max_P=cfg['AUTO_ARIMA_MAX_SP'],
				start_Q=cfg['AUTO_ARIMA_START_SQ'],max_Q=cfg['AUTO_ARIMA_MAX_SQ'],
				max_D=cfg['AUTO_ARIMA_MAX_SD'],d=None,D=D_param,test=cfg['AUTO_ARIMA_TEST'],
				information_criterion=cfg['AUTO_ARIMA_IC'],stepwise=cfg['AUTO_ARIMA_STEPWISE'],
				suppress_warnings=True,error_action='ignore',trace=(logger.getEffectiveLevel()<=logging.DEBUG))
			order=auto_model.order; s_order=auto_model.seasonal_order
			params.update({'auto_arima_successful':True,'auto_arima_order':order,'auto_arima_seasonal_order':s_order})
			logger.info(f"auto_arima found Order: {order} Seasonal Order: {s_order}")
		except Exception as e:
			logger.error(f"auto_arima failed: {e}. Fallback to manual.")
			params.update({'auto_arima_successful':False,'auto_arima_error':str(e)})
			order=cfg['SARIMA_ORDER']; s_order=cfg['SARIMA_SEASONAL_ORDER_NOS']+(s_period,) if s_period>1 else (0,0,0,0)
			logger.info(f"Using manual Order: {order} Seasonal Order: {s_order}")
	else:
		logger.info(f"Using manual Order: {order}")
		if s_period>1: logger.info(f"Using manual Seasonal Order: {s_order}")
		else: logger.info("Using Non-Seasonal SARIMA (s=1)")
	params.update({'final_order':order,'final_seasonal_order':s_order,'enforce_stationarity':cfg['SARIMA_ENFORCE_STATIONARITY'],'enforce_invertibility':cfg['SARIMA_ENFORCE_INVERTIBILITY']})
	y_train=train_data['y']
	if y_train.index.freq is None:
		logger.warning("SARIMA: Train index freq missing. Inferring.")
		inf_freq=pd.infer_freq(y_train.index)
		if inf_freq:
			y_train=y_train.copy()
			try: y_train.index.freq=inf_freq; logger.info(f"Set index freq to inferred: {inf_freq}")
			except Exception as freq_err: logger.error(f"Failed set inferred freq '{inf_freq}': {freq_err}."); return nan_df,params,None
		else: logger.error("SARIMA Err: Cannot determine freq."); return nan_df,params,None
	try:
		logger.info(f"Fitting SARIMAX{order}{s_order}")
		model=SARIMAX(y_train,order=order,seasonal_order=s_order,enforce_stationarity=cfg['SARIMA_ENFORCE_STATIONARITY'],enforce_invertibility=cfg['SARIMA_ENFORCE_INVERTIBILITY'])
		fitted_model=model.fit(disp=False); logger.info("SARIMAX fit complete.")
		pred=fitted_model.get_prediction(start=test_index[0],end=test_index[-1])
		fcst_summ:pd.DataFrame=pred.summary_frame(alpha=1.0-cfg.get('PROPHET_INTERVAL_WIDTH',0.95))
		if not fcst_summ.index.equals(test_index):
			logger.warning("SARIMA fcst index mismatch. Reindexing.")
			fcst_summ=fcst_summ.reindex(test_index)
		fcst_vals:pd.DataFrame=fcst_summ[['mean','mean_ci_lower','mean_ci_upper']].rename(columns={'mean':'yhat','mean_ci_lower':'yhat_lower','mean_ci_upper':'yhat_upper'})
		fcst_vals.index=test_index
		if fcst_vals['yhat'].isnull().any(): logger.warning("SARIMA fcst has NaNs.")
		logger.info("SARIMA forecast generated.")
		return fcst_vals,params,fitted_model
	except(ValueError,np.linalg.LinAlgError) as fit_err:
		logger.error(f"SARIMA Fit/Fcst Err: {type(fit_err).__name__}: {fit_err}. Check orders/data.")
		params['fit_error']=f"{type(fit_err).__name__}: {fit_err}"
		return nan_df,params,None
	except Exception as e:
		logger.error(f"SARIMA Unexpected Err: {type(e).__name__}: {e}",exc_info=True)
		params['fit_error']=f"Unexpected {type(e).__name__}: {e}"
		return nan_df,params,None

def save_sarima_model(model:SARIMAXResultsWrapper,file_path:str):
	"""Saves fitted SARIMAX model via pickle."""
	try:
		os.makedirs(os.path.dirname(file_path),exist_ok=True)
		with open(file_path,'wb') as f: pickle.dump(model,f)
		logger.info(f"SARIMA model saved: {file_path}")
	except Exception as e: logger.error(f"Err saving SARIMA model {file_path}: {e}",exc_info=True)