import pandas as pd, numpy as np, tensorflow as tf, sys, time, os, logging, json, datetime, joblib
from typing import Dict, List, Tuple, Optional, Any
from tensorflow.keras.models import Model
from sklearn.preprocessing import MinMaxScaler
try:
	from src import config, data_loader
	from src.models import sarima_model, prophet_model, nn_models
	from src import evaluation, plotting
	from statsmodels.tsa.statespace.sarimax import SARIMAXResultsWrapper
	from prophet import Prophet
except ImportError as e: print(f"Import Err: {e}\nRun from correct dir or check sys.path."); sys.exit(1)
logger=logging.getLogger()

def set_seeds(seed:int)->None: np.random.seed(seed); tf.random.set_seed(seed); logger.info(f"Set random seeds: {seed}")

class NpEncoder(json.JSONEncoder):
	def default(self,obj):
		if isinstance(obj,np.integer): return int(obj)
		if isinstance(obj,np.floating): return float(obj)
		if isinstance(obj,np.ndarray): return obj.tolist()
		if isinstance(obj,pd.Timestamp): return obj.isoformat()
		if isinstance(obj,(pd.Timedelta,pd.Period)): return str(obj)
		if isinstance(obj,(SARIMAXResultsWrapper,Prophet,Model,MinMaxScaler)): return f"<Object type: {type(obj).__name__}>"
		try: return super(NpEncoder,self).default(obj)
		except TypeError: return f"<Unserializable type: {type(obj).__name__}>"

def save_run_params(params_dict:Dict[str,Any],fpath:str):
	try:
		os.makedirs(os.path.dirname(fpath),exist_ok=True)
		with open(fpath,'w') as f: json.dump(params_dict,f,indent=4,cls=NpEncoder)
		logger.info(f"Run params saved: {fpath}")
	except Exception as e: logger.error(f"Error saving params {fpath}: {e}",exc_info=True)

def run_comparison()->None:
	run_start_time=time.time(); logger.info("--- Starting Forecast Comparison ---")
	try: cfg=config.get_config_dict()
	except Exception as e: logger.error(f"Failed get config dict: {e}",exc_info=True); sys.exit(1)
	run_ts=datetime.datetime.now().strftime("%Y%m%d_%H%M%S"); logger.info(f"Run Timestamp: {run_ts}")
	orig_res_dir=cfg.get('RESULTS_DIR','results'); orig_tuner_dir_n=os.path.basename(cfg.get('KERAS_TUNER_DIR','keras_tuner_dir'))
	run_res_dir=os.path.join(orig_res_dir,run_ts); run_tuner_dir=os.path.join(run_res_dir,orig_tuner_dir_n)
	try: os.makedirs(run_res_dir,exist_ok=True); logger.info(f"Created run results dir: {run_res_dir}")
	except OSError as e: logger.error(f"Fatal Err: Cannot create run dir '{run_res_dir}'. Exit. Err: {e}"); sys.exit(1)
	cfg['RESULTS_DIR']=run_res_dir; cfg['KERAS_TUNER_DIR']=run_tuner_dir; cfg['RUN_IDENTIFIER']=run_ts
	set_seeds(cfg['RANDOM_SEED'])
	logger.info("--- Stage 1: Load & Prep Data ---")
	df:Optional[pd.DataFrame]=None; data_params:Optional[Dict[str,Any]]=None
	train_df:Optional[pd.DataFrame]=None; val_df:Optional[pd.DataFrame]=None; test_df:Optional[pd.DataFrame]=None
	try:
		df,data_params=data_loader.load_and_prepare_data(fpath=cfg['CSV_FILE_PATH'],dt_col=cfg['DATE_COLUMN'],val_col=cfg['VALUE_COLUMN'],cfg=cfg)
		if df is None or data_params.get('status')!='Loaded Successfully': raise ValueError(f"Data load failed. Status: {data_params.get('status','Unknown')}")
		train_df,val_df,test_df=data_loader.split_data_train_val_test(df,cfg['VALIDATION_SIZE'],cfg['TEST_SIZE'])
		logger.info("Data load, prep, split OK.")
	except Exception as e: logger.error("--- Fatal Err: Data Load/Split Failed ---",exc_info=True); logger.error(f"Err details: {e}\nCheck config & CSV."); sys.exit(1)
	logger.info("--- Stage 2: Init Results Storage ---")
	run_params:Dict[str,Any]={'config_settings':cfg,'data_load_summary':data_params,'models':{},'final_forecast_runs':{}}
	eval_fcsts:Dict[str,pd.DataFrame]={}; eval_res_list:List[Dict[str,Any]]=[]; eval_models:Dict[str,Any]={}; future_fcsts:Dict[str,pd.DataFrame]={}
	test_periods:int=len(test_df); test_index:pd.DatetimeIndex=test_df.index
	logger.info(f"Test set: {test_periods} periods from {test_index.min()} to {test_index.max()}.")
	logger.info("--- Stage 3: Run Models (Eval Run) ---")
	models_to_run=cfg.get('MODELS_TO_RUN',[])
	nan_eval_fcst=pd.DataFrame(np.nan,index=test_index,columns=['yhat','yhat_lower','yhat_upper'])
	if 'SARIMA' in models_to_run:
		m_name='SARIMA'; start_tm=time.time(); logger.info(f"--- Run {m_name} (Eval) ---")
		fcst_df:Optional[pd.DataFrame]=None; params:Optional[Dict[str,Any]]=None; model_obj:Optional[Any]=None
		try:
			fcst_df,params,model_obj=sarima_model.run_sarima(train_data=train_df.copy(),test_periods=test_periods,test_index=test_index,cfg=cfg)
			eval_fcsts[m_name]=fcst_df
			if model_obj and cfg.get('SAVE_TRAINED_MODELS'): eval_models[m_name]=model_obj
		except Exception as e:
			logger.error(f"Unexpected err {m_name} main loop (Eval): {e}",exc_info=True); eval_fcsts[m_name]=nan_eval_fcst.copy()
			if params is None: params={'model_type':m_name}; params['run_error']=f"Main Loop Err: {e}"
		runtime=time.time()-start_tm; logger.info(f"--- {m_name} (Eval) Finished. Runtime: {runtime:.2f}s ---")
		if params: params['runtime_seconds']=runtime; run_params['models'][m_name]=params
		else: run_params['models'][m_name]={'model_type':m_name,'runtime_seconds':runtime,'status':'Params missing'}
	if 'Prophet' in models_to_run:
		m_name='Prophet'; start_tm=time.time(); logger.info(f"--- Run {m_name} (Eval) ---")
		fcst_df:Optional[pd.DataFrame]=None; params:Optional[Dict[str,Any]]=None; model_obj:Optional[Any]=None
		try:
			fcst_df,params,model_obj=prophet_model.run_prophet(train_data=train_df.copy(),test_periods=test_periods,test_index=test_index,cfg=cfg)
			eval_fcsts[m_name]=fcst_df
			if model_obj and cfg.get('SAVE_TRAINED_MODELS'): eval_models[m_name]=model_obj
		except Exception as e:
			logger.error(f"Unexpected err {m_name} main loop (Eval): {e}",exc_info=True); eval_fcsts[m_name]=nan_eval_fcst.copy()
			if params is None: params={'model_type':m_name}; params['run_error']=f"Main Loop Err: {e}"
		runtime=time.time()-start_tm; logger.info(f"--- {m_name} (Eval) Finished. Runtime: {runtime:.2f}s ---")
		if params: params['runtime_seconds']=runtime; run_params['models'][m_name]=params
		else: run_params['models'][m_name]={'model_type':m_name,'runtime_seconds':runtime,'status':'Params missing'}
	nn_types=[m for m in ['RNN','LSTM'] if m in models_to_run]
	for m_name in nn_types:
		start_tm=time.time(); logger.info(f"--- Run {m_name} (Eval) ---")
		fcst_ser:Optional[pd.Series]=None; params:Optional[Dict[str,Any]]=None; model_obj:Optional[Model]=None
		try:
			fcst_ser,params,model_obj=nn_models.run_nn_model(m_type=m_name,train_data=train_df.copy(),val_data=val_df.copy() if val_df is not None else None,test_periods=test_periods,test_index=test_index,cfg=cfg)
			eval_fcsts[m_name]=pd.DataFrame({'yhat':fcst_ser,'yhat_lower':np.nan,'yhat_upper':np.nan})
			if model_obj and cfg.get('SAVE_TRAINED_MODELS'): eval_models[m_name]=model_obj
		except Exception as e:
			logger.error(f"Unexpected err {m_name} main loop (Eval): {e}",exc_info=True); eval_fcsts[m_name]=nan_eval_fcst.copy()
			if params is None: params={'model_type':m_name}; params['run_error']=f"Main Loop Err: {e}"
		runtime=time.time()-start_tm; logger.info(f"--- {m_name} (Eval) Finished. Runtime: {runtime:.2f}s ---")
		if params: params['runtime_seconds']=runtime; run_params['models'][m_name]=params
		else: run_params['models'][m_name]={'model_type':m_name,'runtime_seconds':runtime,'status':'Params missing'}
	logger.info("--- Stage 4: Evaluating Forecasts ---")
	point_fcsts_df=pd.DataFrame(index=test_index)
	for model,res_df in eval_fcsts.items():
		if isinstance(res_df,pd.DataFrame) and 'yhat' in res_df: point_fcsts_df[model]=res_df['yhat']
		else: logger.warning(f"No 'yhat' for {model}. Skip eval.")
	metrics_to_calc=cfg['EVALUATION_METRICS']; eval_df:Optional[pd.DataFrame]=None
	for m_name in point_fcsts_df.columns:
		if m_name not in run_params['models']: logger.warning(f"'{m_name}' in fcsts but not param store. Skip eval linkage."); continue
		m_params=run_params['models'][m_name]; nan_metrics={'Model':m_name,**{m:np.nan for m in metrics_to_calc}}
		if point_fcsts_df[m_name].isnull().all():
			logger.warning(f"Skip eval {m_name}: All fcsts are NaN.")
			m_params['evaluation_metrics']={m:np.nan for m in metrics_to_calc}; m_params['evaluation_status']='Skipped - All NaN fcsts'; eval_res_list.append(nan_metrics)
		else:
			try:
				eval_mets:Dict[str,Any]=evaluation.evaluate_forecast(y_true=test_df['y'],y_pred=point_fcsts_df[m_name],model_name=m_name,metrics_list=metrics_to_calc)
				m_params['evaluation_metrics']={k:v for k,v in eval_mets.items() if k!='Model'}; m_params['evaluation_status']='Success'; eval_res_list.append(eval_mets)
			except Exception as e: logger.error(f"Could not eval {m_name}. Err: {e}",exc_info=True); m_params['evaluation_metrics']={m:np.nan for m in metrics_to_calc}; m_params['evaluation_status']=f'Error: {e}'; eval_res_list.append(nan_metrics)
	if eval_res_list:
		eval_df=pd.DataFrame(eval_res_list); runtimes={m:run_params['models'].get(m,{}).get('runtime_seconds',np.nan) for m in eval_df['Model']}
		eval_df['Runtime (s)']=eval_df['Model'].map(runtimes); eval_df=eval_df.set_index('Model')
	else: logger.warning("No models evaluated."); eval_df=pd.DataFrame()
	logger.info("--- Stage 5: Display Eval Results ---")
	logger.info("\n"+"="*20+" Point Forecasts (Test Set) "+"="*20)
	try:
		with pd.option_context('display.max_rows',None,'display.max_columns',None,'display.width',1000): logger.info(f"\n{point_fcsts_df}")
	except Exception as disp_err: logger.error(f"Err display point fcsts: {disp_err}"); logger.info(point_fcsts_df.to_string())
	logger.info("\n"+"="*20+" Evaluation Metrics "+"="*20)
	if not eval_df.empty:
		if 'Runtime (s)' in eval_df.columns: eval_df['Runtime (s)']=pd.to_numeric(eval_df['Runtime (s)'],errors='coerce').map('{:.2f}'.format)
		try:
			with pd.option_context('display.max_rows',None,'display.max_columns',None,'display.width',1000,'display.float_format','{:.4f}'.format): logger.info(f"\n{eval_df}")
		except Exception as disp_err: logger.error(f"Err display eval metrics: {disp_err}"); logger.info(eval_df.to_string(float_format="%.4f"))
	else: logger.info("No models evaluated.")
	logger.info("--- Stage 6: Save Eval Run Results ---")
	res_dir=cfg['RESULTS_DIR']; save_res=cfg['SAVE_RESULTS']
	if save_res:
		logger.info(f"Saving eval results to: '{res_dir}'")
		try:
			if cfg.get('SAVE_MODEL_PARAMETERS',False): param_file=os.path.join(res_dir,"run_parameters.json"); save_run_params(run_params,param_file)
			if not eval_df.empty: eval_path=os.path.join(res_dir,"evaluation_metrics.csv"); eval_df.to_csv(eval_path,float_format='%.6f'); logger.info(f"Saved eval metrics: {eval_path}")
			else: logger.info("Skip eval metrics save (no results).")
			if not point_fcsts_df.empty: points_path=os.path.join(res_dir,"point_forecasts.csv"); point_fcsts_df.to_csv(points_path,float_format='%.6f'); logger.info(f"Saved eval point fcsts: {points_path}")
			else: logger.info("Skip eval point fcsts save (no results).")
			all_eval_fcsts_saved=False
			for m_name,fcst_df in eval_fcsts.items():
				if isinstance(fcst_df,pd.DataFrame) and not fcst_df.empty and not fcst_df.isnull().all().all():
					fcst_path=os.path.join(res_dir,f"full_forecast_{m_name}.csv"); fcst_df.to_csv(fcst_path,float_format='%.6f')
					logger.info(f"Saved eval full fcst {m_name}: {fcst_path}"); all_eval_fcsts_saved=True
				else: logger.debug(f"Skip save eval full fcst {m_name} (empty/NaN/not DF).")
			if not all_eval_fcsts_saved: logger.warning("No eval full forecasts saved.")
			if cfg.get('SAVE_TRAINED_MODELS',False):
				models_dir=os.path.join(res_dir,'saved_models'); os.makedirs(models_dir,exist_ok=True); logger.info(f"Saving eval models to: {models_dir}")
				n_saved=0
				for m_name,model_obj in eval_models.items():
					if model_obj:
						try:
							sp=os.path.join(models_dir,f"model_{m_name}")
							if m_name=='SARIMA': sarima_model.save_sarima_model(model_obj,sp+".pkl")
							elif m_name=='Prophet': prophet_model.save_prophet_model(model_obj,sp+".pkl")
							elif m_name in ['RNN','LSTM']:
								nn_models.save_nn_model(model_obj,sp)
								nn_p=run_params.get('models',{}).get(m_name,{}); scaler=nn_p.get('scaler_object_ref')
								if isinstance(scaler,MinMaxScaler): scaler_p=os.path.join(models_dir,f"scaler_{m_name}.joblib"); joblib.dump(scaler,scaler_p); logger.info(f"Saved eval NN scaler {m_name}: {scaler_p}")
								else: logger.warning(f"Could not find/verify scaler for eval NN {m_name}")
							else: logger.warning(f"Unknown eval model type {m_name}"); continue
							n_saved+=1
						except Exception as save_err: logger.error(f"Failed save eval model {m_name}: {save_err}",exc_info=True)
				logger.info(f"Finished saving eval models ({n_saved}/{len(eval_models)} saved).")
		except Exception as e: logger.error(f"Error during eval results saving: {e}",exc_info=True)
	else: logger.info("Skip Eval Results Save (SAVE_RESULTS=False).")
	logger.info("--- Stage 7: Final Forecasting ---")
	if cfg['RUN_FINAL_FORECAST']:
		logger.info(f"Start final forecast for {cfg['FORECAST_HORIZON']} periods.")
		if df is None: logger.error("Cannot run final forecast: Orig DF 'df' missing."); cfg['RUN_FINAL_FORECAST']=False
		else:
			full_train=df.copy(); horizon=cfg['FORECAST_HORIZON']; final_models:Dict[str,Any]={}
			future_idx:Optional[pd.DatetimeIndex]=None; last_dt=full_train.index[-1]; freq=full_train.index.freq
			if freq:
				try: future_idx=pd.date_range(start=last_dt+freq,periods=horizon,freq=freq); logger.info(f"Generated future index for {horizon} periods after {last_dt}.")
				except Exception as freq_err: logger.error(f"Err gen future date range: {freq_err}. Final fcst skipped.",exc_info=True); cfg['RUN_FINAL_FORECAST']=False
			else: logger.error("Cannot gen future index: Freq missing. Final fcst skipped."); cfg['RUN_FINAL_FORECAST']=False
			if future_idx is not None:
				models_final=cfg.get('MODELS_TO_RUN',[])
				for m_name in models_final:
					start_tm=time.time(); logger.info(f"--- Run Final Fcst {m_name} ---")
					fcst_df:Optional[pd.DataFrame]=None; params:Optional[Dict[str,Any]]=None; model_obj:Optional[Any]=None
					empty_fcst=pd.DataFrame(np.nan,index=future_idx,columns=['yhat','yhat_lower','yhat_upper'])
					try:
						if m_name=='SARIMA':
							_,params,trained_m=sarima_model.run_sarima(train_data=full_train,test_periods=1,test_index=pd.DatetimeIndex([future_idx[0]]),cfg=cfg)
							if trained_m:
								final_models[m_name]=trained_m; logger.info(f"SARIMA retrained. Gen {horizon} step fcst...")
								pred=trained_m.get_forecast(steps=horizon); fcst_summ=pred.summary_frame(alpha=1.0-cfg.get('PROPHET_INTERVAL_WIDTH',0.95))
								fcst_summ.index=future_idx; fcst_df=fcst_summ[['mean','mean_ci_lower','mean_ci_upper']].rename(columns={'mean':'yhat','mean_ci_lower':'yhat_lower','mean_ci_upper':'yhat_upper'})
							else: logger.error("SARIMA retraining failed."); fcst_df=empty_fcst.copy()
						elif m_name=='Prophet':
							fcst_df,params,model_obj=prophet_model.run_prophet(train_data=full_train,test_periods=horizon,test_index=future_idx,cfg=cfg)
							if model_obj: final_models[m_name]=model_obj
							if fcst_df is not None:
								if 'yhat_lower' not in fcst_df: fcst_df['yhat_lower']=np.nan
								if 'yhat_upper' not in fcst_df: fcst_df['yhat_upper']=np.nan
							else: fcst_df=empty_fcst.copy()
						elif m_name in ['RNN','LSTM']:
							final_nn_cfg=cfg.copy(); final_nn_cfg['USE_KERAS_TUNER']=False; logger.info(f"Temp disable KerasTuner for final {m_name} retrain.")
							fcst_ser,params,model_obj=nn_models.run_nn_model(m_type=m_name,train_data=full_train,val_data=None,test_periods=horizon,test_index=future_idx,cfg=final_nn_cfg)
							if isinstance(fcst_ser,pd.Series): fcst_df=pd.DataFrame({'yhat':fcst_ser,'yhat_lower':np.nan,'yhat_upper':np.nan},index=future_idx)
							else: fcst_df=empty_fcst.copy()
							if model_obj: final_models[m_name]=model_obj
						if fcst_df is None: fcst_df=empty_fcst.copy()
						future_fcsts[m_name]=fcst_df.reindex(columns=['yhat','yhat_lower','yhat_upper'])
					except Exception as e:
						logger.error(f"Unexpected err final fcst {m_name}: {e}",exc_info=True); future_fcsts[m_name]=empty_fcst.copy()
						if params is None: params={'model_type':m_name,'status':'Failed'}; params['final_forecast_run_error']=f"Main Loop Err: {e}"
					runtime=time.time()-start_tm; logger.info(f"--- {m_name} Final Fcst Finished. Runtime: {runtime:.2f}s ---")
					if params: params['final_forecast_runtime_seconds']=runtime; run_params['final_forecast_runs'][m_name]=params
				logger.info("\n"+"="*20+" Future Point Forecasts "+"="*20)
				future_points_df=pd.DataFrame(index=future_idx)
				for model,res_df in future_fcsts.items():
					if isinstance(res_df,pd.DataFrame) and 'yhat' in res_df: future_points_df[model]=res_df['yhat']
				try:
					with pd.option_context('display.max_rows',None,'display.max_columns',None,'display.width',1000): logger.info(f"\n{future_points_df}")
				except Exception as disp_err: logger.error(f"Err display future point fcsts: {disp_err}"); logger.info(future_points_df.to_string())
				if save_res:
					logger.info(f"Saving future forecasts to: '{res_dir}'")
					try:
						if not future_points_df.empty: points_path=os.path.join(res_dir,"future_point_forecasts.csv"); future_points_df.to_csv(points_path,float_format='%.6f'); logger.info(f"Saved future point fcsts: {points_path}")
						for m_name,fcst_df in future_fcsts.items():
							if isinstance(fcst_df,pd.DataFrame) and not fcst_df.empty and not fcst_df.isnull().all().all():
								fcst_path=os.path.join(res_dir,f"future_full_forecast_{m_name}.csv"); fcst_df.to_csv(fcst_path,float_format='%.6f'); logger.info(f"Saved full future fcst {m_name}: {fcst_path}")
						if cfg.get('SAVE_TRAINED_MODELS',False):
							final_models_dir=os.path.join(res_dir,'saved_final_models'); os.makedirs(final_models_dir,exist_ok=True); logger.info(f"Saving final models (full data) to: {final_models_dir}")
							n_saved=0
							for m_name,model_obj in final_models.items():
								if model_obj:
									try:
										sp=os.path.join(final_models_dir,f"model_{m_name}")
										if m_name=='SARIMA': sarima_model.save_sarima_model(model_obj,sp+".pkl")
										elif m_name=='Prophet': prophet_model.save_prophet_model(model_obj,sp+".pkl")
										elif m_name in ['RNN','LSTM']:
											nn_models.save_nn_model(model_obj,sp)
											final_nn_p=run_params.get('final_forecast_runs',{}).get(m_name,{}); scaler=final_nn_p.get('scaler_object_ref')
											if isinstance(scaler,MinMaxScaler): scaler_p=os.path.join(final_models_dir,f"scaler_{m_name}.joblib"); joblib.dump(scaler,scaler_p); logger.info(f"Saved final NN scaler {m_name}: {scaler_p}")
											else: logger.warning(f"Could not find/verify scaler for final NN {m_name}")
										else: logger.warning(f"Unknown final model type {m_name}")
										n_saved+=1
									except Exception as save_err: logger.error(f"Failed save final model {m_name}: {save_err}",exc_info=True)
							logger.info(f"Finished saving final models ({n_saved}/{len(final_models)} saved).")
					except Exception as e: logger.error(f"Error during future results saving: {e}",exc_info=True)
	else: logger.info("Skip Final Forecast step (RUN_FINAL_FORECAST=False or freq err).")
	logger.info("--- Stage 8: Generating Plots ---")
	plot_train_val=pd.concat([train_df,val_df]) if val_df is not None else train_df.copy()
	full_data=df.copy() if df is not None else plot_train_val
	eval_metrics_df=eval_df
	try:
		plotting.generate_all_plots(train_df=plot_train_val,test_df=test_df,full_df=full_data,eval_fcsts=eval_fcsts,future_fcsts=future_fcsts,metrics_df=eval_metrics_df,cfg=cfg)
	except ImportError: logger.error("Plotting skipped: matplotlib missing.")
	except Exception as e: logger.error(f"Could not generate plots. Err: {e}",exc_info=True)
	run_end_time=time.time(); total_runtime=run_end_time-run_start_time
	logger.info("--- Forecast Comparison Finished ---"); logger.info(f"Total Run Time: {total_runtime:.2f} seconds"); logger.info(f"Results saved in: {cfg['RESULTS_DIR']}")
	if cfg['SAVE_RESULTS'] and cfg.get('SAVE_MODEL_PARAMETERS',False):
		run_params['total_runtime_seconds']=total_runtime; param_file=os.path.join(cfg['RESULTS_DIR'],"run_parameters.json"); save_run_params(run_params,param_file)

if __name__=="__main__": run_comparison()