import numpy as np, pandas as pd, tensorflow as tf, logging, os, json
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, Input, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam, RMSprop, SGD, Adagrad
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, Optional, List, Union, Any, Dict
_kt_avail = False
try: import keras_tuner as kt; _kt_avail = True
except ImportError: pass
logger=logging.getLogger(__name__)

def create_sequences(data:Optional[np.ndarray],n_steps:int)->Tuple[np.ndarray,np.ndarray]:
	if data is None or len(data)<=n_steps: return np.array([]),np.array([])
	X, y = [], []
	for i in range(len(data)-n_steps): X.append(data[i:i+n_steps]); y.append(data[i+n_steps])
	if not X: return np.array([]),np.array([])
	return np.array(X),np.array(y).flatten()

def get_optimizer(name:str,lr:Optional[float]=None)->keras.optimizers.Optimizer:
	name=name.lower(); kwargs={'learning_rate':lr} if lr is not None else {}
	if name=='adam': return Adam(**kwargs)
	if name=='rmsprop': return RMSprop(**kwargs)
	if name=='sgd': return SGD(**kwargs)
	if name=='adagrad': return Adagrad(**kwargs)
	logger.warning(f"Unsupported optimizer '{name}'. Using Adam."); return Adam()

def build_hypermodel(hp:kt.HyperParameters,m_type:str,n_steps:int,cfg:Dict[str,Any],n_features:int=1)->Model:
	model=Sequential()
	model.add(Input(shape=(n_steps,n_features)))
	hp_units:int=hp.Int('units',min_value=cfg['NN_TUNER_HP_UNITS_MIN'],max_value=cfg['NN_TUNER_HP_UNITS_MAX'],step=cfg['NN_TUNER_HP_UNITS_STEP'])
	hp_act:str=hp.Choice('activation',values=cfg['NN_TUNER_HP_ACTIVATION_CHOICES'])
	layer_name=f"tuned_{m_type}_layer"
	if m_type=='RNN': model.add(SimpleRNN(units=hp_units,activation=hp_act,name=layer_name))
	elif m_type=='LSTM': model.add(LSTM(units=hp_units,activation=hp_act,name=layer_name))
	else: raise ValueError("Invalid m_type. Use 'RNN' or 'LSTM'.")
	if cfg['NN_TUNER_HP_USE_DROPOUT']:
		hp_drop_rate=hp.Float('dropout_rate',min_value=cfg['NN_TUNER_HP_DROPOUT_MIN'],max_value=cfg['NN_TUNER_HP_DROPOUT_MAX'],step=cfg['NN_TUNER_HP_DROPOUT_STEP'])
		model.add(Dropout(rate=hp_drop_rate,name="tuned_dropout"))
	model.add(Dense(1,name="output_dense"))
	hp_lr:float=hp.Float('learning_rate',min_value=cfg['NN_TUNER_HP_LR_MIN'],max_value=cfg['NN_TUNER_HP_LR_MAX'],sampling='log')
	hp_opt:str=hp.Choice('optimizer',values=cfg['NN_TUNER_HP_OPTIMIZER_CHOICES'])
	optimizer=get_optimizer(hp_opt,hp_lr)
	model.compile(optimizer=optimizer,loss=cfg['NN_LOSS_FUNCTION'],metrics=['mae'])
	return model

def train_nn(m_type:str,train_data:pd.DataFrame,val_data:Optional[pd.DataFrame],cfg:Dict[str,Any])->Tuple[Optional[Model],Optional[MinMaxScaler],Optional[Dict[str,Any]]]:
	logger.info(f"--- Starting {m_type} Train ---")
	params:Dict[str,Any]={'model_type':m_type}
	n_steps=cfg['NN_STEPS']; use_tuner=cfg['USE_KERAS_TUNER']
	best_model:Optional[Model]=None; scaler:Optional[MinMaxScaler]=None
	if train_data is None or train_data.empty or train_data['y'].isnull().all():
		logger.warning(f"Skip {m_type}: No train data."); params['status']='Skipped - No training data'
		return None,None,params
	scaler=MinMaxScaler(feature_range=(0,1)); scaled_train:Optional[np.ndarray]=None; scaled_val:Optional[np.ndarray]=None
	try:
		train_y=train_data[['y']].dropna()
		if train_y.empty: logger.warning(f"Skip {m_type}: No non-NaN train data."); params['status']='Skipped - No non-NaN training data'; return None,None,params
		scaled_train=scaler.fit_transform(train_y)
		params.update({'scaler_type':'MinMaxScaler','scaler_feature_range':(0,1),'scaler_object_ref':scaler})
		logger.info(f"Train data scaled. Shape: {scaled_train.shape}")
		if val_data is not None and not val_data.empty and not val_data['y'].isnull().all():
			val_y=val_data[['y']].dropna()
			if not val_y.empty: scaled_val=scaler.transform(val_y); logger.info(f"Val data scaled. Shape: {scaled_val.shape}")
			else: logger.warning("Val data only NaNs after dropna(). Val disabled."); scaled_val=None
		else: scaled_val=None
		tuner_obj=cfg.get('NN_TUNER_OBJECTIVE','val_loss')
		if use_tuner and tuner_obj.startswith('val_') and scaled_val is None:
			logger.error(f"Tuner obj '{tuner_obj}' needs val data, none available. Abort {m_type}."); params['status']='Error - Tuner needs valid val data'; return None,scaler,params
	except Exception as e: logger.error(f"Err scaling {m_type}: {e}. Skip.",exc_info=True); params['status']=f'Error - Scaling failed: {e}'; return None,scaler,params
	X_train,y_train=create_sequences(scaled_train,n_steps)
	X_val,y_val=None,None; val_fit_data:Optional[Tuple[np.ndarray,np.ndarray]]=None
	if scaled_val is not None:
		X_val_t,y_val_t=create_sequences(scaled_val,n_steps)
		if X_val_t.size>0 and y_val_t.size>0:
			X_val,y_val=X_val_t,y_val_t; logger.info(f"Created val seqs: X_val {X_val.shape}, y_val {y_val.shape}"); val_fit_data=(X_val,y_val)
		else: logger.warning(f"Could not create val seqs (Val len {len(scaled_val)} <= n_steps {n_steps}?). Val disabled.")
	if X_train.size==0: logger.error(f"Skip {m_type}: Not enough train data ({len(scaled_train)}) for seqs (n_steps={n_steps})."); params['status']='Error - Not enough data for sequences'; return None,scaler,params
	logger.info(f"Created train seqs: X_train {X_train.shape}, y_train {y_train.shape}"); params['n_steps']=n_steps
	n_features:int=1; X_train=X_train.reshape((X_train.shape[0],X_train.shape[1],n_features))
	if val_fit_data: X_val=X_val.reshape((X_val.shape[0],X_val.shape[1],n_features)); val_fit_data=(X_val,y_val)
	es_patience=cfg['NN_EARLY_STOPPING_PATIENCE']; final_callbacks:List[keras.callbacks.Callback]=[]; es_cb=None
	if es_patience>0:
		mon='val_loss' if val_fit_data else 'loss'
		es_cb=EarlyStopping(monitor=mon,patience=es_patience,verbose=1,mode='min',restore_best_weights=True)
		final_callbacks.append(es_cb); logger.info(f"ES enabled for final fit ({mon}, patience {es_patience})."); params.update({'early_stopping_patience':es_patience,'early_stopping_monitor':mon})
	else: logger.info("ES disabled for final fit."); params['early_stopping_patience']=0
	tf.random.set_seed(cfg['RANDOM_SEED']); epochs=cfg['RNN_EPOCHS'] if m_type=='RNN' else cfg['LSTM_EPOCHS']
	batch_size=cfg['NN_BATCH_SIZE']; verbose=cfg['NN_VERBOSE']; loss_func=cfg['NN_LOSS_FUNCTION']
	params.update({'batch_size':batch_size,'loss_function':loss_func})
	if use_tuner:
		if not _kt_avail: logger.error(f"USE_KERAS_TUNER=True, but KerasTuner missing. Abort {m_type}."); params['status']='Error - KerasTuner missing'; return None,scaler,params
		logger.info(f"--- KerasTuner Search for {m_type} ---"); params['tuning_attempted']=True
		tuner_obj=cfg['NN_TUNER_OBJECTIVE']
		if tuner_obj.startswith('val_') and not val_fit_data: logger.error(f"FATAL: Tuner obj '{tuner_obj}' needs val data, none available. Abort."); params['status']='Error - Tuner obj needs valid val data'; return None,scaler,params
		model_builder=lambda hp: build_hypermodel(hp,m_type=m_type,n_steps=n_steps,cfg=cfg,n_features=n_features)
		tuner_type=cfg['NN_TUNER_TYPE']; tuner_cls_map={'RandomSearch':kt.RandomSearch,'Hyperband':kt.Hyperband,'BayesianOptimization':kt.BayesianOptimization}
		tuner_cls=tuner_cls_map.get(tuner_type)
		if not tuner_cls: logger.warning(f"Unsupported NN_TUNER_TYPE '{tuner_type}'. Using RandomSearch."); tuner_cls,tuner_type=kt.RandomSearch,'RandomSearch'
		proj_name=f"{cfg['NN_TUNER_PROJECT_NAME_PREFIX']}_{m_type}"; tuner_dir=cfg['KERAS_TUNER_DIR']; tuner_ovr=cfg['KERAS_TUNER_OVERWRITE']
		max_trials=cfg['NN_TUNER_MAX_TRIALS']; exec_per_trial=cfg['NN_TUNER_EXECUTIONS_PER_TRIAL']; tuner_epochs=cfg['NN_TUNER_EPOCHS']
		params['tuner_config']={'type':tuner_type,'objective':tuner_obj,'max_trials':max_trials,'executions_per_trial':exec_per_trial,'epochs_per_trial':tuner_epochs,'directory':tuner_dir,'project_name':proj_name,'overwrite':tuner_ovr,'seed':cfg['RANDOM_SEED']}
		tuner=tuner_cls(model_builder,objective=kt.Objective(tuner_obj,direction="min"),max_trials=max_trials,executions_per_trial=exec_per_trial,directory=tuner_dir,project_name=proj_name,seed=cfg['RANDOM_SEED'],overwrite=tuner_ovr)
		logger.info(f"--- Tuner ({tuner_type}) Search Space ---"); tuner.search_space_summary()
		logger.info(f"--- Starting Tuner Search (Max Trials: {max_trials}, Epochs/Trial: {tuner_epochs}) ---")
		logger.info(f"Tuner results in: {os.path.join(tuner_dir,proj_name)}")
		search_callbacks:List[keras.callbacks.Callback]=[]
		if es_patience>0:
			search_es=EarlyStopping(monitor=tuner_obj,patience=es_patience,verbose=1,mode='min',restore_best_weights=False)
			search_callbacks.append(search_es); logger.info(f"ES during search active ({tuner_obj}, patience {es_patience})")
		try:
			tuner.search(X_train,y_train,epochs=tuner_epochs,validation_data=val_fit_data,batch_size=batch_size,callbacks=search_callbacks,verbose=verbose)
			params['tuning_status']='Completed'
		except Exception as e: logger.error(f"KerasTuner search failed {m_type}: {e}",exc_info=True); params.update({'tuning_status':f'Failed: {e}','status':'Error - Tuner search failed'}); return None,scaler,params
		logger.info(f"\n--- KerasTuner Search Complete for {m_type} ---")
		try:
			best_hps_list=tuner.get_best_hyperparameters(num_trials=1)
			if not best_hps_list: logger.error(f"Error: Tuner found no best HPs for {m_type}."); params.update({'tuning_status':'Completed - No best HPs','status':'Error - No best HPs from tuner'}); return None,scaler,params
			best_hps=best_hps_list[0]; params['best_hyperparameters']=best_hps.values
			logger.info(f"\n--- Best HPs Found ({m_type}) ---"); [logger.info(f"  - {p}: {v}") for p,v in sorted(best_hps.values.items())]
			logger.info(f"\n--- Building Best {m_type} Model from Tuner ---"); best_model=tuner.hypermodel.build(best_hps)
		except Exception as e: logger.error(f"Error retrieving/building from tuner: {e}",exc_info=True); params.update({'tuning_status':f'Completed - Error processing: {e}','status':'Error - Tuner result processing failed'}); return None,scaler,params
	else:
		logger.info(f"--- Building {m_type} Manually (Tuner Disabled) ---"); params['tuning_attempted']=False
		manual_model=Sequential(name=f"Manual_{m_type}"); manual_model.add(Input(shape=(n_steps,n_features)))
		units=cfg['NN_UNITS']; activation=cfg['NN_ACTIVATION']; opt_name=cfg['NN_OPTIMIZER']
		add_drop=cfg['NN_ADD_DROPOUT']; drop_rate=cfg['NN_DROPOUT_RATE']
		layer_name=f"manual_{m_type}_layer"
		if m_type=='RNN': manual_model.add(SimpleRNN(units=units,activation=activation,name=layer_name))
		elif m_type=='LSTM': manual_model.add(LSTM(units=units,activation=activation,name=layer_name))
		if add_drop: manual_model.add(Dropout(rate=drop_rate,name="manual_dropout"))
		manual_model.add(Dense(1,name="output_dense"))
		opt_inst=get_optimizer(opt_name); manual_model.compile(optimizer=opt_inst,loss=loss_func,metrics=['mae']); best_model=manual_model
		params['manual_config']={'units':units,'activation':activation,'optimizer':opt_name,'add_dropout':add_drop,'dropout_rate':drop_rate if add_drop else None}
	if best_model is not None:
		logger.info(f"\n--- Final Training ({m_type}, Max Epochs: {epochs}) ---")
		logger.info(f"Final Model Arch Summary ({m_type}):"); best_model.summary(print_fn=logger.info); params['final_epochs_configured']=epochs
		try:
			history=best_model.fit(X_train,y_train,epochs=epochs,validation_data=val_fit_data,batch_size=batch_size,callbacks=final_callbacks,verbose=verbose)
			logger.info(f"--- {m_type} Final Training Complete ---"); params['status']='Trained Successfully'
			actual_epochs=len(history.epoch); params['final_epochs_run']=actual_epochs; final_metrics=history.history
			if es_cb and es_cb.stopped_epoch>0:
				best_val=es_cb.best; mon=es_cb.monitor; best_epoch=es_cb.best_epoch+1
				logger.info(f"Final train stopped early at epoch {actual_epochs}. Best from epoch {best_epoch}.")
				logger.info(f"Best ({mon}): {best_val:.4f}"); params.update({'early_stopping_triggered':True,'best_monitored_value':float(best_val),'best_epoch_number':best_epoch})
			else:
				logger.info(f"Final training ran {actual_epochs} epochs."); params['early_stopping_triggered']=False
				if 'loss' in final_metrics and final_metrics['loss']: params['final_train_loss']=float(final_metrics['loss'][-1])
				if 'val_loss' in final_metrics and final_metrics['val_loss']: params['final_val_loss']=float(final_metrics['val_loss'][-1])
				if 'mae' in final_metrics and final_metrics['mae']: params['final_train_mae']=float(final_metrics['mae'][-1])
				if 'val_mae' in final_metrics and final_metrics['val_mae']: params['final_val_mae']=float(final_metrics['val_mae'][-1])
			return best_model,scaler,params
		except Exception as e: logger.error(f"Error {m_type} final fit: {e}",exc_info=True); params['status']=f'Error - Final fit failed: {e}'; return None,scaler,params
	else: logger.error(f"Error: No {m_type} model for final training."); params['status']='Error - No model for final training'; return None,scaler,params

def forecast_nn(model:Optional[Model],scaler:Optional[MinMaxScaler],train_data:pd.DataFrame,test_periods:int,n_steps:int,m_type:str)->np.ndarray:
	logger.info(f"--- Generating {m_type} forecast ({test_periods} periods) ---")
	n_features:int=1; forecasts=np.full(test_periods,np.nan)
	if model is None or scaler is None: logger.error(f"{m_type} Fcst Err: Model/scaler unavailable."); return forecasts
	train_y_init=train_data[['y']].dropna()
	if len(train_y_init)<n_steps: logger.error(f"{m_type} Fcst Err: Not enough data ({len(train_y_init)}) for init (need {n_steps})."); return forecasts
	try: scaled_train_tail=scaler.transform(train_y_init.iloc[-n_steps:])
	except Exception as e: logger.error(f"{m_type} Fcst Err: Scaling train tail failed: {e}",exc_info=True); return forecasts
	forecasts_scaled:List[float]=[]; current_input:List[List[float]]=scaled_train_tail.tolist()
	logger.info(f"Iterative prediction for {test_periods} steps...")
	for i in range(test_periods):
		try:
			curr_batch:np.ndarray=np.array(current_input).reshape((1,n_steps,n_features))
			pred_scaled:np.ndarray=model.predict(curr_batch,verbose=0)
			pred_val:float=pred_scaled[0,0]; forecasts_scaled.append(pred_val)
			current_input.pop(0); current_input.append([pred_val])
		except Exception as e: logger.error(f"Error {m_type} fcst step {i+1}/{test_periods}: {e}",exc_info=True); break
	num_fcsts=len(forecasts_scaled); logger.info(f"Generated {num_fcsts} scaled forecast(s).")
	if forecasts_scaled:
		fcsts_scaled_arr=np.array(forecasts_scaled).reshape(-1,1)
		try: fcsts_unscaled=scaler.inverse_transform(fcsts_scaled_arr).flatten(); forecasts[:num_fcsts]=fcsts_unscaled; logger.info("Inverse scaling OK.")
		except Exception as e: logger.error(f"Error inverse scaling {m_type}: {e}. Fcsts incomplete/NaN.",exc_info=True)
	logger.info(f"--- {m_type} Forecasting Finished ---")
	return forecasts

def save_nn_model(model:Model,file_path:str):
	try:
		os.makedirs(os.path.dirname(file_path),exist_ok=True)
		fmt='.keras' if hasattr(tf,'__version__') and tf.__version__>='2.7' else '.h5'
		fpath_ext=os.path.splitext(file_path)[0]+fmt
		model.save(fpath_ext); logger.info(f"Keras model saved: {fpath_ext} (format: {fmt})")
	except Exception as e: logger.error(f"Err saving Keras model {fpath_ext}: {e}",exc_info=True)

def run_nn_model(m_type:str,train_data:pd.DataFrame,val_data:Optional[pd.DataFrame],test_periods:int,test_index:pd.DatetimeIndex,cfg:Dict[str,Any])->Tuple[pd.Series,Optional[Dict[str,Any]],Optional[Model]]:
	model:Optional[Model]=None; scaler:Optional[MinMaxScaler]=None
	params:Optional[Dict[str,Any]]={'model_type':m_type,'status':'Not Run','scaler_object_ref':None}
	fcst_vals:np.ndarray=np.full(test_periods,np.nan)
	try:
		model,scaler,params=train_nn(m_type,train_data,val_data,cfg)
		if model and scaler and params and params.get('status')=='Trained Successfully':
			fcst_vals=forecast_nn(model=model,scaler=scaler,train_data=train_data,test_periods=test_periods,n_steps=cfg['NN_STEPS'],m_type=m_type)
		else: status_msg=params.get('status','Unknown Failure') if params else 'Init Failed'; logger.warning(f"{m_type} train failed ({status_msg}), skipping fcst.")
	except Exception as e:
		logger.error(f"{m_type} Error run_nn_model: {e}",exc_info=True)
		if params: params.update({'status':'Error - Overall Run Failed','run_error':str(e)})
	fcst_series=pd.Series(fcst_vals,index=test_index,name=m_type)
	if params is None: params={'model_type':m_type,'status':'Failed before param capture'}
	return fcst_series,params,model