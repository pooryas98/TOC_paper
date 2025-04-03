import numpy as np, pandas as pd, tensorflow as tf, logging, os, json
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, Input, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam, RMSprop, SGD, Adagrad
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, Optional, List, Union, Any, Dict

# Keras Tuner is optional
_kt_avail = False
try:
    import keras_tuner as kt
    _kt_avail = True
except ImportError:
    pass # Keras Tuner not installed, USE_KERAS_TUNER=True will fail later if set

logger=logging.getLogger(__name__)

def create_sequences(data:Optional[np.ndarray],n_steps:int)->Tuple[np.ndarray,np.ndarray]:
	"""Creates input sequences and corresponding target values."""
	if data is None or len(data)<=n_steps:
		logger.debug(f"Data length ({len(data) if data is not None else 0}) <= n_steps ({n_steps}). Cannot create sequences.")
		return np.array([]),np.array([])
	X, y = [], []
	for i in range(len(data)-n_steps):
		X.append(data[i:i+n_steps]);
		y.append(data[i+n_steps])
	if not X:
		logger.debug("No sequences created from data.")
		return np.array([]),np.array([])
	return np.array(X),np.array(y).flatten()

def get_optimizer(name:str,lr:Optional[float]=None)->keras.optimizers.Optimizer:
	"""Gets a Keras optimizer instance by name."""
	name=name.lower();
	kwargs={'learning_rate':lr} if lr is not None else {}
	if name=='adam': return Adam(**kwargs)
	if name=='rmsprop': return RMSprop(**kwargs)
	if name=='sgd': return SGD(**kwargs)
	if name=='adagrad': return Adagrad(**kwargs)
	logger.warning(f"Unsupported optimizer '{name}'. Using Adam with default learning rate.")
	# Return Adam without kwargs if lr wasn't specified or optimizer was unknown
	return Adam() if lr is None else Adam(learning_rate=lr)


def build_hypermodel(hp:kt.HyperParameters,m_type:str,n_steps:int,cfg:Dict[str,Any],n_features:int=1)->Model:
	"""Builds the Keras model structure for hyperparameter tuning."""
	model=Sequential(name=f"Tuned_{m_type}") # Give the model a name
	model.add(Input(shape=(n_steps,n_features)))

	# Tunable hyperparameters
	hp_units:int=hp.Int('units',min_value=cfg['NN_TUNER_HP_UNITS_MIN'],max_value=cfg['NN_TUNER_HP_UNITS_MAX'],step=cfg['NN_TUNER_HP_UNITS_STEP'])
	hp_act:str=hp.Choice('activation',values=cfg['NN_TUNER_HP_ACTIVATION_CHOICES'])

	layer_name=f"tuned_{m_type}_layer"
	if m_type=='RNN': model.add(SimpleRNN(units=hp_units,activation=hp_act,name=layer_name))
	elif m_type=='LSTM': model.add(LSTM(units=hp_units,activation=hp_act,name=layer_name))
	else: raise ValueError("Invalid m_type in build_hypermodel. Use 'RNN' or 'LSTM'.")

    # Optional Tunable Dropout
	if cfg['NN_TUNER_HP_USE_DROPOUT']:
		# Check if dropout hyperparameters are actually defined in config (provide defaults if missing)
		min_drop = cfg.get('NN_TUNER_HP_DROPOUT_MIN', 0.1)
		max_drop = cfg.get('NN_TUNER_HP_DROPOUT_MAX', 0.5)
		step_drop = cfg.get('NN_TUNER_HP_DROPOUT_STEP', 0.1)
		hp_drop_rate=hp.Float('dropout_rate',min_value=min_drop,max_value=max_drop,step=step_drop)
		model.add(Dropout(rate=hp_drop_rate,name="tuned_dropout"))

	model.add(Dense(1,name="output_dense")) # Output layer

    # Tunable optimizer and learning rate
	hp_lr:float=hp.Float('learning_rate',min_value=cfg['NN_TUNER_HP_LR_MIN'],max_value=cfg['NN_TUNER_HP_LR_MAX'],sampling='log')
	hp_opt:str=hp.Choice('optimizer',values=cfg['NN_TUNER_HP_OPTIMIZER_CHOICES'])
	optimizer=get_optimizer(hp_opt,hp_lr)

	model.compile(optimizer=optimizer,loss=cfg['NN_LOSS_FUNCTION'],metrics=['mae']) # Use MAE as a metric
	return model

def train_nn(m_type:str,train_data:pd.DataFrame,val_data:Optional[pd.DataFrame],cfg:Dict[str,Any])->Tuple[Optional[Model],Optional[MinMaxScaler],Optional[Dict[str,Any]]]:
	"""Trains an RNN or LSTM model, optionally using Keras Tuner."""
	logger.info(f"--- Starting {m_type} Train ---")
	params:Dict[str,Any]={'model_type':m_type} # Initialize parameters dictionary
	n_steps=cfg['NN_STEPS']; use_tuner=cfg['USE_KERAS_TUNER']
	best_model:Optional[Model]=None; scaler:Optional[MinMaxScaler]=None

	# --- 1. Data Preparation and Scaling ---
	if train_data is None or train_data.empty or train_data['y'].isnull().all():
		logger.warning(f"Skipping {m_type} training: No valid training data provided."); params['status']='Skipped - No training data'
		return None,None,params

	scaler=MinMaxScaler(feature_range=(0,1)); # Initialize scaler
	scaled_train:Optional[np.ndarray]=None; scaled_val:Optional[np.ndarray]=None
	try:
		# Drop NaNs before scaling
		train_y=train_data[['y']].dropna()
		if train_y.empty:
		    logger.warning(f"Skipping {m_type} training: No non-NaN training data after dropna()."); params['status']='Skipped - No non-NaN training data'; return None,None,params

		scaled_train=scaler.fit_transform(train_y)
		params.update({'scaler_type':'MinMaxScaler','scaler_feature_range':(0,1),'scaler_data_min': float(scaler.data_min_[0]), 'scaler_data_max': float(scaler.data_max_[0]), 'scaler_object_ref':scaler})
		logger.info(f"Training data scaled. Shape: {scaled_train.shape}")

		# Scale validation data if available
		if val_data is not None and not val_data.empty and not val_data['y'].isnull().all():
			val_y=val_data[['y']].dropna()
			if not val_y.empty:
			    scaled_val=scaler.transform(val_y);
			    logger.info(f"Validation data scaled. Shape: {scaled_val.shape}")
			else:
			    logger.warning("Validation data contains only NaNs after dropna(). Validation disabled for training/tuning.")
			    scaled_val=None # Ensure it's None if only NaNs
		else:
		    logger.info("No validation data provided or it's empty/all NaN. Validation disabled for training/tuning.")
		    scaled_val=None # Ensure it's None

		# Check if tuner requires validation data when none is available
		tuner_obj=cfg.get('NN_TUNER_OBJECTIVE','val_loss')
		if use_tuner and tuner_obj.startswith('val_') and scaled_val is None:
			logger.error(f"KerasTuner objective '{tuner_obj}' requires validation data, but none is available or valid. Aborting {m_type} training."); params['status']='Error - Tuner needs valid validation data'; return None,scaler,params

	except Exception as e:
	    logger.error(f"Error during data scaling for {m_type}: {e}. Skipping training.",exc_info=True); params['status']=f'Error - Scaling failed: {e}'; return None,scaler,params

	# --- 2. Create Sequences ---
	X_train,y_train=create_sequences(scaled_train,n_steps)
	X_val,y_val=None,None; val_fit_data:Optional[Tuple[np.ndarray,np.ndarray]]=None

	if scaled_val is not None:
		X_val_t,y_val_t=create_sequences(scaled_val,n_steps)
		# Ensure sequences could be created (validation set might be too small)
		if X_val_t.size>0 and y_val_t.size>0:
			X_val,y_val=X_val_t,y_val_t;
			logger.info(f"Created validation sequences: X_val shape {X_val.shape}, y_val shape {y_val.shape}")
			val_fit_data=(X_val,y_val) # Prepare validation data tuple for Keras fit/search
		else:
		    logger.warning(f"Could not create validation sequences (Validation length {len(scaled_val)} <= n_steps {n_steps}?). Validation disabled.")
		    val_fit_data = None # Disable validation if sequences couldn't be made

	if X_train.size==0:
	    logger.error(f"Skipping {m_type} training: Not enough training data ({len(scaled_train)}) to create sequences with n_steps={n_steps}."); params['status']='Error - Not enough data for sequences'; return None,scaler,params

	logger.info(f"Created training sequences: X_train shape {X_train.shape}, y_train shape {y_train.shape}"); params['n_steps']=n_steps

	# Reshape for RNN/LSTM layers: [samples, time steps, features]
	n_features:int=1; # Univariate time series
	X_train=X_train.reshape((X_train.shape[0],X_train.shape[1],n_features))
	if val_fit_data:
	    X_val=X_val.reshape((X_val.shape[0],X_val.shape[1],n_features));
	    val_fit_data=(X_val,y_val) # Update tuple with reshaped X_val

	# --- 3. Callbacks ---
	es_patience=cfg['NN_EARLY_STOPPING_PATIENCE'];
	final_callbacks:List[keras.callbacks.Callback]=[]; es_cb=None
	if es_patience > 0:
		monitor_metric='val_loss' if val_fit_data else 'loss' # Monitor validation loss if available, else training loss
		es_cb=EarlyStopping(monitor=monitor_metric,patience=es_patience,verbose=1,mode='min',restore_best_weights=True)
		final_callbacks.append(es_cb);
		logger.info(f"EarlyStopping enabled for final model fit (Monitor: '{monitor_metric}', Patience: {es_patience}).");
		params.update({'early_stopping_patience':es_patience,'early_stopping_monitor':monitor_metric})
	else:
	    logger.info("EarlyStopping disabled for final model fit (NN_EARLY_STOPPING_PATIENCE=0)."); params['early_stopping_patience']=0

	# --- 4. Model Building / Tuning ---
	tf.random.set_seed(cfg['RANDOM_SEED']); # Set seed for reproducibility
	epochs=cfg['RNN_EPOCHS'] if m_type=='RNN' else cfg['LSTM_EPOCHS']
	batch_size=cfg['NN_BATCH_SIZE']; verbose=cfg['NN_VERBOSE']; loss_func=cfg['NN_LOSS_FUNCTION']
	params.update({'batch_size':batch_size,'loss_function':loss_func})

	if use_tuner:
		# --- 4a. Keras Tuner ---
		if not _kt_avail:
		    logger.error(f"USE_KERAS_TUNER=True, but KerasTuner library is not installed. Aborting {m_type} training."); params['status']='Error - KerasTuner library missing'; return None,scaler,params

		logger.info(f"--- KerasTuner Search for {m_type} ---"); params['tuning_attempted']=True
		tuner_obj=cfg['NN_TUNER_OBJECTIVE']

		# Double-check validation data presence if needed by objective
		if tuner_obj.startswith('val_') and not val_fit_data:
		    logger.error(f"FATAL: KerasTuner objective '{tuner_obj}' requires validation data, but none is available or valid. Aborting."); params['status']='Error - Tuner objective needs valid validation data'; return None,scaler,params

		# Define the model-building function for the tuner
		model_builder=lambda hp: build_hypermodel(hp,m_type=m_type,n_steps=n_steps,cfg=cfg,n_features=n_features)

		# Get Tuner class
		tuner_type=cfg['NN_TUNER_TYPE'];
		tuner_cls_map={'RandomSearch':kt.RandomSearch,'Hyperband':kt.Hyperband,'BayesianOptimization':kt.BayesianOptimization}
		tuner_cls=tuner_cls_map.get(tuner_type)
		if not tuner_cls:
		    logger.warning(f"Unsupported NN_TUNER_TYPE '{tuner_type}'. Using RandomSearch."); tuner_cls,tuner_type=kt.RandomSearch,'RandomSearch'

		# Tuner configuration
		proj_name=f"{cfg['NN_TUNER_PROJECT_NAME_PREFIX']}_{m_type}"; tuner_dir=cfg['KERAS_TUNER_DIR']; tuner_ovr=cfg['KERAS_TUNER_OVERWRITE']
		max_trials=cfg['NN_TUNER_MAX_TRIALS']; exec_per_trial=cfg['NN_TUNER_EXECUTIONS_PER_TRIAL']; tuner_epochs=cfg['NN_TUNER_EPOCHS']

		# Store tuner config in params
		params['tuner_config']={
		    'type':tuner_type,
		    'objective':tuner_obj,
		    # Note: max_trials/max_epochs stored conditionally below based on tuner type
		    'executions_per_trial':exec_per_trial,
		    'directory':tuner_dir,
		    'project_name':proj_name,
		    'overwrite':tuner_ovr,
		    'seed':cfg['RANDOM_SEED']
		}


		# ----- CORRECTED Tuner Initialization -----
		common_args = {
		    'hypermodel': model_builder,
		    'objective': kt.Objective(tuner_obj, direction="min"),
		    'executions_per_trial': exec_per_trial,
		    'directory': tuner_dir,
		    'project_name': proj_name,
		    'seed': cfg['RANDOM_SEED'],
		    'overwrite': tuner_ovr
		}
		tuner: Optional[kt.BaseTuner] = None # Initialize tuner variable

		try:
		    if tuner_type == 'Hyperband':
		        # Hyperband needs max_epochs for its budget scheduling
		        tuner = tuner_cls(
		            **common_args,
		            max_epochs=tuner_epochs
		            # factor=3, hyperband_iterations=1 # Default Hyperband parameters, can be exposed via config if needed
		        )
		        params['tuner_config']['max_epochs_hyperband'] = tuner_epochs

		    elif tuner_type in ['BayesianOptimization', 'RandomSearch']:
		         # These tuners take max_trials
		        tuner = tuner_cls(
		            **common_args,
		            max_trials=max_trials
		        )
		        params['tuner_config']['max_trials'] = max_trials

		    else: # Fallback for safety, though validation should catch unknown types earlier
		        logger.error(f"Tuner type '{tuner_type}' initialization logic not explicitly defined. Aborting.")
		        params['status'] = f"Error - Unknown tuner type '{tuner_type}'"
		        return None, scaler, params

		except Exception as tuner_init_err:
		    logger.error(f"Failed to initialize Keras Tuner ({tuner_type}): {tuner_init_err}", exc_info=True)
		    params['status'] = f"Error - Tuner initialization failed: {tuner_init_err}"
		    return None, scaler, params
        # ----- END Tuner Initialization Correction -----


		logger.info(f"--- Tuner ({tuner_type}) Search Space Summary ---"); tuner.search_space_summary()
		logger.info(f"--- Starting Tuner Search ({tuner_type}) ---")
		logger.info(f"Objective: {tuner_obj}, Max Trials/Epochs Config: Trials={max_trials}/Epochs(HB)={tuner_epochs}, Exec/Trial: {exec_per_trial}")
		logger.info(f"Tuner results will be stored in: {os.path.join(tuner_dir,proj_name)}")

		# Callbacks for tuner search phase
		search_callbacks:List[keras.callbacks.Callback]=[]
		if es_patience>0:
			# Use a separate ES callback for the search phase, monitor the tuner objective
			# Don't restore best weights here, tuner handles selecting the best trial based on HPs
			search_es=EarlyStopping(monitor=tuner_obj,patience=es_patience,verbose=1,mode='min',restore_best_weights=False)
			search_callbacks.append(search_es);
			logger.info(f"EarlyStopping active during tuner search (Monitor: '{tuner_obj}', Patience: {es_patience})")

		# Run the search
		try:
			tuner.search(X_train,y_train,epochs=tuner_epochs,validation_data=val_fit_data,batch_size=batch_size,callbacks=search_callbacks,verbose=verbose)
			params['tuning_status']='Completed'
		except Exception as e:
		    logger.error(f"KerasTuner search failed for {m_type}: {e}",exc_info=True); params.update({'tuning_status':f'Failed: {e}','status':'Error - Tuner search failed'}); return None,scaler,params

		logger.info(f"\n--- KerasTuner Search Complete for {m_type} ---")

		# Retrieve the best model
		try:
			best_hps_list=tuner.get_best_hyperparameters(num_trials=1)
			if not best_hps_list:
			    logger.error(f"Error: KerasTuner found no best hyperparameters for {m_type}. This might happen if all trials failed."); params.update({'tuning_status':'Completed - No best HPs found','status':'Error - No best HPs from tuner'}); return None,scaler,params

			best_hps=best_hps_list[0];
			params['best_hyperparameters']=best_hps.values # Store the best HPs found
			logger.info(f"\n--- Best Hyperparameters Found ({m_type}) ---");
			for p,v in sorted(best_hps.values.items()): logger.info(f"  - {p}: {v}") # Log sorted HPs

			logger.info(f"\n--- Building Best {m_type} Model from Tuner Results ---");
			best_model=tuner.hypermodel.build(best_hps) # Build the final model using the best HPs

		except Exception as e:
		    logger.error(f"Error retrieving/building best model from tuner for {m_type}: {e}",exc_info=True); params.update({'tuning_status':f'Completed - Error processing results: {e}','status':'Error - Tuner result processing failed'}); return None,scaler,params

	else:
		# --- 4b. Manual Build ---
		logger.info(f"--- Building {m_type} Manually (USE_KERAS_TUNER=False) ---"); params['tuning_attempted']=False
		manual_model=Sequential(name=f"Manual_{m_type}");
		manual_model.add(Input(shape=(n_steps,n_features)))

		# Get manual config
		units=cfg['NN_UNITS']; activation=cfg['NN_ACTIVATION']; opt_name=cfg['NN_OPTIMIZER']
		add_drop=cfg['NN_ADD_DROPOUT']; drop_rate=cfg['NN_DROPOUT_RATE']

		layer_name=f"manual_{m_type}_layer"
		if m_type=='RNN': manual_model.add(SimpleRNN(units=units,activation=activation,name=layer_name))
		elif m_type=='LSTM': manual_model.add(LSTM(units=units,activation=activation,name=layer_name))

		if add_drop: manual_model.add(Dropout(rate=drop_rate,name="manual_dropout"))
		manual_model.add(Dense(1,name="output_dense"))

		opt_inst=get_optimizer(opt_name); # Get optimizer instance (default LR)
		manual_model.compile(optimizer=opt_inst,loss=loss_func,metrics=['mae']);
		best_model=manual_model # Assign the manually built model as the one to train

		# Store manual config in params
		params['manual_config']={'units':units,'activation':activation,'optimizer':opt_name,'add_dropout':add_drop,'dropout_rate':drop_rate if add_drop else None}

	# --- 5. Final Model Training ---
	if best_model is not None:
		logger.info(f"\n--- Final Training ({m_type}, Max Epochs: {epochs}) ---")
		logger.info(f"Final Model Architecture Summary ({m_type}):");
		best_model.summary(print_fn=logger.info); # Log model summary
		params['final_epochs_configured']=epochs # Record max epochs allowed

		try:
			# Fit the final model (either best from tuner or manually built)
			history=best_model.fit(X_train,y_train,epochs=epochs,validation_data=val_fit_data,batch_size=batch_size,callbacks=final_callbacks,verbose=verbose)
			logger.info(f"--- {m_type} Final Training Complete ---"); params['status']='Trained Successfully'

			# Record training history and early stopping info
			actual_epochs=len(history.epoch); params['final_epochs_run']=actual_epochs;
			final_metrics=history.history
			params['training_history'] = {k: [float(val) for val in v] for k, v in final_metrics.items()} # Store history

			if es_cb and es_cb.stopped_epoch > 0:
				best_val=es_cb.best; mon=es_cb.monitor; best_epoch=es_cb.best_epoch+1 # Keras reports 0-based epoch index
				logger.info(f"Final training stopped early at epoch {actual_epochs}. Best weights restored from epoch {best_epoch}.")
				logger.info(f"Best monitored value ({mon}): {best_val:.4f}");
				params.update({'early_stopping_triggered':True,'best_monitored_value':float(best_val),'best_epoch_number':best_epoch})
			else:
				logger.info(f"Final training completed {actual_epochs} epochs (EarlyStopping not triggered or disabled)."); params['early_stopping_triggered']=False
				# Log final metric values if not stopped early
				if 'loss' in final_metrics and final_metrics['loss']: params['final_train_loss']=float(final_metrics['loss'][-1])
				if 'val_loss' in final_metrics and final_metrics['val_loss']: params['final_val_loss']=float(final_metrics['val_loss'][-1])
				if 'mae' in final_metrics and final_metrics['mae']: params['final_train_mae']=float(final_metrics['mae'][-1])
				if 'val_mae' in final_metrics and final_metrics['val_mae']: params['final_val_mae']=float(final_metrics['val_mae'][-1])

			return best_model,scaler,params

		except Exception as e:
		    logger.error(f"Error during {m_type} final model fitting: {e}",exc_info=True); params['status']=f'Error - Final fit failed: {e}'; return None,scaler,params
	else:
	    # This case should ideally not be reached if logic above is correct
	    logger.error(f"Error: No {m_type} model available for final training (Tuner failed or Manual build failed)."); params['status']='Error - No model found for final training'; return None,scaler,params

def forecast_nn(model:Optional[Model],scaler:Optional[MinMaxScaler],train_data:pd.DataFrame,test_periods:int,n_steps:int,m_type:str)->np.ndarray:
	"""Generates forecasts using a trained NN model iteratively."""
	logger.info(f"--- Generating {m_type} forecast ({test_periods} periods) ---")
	n_features:int=1; # Univariate
	forecasts=np.full(test_periods,np.nan) # Initialize forecast array with NaNs

	if model is None or scaler is None:
	    logger.error(f"{m_type} Forecasting Error: Model or Scaler is missing."); return forecasts

	# Need the last 'n_steps' of training data to start forecasting
	train_y_init=train_data[['y']].dropna()
	if len(train_y_init)<n_steps:
	    logger.error(f"{m_type} Forecasting Error: Not enough historical data ({len(train_y_init)}) available to form initial sequence (need {n_steps})."); return forecasts

	try:
		# Scale the last part of the training data
		scaled_train_tail=scaler.transform(train_y_init.iloc[-n_steps:])
	except Exception as e:
	    logger.error(f"{m_type} Forecasting Error: Scaling initial sequence failed: {e}",exc_info=True); return forecasts

	# Iterative forecasting loop
	forecasts_scaled:List[float]=[];
	current_input:List[List[float]]=scaled_train_tail.tolist() # Start with the last known sequence
	logger.info(f"Starting iterative prediction for {test_periods} steps...")
	for i in range(test_periods):
		try:
			# Reshape current input for model prediction
			curr_batch:np.ndarray=np.array(current_input).reshape((1,n_steps,n_features))
			# Predict the next step (scaled)
			pred_scaled:np.ndarray=model.predict(curr_batch,verbose=0) # verbose=0 prevents print spam
			pred_val:float=pred_scaled[0,0]; # Get the single predicted value

			forecasts_scaled.append(pred_val) # Store the scaled prediction

			# Update the input sequence for the next step: remove oldest, add prediction
			current_input.pop(0);
			current_input.append([pred_val])
		except Exception as e:
		    logger.error(f"Error during {m_type} forecast step {i+1}/{test_periods}: {e}",exc_info=True);
		    logger.warning(f"Stopping forecast generation for {m_type} due to error.")
		    break # Stop forecasting if an error occurs

	num_fcsts=len(forecasts_scaled);
	logger.info(f"Generated {num_fcsts} scaled forecast(s).")

	# Inverse transform the forecasts
	if forecasts_scaled:
		fcsts_scaled_arr=np.array(forecasts_scaled).reshape(-1,1)
		try:
			fcsts_unscaled=scaler.inverse_transform(fcsts_scaled_arr).flatten();
			# Fill the forecast array, handling cases where prediction stopped early
			forecasts[:num_fcsts]=fcsts_unscaled;
			logger.info("Inverse scaling of forecasts successful.")
		except Exception as e:
		    logger.error(f"Error inverse scaling {m_type} forecasts: {e}. Forecasts may be incomplete or NaN.",exc_info=True)

	logger.info(f"--- {m_type} Forecasting Finished ---")
	return forecasts

def save_nn_model(model:Model,file_path:str):
	"""Saves the Keras model."""
	try:
		# Ensure directory exists
		os.makedirs(os.path.dirname(file_path),exist_ok=True)
		# Use .keras format if available (preferred), else fallback to .h5
		# Note: file_path provided might already have an extension, we normalize it.
		base_path = os.path.splitext(file_path)[0]
		fmt='.keras' # Keras 3 native format
		# Fallback if needed, though .keras should work widely now
		# if hasattr(tf,'__version__') and tf.__version__>='2.7': fmt='.keras'
		# else: fmt='.h5'
		fpath_ext=base_path+fmt
		model.save(fpath_ext);
		logger.info(f"Keras model saved successfully: {fpath_ext} (format: {fmt})")
	except Exception as e:
	    logger.error(f"Error saving Keras model to {fpath_ext}: {e}",exc_info=True)


def run_nn_model(m_type:str,train_data:pd.DataFrame,val_data:Optional[pd.DataFrame],test_periods:int,test_index:pd.DatetimeIndex,cfg:Dict[str,Any])->Tuple[pd.Series,Optional[Dict[str,Any]],Optional[Model]]:
	"""Orchestrates the training and forecasting for a single NN model."""
	model:Optional[Model]=None; scaler:Optional[MinMaxScaler]=None
	params:Optional[Dict[str,Any]]={'model_type':m_type,'status':'Not Run','scaler_object_ref':None} # Initialize params
	fcst_vals:np.ndarray=np.full(test_periods,np.nan) # Default to NaN forecasts

	try:
		# Train the model
		model,scaler,params=train_nn(m_type,train_data,val_data,cfg)

		# Check if training was successful before forecasting
		if model and scaler and params and params.get('status')=='Trained Successfully':
			# Generate forecasts
			fcst_vals=forecast_nn(model=model,scaler=scaler,train_data=train_data,test_periods=test_periods,n_steps=cfg['NN_STEPS'],m_type=m_type)
			params['forecast_status'] = 'Generated'
		else:
		    # Log failure and skip forecasting
		    status_msg=params.get('status','Unknown Failure') if params else 'Initialization Failed'
		    logger.warning(f"{m_type} training did not complete successfully (Status: '{status_msg}'). Skipping forecast generation.")
		    params['forecast_status'] = 'Skipped - Training Failed'

	except Exception as e:
		# Catch any unexpected errors during the train/forecast orchestration
		logger.error(f"{m_type} Error in run_nn_model orchestration: {e}",exc_info=True)
		if params: # Ensure params exists before updating
		    params.update({'status':'Error - Overall Run Failed','run_error':str(e), 'forecast_status': 'Skipped - Run Error'})
		# Model might be partially trained or None, return what we have

	# Create pandas Series for the forecast
	fcst_series=pd.Series(fcst_vals,index=test_index,name=m_type)

	# Ensure params dict is always returned, even if errors occurred early
	if params is None:
	    params={'model_type':m_type,'status':'Failed before parameter capture'}

	return fcst_series,params,model