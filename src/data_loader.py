import pandas as pd, os, logging
from typing import Tuple, Optional, Dict, Any
logger=logging.getLogger(__name__)

def apply_imputation(df:pd.DataFrame,method:str,val_col:str='y')->pd.DataFrame:
	if method=='none' or method is None: return df
	nan0=df[val_col].isnull().sum()
	if nan0==0: logger.info("No NaNs, imputation skipped."); return df
	logger.info(f"Apply imputation '{method}' to '{val_col}' ({nan0} NaNs).")
	df_imp=df.copy()
	if method=='ffill': df_imp[val_col]=df_imp[val_col].ffill()
	elif method=='bfill': df_imp[val_col]=df_imp[val_col].bfill()
	elif method=='mean': fill_val=df_imp[val_col].mean(); df_imp[val_col]=df_imp[val_col].fillna(fill_val); logger.info(f"Imputed mean: {fill_val:.4f}")
	elif method=='median': fill_val=df_imp[val_col].median(); df_imp[val_col]=df_imp[val_col].fillna(fill_val); logger.info(f"Imputed median: {fill_val:.4f}")
	elif method=='interpolate': df_imp[val_col]=df_imp[val_col].interpolate(method='linear',limit_direction='both')
	else: logger.warning(f"Unknown impute method '{method}'. Not applied."); return df
	nan1=df_imp[val_col].isnull().sum()
	if nan1>0: logger.warning(f"Imputation '{method}' done, but {nan1} NaNs remain.")
	else: logger.info("Imputation complete.")
	return df_imp

def load_and_prepare_data(fpath:str,dt_col:str,val_col:str,cfg:Dict[str,Any])->Tuple[Optional[pd.DataFrame],Optional[Dict[str,Any]]]:
	logger.info(f"Loading data: {fpath}")
	params:Dict[str,Any]={'file_path':fpath,'date_column_config':dt_col,'value_column_config':val_col,'requested_freq':cfg['TIME_SERIES_FREQUENCY'],'imputation_method':cfg['DATA_IMPUTATION_METHOD']}
	if not os.path.exists(fpath): logger.error(f"Fatal Err: File '{fpath}' not found."); params['status']='Error - File not found'; return None,params
	try:
		df:pd.DataFrame=pd.read_csv(fpath,parse_dates=[dt_col],index_col=dt_col)
		logger.info(f"CSV loaded. Cols: {df.columns.tolist()}")
		if val_col not in df.columns: logger.error(f"Fatal Err: Val col '{val_col}' not in CSV cols: {df.columns.tolist()}"); params['status']=f"Error - Value column '{val_col}' not found"; return None,params
		df=df[[val_col]].rename(columns={val_col:'y'})
	except KeyError as e: logger.error(f"KeyError CSV load: {e}. Check DATE_COLUMN ('{dt_col}')."); params['status']=f'Error - KeyError load index/date: {e}'; return None,params
	except ValueError as e: logger.error(f"ValueError CSV load/parse: {e}. Check date fmt in '{dt_col}'."); params['status']=f'Error - ValueError load/parse: {e}'; return None,params
	except Exception as e: logger.error(f"Error load/process CSV: {e}",exc_info=True); params['status']=f'Error - CSV Load/Process: {e}'; return None,params
	df=df.sort_index()
	if not isinstance(df.index,pd.DatetimeIndex): logger.error("Fatal Err: Index not DatetimeIndex."); params['status']='Error - Index not DatetimeIndex'; return None,params
	if not pd.api.types.is_numeric_dtype(df['y']):
		logger.warning(f"Val col '{val_col}' ('y') not numeric. Converting.")
		orig_dtype=df['y'].dtype; df['y']=pd.to_numeric(df['y'],errors='coerce')
		cnv_nan=df['y'].isnull().sum(); params['value_col_original_dtype']=str(orig_dtype)
		if cnv_nan>0: logger.warning(f"Coercion to numeric -> {cnv_nan} NaNs.")
	imp_method=cfg['DATA_IMPUTATION_METHOD']; nan0=df['y'].isnull().sum(); params['nan_count_before_imputation']=nan0
	if nan0>0 and imp_method!='none': df=apply_imputation(df,imp_method,'y'); params['nan_count_after_imputation']=df['y'].isnull().sum()
	elif imp_method=='none' and nan0>0: logger.warning(f"Data has {nan0} NaNs; imputation 'none'. Some models may fail.")
	spec_freq=cfg['TIME_SERIES_FREQUENCY']; orig_freq=None
	try: orig_freq=pd.infer_freq(df.index); logger.info(f"Inferred freq: {orig_freq}")
	except Exception as infer_err: logger.warning(f"Could not infer freq: {infer_err}")
	final_freq:Optional[str]=None
	if spec_freq:
		logger.info(f"Freq specified: {spec_freq}")
		if orig_freq==spec_freq: logger.info("Data freq matches specified."); final_freq=spec_freq
		elif orig_freq: logger.warning(f"Inferred freq ('{orig_freq}') differs from specified ('{spec_freq}'). Using inferred."); final_freq=orig_freq
		else: logger.warning(f"Could not infer freq; try set specified '{spec_freq}'."); final_freq=spec_freq
	elif orig_freq: logger.info(f"Using inferred freq: {orig_freq}"); final_freq=orig_freq
	else: logger.warning("Could not infer/find specified freq. Freq models might error.")
	if final_freq:
		try:
			df_copy=df.copy(); df_copy.index=pd.DatetimeIndex(df_copy.index,freq=final_freq); df=df_copy
			if df.index.freqstr!=final_freq: logger.warning(f"Attempted set freq '{final_freq}', but index.freqstr is '{df.index.freqstr}'.")
		except ValueError as set_freq_err: logger.error(f"Failed set freq '{final_freq}': {set_freq_err}. Freq remains None."); final_freq=None; df.index.freq=None
	else: df.index.freq=None
	final_freq_str:Optional[str]=df.index.freqstr; logger.info(f"Final freq set on DF index: {final_freq_str}")
	params['final_frequency_set']=final_freq_str
	nan_final=df['y'].isnull().sum(); data_len=len(df)
	if nan_final>0: logger.warning(f"Data still has {nan_final} NaNs.")
	params.update({'nan_count_final':nan_final,'data_start_date':df.index.min().strftime('%Y-%m-%d %H:%M:%S'),'data_end_date':df.index.max().strftime('%Y-%m-%d %H:%M:%S'),'num_observations':data_len,'status':'Loaded Successfully'})
	logger.info(f"Data loaded: {data_len} obs from {params['data_start_date']} to {params['data_end_date']}.")
	logger.debug(f"Final data head:\n{df.head()}")
	return df,params

def split_data_train_val_test(df:pd.DataFrame,val_size:int,test_size:int)->Tuple[pd.DataFrame,Optional[pd.DataFrame],pd.DataFrame]:
	n:int=len(df); min_train_size=1
	if val_size<0 or test_size<=0: raise ValueError("val_size>=0 and test_size>0 required")
	if n<test_size+val_size+min_train_size: raise ValueError(f"Not enough data ({n}) for val({val_size}), test({test_size}), min train({min_train_size}).")
	test_idx:int=n-test_size; val_idx:int=test_idx-val_size
	test_df:pd.DataFrame=df.iloc[test_idx:]; train_df:pd.DataFrame; val_df:Optional[pd.DataFrame]=None
	if val_size>0:
		if val_idx<0: raise ValueError(f"Val size ({val_size}) too large.")
		val_df=df.iloc[val_idx:test_idx]; train_df=df.iloc[:val_idx]
		if train_df.empty: raise ValueError(f"Split resulted in empty train (n={n}, val={val_size}, test={test_size}).")
		logger.info(f"Split data: Train {len(train_df)} ({train_df.index.min()} - {train_df.index.max()})")
		logger.info(f"          : Val {len(val_df)} ({val_df.index.min()} - {val_df.index.max()})")
	else:
		if test_idx<=0: raise ValueError(f"Test size ({test_size}) too large, no train data (n={n}).")
		train_df=df.iloc[:test_idx]
		logger.info(f"Split data: Train {len(train_df)} ({train_df.index.min()} - {train_df.index.max()})")
		logger.info("          : Validation disabled (size=0).")
	logger.info(f"          : Test {len(test_df)} ({test_df.index.min()} - {test_df.index.max()})")
	return train_df,val_df,test_df