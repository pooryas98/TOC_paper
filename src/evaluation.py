import numpy as np, pandas as pd, logging
from sklearn.metrics import mean_absolute_error, mean_squared_error
from typing import Dict, Any, List, Optional
logger=logging.getLogger(__name__)

def calculate_mape(y_true:np.ndarray,y_pred:np.ndarray)->float:
	y_true,y_pred=np.array(y_true,dtype=float),np.array(y_pred,dtype=float)
	mask=(y_true!=0)&(~np.isnan(y_true))&(~np.isnan(y_pred))
	removed=len(y_true)-np.sum(mask)
	if removed>0: logger.debug(f"MAPE Calc: Ignored {removed} pair(s) (zero/NaN).")
	y_true_f,y_pred_f=y_true[mask],y_pred[mask]
	if len(y_true_f)==0: logger.warning("MAPE Warn: No valid points after filter."); return np.nan
	try: return np.mean(np.abs((y_true_f-y_pred_f)/y_true_f))*100
	except ZeroDivisionError: logger.error("MAPE Err: Division by zero."); return np.nan

def evaluate_forecast(y_true:pd.Series,y_pred:pd.Series,model_name:str,metrics_list:List[str])->Dict[str,Any]:
	res:Dict[str,Any]={'Model':model_name}
	y_pred_aligned:pd.Series=y_pred.reindex(y_true.index)
	valid_mask:pd.Series=~y_pred_aligned.isnull()
	y_true_eval:np.ndarray=y_true[valid_mask].values; y_pred_eval:np.ndarray=y_pred_aligned[valid_mask].values
	n_valid=len(y_true_eval); n_total=len(y_true)
	res['Num_Forecast_Points']=n_total; res['Num_Valid_Points_Eval']=n_valid
	if n_valid==0:
		logger.warning(f"{model_name} Eval Warn: No valid (non-NaN) preds. Cannot calc metrics.")
		for metric in metrics_list: res[metric]=np.nan
		return res
	if n_valid<n_total: logger.warning(f"{model_name} Eval Warn: Only {n_valid}/{n_total} points used for eval (NaN preds).")
	log_parts=[]; errors=[]
	if 'MAE' in metrics_list:
		try: mae=mean_absolute_error(y_true_eval,y_pred_eval); res['MAE']=mae; log_parts.append(f"MAE: {mae:.4f}")
		except Exception as e: res['MAE']=np.nan; errors.append(f"MAE Err: {e}")
	if 'RMSE' in metrics_list:
		try: rmse=np.sqrt(mean_squared_error(y_true_eval,y_pred_eval)); res['RMSE']=rmse; log_parts.append(f"RMSE: {rmse:.4f}")
		except Exception as e: res['RMSE']=np.nan; errors.append(f"RMSE Err: {e}")
	if 'MAPE' in metrics_list:
		try: mape=calculate_mape(y_true_eval,y_pred_eval); res['MAPE']=mape; log_parts.append(f"MAPE: {mape:.4f}%")
		except Exception as e: res['MAPE']=np.nan; errors.append(f"MAPE Err: {e}")
	if log_parts: logger.info(f"{model_name} Eval - {', '.join(log_parts)}")
	if errors: logger.error(f"{model_name} Eval Errors: {'; '.join(errors)}")
	return res