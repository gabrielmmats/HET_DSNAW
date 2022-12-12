import numpy as np
from sklearn.model_selection import ParameterGrid
from sklearn.base import clone
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
import sklearn.metrics as metrics

@ignore_warnings(category=ConvergenceWarning)
def validate_params(params, estimator, x_train, y_train, x_val, y_val):
     
    forecaster = clone(estimator).set_params(**params)
    forecaster.fit(x_train, y_train)
    y_pred = forecaster.predict(x_val)
    rmse = np.sqrt(metrics.mean_squared_error(y_val, y_pred))

    return params, rmse

def validate_ds_params(params, df, pred_results, val_index, test_index, method="dsnaw", euclidians=None):
    k = params['k']
    n = params['n']
    comb = params['comb']   
    
    y_dsnaw = np.zeros(test_index-val_index)
    for i in range(val_index, test_index):
        rmse_roc = []
        if method=="dsla":
            real_roc = df['y'].loc[euclidians[i-val_index, :k]].values
        else:
            real_roc = df['y'].loc[i-k:i-1].values
        for j in range(0, len(pred_results)):      
            pred_roc = pred_results[j]['y'].loc[i-k:i-1].values
            rmse_roc.append((j, np.sqrt(metrics.mean_squared_error(pred_roc, real_roc))))
        sorted_rmse_roc = sorted(rmse_roc, key=lambda x: x[1])
        comb_values = np.zeros(n)
        for j in range(0, n):
            comb_values[j] = pred_results[sorted_rmse_roc[j][0]]['y'].loc[i]
        if comb == 'median':
            y_dsnaw[i-val_index] = np.median(comb_values)
        else:
            y_dsnaw[i-val_index] = np.average(comb_values)
            
    rmse = np.sqrt(metrics.mean_squared_error(y_dsnaw, df['y'].loc[val_index:test_index-1].values))

    return params, rmse

def sort_euclidian_distances(test_i, df, k, lags, val_index, test_index):
    w = df['y'].loc[test_i-lags:test_i-1].values
    distances = []
    if test_i < test_index:
        idx = val_index
    else:
        idx = test_index
    for i in range(lags, idx):
        distances.append((i, np.linalg.norm(df['y'].loc[i-lags:i-1].values - w)))
    sorted_distances = sorted(distances, key=lambda x: x[1])
    return test_i, [s[0] for s in sorted_distances[:k]]
        
        
