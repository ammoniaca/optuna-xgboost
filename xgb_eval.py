from sklearn.model_selection import LeaveOneOut
import xgboost as xgb
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, max_error, r2_score
import numpy as np


def nse(y_pred, y_true):
    """Nash-Sutcliffe Efficiency (NSE) as per `Nash and Sutcliffe, 1970
    <https://doi.org/10.1016/0022-1694(70)90255-6>`_.

    """
    return 1 - (np.sum((y_true - y_pred) ** 2, axis=0, dtype=np.float64)
                / np.sum((y_true - np.mean(y_true)) ** 2, dtype=np.float64))


def xgb_regression_loocv_eval(trial_best_params: dict, X, y):
    # create loocv procedure
    cv = LeaveOneOut()
    # enumerate splits
    y_true, y_pred = list(), list()
    # run loocv procedure
    results = {}
    for train_ix, test_ix in cv.split(X):
        # split data
        X_train, X_test = X[train_ix, :], X[test_ix, :]
        y_train, y_test = y[train_ix], y[test_ix]
        # fit model
        # model = xgb.XGBRegressor(object="reg:squarederror", **trial_best_params)
        model = xgb.XGBRegressor(**trial_best_params)
        model.fit(X_train, y_train)
        # evaluate model
        yhat = model.predict(X_test)
        # store
        y_true.append(y_test[0])
        y_pred.append(yhat[0])
    results["rmse"] = root_mean_squared_error(y_true, y_pred)
    results["mae"] = mean_absolute_error(y_true, y_pred)
    results["max_error_score"] = max_error(y_true, y_pred)
    results["r2"] = r2_score(y_true, y_pred)
    results["nse"] = nse(np.array(y_pred), np.array(y_true))
    results["y_true"] = [x.tolist() for x in y_true]
    results["y_pred"] = [x.tolist() for x in y_pred]
    return results
    # print(f"rmse: {rmse}")
    # print(f"mae: {mae}")
    # print(f"max_error_score: {max_error_score}")
    # print(f"r2: {r2}")
    # print("")
