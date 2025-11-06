
import numpy as np
def mae(y, yhat): return float(np.mean(np.abs(y - yhat)))
def rmse(y, yhat): return float(np.sqrt(np.mean((y - yhat)**2)))
def mape(y, yhat):
    y = np.asarray(y); yhat = np.asarray(yhat)
    return float(np.mean(np.abs((y - yhat) / np.maximum(1e-9, np.abs(y))))) * 100.0
def r2(y, yhat):
    y = np.asarray(y); yhat = np.asarray(yhat)
    ss_res = np.sum((y - yhat)**2)
    ss_tot = np.sum((y - np.mean(y))**2) + 1e-12
    return float(1 - ss_res/ss_tot)
def directional_accuracy(y_prev, y, yhat):
    dy_true = y - y_prev; dy_pred = yhat - y_prev
    return float(np.mean(np.sign(dy_true) == np.sign(dy_pred))) * 100.0
def summarize(y, yhat, y_prev):
    return {"MAE": mae(y,yhat), "RMSE": rmse(y,yhat), "MAPE": mape(y,yhat), "R2": r2(y,yhat),
            "DirectionalAccuracy(%)": directional_accuracy(y_prev, y, yhat)}
