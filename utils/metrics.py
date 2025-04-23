import numpy as np


def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))

def MAPE_zeroPro(pred, true):
    zeroPro = np.divide((pred - true), true, out=np.zeros_like(true), where=true!=0)
    return np.mean(np.abs(zeroPro))

def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))

def MSPE_zeroPro(pred, true):
    zeroPro = np.divide((pred - true), true, out=np.zeros_like(true), where=true!=0)
    return np.mean(np.abs(zeroPro))

def MAPE_Mask(pred, true, mask):
    """
        Mean absolute percentage. Assumes ``y >= 0``.
        Defined as ``(y - y_pred).abs() / y.abs()``
    """

    loss = np.abs(pred - true) / (np.abs(true) + 1)
    loss *= mask
    non_zero_len = mask.sum()
    return np.sum(loss)/non_zero_len

def MSPE_Mask(pred, true, mask):

    loss = np.square(pred - true) / (np.abs(true) + 1)
    loss *= mask
    non_zero_len = mask.sum()
    return np.sum(loss)/non_zero_len

def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)

    return mae, mse, rmse, mape, mspe

def metric_Mask(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    real_y_true_mask = (1 - (true == 0))
    mape = MAPE_Mask(pred, true, real_y_true_mask)
    mspe = MSPE_Mask(pred, true, real_y_true_mask)
    return mae, mse, rmse, mape, mspe