# Import required dependencies
import numpy as np

def MAE(y_true: np.ndarray, y_pred: np.ndarray) -> np.float64:
    """ Implementation of the Mean Absolute Error (MAE) """
    return np.mean(abs(y_true - y_pred))

def MSE(y_true: np.ndarray, y_pred: np.ndarray) -> np.float64:
    """ Implementation of the Mean Squared Error (MSE) """
    return np.mean((y_true - y_pred) ** 2)

def R2(y_true: np.ndarray, y_pred: np.ndarray) -> np.float64:
    """ Implementation of the R2 metric """
    mean_true = np.mean(y_true)

    regressor = np.sum((y_true - y_pred) ** 2)
    mean_based_estimator = np.sum((y_true - mean_true) ** 2)

    return 1 - (regressor / mean_based_estimator)

def Corr(y_true: np.ndarray, y_pred: np.ndarray) -> np.float64:
    """ Implementation of the Pearson's Correlation Coefficient """
    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)

    var_true = np.mean((y_true - mean_true) ** 2)
    var_pred = np.mean((y_pred - mean_pred) ** 2)
    cov = np.mean((y_true - mean_true) * (y_pred - mean_pred))

    return cov / np.sqrt(var_true * var_pred)
