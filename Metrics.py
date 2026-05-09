# Import required dependencies
import numpy as np

def MAE(y_true: np.ndarray, y_pred: np.ndarray) -> np.float64:
    """ Implementation of the Mean Absolute Error (MAE) """
    sum = 0

    for i in range(len(y_true)):
        sum += abs(y_true[i] - y_pred[i])

    return sum / len(y_true)

def MSE(y_true: np.ndarray, y_pred: np.ndarray) -> np.float64:
    """ Implementation of the Mean Squared Error (MSE) """
    sum = 0

    for i in range(len(y_true)):
        sum += (y_true[i] - y_pred[i]) ** 2

    return sum / len(y_true)

def R2(y_true: np.ndarray, y_pred: np.ndarray) -> np.float64:
    """ Implementation of the R2 metric """
    mean_true = 0

    for i in range(len(y_true)):
        mean_true += y_true[i] / len(y_true) 

    regressor = 0
    mean_based_estimator = 0

    for i in range(len(y_true)):
        regressor += (y_true[i] - y_pred[i]) ** 2
        mean_based_estimator += (y_true[i] - mean_true) ** 2

    return 1 - (regressor / mean_based_estimator)

def Corr(y_true: np.ndarray, y_pred: np.ndarray) -> np.float64:
    """ Implementation of the Pearson's Correlation Coefficient """
    mean_true = 0
    mean_pred = 0

    for i in range(len(y_true)):
        mean_true += y_true[i] / len(y_true)
        mean_pred += y_pred[i] / len(y_true)

    var_true = 0
    var_pred = 0
    cov_true_pred = 0

    for i in range(len(y_true)):
        var_true += (y_true[i] - mean_true) ** 2 / len(y_true)
        var_pred += (y_pred[i] - mean_pred) ** 2 / len(y_true)
        cov_true_pred += (y_true[i] - mean_true) * (y_pred[i] - mean_pred) / len(y_true)

    return cov_true_pred / np.sqrt(var_true * var_pred)
