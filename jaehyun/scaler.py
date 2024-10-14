import numpy as np


class LogScaler:

    def __init__(self):
        pass

    def fit_transform(
        self,
        X: np.ndarray
    ):

        return np.log(1 + X)

    def inverse_transform(
        self,
        X_new
    ):

        return np.exp(X_new) - 1

class TimeseriesMinMaxScaler:

    def __init__(self):

        self.max: float
        self.min: float
        self.scale = 1000

    def fit_transform(
        self,
        X: np.ndarray
    ):

        self.max = np.max(X)
        self.min = np.min(X)

        X_new = (X - self.min)/(self.max - self.min)
        X_new = X_new*self.scale

        return X_new

    def inverse_transform(
        self,
        X_new
    ):

        X_new = X_new/self.scale

        return (X_new*(self.max - self.min) + self.min)
