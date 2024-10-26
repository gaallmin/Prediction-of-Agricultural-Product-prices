from sklearn.linear_model import Lasso
from sklearn.pipeline import make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import VotingRegressor

import numpy as np


class LastPredictor(BaseEstimator, RegressorMixin):
    def fit(self, X, y=None):
        # Calculate the mean of each feature and store it
        self.mean_ = np.mean(X, axis=1)
        return self

    def predict(self, X):
        # Return the stored mean as a prediction for each sample

        return X[:, -1].flatten()

class LastLassoRegressor(BaseEstimator, RegressorMixin):
    def __init__(self):

        self.first = LastPredictor()

        lasso = [Lasso(alpha=1, tol=1e-7, selection='random')]*10
        self.second = VotingRegressor(
            estimators=[
                (f'second_{i}', lasso[i]) for i in range(10)
            ],
        )


    def fit(self, X, y):

        self.first.fit(X, y)
        residuals = y - self.first.predict(X)
        self.second.fit(X, residuals)

        return self

    def predict(self, X):

        first_preds = self.first.predict(X)
        second_preds = self.second.predict(X)
        return first_preds + second_preds


class ThreeLassoRegressor(BaseEstimator, RegressorMixin):
    def __init__(self):

        self.model = LastLassoRegressor()

    def fit(self, X, y):

        self.model.fit(X, y[:, 0])
        return self

    def predict(self, X):

        preds = self.model.predict(X)

        print(X.shape)
        print(preds.shape)

        new_X = np.vstack((X, [preds]))
        raise ValueError("test")

        return (1 - self.mean_percentage)*common_preds + self.mean_percentage*mean_preds



class CommonCombinedRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, no_common: bool = False):

        self.no_common = no_common
        self.mean_percentage = 1

        if not self.no_common:
            self.mean_percentage = 0
            self.common_model = LastLassoRegressor()

        self.mean_model = LastLassoRegressor()

    def fit(self, X, y):

        if not self.no_common:
            self.common_model.fit(X['common'], y['common'])

        self.mean_model.fit(X['mean'], y['mean'])

        return self

    def predict(self, X):

        if not self.no_common:
            common_preds = self.common_model.predict(X['common'])
        else:
            common_preds = 0

        mean_preds = self.mean_model.predict(X['mean'])

        return (1 - self.mean_percentage)*common_preds + self.mean_percentage*mean_preds

    def predict_common(self, X):
        if not self.no_common:
            return self.common_model.predict(X)
        else:
            return 0

    def predict_mean(self, X):
        return self.mean_model.predict(X)
